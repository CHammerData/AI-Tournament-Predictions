#!/usr/bin/env python3
"""
Phase 2a — Feature Engineering
Loads processed YAML files, reconstructs tournament matchups via bracket
structure, computes feature differentials, and exports matchup CSVs.

Outputs:
    data/processed/mens_matchups.csv
    data/processed/womens_matchups.csv

Rounds covered:
    R1 (First Round) through R4 (Elite Eight)  — exact reconstruction
    R6 (Championship)                           — exact (2 teams)
    R5 (Final Four semis)                       — skipped; requires bracket
                                                  draw data not in YAML
"""

import yaml
import pandas as pd
from pathlib import Path
from collections import defaultdict

# ── Paths ──────────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"

# ── Constants ──────────────────────────────────────────────────────────────────

# Team order within a 16-slot regional bracket; adjacent pairs play each other
# in R1, winners of adjacent pairs play in R2, etc.
SEED_ORDER = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]

# Maps round_reached -> round number in which the team was eliminated.
# "Champion" uses 7 so won_round() returns True for all rounds 1-6.
ELIM_ROUND = {
    "First Round":      1,
    "Second Round":     2,
    "Sweet Sixteen":    3,
    "Elite Eight":      4,
    "Final Four":       5,
    "Championship Game": 6,
    "Champion":         7,
}

ROUND_NAMES = {
    1: "First Round",
    2: "Second Round",
    3: "Sweet Sixteen",
    4: "Elite Eight",
    5: "Final Four",
    6: "Championship",
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def won_round(team: dict, round_num: int) -> bool:
    """Return True if team won in round_num (i.e. was eliminated after it)."""
    elim = ELIM_ROUND.get(team["tournament_result"]["round_reached"], 0)
    return elim > round_num


def parse_record(record_str) -> tuple[int, int]:
    """Parse 'W-L' string -> (wins, losses). Returns (0, 0) on bad input."""
    s = str(record_str or "")
    if "-" not in s:
        return 0, 0
    parts = s.split("-", 1)
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return 0, 0


def safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val) if val is not None else default
    except (TypeError, ValueError):
        return default


def safe_int(val, default: int = 0) -> int:
    try:
        return int(val) if val is not None else default
    except (TypeError, ValueError):
        return default

# ── Data Loading ───────────────────────────────────────────────────────────────

def load_all_teams(gender: str) -> list[dict]:
    """Load every team record from processed YAML files for one gender."""
    teams = []
    gender_dir = PROCESSED_DIR / gender
    for yaml_file in sorted(gender_dir.glob("*.yaml")):
        with open(yaml_file, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        for team in data.get("teams", []):
            teams.append(team)
    return teams

# ── Feature Extraction ─────────────────────────────────────────────────────────

def get_features(team: dict) -> dict:
    """Extract a flat numeric feature dict from a team YAML record."""
    overall_w, overall_l = parse_record(team.get("record", {}).get("overall"))
    conf_w, conf_l = parse_record(team.get("record", {}).get("conference"))
    total_g = overall_w + overall_l
    conf_g  = conf_w  + conf_l

    adv = team.get("advanced") or {}
    sos = team.get("strength_of_schedule") or {}

    # Prospect count: mens uses nba_prospects, womens uses wnba_prospects
    prospects = safe_int(team.get("nba_prospects") or team.get("wnba_prospects"))

    return {
        "team":                team["team"],
        "year":                team["year"],
        "region":              team["region"],
        "seed":                safe_int(team.get("seed")),
        "conf_champion":       int(bool(team.get("conf_champion"))),
        "conf_tourn_champion": int(bool(team.get("conf_tourn_champion"))),
        "win_pct":             overall_w / total_g if total_g > 0 else 0.0,
        "conf_win_pct":        conf_w / conf_g     if conf_g  > 0 else 0.0,
        "sos":                 safe_float(sos.get("rating")),
        "srs":                 safe_float(adv.get("srs")),
        "ortg":                safe_float(adv.get("offensive_rating")),
        "drtg":                safe_float(adv.get("defensive_rating")),
        "pace":                safe_float(adv.get("pace")),
        "efg_pct":             safe_float(adv.get("efg_pct")),
        "tov_pct":             safe_float(adv.get("tov_pct")),
        "orb_pct":             safe_float(adv.get("orb_pct")),
        "ft_rate":             safe_float(adv.get("ft_rate")),
        "prospects":           prospects,
    }

# ── Bracket Reconstruction ─────────────────────────────────────────────────────

def simulate_regional_bracket(
    bracket_order: list[dict | None],
) -> list[tuple[dict, dict, int]]:
    """
    Simulate rounds 1-4 within a single regional bracket.

    bracket_order: 16 slots ordered by SEED_ORDER position; None marks a
    missing seed (e.g. First Four casualty not recorded in the dataset).
    Adjacent pairs play each other in R1; winners advance paired against
    the next group's winner in R2, etc.

    Returns list of (winner, loser, round_num). Bye slots (None) auto-advance
    their opponent without producing a matchup row.
    """
    matchups: list[tuple[dict, dict, int]] = []
    current: list[dict | None] = bracket_order[:]
    round_num = 1

    while round_num <= 4 and len(current) > 1:
        next_round: list[dict | None] = []
        for i in range(0, len(current), 2):
            a = current[i]     if i     < len(current) else None
            b = current[i + 1] if i + 1 < len(current) else None

            if a is None and b is None:
                next_round.append(None)
            elif a is None:
                next_round.append(b)   # b gets a bye
            elif b is None:
                next_round.append(a)   # a gets a bye
            else:
                if won_round(a, round_num):
                    winner, loser = a, b
                else:
                    winner, loser = b, a
                matchups.append((winner, loser, round_num))
                next_round.append(winner)

        current = next_round
        round_num += 1

    return matchups


def reconstruct_year_matchups(
    year_teams: list[dict],
) -> list[tuple[dict, dict, int]]:
    """
    Reconstruct all reconstructable matchups for a single tournament year.
    Covers R1-R4 (within-region) and R6 (Championship).
    R5 (Final Four semis) is omitted — bracket draw data required.
    """
    matchups: list[tuple[dict, dict, int]] = []

    # ── R1-R4: within-region bracket ──────────────────────────────────────────
    by_region: dict[str, list[dict]] = defaultdict(list)
    for team in year_teams:
        by_region[team["region"]].append(team)

    for region_teams in by_region.values():
        seed_to_team = {t["seed"]: t for t in region_teams}
        # None fills missing seeds (e.g. First Four casualties); preserve
        # bracket positions so subsequent round pairings remain correct.
        bracket_order = [seed_to_team.get(s) for s in SEED_ORDER]
        real_teams = [t for t in bracket_order if t is not None]
        if len(real_teams) < 2:
            continue
        matchups.extend(simulate_regional_bracket(bracket_order))

    # ── R6: Championship ───────────────────────────────────────────────────────
    finalists = [
        t for t in year_teams
        if t["tournament_result"]["round_reached"] in ("Championship Game", "Champion")
    ]
    if len(finalists) == 2:
        a, b = finalists
        if won_round(a, 6):
            matchups.append((a, b, 6))
        else:
            matchups.append((b, a, 6))

    return matchups

# ── Matchup Row Builder ────────────────────────────────────────────────────────

def create_matchup_row(
    team_a: dict,
    team_b: dict,
    round_num: int,
    a_won: bool,
) -> dict:
    """
    Build one training row for a matchup.

    Differentials are always (A - B) oriented so that a positive value
    indicates team A has the advantage on that metric.
    Exception: seed_diff = seed_b - seed_a (lower seed = better, so positive
    means A is the higher-ranked team).
    """
    af = get_features(team_a)
    bf = get_features(team_b)

    return {
        # Metadata
        "year":                    af["year"],
        "round":                   round_num,
        "round_name":              ROUND_NAMES.get(round_num, f"Round {round_num}"),
        "team_a":                  af["team"],
        "team_b":                  bf["team"],
        "seed_a":                  af["seed"],
        "seed_b":                  bf["seed"],
        # Differentials (positive = team A has advantage)
        "seed_diff":               bf["seed"]      - af["seed"],       # lower seed = better rank
        "srs_diff":                af["srs"]        - bf["srs"],
        "sos_diff":                af["sos"]        - bf["sos"],
        "ortg_diff":               af["ortg"]       - bf["ortg"],
        "drtg_diff":               bf["drtg"]       - af["drtg"],      # lower drtg = better defense
        "pace_diff":               af["pace"]       - bf["pace"],
        "efg_pct_diff":            af["efg_pct"]    - bf["efg_pct"],
        "tov_pct_diff":            bf["tov_pct"]    - af["tov_pct"],   # lower tov = better
        "orb_pct_diff":            af["orb_pct"]    - bf["orb_pct"],
        "ft_rate_diff":            af["ft_rate"]    - bf["ft_rate"],
        "win_pct_diff":            af["win_pct"]    - bf["win_pct"],
        "conf_win_pct_diff":       af["conf_win_pct"] - bf["conf_win_pct"],
        "prospect_diff":           af["prospects"]  - bf["prospects"],
        # Raw flags
        "conf_champion_a":         af["conf_champion"],
        "conf_champion_b":         bf["conf_champion"],
        "conf_tourn_champion_a":   af["conf_tourn_champion"],
        "conf_tourn_champion_b":   bf["conf_tourn_champion"],
        # Label
        "outcome":                 1 if a_won else 0,
    }

# ── Dataset Builder ────────────────────────────────────────────────────────────

def build_matchup_dataset(teams: list[dict]) -> pd.DataFrame:
    """
    Build full matchup DataFrame from all team records.
    Each game produces two mirror rows (both team perspectives) so the
    model sees both winner-as-A and loser-as-A examples symmetrically.
    """
    rows: list[dict] = []

    by_year: dict[int, list[dict]] = defaultdict(list)
    for team in teams:
        by_year[team["year"]].append(team)

    for year in sorted(by_year):
        matchups = reconstruct_year_matchups(by_year[year])
        for winner, loser, rnd in matchups:
            rows.append(create_matchup_row(winner, loser, rnd, a_won=True))
            rows.append(create_matchup_row(loser, winner, rnd, a_won=False))

    return pd.DataFrame(rows)

# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    for gender in ("mens", "womens"):
        print(f"\n{'='*50}")
        print(f"Processing {gender} tournament data")
        print(f"{'='*50}")

        teams = load_all_teams(gender)
        print(f"  Loaded {len(teams)} team-seasons")

        df = build_matchup_dataset(teams)

        # Sanity check: each game produces 2 rows, outcome should be 50/50
        n_games   = len(df) // 2
        n_rounds  = df["round"].nunique()
        win_rate  = df["outcome"].mean()
        print(f"  Games reconstructed : {n_games}")
        print(f"  Rows (×2 mirror)    : {len(df)}")
        print(f"  Rounds covered      : {sorted(df['round'].unique())}")
        print(f"  Outcome balance     : {win_rate:.3f} (should be 0.500)")

        # Per-round breakdown
        print("\n  Matchups per round:")
        for rnd, group in df[df["outcome"] == 1].groupby("round"):
            print(f"    R{rnd} ({ROUND_NAMES.get(rnd, '?'):20s}): {len(group):3d} games")

        out_path = PROCESSED_DIR / f"{gender}_matchups.csv"
        df.to_csv(out_path, index=False)
        print(f"\n  Saved -> {out_path}")


if __name__ == "__main__":
    main()