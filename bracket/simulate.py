#!/usr/bin/env python3
"""
Phase 4a -- Bracket Simulation (Monte Carlo)

Loads the current-year team YAML, pre-computes all pairwise win probabilities,
runs N Monte Carlo simulations to derive realistic per-round reach probabilities
for every team, then outputs the deterministic "best bracket" plus the full
probability distributions and expected ESPN scores.

Outputs:
  bracket/brackets/{gender}_{year}.json
  bracket/brackets/{gender}_{year}.csv
  bracket/brackets/{gender}_{year}_mc_probs.csv

Usage:
    python bracket/simulate.py
    python bracket/simulate.py --gender mens
    python bracket/simulate.py --year 2026
    python bracket/simulate.py --sims 25000
"""

import argparse
import json
import pickle
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import yaml

# -- Paths ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR     = PROJECT_ROOT / "data" / "processed"
MODELS_DIR   = PROJECT_ROOT / "models" / "saved"
BRACKETS_DIR = Path(__file__).parent / "brackets"
BRACKETS_DIR.mkdir(parents=True, exist_ok=True)

SEED_ORDER = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]

ROUND_NAMES = {
    1: "First Round",
    2: "Second Round",
    3: "Sweet Sixteen",
    4: "Elite Eight",
    5: "Final Four",
    6: "Championship",
    7: "Champion",
}

# ESPN standard bracket scoring (points per correct pick per round)
ESPN_POINTS = {1: 10, 2: 20, 3: 40, 4: 80, 5: 160, 6: 320}

ALL_FEATURES = [
    "seed_diff", "srs_diff", "sos_diff", "ortg_diff", "drtg_diff",
    "pace_diff", "efg_pct_diff", "tov_pct_diff", "orb_pct_diff",
    "ft_rate_diff", "win_pct_diff", "conf_win_pct_diff", "prospect_diff",
    "conf_champion_a", "conf_champion_b",
    "conf_tourn_champion_a", "conf_tourn_champion_b",
]

# -- Data & Model Loading ------------------------------------------------------

def load_teams(gender: str, year: int) -> list[dict]:
    path = DATA_DIR / gender / f"{year}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No YAML for {gender}/{year}. Run scrapers first.")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["teams"]


def load_model(gender: str) -> dict:
    path = MODELS_DIR / f"{gender}_model.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No model at {path}. Run models/train.py first.")
    with open(path, "rb") as f:
        return pickle.load(f)


def _ensemble_proba(model: dict, X: np.ndarray) -> np.ndarray:
    return (model["lr"].predict_proba(X)[:, 1] +
            model["xgb"].predict_proba(X)[:, 1]) / 2

# -- Feature Extraction --------------------------------------------------------

def parse_record(s) -> tuple[int, int]:
    s = str(s or "")
    if "-" not in s:
        return 0, 0
    try:
        w, l = s.split("-", 1)
        return int(w), int(l)
    except ValueError:
        return 0, 0


def _flat_features(team: dict) -> dict:
    ow, ol = parse_record(team.get("record", {}).get("overall"))
    cw, cl = parse_record(team.get("record", {}).get("conference"))
    tg = ow + ol
    cg = cw + cl
    adv = team.get("advanced") or {}
    sos = team.get("strength_of_schedule") or {}
    return {
        "seed":                int(team.get("seed", 0)),
        "conf_champion":       int(bool(team.get("conf_champion"))),
        "conf_tourn_champion": int(bool(team.get("conf_tourn_champion"))),
        "win_pct":             ow / tg if tg > 0 else 0.0,
        "conf_win_pct":        cw / cg if cg > 0 else 0.0,
        "sos":      float(sos.get("rating") or 0),
        "srs":      float(adv.get("srs") or 0),
        "ortg":     float(adv.get("offensive_rating") or 0),
        "drtg":     float(adv.get("defensive_rating") or 0),
        "pace":     float(adv.get("pace") or 0),
        "efg_pct":  float(adv.get("efg_pct") or 0),
        "tov_pct":  float(adv.get("tov_pct") or 0),
        "orb_pct":  float(adv.get("orb_pct") or 0),
        "ft_rate":  float(adv.get("ft_rate") or 0),
        "prospects": int(team.get("nba_prospects") or team.get("wnba_prospects") or 0),
    }


def _matchup_row(af: dict, bf: dict) -> list[float]:
    return [
        bf["seed"]      - af["seed"],
        af["srs"]       - bf["srs"],
        af["sos"]       - bf["sos"],
        af["ortg"]      - bf["ortg"],
        bf["drtg"]      - af["drtg"],
        af["pace"]      - bf["pace"],
        af["efg_pct"]   - bf["efg_pct"],
        bf["tov_pct"]   - af["tov_pct"],
        af["orb_pct"]   - bf["orb_pct"],
        af["ft_rate"]   - bf["ft_rate"],
        af["win_pct"]   - bf["win_pct"],
        af["conf_win_pct"] - bf["conf_win_pct"],
        af["prospects"] - bf["prospects"],
        af["conf_champion"],
        bf["conf_champion"],
        af["conf_tourn_champion"],
        bf["conf_tourn_champion"],
    ]

# -- Pairwise probability pre-computation --------------------------------------

def precompute_probs(teams: list[dict], model: dict) -> dict[tuple[str, str], float]:
    """
    Compute P(team_i beats team_j) for every ordered pair in one batched
    model call. Returns dict keyed by (name_i, name_j).

    60 teams -> 3,540 predictions total, run once before any simulation.
    """
    feats = {t["team"]: _flat_features(t) for t in teams}
    names = [t["team"] for t in teams]

    rows, pairs = [], []
    for a in names:
        for b in names:
            if a != b:
                rows.append(_matchup_row(feats[a], feats[b]))
                pairs.append((a, b))

    probs = _ensemble_proba(model, np.array(rows))
    return {pair: float(p) for pair, p in zip(pairs, probs)}

# -- Single bracket simulation -------------------------------------------------

def _play(a: str, b: str, prob_lookup: dict, stochastic: bool) -> str:
    """Return winner name for a game between a and b."""
    p_a = prob_lookup[(a, b)]
    if stochastic:
        return a if random.random() < p_a else b
    return a if p_a >= 0.5 else b


def simulate_one(
    teams: list[dict],
    prob_lookup: dict[tuple[str, str], float],
    stochastic: bool = True,
) -> dict[str, int]:
    """
    Simulate one full bracket.
    Returns {team_name: round_reached} where round_reached is the last round
    the team WON (i.e. the champion gets 6, finalists get 5, etc.).
    Teams that lose in R1 get 0.
    """
    round_reached = {t["team"]: 0 for t in teams}

    by_region = defaultdict(list)
    for t in teams:
        by_region[t["region"]].append(t)

    regional_winners = []

    # R1-R4: within-region bracket
    for region in sorted(by_region.keys()):
        region_teams = by_region[region]
        seed_to_name = {t["seed"]: t["team"] for t in region_teams}
        bracket = [seed_to_name.get(s) for s in SEED_ORDER]  # None = bye slot

        current = bracket[:]
        for rnd in range(1, 5):
            if sum(1 for x in current if x) <= 1:
                break
            nxt = []
            for i in range(0, len(current), 2):
                a = current[i]     if i     < len(current) else None
                b = current[i + 1] if i + 1 < len(current) else None
                if a is None and b is None:
                    nxt.append(None)
                elif a is None:
                    nxt.append(b)
                elif b is None:
                    nxt.append(a)
                else:
                    w = _play(a, b, prob_lookup, stochastic)
                    round_reached[w] = max(round_reached[w], rnd)
                    nxt.append(w)
            current = nxt

        rw = next((x for x in current if x), None)
        if rw:
            regional_winners.append(rw)

    # R5: Final Four (pair by sorted region order)
    regional_winners.sort()
    ff_winners = []
    for i in range(0, len(regional_winners), 2):
        if i + 1 >= len(regional_winners):
            ff_winners.append(regional_winners[i])
            break
        a, b = regional_winners[i], regional_winners[i + 1]
        w = _play(a, b, prob_lookup, stochastic)
        round_reached[w] = max(round_reached[w], 5)
        ff_winners.append(w)

    # R6: Championship
    if len(ff_winners) == 2:
        w = _play(ff_winners[0], ff_winners[1], prob_lookup, stochastic)
        round_reached[w] = max(round_reached[w], 6)

    return round_reached

# -- Monte Carlo ---------------------------------------------------------------

def monte_carlo(
    teams: list[dict],
    prob_lookup: dict,
    n_sims: int = 10_000,
) -> dict[str, dict[int, float]]:
    """
    Run n_sims stochastic bracket simulations.
    Returns {team_name: {round: P(team_reaches_that_round_or_further)}}
    where round 7 = P(champion).
    """
    # Count how many times each team reached each round
    reach_counts: dict[str, dict[int, int]] = {
        t["team"]: defaultdict(int) for t in teams
    }

    for _ in range(n_sims):
        results = simulate_one(teams, prob_lookup, stochastic=True)
        for team, best_round in results.items():
            # Credit all rounds up to and including best_round reached
            for r in range(1, best_round + 2):   # +2: reaching the next round
                reach_counts[team][r] += 1

    # Convert counts to probabilities
    team_probs = {}
    for team, counts in reach_counts.items():
        team_probs[team] = {r: round(c / n_sims, 4) for r, c in counts.items()}

    return team_probs

# -- Deterministic bracket (for output) ----------------------------------------

def deterministic_bracket(
    teams: list[dict],
    prob_lookup: dict,
) -> tuple[list[dict], str]:
    """
    Play through the bracket deterministically (highest-probability team wins).
    Returns (games_list, champion_name).
    Each game is a dict with round, winner, loser, seeds, win_prob, upset flag.
    """
    by_region = defaultdict(list)
    for t in teams:
        by_region[t["region"]].append(t)

    seed_map = {t["team"]: t["seed"] for t in teams}
    games = []
    regional_winners = []

    for region in sorted(by_region.keys()):
        region_teams = by_region[region]
        seed_to_name = {t["seed"]: t["team"] for t in region_teams}
        bracket = [seed_to_name.get(s) for s in SEED_ORDER]

        current = bracket[:]
        for rnd in range(1, 5):
            if sum(1 for x in current if x) <= 1:
                break
            nxt = []
            for i in range(0, len(current), 2):
                a = current[i]     if i     < len(current) else None
                b = current[i + 1] if i + 1 < len(current) else None
                if a is None and b is None:
                    nxt.append(None)
                elif a is None:
                    nxt.append(b)
                elif b is None:
                    nxt.append(a)
                else:
                    p_a = prob_lookup[(a, b)]
                    w = a if p_a >= 0.5 else b
                    l = b if p_a >= 0.5 else a
                    games.append({
                        "round": rnd,
                        "round_name": ROUND_NAMES[rnd],
                        "region": region,
                        "winner": w, "loser": l,
                        "winner_seed": seed_map[w], "loser_seed": seed_map[l],
                        "win_prob": round(max(p_a, 1 - p_a), 3),
                        "upset": seed_map[w] > seed_map[l],
                    })
                    nxt.append(w)
            current = nxt

        rw = next((x for x in current if x), None)
        if rw:
            regional_winners.append(rw)

    # R5
    regional_winners.sort()
    ff_winners = []
    for i in range(0, len(regional_winners), 2):
        if i + 1 >= len(regional_winners):
            ff_winners.append(regional_winners[i])
            break
        a, b = regional_winners[i], regional_winners[i + 1]
        p_a = prob_lookup[(a, b)]
        w = a if p_a >= 0.5 else b
        l = b if p_a >= 0.5 else a
        games.append({
            "round": 5, "round_name": "Final Four",
            "region": f"{by_region_of(a, teams)} vs {by_region_of(b, teams)}",
            "winner": w, "loser": l,
            "winner_seed": seed_map[w], "loser_seed": seed_map[l],
            "win_prob": round(max(p_a, 1 - p_a), 3),
            "upset": seed_map[w] > seed_map[l],
        })
        ff_winners.append(w)

    # R6
    champion = "Unknown"
    if len(ff_winners) == 2:
        a, b = ff_winners[0], ff_winners[1]
        p_a = prob_lookup[(a, b)]
        w = a if p_a >= 0.5 else b
        l = b if p_a >= 0.5 else a
        games.append({
            "round": 6, "round_name": "Championship",
            "region": "National",
            "winner": w, "loser": l,
            "winner_seed": seed_map[w], "loser_seed": seed_map[l],
            "win_prob": round(max(p_a, 1 - p_a), 3),
            "upset": seed_map[w] > seed_map[l],
        })
        champion = w

    return games, champion


def by_region_of(team_name: str, teams: list[dict]) -> str:
    for t in teams:
        if t["team"] == team_name:
            return t["region"]
    return "?"

# -- Output --------------------------------------------------------------------

ESPN_MAX = sum(ESPN_POINTS[r] * (2 ** (6 - r)) for r in range(1, 7))  # 1920 pts

def expected_espn_score(games: list[dict], mc_probs: dict) -> float:
    """
    E[ESPN score] = sum over all bracket picks of:
        P(team reaches that round) * points_for_round
    The deterministic bracket picks the highest-probability team in every slot,
    so our pick in round R is the team we predict to win.
    """
    total = 0.0
    for g in games:
        rnd = g["round"]
        winner = g["winner"]
        # P(our pick is correct) = P(winner actually reaches round rnd+1)
        p_correct = mc_probs.get(winner, {}).get(rnd + 1, 0.0)
        total += p_correct * ESPN_POINTS.get(rnd, 0)
    return round(total, 1)


def print_results(
    games: list[dict],
    champion: str,
    mc_probs: dict,
    teams: list[dict],
    gender: str,
    year: int,
    n_sims: int,
) -> None:
    seed_map = {t["team"]: t["seed"] for t in teams}

    print(f"\n{'='*68}")
    print(f"  {year} {gender.upper()} TOURNAMENT -- MODEL PREDICTIONS")
    print(f"  Monte Carlo: {n_sims:,} simulations")
    print(f"{'='*68}")

    # Group games by round
    by_round = defaultdict(list)
    for g in games:
        by_round[g["round"]].append(g)

    for rnd in sorted(by_round.keys()):
        rnd_games = by_round[rnd]
        rname = ROUND_NAMES.get(rnd, f"R{rnd}")
        upsets = sum(1 for g in rnd_games if g["upset"])
        print(f"\n  -- {rname} ({len(rnd_games)} games, {upsets} upsets) --")
        for g in rnd_games:
            upset_flag = "  *** UPSET ***" if g["upset"] else ""
            # MC probability for this game's winner reaching the next round
            p_correct = mc_probs.get(g["winner"], {}).get(rnd + 1, 0.0)
            print(
                f"    ({g['winner_seed']:2d}) {g['winner']:<24}"
                f"  def.  ({g['loser_seed']:2d}) {g['loser']:<24}"
                f"  head2head={g['win_prob']:.0%}  mc={p_correct:.0%}"
                f"{upset_flag}"
            )

    exp_score = expected_espn_score(games, mc_probs)
    print(f"\n{'='*68}")
    print(f"  PREDICTED CHAMPION: ({seed_map.get(champion,'?')}) {champion}")
    print(f"  Expected ESPN score: {exp_score:.0f} / {ESPN_MAX} pts")
    print(f"{'='*68}")

    # Championship probability table — all teams with > 1%
    print(f"\n  Championship probabilities (MC, > 1%):")
    print(f"  {'Seed':>4}  {'Team':<28} {'Champ%':>7}  {'Final4%':>7}  {'Elite8%':>7}")
    print(f"  {'-'*62}")
    ranked = sorted(
        [(t, mc_probs.get(t, {})) for t in [x["team"] for x in teams]],
        key=lambda x: -x[1].get(7, 0),
    )
    for team, probs in ranked:
        p_champ  = probs.get(7, 0.0)
        p_final4 = probs.get(6, 0.0)
        p_elite8 = probs.get(5, 0.0)
        if p_champ < 0.01:
            continue
        seed = seed_map.get(team, "?")
        bar = "#" * int(p_champ * 30)
        print(f"  ({seed:>2})  {team:<28} {p_champ:>6.1%}  {p_final4:>7.1%}  {p_elite8:>7.1%}  {bar}")


def save_outputs(
    games: list[dict],
    champion: str,
    mc_probs: dict,
    teams: list[dict],
    gender: str,
    year: int,
) -> None:
    seed_map = {t["team"]: t["seed"] for t in teams}

    # Main bracket JSON
    json_out = {
        "gender":    gender,
        "year":      year,
        "champion":  champion,
        "expected_espn_score": expected_espn_score(games, mc_probs),
        "games":     games,
        "team_probs": {
            team: {
                "p_champion":   mc_probs.get(team, {}).get(7, 0.0),
                "p_final_four": mc_probs.get(team, {}).get(6, 0.0),
                "p_elite_eight": mc_probs.get(team, {}).get(5, 0.0),
                "p_sweet_sixteen": mc_probs.get(team, {}).get(4, 0.0),
                "p_round2":     mc_probs.get(team, {}).get(3, 0.0),
                "p_round1_win": mc_probs.get(team, {}).get(2, 0.0),
            }
            for team in [t["team"] for t in teams]
        },
    }
    json_path = BRACKETS_DIR / f"{gender}_{year}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_out, f, indent=2)

    # Games CSV
    csv_path = BRACKETS_DIR / f"{gender}_{year}.csv"
    pd.DataFrame(games).to_csv(csv_path, index=False)

    # MC probability CSV — one row per team, all round probabilities
    mc_rows = []
    for team_data in teams:
        name = team_data["team"]
        probs = mc_probs.get(name, {})
        mc_rows.append({
            "team":            name,
            "seed":            seed_map[name],
            "p_r1_win":        probs.get(2, 0.0),
            "p_round_of_32":   probs.get(3, 0.0),
            "p_sweet_sixteen": probs.get(4, 0.0),
            "p_elite_eight":   probs.get(5, 0.0),
            "p_final_four":    probs.get(6, 0.0),
            "p_champion":      probs.get(7, 0.0),
        })
    mc_rows.sort(key=lambda r: -r["p_champion"])
    mc_csv_path = BRACKETS_DIR / f"{gender}_{year}_mc_probs.csv"
    pd.DataFrame(mc_rows).to_csv(mc_csv_path, index=False)

    print(f"\n  Saved -> {json_path.name}")
    print(f"  Saved -> {csv_path.name}")
    print(f"  Saved -> {mc_csv_path.name}")

# -- Main ----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate NCAA tournament bracket (Monte Carlo)")
    parser.add_argument("--gender", choices=["mens", "womens", "both"], default="both")
    parser.add_argument("--year",   type=int,  default=2026)
    parser.add_argument("--sims",   type=int,  default=10_000,
                        help="Number of Monte Carlo simulations (default: 10000)")
    parser.add_argument("--seed",   type=int,  default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()
    random.seed(args.seed)

    genders = ["mens", "womens"] if args.gender == "both" else [args.gender]

    for gender in genders:
        print(f"\nLoading {gender} {args.year} data and model...")
        teams = load_teams(gender, args.year)
        model = load_model(gender)

        print(f"  Pre-computing {len(teams) * (len(teams)-1):,} pairwise win probabilities...")
        prob_lookup = precompute_probs(teams, model)

        print(f"  Running {args.sims:,} Monte Carlo simulations...")
        mc_probs = monte_carlo(teams, prob_lookup, n_sims=args.sims)

        print(f"  Building deterministic bracket...")
        games, champion = deterministic_bracket(teams, prob_lookup)

        print_results(games, champion, mc_probs, teams, gender, args.year, args.sims)
        save_outputs(games, champion, mc_probs, teams, gender, args.year)


if __name__ == "__main__":
    main()