#!/usr/bin/env python3
"""
Phase 4a — Bracket Simulation

Loads the current-year team YAML, runs the trained ensemble model to compute
win probabilities for every possible matchup, then simulates the full bracket
using those probabilities. Outputs:
  - Console bracket summary with win probabilities per round
  - bracket/brackets/{gender}_2026.json
  - bracket/brackets/{gender}_2026.csv

Usage:
    python bracket/simulate.py
    python bracket/simulate.py --gender mens
    python bracket/simulate.py --gender womens
    python bracket/simulate.py --year 2026
"""

import argparse
import json
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import yaml

# -- Paths ----------------------------------------------------------------------

PROJECT_ROOT  = Path(__file__).parent.parent
DATA_DIR      = PROJECT_ROOT / "data" / "processed"
MODELS_DIR    = PROJECT_ROOT / "models" / "saved"
BRACKETS_DIR  = Path(__file__).parent / "brackets"
BRACKETS_DIR.mkdir(parents=True, exist_ok=True)

SEED_ORDER = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]

ROUND_NAMES = {
    1: "First Round",
    2: "Second Round",
    3: "Sweet Sixteen",
    4: "Elite Eight",
    5: "Final Four",
    6: "Championship",
}

ALL_FEATURES = [
    "seed_diff", "srs_diff", "sos_diff", "ortg_diff", "drtg_diff",
    "pace_diff", "efg_pct_diff", "tov_pct_diff", "orb_pct_diff",
    "ft_rate_diff", "win_pct_diff", "conf_win_pct_diff", "prospect_diff",
    "conf_champion_a", "conf_champion_b",
    "conf_tourn_champion_a", "conf_tourn_champion_b",
]

# -- Data & Model Loading ------------------------------------------------------─

def load_teams(gender: str, year: int) -> list[dict]:
    path = DATA_DIR / gender / f"{year}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No processed YAML for {gender}/{year}. Run scrapers first.")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["teams"]


def load_model(gender: str) -> dict:
    path = MODELS_DIR / f"{gender}_model.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No trained model at {path}. Run models/train.py first.")
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_proba(model: dict, X: np.ndarray) -> np.ndarray:
    """Ensemble: average LR and XGBoost probabilities."""
    p_lr  = model["lr"].predict_proba(X)[:, 1]
    p_xgb = model["xgb"].predict_proba(X)[:, 1]
    return (p_lr + p_xgb) / 2

# -- Feature Extraction --------------------------------------------------------─

def parse_record(s) -> tuple[int, int]:
    s = str(s or "")
    if "-" not in s:
        return 0, 0
    parts = s.split("-", 1)
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return 0, 0


def get_features(team: dict) -> dict:
    overall_w, overall_l = parse_record(team.get("record", {}).get("overall"))
    conf_w,    conf_l    = parse_record(team.get("record", {}).get("conference"))
    total_g = overall_w + overall_l
    conf_g  = conf_w + conf_l
    adv = team.get("advanced") or {}
    sos = team.get("strength_of_schedule") or {}
    prospects = int(team.get("nba_prospects") or team.get("wnba_prospects") or 0)
    return {
        "team":                team["team"],
        "seed":                int(team.get("seed", 0)),
        "region":              team["region"],
        "conf_champion":       int(bool(team.get("conf_champion"))),
        "conf_tourn_champion": int(bool(team.get("conf_tourn_champion"))),
        "win_pct":             overall_w / total_g if total_g > 0 else 0.0,
        "conf_win_pct":        conf_w / conf_g     if conf_g  > 0 else 0.0,
        "sos":    float(sos.get("rating") or 0),
        "srs":    float(adv.get("srs") or 0),
        "ortg":   float(adv.get("offensive_rating") or 0),
        "drtg":   float(adv.get("defensive_rating") or 0),
        "pace":   float(adv.get("pace") or 0),
        "efg_pct":  float(adv.get("efg_pct") or 0),
        "tov_pct":  float(adv.get("tov_pct") or 0),
        "orb_pct":  float(adv.get("orb_pct") or 0),
        "ft_rate":  float(adv.get("ft_rate") or 0),
        "prospects": prospects,
    }


def matchup_features(a: dict, b: dict) -> dict:
    """Build feature row for matchup (a vs b); positive diff = a has advantage."""
    af = get_features(a)
    bf = get_features(b)
    return {
        "seed_diff":             bf["seed"]      - af["seed"],
        "srs_diff":              af["srs"]        - bf["srs"],
        "sos_diff":              af["sos"]        - bf["sos"],
        "ortg_diff":             af["ortg"]       - bf["ortg"],
        "drtg_diff":             bf["drtg"]       - af["drtg"],
        "pace_diff":             af["pace"]       - bf["pace"],
        "efg_pct_diff":          af["efg_pct"]    - bf["efg_pct"],
        "tov_pct_diff":          bf["tov_pct"]    - af["tov_pct"],
        "orb_pct_diff":          af["orb_pct"]    - bf["orb_pct"],
        "ft_rate_diff":          af["ft_rate"]    - bf["ft_rate"],
        "win_pct_diff":          af["win_pct"]    - bf["win_pct"],
        "conf_win_pct_diff":     af["conf_win_pct"] - bf["conf_win_pct"],
        "prospect_diff":         af["prospects"]  - bf["prospects"],
        "conf_champion_a":       af["conf_champion"],
        "conf_champion_b":       bf["conf_champion"],
        "conf_tourn_champion_a": af["conf_tourn_champion"],
        "conf_tourn_champion_b": bf["conf_tourn_champion"],
    }


def win_probability(model: dict, team_a: dict, team_b: dict) -> float:
    """Return P(team_a beats team_b)."""
    feats = matchup_features(team_a, team_b)
    X = np.array([[feats[f] for f in ALL_FEATURES]])
    return float(predict_proba(model, X)[0])

# -- Bracket Simulation --------------------------------------------------------─

def simulate_bracket(teams: list[dict], model: dict) -> dict:
    """
    Simulate the full bracket deterministically (always advance the higher
    win-probability team). Returns a dict with per-round results and
    probability breakdowns for every team.
    """
    by_region = defaultdict(list)
    for t in teams:
        by_region[t["region"]].append(t)

    round_results = {}   # round_num -> list of {winner, loser, prob}
    team_probs    = {}   # team_name -> {round: prob_of_reaching_round}

    # Initialize each team's probability of reaching R1 as 1.0
    for t in teams:
        team_probs[t["team"]] = {1: 1.0}

    # -- Regional rounds (R1-R4) ------------------------------------------------
    regional_winners = []

    for region, region_teams in sorted(by_region.items()):
        seed_to_team = {t["seed"]: t for t in region_teams}
        bracket = [seed_to_team.get(s) for s in SEED_ORDER]  # None = missing seed

        current = bracket[:]
        round_num = 1

        while round_num <= 4 and len([t for t in current if t]) > 1:
            next_round = []
            games = []

            for i in range(0, len(current), 2):
                a = current[i]     if i     < len(current) else None
                b = current[i + 1] if i + 1 < len(current) else None

                if a is None and b is None:
                    next_round.append(None)
                elif a is None:
                    next_round.append(b)
                elif b is None:
                    next_round.append(a)
                else:
                    p_a = win_probability(model, a, b)
                    winner = a if p_a >= 0.5 else b
                    loser  = b if p_a >= 0.5 else a
                    prob   = max(p_a, 1 - p_a)

                    games.append({
                        "round":       round_num,
                        "round_name":  ROUND_NAMES[round_num],
                        "region":      region,
                        "winner":      winner["team"],
                        "loser":       loser["team"],
                        "winner_seed": winner["seed"],
                        "loser_seed":  loser["seed"],
                        "win_prob":    round(prob, 3),
                        "upset":       winner["seed"] > loser["seed"],
                    })

                    # Track reach-probability for next round
                    prev_prob = team_probs[winner["team"]].get(round_num, 1.0)
                    team_probs[winner["team"]][round_num + 1] = round(prev_prob * prob, 4)

                    next_round.append(winner)

            round_results.setdefault(round_num, []).extend(games)
            current = next_round
            round_num += 1

        # Regional winner is the last remaining team
        regional_winner = next((t for t in current if t is not None), None)
        if regional_winner:
            regional_winners.append(regional_winner)

    # -- Final Four (R5) --------------------------------------------------------
    # Pair regional winners: sort by region name and pair [0,1] vs [2,3]
    # (Consistent pairing — may not match actual bracket draw but is deterministic)
    regional_winners.sort(key=lambda t: t["region"])
    final_four_winners = []

    for i in range(0, len(regional_winners), 2):
        if i + 1 >= len(regional_winners):
            final_four_winners.append(regional_winners[i])
            break
        a, b   = regional_winners[i], regional_winners[i + 1]
        p_a    = win_probability(model, a, b)
        winner = a if p_a >= 0.5 else b
        loser  = b if p_a >= 0.5 else a
        prob   = max(p_a, 1 - p_a)

        round_results.setdefault(5, []).append({
            "round": 5, "round_name": "Final Four",
            "region": f"{a['region']} vs {b['region']}",
            "winner": winner["team"], "loser": loser["team"],
            "winner_seed": winner["seed"], "loser_seed": loser["seed"],
            "win_prob": round(prob, 3),
            "upset": winner["seed"] > loser["seed"],
        })
        prev = team_probs[winner["team"]].get(5, 1.0)
        team_probs[winner["team"]][6] = round(prev * prob, 4)
        final_four_winners.append(winner)

    # -- Championship (R6) ----------------------------------------------------─
    if len(final_four_winners) == 2:
        a, b   = final_four_winners[0], final_four_winners[1]
        p_a    = win_probability(model, a, b)
        winner = a if p_a >= 0.5 else b
        loser  = b if p_a >= 0.5 else a
        prob   = max(p_a, 1 - p_a)

        round_results[6] = [{
            "round": 6, "round_name": "Championship",
            "region": "National",
            "winner": winner["team"], "loser": loser["team"],
            "winner_seed": winner["seed"], "loser_seed": loser["seed"],
            "win_prob": round(prob, 3),
            "upset": winner["seed"] > loser["seed"],
        }]
        team_probs[winner["team"]][7] = round(
            team_probs[winner["team"]].get(6, 1.0) * prob, 4
        )

    champion = round_results[6][0]["winner"] if 6 in round_results else "Unknown"

    return {
        "champion":      champion,
        "round_results": round_results,
        "team_probs":    team_probs,
    }

# -- Output --------------------------------------------------------------------─

def print_bracket(result: dict, teams: list[dict], gender: str, year: int) -> None:
    """Print a readable bracket summary to the console."""
    seed_map = {t["team"]: t["seed"] for t in teams}
    region_map = {t["team"]: t["region"] for t in teams}

    print(f"\n{'='*65}")
    print(f"  {year} {gender.upper()} TOURNAMENT — MODEL PREDICTIONS")
    print(f"{'='*65}")

    for rnd in sorted(result["round_results"].keys()):
        games = result["round_results"][rnd]
        rname = ROUND_NAMES.get(rnd, f"Round {rnd}")
        upsets = [g for g in games if g["upset"]]

        print(f"\n  -- {rname} ({len(games)} games, {len(upsets)} upsets predicted) --")
        for g in games:
            upset_flag = " *** UPSET ***" if g["upset"] else ""
            print(
                f"    ({g['winner_seed']:2d}) {g['winner']:<25}"
                f"  def.  ({g['loser_seed']:2d}) {g['loser']:<25}"
                f"  {g['win_prob']:.0%}{upset_flag}"
            )

    print(f"\n{'='*65}")
    print(f"  PREDICTED CHAMPION: ({seed_map.get(result['champion'], '?')}) {result['champion']}")
    print(f"{'='*65}")

    # Top teams by championship probability
    print(f"\n  Championship probability (top 16):")
    champ_probs = []
    for team, probs in result["team_probs"].items():
        # P(champion) = P(reaching R7) if tracked, else P(reaching R6) * last win prob
        p_champ = probs.get(7, 0.0)
        if p_champ == 0 and probs.get(6, 0):
            p_champ = probs.get(6, 0.0)  # finalist but champion prob not stored
        champ_probs.append((team, p_champ, seed_map.get(team, "?")))

    champ_probs.sort(key=lambda x: -x[1])
    for team, prob, seed in champ_probs[:16]:
        bar = "#" * int(prob * 40)
        print(f"    ({seed:>2}) {team:<28} {prob:>5.1%}  {bar}")


def save_outputs(result: dict, teams: list[dict], gender: str, year: int) -> None:
    seed_map   = {t["team"]: t["seed"]   for t in teams}
    region_map = {t["team"]: t["region"] for t in teams}

    # JSON
    json_out = {
        "gender":    gender,
        "year":      year,
        "champion":  result["champion"],
        "rounds":    result["round_results"],
        "team_championship_probability": {
            team: probs.get(7, probs.get(6, 0.0))
            for team, probs in result["team_probs"].items()
        },
    }
    json_path = BRACKETS_DIR / f"{gender}_{year}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_out, f, indent=2)

    # CSV — one row per game
    rows = []
    for rnd, games in result["round_results"].items():
        for g in games:
            rows.append(g)
    csv_path = BRACKETS_DIR / f"{gender}_{year}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print(f"\n  Saved -> {json_path.name}")
    print(f"  Saved -> {csv_path.name}")

# -- Main ----------------------------------------------------------------------─

def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate NCAA tournament bracket")
    parser.add_argument("--gender", choices=["mens", "womens", "both"], default="both")
    parser.add_argument("--year",   type=int, default=2026)
    args = parser.parse_args()

    genders = ["mens", "womens"] if args.gender == "both" else [args.gender]

    for gender in genders:
        teams = load_teams(gender, args.year)
        model = load_model(gender)
        result = simulate_bracket(teams, model)
        print_bracket(result, teams, gender, args.year)
        save_outputs(result, teams, gender, args.year)


if __name__ == "__main__":
    main()