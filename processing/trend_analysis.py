#!/usr/bin/env python3
"""
Phase 2b — Trend Analysis
Reads matchup CSVs produced by feature_engineering.py and generates:
  - Historical win rates by seed pairing
  - Upset rates by round
  - Conference champion win rates
  - Feature correlation with outcome
  - Summary YAML files
  - Plots saved to data/trends/plots/
"""

import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for file output
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────

DATA_DIR    = Path(__file__).parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
TRENDS_DIR  = DATA_DIR / "trends"
PLOTS_DIR   = TRENDS_DIR / "plots"

ROUND_NAMES = {
    1: "First Round",
    2: "Second Round",
    3: "Sweet Sixteen",
    4: "Elite Eight",
    5: "Final Four",
    6: "Championship",
}

DIFF_FEATURES = [
    "seed_diff", "srs_diff", "sos_diff", "ortg_diff", "drtg_diff",
    "efg_pct_diff", "tov_pct_diff", "orb_pct_diff", "ft_rate_diff",
    "win_pct_diff", "conf_win_pct_diff", "prospect_diff",
]

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_matchups(gender: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{gender}_matchups.csv"
    df = pd.read_csv(path)
    # Keep only one perspective per game to avoid double-counting in win rates
    # (outcome==1 rows represent the actual winner as team_a)
    return df


def pct(x: float) -> float:
    """Round to 3 decimal places."""
    return round(float(x), 3)


def _save_yaml(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

# ── Analysis Functions ─────────────────────────────────────────────────────────

def seed_win_rates(df: pd.DataFrame) -> dict:
    """
    Win rates for each seed matchup (1v16, 2v15, … 8v9) across all rounds.
    Uses only outcome==1 rows (winner perspective) to get unique games.
    """
    wins_df = df[df["outcome"] == 1].copy()

    # Canonical seed pair: (lower_seed, higher_seed)
    wins_df["lo_seed"] = wins_df[["seed_a", "seed_b"]].min(axis=1)
    wins_df["hi_seed"] = wins_df[["seed_a", "seed_b"]].max(axis=1)
    wins_df["lo_won"]  = (wins_df["seed_a"] == wins_df["lo_seed"]).astype(int)

    results = {}
    for (lo, hi), group in wins_df.groupby(["lo_seed", "hi_seed"]):
        total    = len(group)
        lo_wins  = group["lo_won"].sum()
        results[f"{int(lo)}v{int(hi)}"] = {
            "games":            int(total),
            "lower_seed_wins":  int(lo_wins),
            "lower_seed_win_pct": pct(lo_wins / total) if total > 0 else None,
            "upset_pct":        pct(1 - lo_wins / total) if total > 0 else None,
        }
    return dict(sorted(results.items(), key=lambda x: (int(x[0].split("v")[0]), int(x[0].split("v")[1]))))


def upset_rates_by_round(df: pd.DataFrame) -> dict:
    """
    Upset rate per round. An upset = lower seed number (better team) loses.
    Uses outcome==1 rows only.
    """
    wins_df = df[df["outcome"] == 1].copy()
    wins_df["upset"] = (wins_df["seed_a"] > wins_df["seed_b"]).astype(int)  # A won but A is worse seed

    results = {}
    for rnd, group in wins_df.groupby("round"):
        total  = len(group)
        upsets = group["upset"].sum()
        results[ROUND_NAMES.get(rnd, f"Round {rnd}")] = {
            "games":      int(total),
            "upsets":     int(upsets),
            "upset_rate": pct(upsets / total) if total > 0 else None,
        }
    return results


def conf_champion_win_rates(df: pd.DataFrame) -> dict:
    """Win rate for conference regular-season champions and tournament champions."""
    wins_df = df[df["outcome"] == 1]
    results = {}
    for label, col in [("regular_season_champ", "conf_champion_a"),
                       ("tournament_champ",      "conf_tourn_champion_a")]:
        # All games involving a conf champ as team_a
        champ_games = df[df[col] == 1]
        if len(champ_games) == 0:
            continue
        win_count = champ_games["outcome"].sum()
        results[label] = {
            "games":   int(len(champ_games)),
            "wins":    int(win_count),
            "win_pct": pct(win_count / len(champ_games)),
        }
    return results


def feature_correlations(df: pd.DataFrame) -> dict:
    """Pearson correlation of each feature differential with outcome."""
    available = [f for f in DIFF_FEATURES if f in df.columns]
    corr = df[available + ["outcome"]].corr()["outcome"].drop("outcome")
    return {feat: pct(val) for feat, val in corr.sort_values(ascending=False).items()}


def yearly_summary(df: pd.DataFrame) -> dict:
    """Per-year upset count and champion seed."""
    wins_df = df[df["outcome"] == 1].copy()
    wins_df["upset"] = (wins_df["seed_a"] > wins_df["seed_b"]).astype(int)

    results = {}
    for year, group in wins_df.groupby("year"):
        total_games = len(group)
        upsets      = group["upset"].sum()

        # Champion: seed_a of round 6 winner (if available)
        champ_row = group[group["round"] == 6]
        champ_seed = int(champ_row["seed_a"].iloc[0]) if len(champ_row) > 0 else None

        results[int(year)] = {
            "games":       int(total_games),
            "upsets":      int(upsets),
            "upset_rate":  pct(upsets / total_games) if total_games > 0 else None,
            "champ_seed":  champ_seed,
        }
    return dict(sorted(results.items()))

# ── Plot Generators ────────────────────────────────────────────────────────────

def _style() -> None:
    sns.set_theme(style="whitegrid", palette="muted")


def plot_seed_win_rates(seed_rates: dict, gender: str) -> None:
    """Bar chart: win rate for lower (better) seed in each matchup."""
    _style()
    labels = list(seed_rates.keys())
    win_pcts = [seed_rates[k]["lower_seed_win_pct"] or 0 for k in labels]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(labels, win_pcts, color=sns.color_palette("Blues_d", len(labels)))
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1, label="50% (coin flip)")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Seed Matchup")
    ax.set_ylabel("Lower-Seed Win Rate")
    ax.set_title(f"{gender.title()} Tournament — Win Rate by Seed Matchup (2014-2025)")
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    _save_plot(fig, PLOTS_DIR / f"{gender}_seed_win_rates.png")


def plot_upset_rates_by_round(upset_rates: dict, gender: str) -> None:
    """Bar chart: upset rate per round."""
    _style()
    rounds = list(upset_rates.keys())
    rates  = [upset_rates[r]["upset_rate"] or 0 for r in rounds]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(rounds, rates, color=sns.color_palette("Oranges_d", len(rounds)))
    ax.set_ylim(0, max(rates) * 1.2 if rates else 0.5)
    ax.set_xlabel("Round")
    ax.set_ylabel("Upset Rate")
    ax.set_title(f"{gender.title()} Tournament — Upset Rate by Round (2014-2025)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    _save_plot(fig, PLOTS_DIR / f"{gender}_upset_rates_by_round.png")


def plot_feature_correlations(corr: dict, gender: str) -> None:
    """Horizontal bar chart: feature correlations with outcome."""
    _style()
    features = list(corr.keys())
    values   = list(corr.values())

    colors = ["steelblue" if v >= 0 else "tomato" for v in values]
    fig, ax = plt.subplots(figsize=(8, len(features) * 0.4 + 1))
    ax.barh(features, values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Pearson Correlation with Outcome")
    ax.set_title(f"{gender.title()} — Feature Correlation with Win (2014-2025)")
    plt.tight_layout()
    _save_plot(fig, PLOTS_DIR / f"{gender}_feature_correlations.png")


def plot_correlation_heatmap(df: pd.DataFrame, gender: str) -> None:
    """Heatmap of correlation matrix across all numeric features + outcome."""
    _style()
    available = [f for f in DIFF_FEATURES if f in df.columns] + ["outcome"]
    corr_matrix = df[available].corr()

    fig, ax = plt.subplots(figsize=(len(available) * 0.7 + 1, len(available) * 0.7 + 1))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(f"{gender.title()} — Feature Correlation Matrix")
    plt.tight_layout()
    _save_plot(fig, PLOTS_DIR / f"{gender}_correlation_heatmap.png")


def _save_plot(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Plot saved -> {path.name}")

# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    TRENDS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    for gender in ("mens", "womens"):
        print(f"\n{'='*50}")
        print(f"Trend analysis: {gender}")
        print(f"{'='*50}")

        df = load_matchups(gender)
        print(f"  Loaded {len(df)} matchup rows ({len(df)//2} games)")

        seed_rates  = seed_win_rates(df)
        upset_by_rnd = upset_rates_by_round(df)
        conf_rates  = conf_champion_win_rates(df)
        corr        = feature_correlations(df)
        yearly      = yearly_summary(df)

        # ── Print summary ──────────────────────────────────────────────────────
        print("\n  Seed win rates (lower seed = better team):")
        for matchup, stats in seed_rates.items():
            print(f"    {matchup:6s}  lower-seed wins {stats['lower_seed_win_pct']:.1%}"
                  f"  ({stats['games']} games)")

        print("\n  Upset rates by round:")
        for rnd, stats in upset_by_rnd.items():
            print(f"    {rnd:20s}: {stats['upset_rate']:.1%}  ({stats['games']} games)")

        print("\n  Top feature correlations with winning:")
        for feat, val in list(corr.items())[:5]:
            print(f"    {feat:25s}: {val:+.3f}")

        # ── Save YAML ──────────────────────────────────────────────────────────
        summary = {
            "gender":                  gender,
            "years_covered":           sorted(yearly.keys()),
            "total_games":             sum(v["games"] for v in yearly.values()),
            "seed_win_rates":          seed_rates,
            "upset_rates_by_round":    upset_by_rnd,
            "conf_champion_win_rates": conf_rates,
            "feature_correlations":    corr,
            "yearly_summary":          yearly,
        }
        out_path = TRENDS_DIR / f"{gender}_trends.yaml"
        _save_yaml(summary, out_path)
        print(f"\n  Saved -> {out_path}")

        # ── Generate plots ─────────────────────────────────────────────────────
        print("\n  Generating plots:")
        plot_seed_win_rates(seed_rates, gender)
        plot_upset_rates_by_round(upset_by_rnd, gender)
        plot_feature_correlations(corr, gender)
        plot_correlation_heatmap(df, gender)


if __name__ == "__main__":
    main()