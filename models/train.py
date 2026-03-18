#!/usr/bin/env python3
"""
Phase 3a — Model Training

Trains separate Logistic Regression + XGBoost ensemble models for the men's
and women's tournaments. Uses leave-one-year-out (LOYO) cross-validation to
simulate real bracket conditions (train on past, predict a future year).

Outputs:
    models/saved/mens_model.pkl
    models/saved/womens_model.pkl
    models/saved/mens_eval.yaml
    models/saved/womens_eval.yaml
"""

import pickle
import warnings
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, log_loss, brier_score_loss,
    roc_auc_score, confusion_matrix,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────

DATA_DIR   = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent / "saved"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Feature columns ────────────────────────────────────────────────────────────

# All differential features + binary flags
# Note: we use ONE perspective per game (outcome==1 rows only) so the model
# sees balanced, non-redundant data. Mirror rows would just double each game
# identically and inflate cross-val metrics without adding information.
DIFF_FEATURES = [
    "seed_diff",
    "srs_diff",
    "sos_diff",
    "ortg_diff",
    "drtg_diff",
    "pace_diff",
    "efg_pct_diff",
    "tov_pct_diff",
    "orb_pct_diff",
    "ft_rate_diff",
    "win_pct_diff",
    "conf_win_pct_diff",
    "prospect_diff",
]

FLAG_FEATURES = [
    "conf_champion_a",
    "conf_champion_b",
    "conf_tourn_champion_a",
    "conf_tourn_champion_b",
]

ALL_FEATURES = DIFF_FEATURES + FLAG_FEATURES

ROUND_NAMES = {
    1: "First Round",
    2: "Second Round",
    3: "Sweet Sixteen",
    4: "Elite Eight",
    6: "Championship",
}

# ── Data Loading ───────────────────────────────────────────────────────────────

def load_data(gender: str) -> pd.DataFrame:
    """
    Load matchup CSV with both mirror rows (outcome 0 and 1).
    Both perspectives of the same game are needed so the model sees a balanced
    50/50 target. LOYO CV splits by year so both rows of a game stay together
    in the same fold — no leakage.
    """
    path = DATA_DIR / "processed" / f"{gender}_matchups.csv"
    return pd.read_csv(path).reset_index(drop=True)

# ── Model Definitions ──────────────────────────────────────────────────────────

def build_logreg() -> Pipeline:
    """Logistic Regression with standard scaling and L2 regularization."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=0.5,
            max_iter=1000,
            random_state=42,
            solver="lbfgs",
        )),
    ])


def build_xgb() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,   # regularize on small dataset
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )

# ── Leave-One-Year-Out CV ──────────────────────────────────────────────────────

def loyo_cv(df: pd.DataFrame, gender: str) -> dict:
    """
    Train on all years except one, evaluate on the held-out year.
    Trains both LR and XGB; final prediction is soft-vote ensemble average.
    Returns per-year and aggregate metrics.
    """
    years = sorted(df["year"].unique())
    X = df[ALL_FEATURES].values
    y = df["outcome"].values  # 0 or 1

    all_preds_lr  = np.zeros(len(df))
    all_preds_xgb = np.zeros(len(df))
    all_preds_ens = np.zeros(len(df))

    per_year: dict = {}

    for held_out in years:
        train_mask = df["year"] != held_out
        test_mask  = df["year"] == held_out

        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]

        # Logistic Regression
        lr = build_logreg()
        lr.fit(X_train, y_train)
        p_lr = lr.predict_proba(X_test)[:, 1]

        # XGBoost
        xgb = build_xgb()
        xgb.fit(X_train, y_train,
                eval_set=[(X_train, y_train)],
                verbose=False)
        p_xgb = xgb.predict_proba(X_test)[:, 1]

        # Ensemble (equal weight soft vote)
        p_ens = (p_lr + p_xgb) / 2

        all_preds_lr[test_mask]  = p_lr
        all_preds_xgb[test_mask] = p_xgb
        all_preds_ens[test_mask] = p_ens

        # Per-year stats
        n = test_mask.sum()
        per_year[int(held_out)] = {
            "n_games":   int(n),
            "acc_lr":    round(accuracy_score(y_test, p_lr  >= 0.5), 3),
            "acc_xgb":   round(accuracy_score(y_test, p_xgb >= 0.5), 3),
            "acc_ens":   round(accuracy_score(y_test, p_ens >= 0.5), 3),
            "brier_ens": round(brier_score_loss(y_test, p_ens), 3),
        }

    labels = df["outcome"].values

    # Aggregate metrics
    def metrics(preds, label):
        acc    = accuracy_score(labels, preds >= 0.5)
        ll     = log_loss(labels, preds)
        brier  = brier_score_loss(labels, preds)
        auc    = roc_auc_score(labels, preds)
        return {
            f"{label}_accuracy":  round(acc,   3),
            f"{label}_log_loss":  round(ll,    3),
            f"{label}_brier":     round(brier, 3),
            f"{label}_auc":       round(auc,   3),
        }

    agg = {}
    agg.update(metrics(all_preds_lr,  "lr"))
    agg.update(metrics(all_preds_xgb, "xgb"))
    agg.update(metrics(all_preds_ens, "ens"))

    # Per-round accuracy (ensemble) — use outcome==1 rows to count unique games
    per_round = {}
    wins_mask = df["outcome"] == 1
    for rnd, rname in ROUND_NAMES.items():
        mask = (df["round"] == rnd) & wins_mask
        if mask.sum() == 0:
            continue
        acc = accuracy_score(labels[mask], all_preds_ens[mask] >= 0.5)
        per_round[rname] = {
            "games": int(mask.sum()),
            "accuracy": round(acc, 3),
        }

    # Seed-only baseline (predict lower seed number always wins)
    seed_preds = (df["seed_diff"] > 0).astype(int).values
    seed_acc   = accuracy_score(labels, seed_preds)

    return {
        "gender":          gender,
        "years":           [int(y_) for y_ in years],
        "n_games":         len(df),
        "seed_baseline":   round(seed_acc, 3),
        "aggregate":       agg,
        "per_round_ens":   per_round,
        "per_year":        per_year,
    }

# ── Final Model Training ───────────────────────────────────────────────────────

def train_final_model(df: pd.DataFrame) -> dict:
    """
    Train final LR + XGB ensemble on ALL available data.
    Returns {'lr': pipeline, 'xgb': model, 'features': [...]}
    """
    X = df[ALL_FEATURES].values
    y = df["outcome"].values

    lr = build_logreg()
    lr.fit(X, y)

    xgb = build_xgb()
    xgb.fit(X, y, verbose=False)

    return {"lr": lr, "xgb": xgb, "features": ALL_FEATURES}


def feature_importances(model: dict, gender: str) -> dict:
    """Extract and rank feature importances from XGBoost."""
    xgb     = model["xgb"]
    features = model["features"]
    scores  = xgb.feature_importances_

    ranked = sorted(
        zip(features, scores.tolist()),
        key=lambda x: -x[1],
    )
    return {feat: round(float(imp), 4) for feat, imp in ranked}

# ── Prediction Helper ──────────────────────────────────────────────────────────

def predict_proba(model: dict, X: np.ndarray) -> np.ndarray:
    """Ensemble prediction: average of LR and XGB probabilities."""
    p_lr  = model["lr"].predict_proba(X)[:, 1]
    p_xgb = model["xgb"].predict_proba(X)[:, 1]
    return (p_lr + p_xgb) / 2

# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    for gender in ("mens", "womens"):
        print(f"\n{'='*55}")
        print(f"  Training {gender} model")
        print(f"{'='*55}")

        df = load_data(gender)
        print(f"  Games: {len(df)}  |  Years: {sorted(df['year'].unique())}")

        # ── LOYO cross-validation ──────────────────────────────────────────────
        print("\n  Running leave-one-year-out CV...")
        results = loyo_cv(df, gender)

        agg = results["aggregate"]
        baseline = results["seed_baseline"]
        print(f"\n  Seed-only baseline accuracy : {baseline:.1%}")
        print(f"\n  {'Model':<12} {'Accuracy':>10} {'Log-Loss':>10} {'Brier':>8} {'AUC':>8}")
        print(f"  {'-'*50}")
        for label in ("lr", "xgb", "ens"):
            acc   = agg[f"{label}_accuracy"]
            ll    = agg[f"{label}_log_loss"]
            brier = agg[f"{label}_brier"]
            auc   = agg[f"{label}_auc"]
            name  = {"lr": "LogReg", "xgb": "XGBoost", "ens": "Ensemble"}[label]
            marker = " <--" if label == "ens" else ""
            print(f"  {name:<12} {acc:>10.1%} {ll:>10.3f} {brier:>8.3f} {auc:>8.3f}{marker}")

        print(f"\n  Per-round accuracy (ensemble):")
        for rnd, stats in results["per_round_ens"].items():
            print(f"    {rnd:20s}: {stats['accuracy']:.1%}  ({stats['games']} games)")

        print(f"\n  Per-year accuracy (ensemble):")
        for yr, stats in results["per_year"].items():
            print(f"    {yr}: {stats['acc_ens']:.0%}  (brier={stats['brier_ens']:.3f}, n={stats['n_games']})")

        # ── Train final model on all data ──────────────────────────────────────
        print("\n  Training final model on full dataset...")
        model = train_final_model(df)

        importances = feature_importances(model, gender)
        print(f"\n  XGBoost feature importances:")
        for feat, imp in importances.items():
            bar = "#" * int(imp * 200)
            print(f"    {feat:25s} {imp:.4f}  {bar}")

        # ── Save model ─────────────────────────────────────────────────────────
        model_path = MODELS_DIR / f"{gender}_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"\n  Model saved -> {model_path}")

        # ── Save eval YAML ─────────────────────────────────────────────────────
        results["xgb_feature_importances"] = importances
        eval_path = MODELS_DIR / f"{gender}_eval.yaml"
        with open(eval_path, "w", encoding="utf-8") as f:
            yaml.dump(results, f, default_flow_style=False, sort_keys=False)
        print(f"  Eval saved  -> {eval_path}")


if __name__ == "__main__":
    main()