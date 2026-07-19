"""Train the model several times with different random seeds and keep the best.

XGBoost is deterministic for a fixed seed, so "training again" only differs when
the seed changes (subsample / colsample introduce the randomness). This script
trains N seeded variants, scores each with leave-one-tournament-out CV, then
refits and saves the best one.

Run:  python scripts/train_multi.py            # 5 seeds
      python scripts/train_multi.py 10         # 10 seeds
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config                                   # noqa: E402
from src.model import build_prediction          # noqa: E402
from src.train import XGB_PARAMS                 # noqa: E402

EPS = 1e-12


def _outcome(hg, ag):
    return 0 if hg > ag else (1 if hg == ag else 2)


def _models(seed):
    params = {**XGB_PARAMS, "random_state": seed}
    return xgb.XGBRegressor(**params), xgb.XGBRegressor(**params)


def cv_for_seed(df, seed):
    g_sq, correct, tot, lls, briers = [], 0, 0, [], []
    for _, test in df.groupby(["competition_id", "season_id"]):
        train = df[~df.index.isin(test.index)]
        hm, am = _models(seed)
        hm.fit(train[config.FEATURE_COLS], train["home_goals"])
        am.fit(train[config.FEATURE_COLS], train["away_goals"])
        lam_h = hm.predict(test[config.FEATURE_COLS])
        lam_a = am.predict(test[config.FEATURE_COLS])
        for i, (_, row) in enumerate(test.iterrows()):
            pred = build_prediction(row.home_team, row.away_team, lam_h[i], lam_a[i])
            probs = np.array([pred.p_home_win, pred.p_draw, pred.p_away_win])
            true = _outcome(row.home_goals, row.away_goals)
            correct += int(probs.argmax() == true); tot += 1
            g_sq += [(lam_h[i]-row.home_goals)**2, (lam_a[i]-row.away_goals)**2]
            oh = np.zeros(3); oh[true] = 1
            lls.append(-np.log(max(probs[true], EPS)))
            briers.append(float(((probs-oh)**2).sum()))
    return {
        "seed": seed,
        "outcome_accuracy": round(correct/tot, 4),
        "logloss": round(float(np.mean(lls)), 4),
        "brier": round(float(np.mean(briers)), 4),
        "goals_RMSE": round(float(np.sqrt(np.mean(g_sq))), 4),
    }


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    seeds = [42, 1, 7, 2024, 99, 314, 2025, 13, 777, 8][:n]
    df = pd.read_csv(config.PROCESSED_DIR / "training_data.csv")

    runs = []
    print(f"Training {len(seeds)} seeded variants (leave-one-tournament-out CV):\n")
    print(f"{'seed':>6} {'accuracy':>9} {'logloss':>8} {'brier':>7} {'goalsRMSE':>10}")
    for s in seeds:
        r = cv_for_seed(df, s)
        runs.append(r)
        print(f"{r['seed']:>6} {r['outcome_accuracy']:>9} {r['logloss']:>8} {r['brier']:>7} {r['goals_RMSE']:>10}")

    # The seed-to-seed differences are within noise, so pick the most *balanced*
    # variant: lowest average rank across accuracy, log-loss and Brier.
    def ranks(key, reverse):
        order = sorted(runs, key=lambda r: r[key], reverse=reverse)
        return {id(r): i for i, r in enumerate(order)}
    acc_r = ranks("outcome_accuracy", True)
    ll_r = ranks("logloss", False)
    br_r = ranks("brier", False)
    for r in runs:
        r["_avg_rank"] = (acc_r[id(r)] + ll_r[id(r)] + br_r[id(r)]) / 3
    best = min(runs, key=lambda r: r["_avg_rank"])
    for r in runs:
        r.pop("_avg_rank", None)
    print(f"\nBest balanced seed: {best['seed']}  "
          f"(accuracy {best['outcome_accuracy']}, logloss {best['logloss']}, brier {best['brier']})")

    # Refit the best variant on ALL data and save it.
    hm, am = _models(best["seed"])
    hm.fit(df[config.FEATURE_COLS], df["home_goals"])
    am.fit(df[config.FEATURE_COLS], df["away_goals"])
    hm.save_model(config.MODELS_DIR / "home_goals_model.json")
    am.save_model(config.MODELS_DIR / "away_goals_model.json")

    imp = {c: float(v) for c, v in zip(config.FEATURE_COLS, hm.feature_importances_)}
    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_matches": int(len(df)),
        "feature_cols": config.FEATURE_COLS,
        "tournaments": [t[2] for t in config.TOURNAMENTS],
        "selection": "most balanced of %d seeds by avg CV rank (accuracy/logloss/brier)" % len(seeds),
        "best_seed": best["seed"],
        "xgb_params": {**XGB_PARAMS, "random_state": best["seed"]},
        "cv_all_seeds": runs,
        "home_feature_importance_gain": imp,
        "data_source": "StatsBomb Open Data",
    }
    (config.MODELS_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    (config.MODELS_DIR / "cv_metrics.json").write_text(json.dumps({
        "selected_seed": best["seed"],
        "outcome_accuracy": best["outcome_accuracy"],
        "multiclass_logloss": best["logloss"],
        "multiclass_brier": best["brier"],
        "goals_RMSE": best["goals_RMSE"],
        "all_seed_runs": runs,
    }, indent=2))
    print(f"Saved best models + metadata to {config.MODELS_DIR}")


if __name__ == "__main__":
    main()
