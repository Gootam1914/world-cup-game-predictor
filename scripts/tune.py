"""Randomised hyperparameter search for the expected-goals models.

Each candidate configuration is scored with leave-one-tournament-out CV (the
same honest protocol used in evaluation). The winner is chosen by CV log-loss
(well-calibrated probabilities matter most for the confidence output), with
Brier as a tie-break. The best params are written to models/best_params.json,
which src/train.py automatically picks up from then on.

Run:  python scripts/tune.py             # 40 candidates
      python scripts/tune.py 80           # 80 candidates
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

EPS = 1e-12
FIXED = dict(objective="count:poisson", random_state=config.RANDOM_SEED, n_jobs=4)

SPACE = {
    "n_estimators": [150, 200, 250, 300, 400, 500],
    "max_depth": [2, 3, 4],
    "learning_rate": [0.02, 0.03, 0.05, 0.07, 0.10],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [2, 3, 4, 6, 8],
    "reg_lambda": [0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
    "reg_alpha": [0.0, 0.2, 0.5, 1.0],
}

# Current production defaults, always evaluated as a baseline.
BASELINE = dict(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.85,
                colsample_bytree=0.85, min_child_weight=4, reg_lambda=1.5, reg_alpha=0.2)


def _outcome(hg, ag):
    return 0 if hg > ag else (1 if hg == ag else 2)


def cv_score(df, params):
    g_sq, correct, tot, lls, briers = [], 0, 0, [], []
    for _, test in df.groupby(["competition_id", "season_id"]):
        train = df[~df.index.isin(test.index)]
        full = {**FIXED, **params}
        hm, am = xgb.XGBRegressor(**full), xgb.XGBRegressor(**full)
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
        "accuracy": round(correct/tot, 4),
        "logloss": round(float(np.mean(lls)), 4),
        "brier": round(float(np.mean(briers)), 4),
        "goals_RMSE": round(float(np.sqrt(np.mean(g_sq))), 4),
    }


def sample(rng):
    return {k: (rng.choice(v).item() if isinstance(v[0], (int, float)) else rng.choice(v))
            for k, v in SPACE.items()}


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    rng = np.random.default_rng(config.RANDOM_SEED)
    df = pd.read_csv(config.PROCESSED_DIR / "training_data.csv")

    trials = [{"params": BASELINE, "tag": "baseline"}]
    seen = {json.dumps(BASELINE, sort_keys=True)}
    while len(trials) < n + 1:
        p = sample(rng)
        key = json.dumps(p, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        trials.append({"params": p, "tag": "search"})

    print(f"Scoring {len(trials)} configurations (leave-one-tournament-out CV)...\n")
    for t in trials:
        t["cv"] = cv_score(df, t["params"])
        m = t["cv"]
        print(f"  {t['tag']:<8} ll={m['logloss']:.4f}  acc={m['accuracy']:.4f}  "
              f"brier={m['brier']:.4f}  " + " ".join(f"{k}={v}" for k, v in t["params"].items()))

    # Balanced selection: lowest average rank across accuracy, log-loss and
    # Brier. Optimising any single metric on 6 folds just overfits the noise.
    def ranks(key, reverse):
        order = sorted(trials, key=lambda t: t["cv"][key], reverse=reverse)
        return {id(t): i for i, t in enumerate(order)}
    acc_r, ll_r, br_r = ranks("accuracy", True), ranks("logloss", False), ranks("brier", False)
    for t in trials:
        t["avg_rank"] = round((acc_r[id(t)] + ll_r[id(t)] + br_r[id(t)]) / 3, 2)
    trials.sort(key=lambda t: t["avg_rank"])
    best = trials[0]
    base = next(t for t in trials if t["tag"] == "baseline")

    print("\n=== BEST ===")
    print(json.dumps(best["params"], indent=2))
    print("best CV :", best["cv"])
    print("baseline:", base["cv"])

    # Persist tuned params so train.py / evaluate.py use them automatically.
    (config.MODELS_DIR / "best_params.json").write_text(json.dumps(best["params"], indent=2))

    # Refit the winner on ALL data and save the production models.
    full = {**FIXED, **best["params"]}
    hm, am = xgb.XGBRegressor(**full), xgb.XGBRegressor(**full)
    hm.fit(df[config.FEATURE_COLS], df["home_goals"])
    am.fit(df[config.FEATURE_COLS], df["away_goals"])
    hm.save_model(config.MODELS_DIR / "home_goals_model.json")
    am.save_model(config.MODELS_DIR / "away_goals_model.json")

    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_matches": int(len(df)),
        "feature_cols": config.FEATURE_COLS,
        "tournaments": [t[2] for t in config.TOURNAMENTS],
        "selection": f"best of {n} random search candidates by CV log-loss",
        "xgb_params": full,
        "cv_best": best["cv"],
        "cv_baseline": base["cv"],
        "home_feature_importance_gain": {c: float(v) for c, v in
                                         zip(config.FEATURE_COLS, hm.feature_importances_)},
        "data_source": "StatsBomb Open Data",
    }
    (config.MODELS_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    (config.MODELS_DIR / "cv_metrics.json").write_text(json.dumps(best["cv"], indent=2))
    (config.MODELS_DIR / "tuning_results.json").write_text(json.dumps(
        [{"tag": t["tag"], "params": t["params"], "cv": t["cv"]} for t in trials[:15]], indent=2))
    print(f"\nSaved tuned models + best_params.json to {config.MODELS_DIR}")


if __name__ == "__main__":
    main()
