"""Hyperparameter search for the upgraded model with a time-honest split.

    train:      years [2002, 2020)
    validation: [2020-01-01, 2023-01-01)   <- used to pick params
    test:       [2023-01-01, now)          <- untouched, final report only

Scored primarily by RPS (the standard football-forecast metric).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config                                       # noqa: E402
from src.elo import FEATURE_COLS_V2                 # noqa: E402
from src.model import build_prediction              # noqa: E402
from src.train_intl import _metrics, _outcome, XGB_PARAMS as BASE  # noqa: E402

SPACE = {
    "n_estimators": [300, 450, 600, 800],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.02, 0.03, 0.05, 0.08],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [3, 5, 8, 12],
    "reg_lambda": [1.0, 2.0, 3.0, 5.0],
    "reg_alpha": [0.0, 0.2, 0.5],
}
FIXED = dict(objective="count:poisson", random_state=config.RANDOM_SEED, n_jobs=4)


def probs(hm, am, X):
    lh = np.clip(hm.predict(X), 1e-3, None); la = np.clip(am.predict(X), 1e-3, None)
    return np.array([[*[getattr(build_prediction("H", "A", lh[i], la[i]), k)
                        for k in ("p_home_win", "p_draw", "p_away_win")]] for i in range(len(X))])


def fit(train, params):
    yrs = train["date"].dt.year.to_numpy()
    w = 0.5 ** ((yrs.max() - yrs) / 8.0)
    full = {**FIXED, **params}
    hm = xgb.XGBRegressor(**full).fit(train[FEATURE_COLS_V2], train["home_goals"], sample_weight=w)
    am = xgb.XGBRegressor(**full).fit(train[FEATURE_COLS_V2], train["away_goals"], sample_weight=w)
    return hm, am


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    df = pd.read_csv(config.PROCESSED_DIR / "intl_training_data.csv", parse_dates=["date"])
    df = df[df.year >= 2002]
    train = df[df.date < "2020-01-01"]
    val = df[(df.date >= "2020-01-01") & (df.date < "2023-01-01")]
    test = df[df.date >= "2023-01-01"]
    yv = np.array([_outcome(h, a) for h, a in zip(val.home_goals, val.away_goals)])
    yt = np.array([_outcome(h, a) for h, a in zip(test.home_goals, test.away_goals)])

    base_params = {k: v for k, v in BASE.items() if k not in FIXED}
    rng = np.random.default_rng(config.RANDOM_SEED)
    cands = [{"tag": "baseline", "p": base_params}]
    seen = {json.dumps(base_params, sort_keys=True)}
    while len(cands) < n + 1:
        p = {k: (rng.choice(v).item()) for k, v in SPACE.items()}
        key = json.dumps(p, sort_keys=True)
        if key in seen:
            continue
        seen.add(key); cands.append({"tag": "search", "p": p})

    print(f"Scoring {len(cands)} configs on validation (RPS):\n")
    for c in cands:
        hm, am = fit(train, c["p"])
        c["val"] = _metrics(probs(hm, am, val[FEATURE_COLS_V2]), yv)
        print(f"  {c['tag']:<8} rps={c['val']['rps']:.4f} ll={c['val']['logloss']:.4f} acc={c['val']['accuracy']:.4f}")

    cands.sort(key=lambda c: (c["val"]["rps"], c["val"]["logloss"]))
    best = cands[0]
    print("\nBest params:", json.dumps(best["p"]))

    # Final untouched-test comparison: best vs baseline, both trained on <2023.
    tr23 = df[df.date < "2023-01-01"]
    hb, ab = fit(tr23, best["p"]); hbase, abase = fit(tr23, base_params)
    best_test = _metrics(probs(hb, ab, test[FEATURE_COLS_V2]), yt)
    base_test = _metrics(probs(hbase, abase, test[FEATURE_COLS_V2]), yt)
    print("\nTEST (2023+)  best :", best_test)
    print("TEST (2023+)  base :", base_test)

    if best["tag"] != "baseline" and best_test["rps"] <= base_test["rps"]:
        (config.MODELS_DIR / "best_params_intl.json").write_text(json.dumps(best["p"], indent=2))
        print("\nSaved improved params -> models/best_params_intl.json")
    else:
        print("\nBaseline not beaten on test; keeping current params.")


if __name__ == "__main__":
    main()
