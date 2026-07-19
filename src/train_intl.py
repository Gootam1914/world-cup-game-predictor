"""Train the upgraded model: a Poisson goals model + a direct outcome
classifier, blended into an ensemble, on ~150 years of results.

The two expected-goals regressors give the *scoreline*; the ensemble of their
implied win/draw/loss probabilities with a direct multiclass classifier gives
the *outcome probabilities* and confidence. Evaluated with an honest
chronological backtest (train pre-2023, test on unseen 2023+ matches).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

import config
from src.elo import FEATURE_COLS_V2
from src.model import build_prediction, DC_RHO

TRAIN_FROM = 1994           # widened window -> more data
TEST_CUTOFF = "2023-01-01"
HALF_LIFE_YEARS = 8.0
BLEND = (0.60, 0.40)        # (goals-model, classifier) weights for outcome probs

XGB_PARAMS = dict(
    objective="count:poisson", n_estimators=450, max_depth=4, learning_rate=0.03,
    subsample=0.85, colsample_bytree=0.85, min_child_weight=5,
    reg_lambda=2.0, reg_alpha=0.2, random_state=config.RANDOM_SEED, n_jobs=4,
)
CLF_PARAMS = dict(
    objective="multi:softprob", num_class=3, n_estimators=400, max_depth=4,
    learning_rate=0.03, subsample=0.85, colsample_bytree=0.85, min_child_weight=5,
    reg_lambda=2.0, reg_alpha=0.2, random_state=config.RANDOM_SEED, n_jobs=4,
)


def _outcome(hg, ag):
    return 0 if hg > ag else (1 if hg == ag else 2)


def _metrics(probs, trues):
    probs = np.asarray(probs); trues = np.asarray(trues)
    acc = float(np.mean(probs.argmax(1) == trues))
    oh = np.eye(3)[trues]
    ll = float(np.mean(-np.log(np.clip(probs[np.arange(len(trues)), trues], 1e-12, 1))))
    brier = float(np.mean(((probs - oh) ** 2).sum(1)))
    cum_p = np.cumsum(probs, 1); cum_o = np.cumsum(oh, 1)
    rps = float(np.mean(((cum_p - cum_o) ** 2)[:, :2].sum(1)))
    return {"accuracy": round(acc, 4), "logloss": round(ll, 4),
            "brier": round(brier, 4), "rps": round(rps, 4)}


def _load():
    df = pd.read_csv(config.PROCESSED_DIR / "intl_training_data.csv", parse_dates=["date"])
    return df[df.year >= TRAIN_FROM].reset_index(drop=True)


def _weights(train):
    yrs = train["date"].dt.year.to_numpy()
    return 0.5 ** ((yrs.max() - yrs) / HALF_LIFE_YEARS)


def _fit(train):
    w = _weights(train)
    hm = xgb.XGBRegressor(**XGB_PARAMS).fit(train[FEATURE_COLS_V2], train["home_goals"], sample_weight=w)
    am = xgb.XGBRegressor(**XGB_PARAMS).fit(train[FEATURE_COLS_V2], train["away_goals"], sample_weight=w)
    y = [_outcome(h, a) for h, a in zip(train.home_goals, train.away_goals)]
    clf = xgb.XGBClassifier(**CLF_PARAMS).fit(train[FEATURE_COLS_V2], y, sample_weight=w)
    return hm, am, clf


def goals_probs(hm, am, X):
    lh = np.clip(hm.predict(X), 1e-3, None); la = np.clip(am.predict(X), 1e-3, None)
    return np.array([[getattr(build_prediction("H", "A", lh[i], la[i], rho=DC_RHO), k)
                      for k in ("p_home_win", "p_draw", "p_away_win")] for i in range(len(X))])


def ensemble_probs(hm, am, clf, X):
    p1 = goals_probs(hm, am, X)
    p2 = clf.predict_proba(X)
    p = BLEND[0] * p1 + BLEND[1] * p2
    return p / p.sum(1, keepdims=True)


def backtest():
    df = _load()
    train = df[df.date < TEST_CUTOFF]
    test = df[df.date >= TEST_CUTOFF]
    trues = np.array([_outcome(h, a) for h, a in zip(test.home_goals, test.away_goals)])
    hm, am, clf = _fit(train)

    ens = ensemble_probs(hm, am, clf, test[FEATURE_COLS_V2])
    goals_only = goals_probs(hm, am, test[FEATURE_COLS_V2])

    elo = LogisticRegression(max_iter=1000).fit(
        train[["elo_diff"]], [_outcome(h, a) for h, a in zip(train.home_goals, train.away_goals)])
    ep = np.zeros((len(test), 3))
    for j, c in enumerate(elo.classes_):
        ep[:, c] = elo.predict_proba(test[["elo_diff"]])[:, j]

    return {
        "train_matches": int(len(train)), "test_matches": int(len(test)),
        "test_period": f"{TEST_CUTOFF} to {str(df.date.max().date())}",
        "ensemble_model": _metrics(ens, trues),
        "goals_model_only": _metrics(goals_only, trues),
        "elo_only_baseline": _metrics(ep, trues),
        "always_home_baseline": {"accuracy": round(float(np.mean(trues == 0)), 4)},
    }


def train_and_save():
    results = backtest()
    df = _load()
    hm, am, clf = _fit(df)
    hm.save_model(config.MODELS_DIR / "intl_home_model.json")
    am.save_model(config.MODELS_DIR / "intl_away_model.json")
    clf.save_model(config.MODELS_DIR / "intl_clf_model.json")

    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model": "Ensemble: XGBoost Poisson goals (0.6) + XGBoost outcome classifier (0.4)",
        "feature_cols": FEATURE_COLS_V2, "blend_weights": BLEND,
        "train_from_year": TRAIN_FROM, "n_train_matches": int(len(df)),
        "backtest": results, "data_source": "martj42/international_results (~49k matches)",
    }
    (config.MODELS_DIR / "intl_metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(results, indent=2))
    print("\nSaved ensemble models to", config.MODELS_DIR)
    return results


if __name__ == "__main__":
    train_and_save()
