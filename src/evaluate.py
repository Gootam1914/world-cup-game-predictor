"""Leave-one-tournament-out cross-validation.

Holding out an entire tournament at a time is the honest test for this model:
it measures how well team-strength features transfer to an unseen competition,
which is exactly the World-Cup use case.

Run:  python -m src.evaluate
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import config
from src.model import build_prediction
from src.train import load_training_frame, make_models

EPS = 1e-12


def _outcome(hg: int, ag: int) -> int:
    return 0 if hg > ag else (1 if hg == ag else 2)  # home / draw / away


def cross_validate() -> dict:
    df = load_training_frame()
    folds = df.groupby(["competition_id", "season_id"])

    g_abs, g_sq = [], []          # goal errors
    correct = tot = 0
    loglosses, briers = [], []
    per_tourney = {}

    for (cid, sid), test in folds:
        train = df[~df.index.isin(test.index)]
        hm, am = make_models()
        hm.fit(train[config.FEATURE_COLS], train["home_goals"])
        am.fit(train[config.FEATURE_COLS], train["away_goals"])

        lam_h = hm.predict(test[config.FEATURE_COLS])
        lam_a = am.predict(test[config.FEATURE_COLS])

        f_correct = f_tot = 0
        for i, (_, row) in enumerate(test.iterrows()):
            pred = build_prediction(row.home_team, row.away_team, lam_h[i], lam_a[i])
            probs = np.array([pred.p_home_win, pred.p_draw, pred.p_away_win])
            pred_cls = int(probs.argmax())
            true_cls = _outcome(row.home_goals, row.away_goals)

            g_abs += [abs(lam_h[i] - row.home_goals), abs(lam_a[i] - row.away_goals)]
            g_sq += [(lam_h[i] - row.home_goals) ** 2, (lam_a[i] - row.away_goals) ** 2]

            correct += pred_cls == true_cls
            f_correct += pred_cls == true_cls
            tot += 1
            f_tot += 1

            onehot = np.zeros(3); onehot[true_cls] = 1
            loglosses.append(-np.log(max(probs[true_cls], EPS)))
            briers.append(float(((probs - onehot) ** 2).sum()))

        label = next(t[2] for t in config.TOURNAMENTS if t[0] == cid and t[1] == sid)
        per_tourney[label] = round(f_correct / f_tot, 3)

    metrics = {
        "n_matches": int(tot),
        "goals_MAE": round(float(np.mean(g_abs)), 3),
        "goals_RMSE": round(float(np.sqrt(np.mean(g_sq))), 3),
        "outcome_accuracy": round(correct / tot, 3),
        "multiclass_logloss": round(float(np.mean(loglosses)), 3),
        "multiclass_brier": round(float(np.mean(briers)), 3),
        "accuracy_by_tournament": per_tourney,
        "baseline_always_home_win": round(_baseline(df, 0), 3),
        "baseline_always_draw": round(_baseline(df, 1), 3),
    }
    return metrics


def _baseline(df: pd.DataFrame, cls: int) -> float:
    trues = [_outcome(r.home_goals, r.away_goals) for _, r in df.iterrows()]
    return float(np.mean([t == cls for t in trues]))


if __name__ == "__main__":
    m = cross_validate()
    (config.MODELS_DIR / "cv_metrics.json").write_text(json.dumps(m, indent=2))
    print(json.dumps(m, indent=2))
