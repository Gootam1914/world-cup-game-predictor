"""Serving-time prediction: pick two teams, get a full match forecast.

Loads the trained models once and reuses cached team profiles. Used by the
FastAPI backend and runnable directly from the command line:

    python -m src.predict Argentina Spain
"""
from __future__ import annotations

import functools
import sys

import pandas as pd
import xgboost as xgb

import config
from src import features as F
from src import statsbomb_loader as L
from src.model import Prediction, build_prediction


@functools.lru_cache(maxsize=1)
def _load():
    hm = xgb.XGBRegressor()
    hm.load_model(config.MODELS_DIR / "home_goals_model.json")
    am = xgb.XGBRegressor()
    am.load_model(config.MODELS_DIR / "away_goals_model.json")
    stats = pd.read_csv(config.PROCESSED_DIR / "team_match_stats.csv")
    matches = F.load_matches_cached()
    return hm, am, stats, matches


def available_teams() -> list[dict]:
    """Teams the model knows about, with flag code + confederation."""
    _, _, stats, _ = _load()
    teams = sorted(stats["team"].unique())
    return [
        {
            "name": t,
            "flag": config.get_flag_code(t),
            "confederation": config.get_confederation(t),
            "matches": int((stats["team"] == t).sum()),
        }
        for t in teams
    ]


def predict_match(home: str, away: str, neutral: bool = True) -> Prediction:
    hm, am, stats, matches = _load()
    X = F.build_matchup_features(home, away, int(neutral), stats=stats, matches=matches)
    lam_home = float(hm.predict(X)[0])
    lam_away = float(am.predict(X)[0])
    return build_prediction(home, away, lam_home, lam_away)


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        home, away = sys.argv[1], sys.argv[2]
    else:
        home, away = "Argentina", "Spain"
    pred = predict_match(home, away)
    d = pred.to_dict()
    print(f"\n{home} vs {away}  (neutral venue)")
    print(f"  Expected goals : {d['exp_home_goals']} - {d['exp_away_goals']}")
    print(f"  Likely score   : {d['likely_home_goals']} - {d['likely_away_goals']}")
    print(f"  Win / Draw / Win: {d['p_home_win']*100:.0f}% / "
          f"{d['p_draw']*100:.0f}% / {d['p_away_win']*100:.0f}%")
    print(f"  Confidence     : {d['confidence']}% ({d['confidence_band']})")
    print(f"  Top scorelines : "
          + ", ".join(f"{h}-{a} ({p*100:.0f}%)" for h, a, p in d["top_scorelines"][:3]))
