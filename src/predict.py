"""Serving-time prediction using the upgraded Elo + form model.

Loads the XGBoost goal models trained on ~49k international matches plus each
team's current state (Elo rating, recent form, momentum) and produces a full
forecast for any matchup: projected score, win/draw/loss probabilities and a
confidence score. Applies the Dixon-Coles low-score correction.

    python -m src.predict Argentina Spain
"""
from __future__ import annotations

import functools
import json
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

import config
from src.elo import FEATURE_COLS_V2, CONF_CODES, HOME_ADV
from src.model import build_prediction, Prediction, DC_RHO, _confidence_band
from src.train_intl import BLEND

DEFAULT_IMPORTANCE = 4      # treat matchups as major-tournament (World Cup) games
ACTIVE_SINCE_YEAR = 2022    # only surface teams that have played recently


@functools.lru_cache(maxsize=1)
def _load():
    hm = xgb.XGBRegressor(); hm.load_model(config.MODELS_DIR / "intl_home_model.json")
    am = xgb.XGBRegressor(); am.load_model(config.MODELS_DIR / "intl_away_model.json")
    clf = xgb.XGBClassifier(); clf.load_model(config.MODELS_DIR / "intl_clf_model.json")
    ratings = json.loads((config.PROCESSED_DIR / "current_ratings.json").read_text())
    flags = json.loads((config.ROOT / "data" / "team_flags.json").read_text())
    results = pd.read_csv(config.INTL_RESULTS_CSV, parse_dates=["date"])
    return hm, am, clf, ratings, flags, results


def _flag(team: str) -> str:
    return _load()[4].get(team, "un")


def get_flag(team: str) -> str:
    return _flag(team)


@functools.lru_cache(maxsize=4096)
def _h2h(home: str, away: str):
    results = _load()[5]
    m = results[(((results.home_team == home) & (results.away_team == away)) |
                 ((results.home_team == away) & (results.away_team == home)))].tail(10)
    if m.empty:
        return 0.5, 0
    rate = []
    for r in m.itertuples():
        gd = r.home_score - r.away_score
        res = 1.0 if gd > 0 else (0.5 if gd == 0 else 0.0)
        rate.append(res if r.home_team == home else 1 - res)
    return float(np.mean(rate)), len(m)


def available_teams() -> list[dict]:
    _, _, _, ratings, flags, _ = _load()
    out = []
    for name, s in ratings.items():
        if name not in flags:
            continue
        yr = int(s["last_date"][:4]) if s.get("last_date") else 0
        if yr < ACTIVE_SINCE_YEAR:
            continue
        out.append({"name": name, "flag": flags[name],
                    "confederation": s["confederation"], "elo": round(s["elo"])})
    return sorted(out, key=lambda t: -t["elo"])


def _feature_row(home: str, away: str, neutral: bool, importance: int) -> pd.DataFrame:
    ratings = _load()[3]
    h, a = ratings[home], ratings[away]
    ha = 0.0 if neutral else HOME_ADV
    h2h_rate, h2h_played = _h2h(home, away)
    row = {
        "elo_home": h["elo"], "elo_away": a["elo"],
        "elo_diff": h["elo"] + ha - a["elo"],
        "home_gf_avg": h["gf_avg"], "home_ga_avg": h["ga_avg"], "home_pts_avg": h["pts_avg"],
        "away_gf_avg": a["gf_avg"], "away_ga_avg": a["ga_avg"], "away_pts_avg": a["pts_avg"],
        "home_rest_days": 7, "away_rest_days": 7,
        "h2h_home_rate": h2h_rate, "h2h_played": h2h_played,
        "neutral": int(neutral), "tournament_importance": importance,
        "home_confederation": CONF_CODES.get(h["confederation"], 6),
        "away_confederation": CONF_CODES.get(a["confederation"], 6),
        "home_elo_momentum": h.get("elo_momentum", 0.0),
        "away_elo_momentum": a.get("elo_momentum", 0.0),
    }
    return pd.DataFrame([row])[FEATURE_COLS_V2]


def predict_match(home: str, away: str, neutral: bool = True,
                  importance: int = DEFAULT_IMPORTANCE,
                  use_market: bool = False) -> Prediction:
    hm, am, clf, ratings, _, _ = _load()
    if home not in ratings or away not in ratings:
        raise ValueError("Unknown team")
    X = _feature_row(home, away, neutral, importance)
    lam_h = float(np.clip(hm.predict(X)[0], 1e-3, None))
    lam_a = float(np.clip(am.predict(X)[0], 1e-3, None))
    # Scoreline + goals-model probabilities from the Poisson model.
    pred = build_prediction(home, away, lam_h, lam_a, rho=DC_RHO)
    # Blend outcome probabilities with the direct classifier (the ensemble).
    p_goals = np.array([pred.p_home_win, pred.p_draw, pred.p_away_win])
    p_clf = clf.predict_proba(X)[0]
    blend = BLEND[0] * p_goals + BLEND[1] * p_clf
    blend = blend / blend.sum()

    # Optional: blend in live betting-market odds (the sharpest available
    # signal) via a logarithmic opinion pool -> genuinely higher confidence.
    if use_market:
        from src import market_odds
        info = market_odds.fetch_match_odds(home, away)
        if info:
            mp = info["market_probs"]
            blend = np.array(market_odds.log_opinion_pool(
                blend, [mp["home"], mp["draw"], mp["away"]]))
            blend = blend / blend.sum()
            pred.market_used = True
            pred.market_probs = mp

    pred.p_home_win, pred.p_draw, pred.p_away_win = (round(float(x), 4) for x in blend)
    pred.confidence = round(float(blend.max()) * 100, 1)
    pred.confidence_band = _confidence_band(pred.confidence)
    return pred


def team_elo(team: str) -> int:
    return round(_load()[3].get(team, {}).get("elo", 1500))


if __name__ == "__main__":
    home = sys.argv[1] if len(sys.argv) > 1 else "Argentina"
    away = sys.argv[2] if len(sys.argv) > 2 else "Spain"
    p = predict_match(home, away).to_dict()
    print(f"\n{home} (Elo {team_elo(home)})  vs  {away} (Elo {team_elo(away)})  [neutral]")
    print(f"  Projected score : {p['likely_home_goals']} - {p['likely_away_goals']}"
          f"   (xG {p['exp_home_goals']} - {p['exp_away_goals']})")
    print(f"  Win / Draw / Win: {p['p_home_win']*100:.0f}% / {p['p_draw']*100:.0f}% / {p['p_away_win']*100:.0f}%")
    print(f"  Confidence      : {p['confidence']}% ({p['confidence_band']})")
