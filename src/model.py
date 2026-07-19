"""Scoreline maths shared by training, evaluation and serving.

The two XGBoost regressors output expected goals (lambda) for each team. We
treat the two scores as independent Poisson variables and enumerate the full
score matrix to derive: the most likely scoreline, win/draw/loss probabilities,
and a confidence score.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy.stats import poisson

from config import MAX_GOALS


@dataclass
class Prediction:
    home_team: str
    away_team: str
    exp_home_goals: float
    exp_away_goals: float
    likely_home_goals: int
    likely_away_goals: int
    p_home_win: float
    p_draw: float
    p_away_win: float
    confidence: float          # 0-100, probability mass on the predicted outcome
    confidence_band: str       # Low / Medium / High
    top_scorelines: list       # [(h, a, prob), ...] most probable exact scores

    def to_dict(self) -> dict:
        return asdict(self)


def _score_matrix(lam_home: float, lam_away: float) -> np.ndarray:
    lam_home = max(float(lam_home), 1e-6)
    lam_away = max(float(lam_away), 1e-6)
    h = poisson.pmf(np.arange(MAX_GOALS + 1), lam_home)
    a = poisson.pmf(np.arange(MAX_GOALS + 1), lam_away)
    return np.outer(h, a)  # M[i, j] = P(home=i, away=j)


def _confidence_band(conf: float) -> str:
    if conf >= 60:
        return "High"
    if conf >= 45:
        return "Medium"
    return "Low"


def build_prediction(home_team: str, away_team: str,
                     lam_home: float, lam_away: float) -> Prediction:
    m = _score_matrix(lam_home, lam_away)
    p_home = float(np.tril(m, -1).sum())   # home > away
    p_away = float(np.triu(m, 1).sum())    # away > home
    p_draw = float(np.trace(m))
    # renormalise (Poisson tail beyond MAX_GOALS is tiny but keep it clean)
    tot = p_home + p_draw + p_away
    p_home, p_draw, p_away = p_home / tot, p_draw / tot, p_away / tot

    hi, ai = np.unravel_index(int(m.argmax()), m.shape)

    flat = [(i, j, float(m[i, j])) for i in range(m.shape[0]) for j in range(m.shape[1])]
    flat.sort(key=lambda t: t[2], reverse=True)
    top = [(int(i), int(j), round(p / tot, 4)) for i, j, p in flat[:5]]

    confidence = max(p_home, p_draw, p_away) * 100.0
    return Prediction(
        home_team=home_team,
        away_team=away_team,
        exp_home_goals=round(float(lam_home), 2),
        exp_away_goals=round(float(lam_away), 2),
        likely_home_goals=int(hi),
        likely_away_goals=int(ai),
        p_home_win=round(p_home, 4),
        p_draw=round(p_draw, 4),
        p_away_win=round(p_away, 4),
        confidence=round(confidence, 1),
        confidence_band=_confidence_band(confidence),
        top_scorelines=top,
    )
