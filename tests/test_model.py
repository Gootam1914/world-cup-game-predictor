"""Sanity tests for the scoreline maths and the prediction pipeline.

Run:  python -m pytest -q     (or)     python tests/test_model.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.model import build_prediction


def test_probabilities_sum_to_one():
    p = build_prediction("A", "B", 1.4, 1.1)
    total = p.p_home_win + p.p_draw + p.p_away_win
    # values are rounded to 4dp for display, so allow a small tolerance
    assert abs(total - 1.0) < 2e-3


def test_stronger_team_favoured():
    p = build_prediction("Strong", "Weak", 2.4, 0.6)
    assert p.p_home_win > p.p_away_win
    assert p.likely_home_goals >= p.likely_away_goals
    assert p.confidence_band in {"Low", "Medium", "High"}


def test_symmetry():
    p1 = build_prediction("A", "B", 1.8, 1.0)
    p2 = build_prediction("B", "A", 1.0, 1.8)
    assert abs(p1.p_home_win - p2.p_away_win) < 1e-9
    assert abs(p1.p_draw - p2.p_draw) < 1e-9


def test_feature_schema_is_twenty():
    assert len(config.FEATURE_COLS) == 20
    assert config.FEATURE_COLS[-1] == "is_neutral"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
    print("All tests passed.")
