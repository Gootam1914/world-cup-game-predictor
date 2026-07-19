"""Smoke tests for the upgraded Elo + form prediction pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import predict as P
from src.elo import FEATURE_COLS_V2


def test_active_team_list():
    teams = P.available_teams()
    assert len(teams) > 100
    names = {t["name"] for t in teams}
    assert {"Argentina", "Spain", "Brazil", "France"} <= names
    # sorted strongest-first
    assert teams[0]["elo"] >= teams[-1]["elo"]


def test_probabilities_sum_to_one():
    p = P.predict_match("Argentina", "Spain").to_dict()
    total = p["p_home_win"] + p["p_draw"] + p["p_away_win"]
    assert abs(total - 1.0) < 2e-3


def test_strong_beats_weak():
    # Brazil vs a minnow: Brazil should be a heavy favourite, high confidence.
    p = P.predict_match("Brazil", "Gibraltar").to_dict()
    assert p["p_home_win"] > 0.7
    assert p["confidence_band"] in {"Medium", "High"}


def test_feature_vector_shape():
    X = P._feature_row("France", "Morocco", neutral=True, importance=4)
    assert list(X.columns) == FEATURE_COLS_V2
    assert X.shape == (1, len(FEATURE_COLS_V2))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"PASS  {name}")
    print("All predict tests passed.")
