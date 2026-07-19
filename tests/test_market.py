"""Tests for the market-odds vig removal and blending math (run offline)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.market_odds import shin_devig, multiplicative_devig, log_opinion_pool


def test_shin_devig_sums_to_one():
    p = shin_devig(2.7, 3.3, 2.6)
    assert abs(sum(p) - 1.0) < 1e-9
    assert all(0 < x < 1 for x in p)


def test_shin_removes_overround():
    # booked probabilities sum to > 1; de-vigged must sum to exactly 1
    oh, od, oa = 1.5, 4.0, 7.0
    booked = 1 / oh + 1 / od + 1 / oa
    assert booked > 1.0
    p = shin_devig(oh, od, oa)
    assert abs(sum(p) - 1.0) < 1e-9
    assert p[0] > p[1] > p[2]  # favourite ordering preserved


def test_log_pool_moves_toward_market():
    model = [0.45, 0.30, 0.25]
    market = [0.60, 0.25, 0.15]
    blended = log_opinion_pool(model, market, w_market=0.6)
    assert abs(sum(blended) - 1.0) < 1e-9
    assert blended[0] > model[0]          # pulled toward sharper market
    assert max(blended) > max(model)      # confidence increased


def test_log_pool_endpoints():
    model, market = [0.4, 0.35, 0.25], [0.7, 0.2, 0.1]
    assert log_opinion_pool(model, market, 0.0) == model
    b1 = log_opinion_pool(model, market, 1.0)
    assert all(abs(a - b) < 1e-9 for a, b in zip(b1, market))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"PASS  {name}")
    print("All market tests passed.")
