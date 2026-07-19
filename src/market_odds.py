"""Market-odds integration — the single biggest lever for justified confidence.

Research consensus (Wunderlich & Memmert 2018; Hvattum & Arntzen 2010; Štrumbelj
2014): pre-match betting odds are the strongest, best-calibrated predictor of
football outcomes and beat rating/statistical models head to head. This module:

1. fetches live 1X2 odds for a real fixture from The Odds API (free tier, key in
   .env) across the national-team competition keys;
2. removes the bookmaker margin ("vig") with **Shin's method**, which the
   literature finds gives the best-calibrated true probabilities;
3. blends the market probabilities with the model's via a **logarithmic opinion
   pool** (sharper than a linear average when both are calibrated).

Everything degrades gracefully: no key, no internet, or no scheduled fixture ->
the app simply uses the pure model.

Get a free key (500 req/mo) at https://the-odds-api.com and put it in .env:
    ODDS_API_KEY=your_key_here
"""
from __future__ import annotations

import math
import os
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

BASE = "https://api.the-odds-api.com/v4"

# National-team competition sport keys (The Odds API). Only in-season keys
# return data; out-of-season keys return an empty list (and cost no credits).
SOCCER_INTL_KEYS = [
    "soccer_fifa_world_cup",
    "soccer_fifa_world_cup_qualifiers_europe",
    "soccer_fifa_world_cup_qualifiers_south_america",
    "soccer_uefa_nations_league",
    "soccer_uefa_european_championship",
    "soccer_uefa_euro_qualification",
    "soccer_conmebol_copa_america",
    "soccer_africa_cup_of_nations",
    "soccer_concacaf_gold_cup",
]

# Map our team names -> names The Odds API is likely to use.
NAME_ALIASES = {
    "United States": "USA", "South Korea": "Korea Republic",
    "North Korea": "Korea DPR", "Côte d'Ivoire": "Ivory Coast",
    "Congo DR": "DR Congo", "Czech Republic": "Czechia",
    "Cape Verde Islands": "Cape Verde",
}

MARKET_BLEND_WEIGHT = float(os.environ.get("MARKET_BLEND_WEIGHT", "0.6"))


def _key() -> Optional[str]:
    return os.environ.get("ODDS_API_KEY") or None


def has_key() -> bool:
    return bool(_key())


# --------------------------------------------------------------------------- #
# Vig removal — Shin's method
# --------------------------------------------------------------------------- #
def shin_devig(odds_home: float, odds_draw: float, odds_away: float) -> tuple[float, float, float]:
    """Return de-vigged (true) probabilities from three decimal 1X2 odds.

    Shin's model treats the margin as protection against informed bettors and
    solves for z (the share of "smart money"). Solved here by bisection.
    """
    pis = [1.0 / odds_home, 1.0 / odds_draw, 1.0 / odds_away]
    book = sum(pis)  # > 1 (the overround)

    def p_of_z(z: float) -> list[float]:
        out = []
        for pi in pis:
            num = math.sqrt(z * z + 4 * (1 - z) * pi * pi / book) - z
            out.append(num / (2 * (1 - z)))
        return out

    lo, hi = 0.0, 0.5
    for _ in range(60):
        mid = (lo + hi) / 2
        s = sum(p_of_z(mid))
        if s > 1:
            lo = mid
        else:
            hi = mid
    p = p_of_z((lo + hi) / 2)
    total = sum(p)
    return tuple(x / total for x in p)  # type: ignore


def multiplicative_devig(oh: float, od: float, oa: float) -> tuple[float, float, float]:
    pis = [1 / oh, 1 / od, 1 / oa]
    s = sum(pis)
    return tuple(p / s for p in pis)  # type: ignore


# --------------------------------------------------------------------------- #
# Blending — logarithmic opinion pool
# --------------------------------------------------------------------------- #
def log_opinion_pool(p_model, p_market, w_market: float = MARKET_BLEND_WEIGHT):
    """Weighted geometric mean of two probability vectors, renormalised."""
    w = min(max(w_market, 0.0), 1.0)
    blended = [ (m ** (1 - w)) * (k ** w) for m, k in zip(p_model, p_market) ]
    s = sum(blended)
    return [b / s for b in blended]


# --------------------------------------------------------------------------- #
# Fetching
# --------------------------------------------------------------------------- #
def _norm(name: str) -> str:
    return NAME_ALIASES.get(name, name).lower().strip()


def fetch_match_odds(home: str, away: str, regions: str = "eu") -> Optional[dict]:
    """Find a scheduled fixture between the two teams and return de-vigged
    market probabilities (averaged across bookmakers). None if unavailable.
    """
    if not has_key():
        return None
    h, a = _norm(home), _norm(away)
    try:
        for key in SOCCER_INTL_KEYS:
            resp = requests.get(
                f"{BASE}/sports/{key}/odds",
                params={"apiKey": _key(), "regions": regions,
                        "markets": "h2h", "oddsFormat": "decimal"},
                timeout=20,
            )
            if resp.status_code != 200:
                continue
            for event in resp.json():
                teams = {_norm(event.get("home_team", "")), _norm(event.get("away_team", ""))}
                if h in teams and a in teams:
                    return _extract(event, home, away, key)
    except requests.RequestException as exc:
        print(f"[odds] request failed: {exc}")
    return None


def _extract(event: dict, home: str, away: str, key: str) -> Optional[dict]:
    hs, ds, as_ = [], [], []
    ev_home = event.get("home_team", "")
    for bk in event.get("bookmakers", []):
        for mk in bk.get("markets", []):
            if mk.get("key") != "h2h":
                continue
            price = {o["name"]: o["price"] for o in mk.get("outcomes", [])}
            draw = price.get("Draw")
            # orient odds to OUR home/away regardless of the fixture's designation
            oh = price.get(ev_home if _norm(ev_home) == _norm(home) else None)
            # robust lookup by matching names
            names = {_norm(k): v for k, v in price.items()}
            oh = names.get(_norm(home)); oa = names.get(_norm(away))
            if oh and oa and draw:
                hs.append(oh); ds.append(draw); as_.append(oa)
    if not hs:
        return None
    avg = lambda xs: sum(xs) / len(xs)
    p = shin_devig(avg(hs), avg(ds), avg(as_))
    return {
        "competition": key,
        "n_bookmakers": len(hs),
        "market_probs": {"home": round(p[0], 4), "draw": round(p[1], 4), "away": round(p[2], 4)},
        "avg_odds": {"home": round(avg(hs), 2), "draw": round(avg(ds), 2), "away": round(avg(as_), 2)},
    }


if __name__ == "__main__":
    # Demo the vig-removal math (works offline).
    demo = shin_devig(2.7, 3.3, 2.6)
    print("Shin de-vig of odds 2.70/3.30/2.60 ->",
          [round(x, 3) for x in demo], "sum", round(sum(demo), 4))
    print("Key configured:", has_key())
