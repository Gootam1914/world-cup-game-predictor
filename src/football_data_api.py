"""Gap-fill recent international results from football-data.org (optional).

StatsBomb publishes rich event data but with a lag. This module appends recent
international fixtures/results (World Cup, Euro, friendlies) so the team-strength
profiles can be kept as current as possible. It only fills in match-level goals
(no event detail), which still feeds goals_for / goals_against.

Needs a free key from https://www.football-data.org/ in .env:
    FOOTBALL_DATA_ORG_KEY=your_key_here

Everything degrades gracefully when the key is missing.

Usage:
    python -m src.football_data_api
"""
from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv

import config

load_dotenv()

BASE = "https://api.football-data.org/v4"
# football-data.org competition codes for international tournaments
INTL_COMPETITIONS = {"WC": "FIFA World Cup", "EC": "European Championship"}


def _key() -> Optional[str]:
    return os.environ.get("FOOTBALL_DATA_ORG_KEY") or None


def has_key() -> bool:
    return bool(_key())


def _get(path: str, params: dict) -> dict:
    resp = requests.get(
        f"{BASE}/{path}",
        headers={"X-Auth-Token": _key()},
        params=params,
        timeout=20,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_recent_results(days_back: int = 540) -> Optional[pd.DataFrame]:
    """Finished international matches in the last ``days_back`` days.

    Returns a DataFrame [match_date, home_team, away_team, home_score,
    away_score, competition] or None if unavailable.
    """
    if not has_key():
        print("[football-data] FOOTBALL_DATA_ORG_KEY not set — skipping gap-fill.")
        return None

    date_to = date.today()
    date_from = date_to - timedelta(days=days_back)
    rows = []
    try:
        for code, label in INTL_COMPETITIONS.items():
            data = _get(
                f"competitions/{code}/matches",
                {"status": "FINISHED",
                 "dateFrom": date_from.isoformat(),
                 "dateTo": date_to.isoformat()},
            )
            for m in data.get("matches", []):
                ft = m["score"]["fullTime"]
                if ft["home"] is None:
                    continue
                rows.append(
                    {
                        "match_date": m["utcDate"][:10],
                        "home_team": m["homeTeam"]["name"],
                        "away_team": m["awayTeam"]["name"],
                        "home_score": ft["home"],
                        "away_score": ft["away"],
                        "competition": label,
                    }
                )
    except requests.RequestException as exc:
        print(f"[football-data] request failed: {exc}")
        return None

    if not rows:
        return None
    df = pd.DataFrame(rows)
    out = config.PROCESSED_DIR / "recent_results.csv"
    df.to_csv(out, index=False)
    print(f"[football-data] wrote {len(df)} recent results -> {out}")
    return df


if __name__ == "__main__":
    df = fetch_recent_results()
    if df is not None:
        print(df.tail(10).to_string(index=False))
    else:
        print("No data returned (missing key or offline). Model still works without it.")
