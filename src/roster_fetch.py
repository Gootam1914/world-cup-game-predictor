"""Fetch CURRENT national-team squads from API-Football (optional).

This is the "current rosters" layer. It needs an API key and live internet, so
it runs on YOUR machine, not in the training sandbox. Everything degrades
gracefully when ``API_FOOTBALL_KEY`` is missing: the app still works using the
model's historical team-strength features.

Get a free key at https://www.api-football.com/ and put it in .env:
    API_FOOTBALL_KEY=your_key_here

Usage:
    python -m src.roster_fetch Argentina
"""
from __future__ import annotations

import json
import os
import sys
from typing import Optional

import requests
from dotenv import load_dotenv

import config

load_dotenv()

BASE = "https://v3.football.api-sports.io"
_CACHE = config.ASSETS_DIR / "current_squads.json"


def _key() -> Optional[str]:
    return os.environ.get("API_FOOTBALL_KEY") or None


def _headers() -> dict:
    return {"x-apisports-key": _key()}


def has_key() -> bool:
    return bool(_key())


def _get(path: str, params: dict) -> dict:
    resp = requests.get(f"{BASE}/{path}", headers=_headers(), params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()


def _team_id(team_name: str) -> Optional[int]:
    data = _get("teams", {"name": team_name})
    results = data.get("response", [])
    if not results:  # national teams sometimes need the country param
        data = _get("teams", {"search": team_name})
        results = data.get("response", [])
    for r in results:
        if r["team"].get("national"):
            return r["team"]["id"]
    return results[0]["team"]["id"] if results else None


def get_current_squad(team_name: str, use_cache: bool = True) -> Optional[list[dict]]:
    """Return a list of {name, age, number, position} for the current squad.

    Returns None (and prints a hint) if no key is configured or the call fails.
    """
    if use_cache and _CACHE.exists():
        cached = json.loads(_CACHE.read_text())
        if team_name in cached:
            return cached[team_name]

    if not has_key():
        print("[roster] API_FOOTBALL_KEY not set — skipping live roster fetch.")
        return None

    try:
        tid = _team_id(team_name)
        if tid is None:
            print(f"[roster] no team id found for {team_name!r}")
            return None
        data = _get("players/squads", {"team": tid})
        players = data.get("response", [{}])[0].get("players", [])
        squad = [
            {
                "name": p.get("name"),
                "age": p.get("age"),
                "number": p.get("number"),
                "position": p.get("position"),
            }
            for p in players
        ]
        _update_cache(team_name, squad)
        return squad
    except requests.RequestException as exc:
        print(f"[roster] request failed for {team_name}: {exc}")
        return None


def _update_cache(team_name: str, squad: list[dict]) -> None:
    cache = json.loads(_CACHE.read_text()) if _CACHE.exists() else {}
    cache[team_name] = squad
    _CACHE.write_text(json.dumps(cache, indent=2))


if __name__ == "__main__":
    team = sys.argv[1] if len(sys.argv) > 1 else "Argentina"
    squad = get_current_squad(team)
    if squad:
        print(f"{team}: {len(squad)} players")
        for p in squad[:11]:
            print(f"  {p['number'] or '-':>3}  {p['name']}  ({p['position']})")
    else:
        print("No squad returned (missing key or offline). App still works without it.")
