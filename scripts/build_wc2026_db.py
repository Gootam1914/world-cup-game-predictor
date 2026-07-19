"""Build a curated 2026 FIFA World Cup database from the real results.

Extracts every played 2026 World Cup match from the international results
dataset and computes a per-team tournament summary (played, W/D/L, goals for/
against, goal difference, points), joined with each team's current Elo rating.
All figures are REAL match results — nothing is fabricated. Unplayed fixtures
(e.g. the final before it kicks off) are listed separately with no score.

Outputs:
    data/world_cup_2026_results.csv
    data/world_cup_2026_standings.csv

Run:  python scripts/build_wc2026_db.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def main():
    df = pd.read_csv(config.INTL_RESULTS_CSV, parse_dates=["date"])
    wc = df[(df.tournament == "FIFA World Cup") & (df.date >= "2026-06-01")].copy()
    wc = wc.sort_values("date")

    played = wc[wc.home_score.notna()].copy()
    played["home_score"] = played.home_score.astype(int)
    played["away_score"] = played.away_score.astype(int)
    unplayed = wc[wc.home_score.isna()]

    results = wc[["date", "home_team", "away_team", "home_score", "away_score", "city", "country"]]
    results.to_csv(config.ROOT / "data" / "world_cup_2026_results.csv", index=False)

    # Per-team tournament summary from the played matches.
    rows = {}
    def bump(team):
        rows.setdefault(team, dict(P=0, W=0, D=0, L=0, GF=0, GA=0))
        return rows[team]
    for r in played.itertuples():
        h, a = bump(r.home_team), bump(r.away_team)
        h["P"] += 1; a["P"] += 1
        h["GF"] += r.home_score; h["GA"] += r.away_score
        a["GF"] += r.away_score; a["GA"] += r.home_score
        if r.home_score > r.away_score:
            h["W"] += 1; a["L"] += 1
        elif r.home_score < r.away_score:
            a["W"] += 1; h["L"] += 1
        else:
            h["D"] += 1; a["D"] += 1

    ratings = json.loads((config.PROCESSED_DIR / "current_ratings.json").read_text())
    table = []
    for team, s in rows.items():
        table.append({
            "team": team, **s, "GD": s["GF"] - s["GA"],
            "points": s["W"] * 3 + s["D"],
            "current_elo": round(ratings.get(team, {}).get("elo", 1500)),
        })
    standings = pd.DataFrame(table).sort_values(
        ["points", "GD", "GF"], ascending=False).reset_index(drop=True)
    standings.to_csv(config.ROOT / "data" / "world_cup_2026_standings.csv", index=False)

    print(f"2026 World Cup: {len(played)} played, {len(unplayed)} unplayed, {len(rows)} teams")
    print("\nTop of the tournament by points:")
    print(standings.head(8).to_string(index=False))
    if len(unplayed):
        print("\nStill to play:")
        print(unplayed[["date", "home_team", "away_team"]].to_string(index=False))


if __name__ == "__main__":
    main()
