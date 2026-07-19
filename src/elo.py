"""Elo ratings + match features from 150 years of international results.

This is the engine behind the upgraded model. In a single chronological pass
over ~49k matches (martj42/international_results) it computes, for every match,
the *pre-match* state of both teams — Elo rating, recent form, rest days,
head-to-head record — with zero leakage (only information available before
kick-off is used). It also emits each team's current state so the app can
predict today's matchups with up-to-date ratings.

Elo follows the World Football Elo conventions (eloratings.net): a
margin-of-victory multiplier and a tournament-importance weight K.
"""
from __future__ import annotations

import json
from collections import defaultdict, deque

import numpy as np
import pandas as pd

import config

HOME_ADV = 65.0          # Elo points added to the home side when not neutral
BASE_RATING = 1500.0
FORM_WINDOW = 10         # matches used for rolling form
REST_CAP = 45            # days

FEATURE_COLS_V2 = [
    "elo_home", "elo_away", "elo_diff",
    "home_gf_avg", "home_ga_avg", "home_pts_avg",
    "away_gf_avg", "away_ga_avg", "away_pts_avg",
    "home_rest_days", "away_rest_days",
    "h2h_home_rate", "h2h_played",
    "neutral", "tournament_importance",
    "home_confederation", "away_confederation",
    "home_elo_momentum", "away_elo_momentum",
]


# --------------------------------------------------------------------------- #
# Tournament weighting
# --------------------------------------------------------------------------- #
def k_factor(t: str) -> float:
    t = str(t)
    if t == "FIFA World Cup":
        return 60.0
    if t in ("UEFA Euro", "Copa América", "Copa America", "African Cup of Nations",
             "AFC Asian Cup", "Gold Cup", "CONCACAF Championship", "Confederations Cup"):
        return 50.0
    if "Nations League" in t:
        return 45.0
    if "qualification" in t:
        return 40.0
    if t == "Friendly":
        return 20.0
    return 30.0


def tournament_importance(t: str) -> int:
    t = str(t)
    if t == "FIFA World Cup":
        return 4
    if t in ("UEFA Euro", "Copa América", "Copa America", "African Cup of Nations",
             "AFC Asian Cup", "Gold Cup", "CONCACAF Championship"):
        return 3
    if "qualification" in t or "Nations League" in t:
        return 2
    if t == "Friendly":
        return 0
    return 1


def _gd_multiplier(gd: int) -> float:
    gd = abs(int(gd))
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11 + gd) / 8.0


# --------------------------------------------------------------------------- #
# Confederation inference (from the continental competitions each team plays)
# --------------------------------------------------------------------------- #
CONF_CODES = {"UEFA": 0, "CONMEBOL": 1, "CONCACAF": 2, "CAF": 3, "AFC": 4, "OFC": 5, "OTHER": 6}


def infer_confederations(df: pd.DataFrame) -> dict:
    def conf_of(t: str):
        t = str(t)
        if "UEFA" in t or "European" in t:
            return "UEFA"
        if "CONMEBOL" in t or "Copa Am" in t:
            return "CONMEBOL"
        if "CONCACAF" in t or "Gold Cup" in t or "Caribbean" in t or "UNCAF" in t:
            return "CONCACAF"
        if "African" in t or "Africa" in t or "CECAFA" in t or "COSAFA" in t or "WAFU" in t:
            return "CAF"
        if "AFC" in t or "Asian" in t or "AFF" in t or "SAFF" in t or "Gulf" in t:
            return "AFC"
        if "OFC" in t or "Oceania" in t or "Pacific" in t:
            return "OFC"
        return None

    tally = defaultdict(lambda: defaultdict(int))
    for r in df.itertuples():
        c = conf_of(r.tournament)
        if c:
            tally[r.home_team][c] += 1
            tally[r.away_team][c] += 1
    out = {}
    for team, counts in tally.items():
        out[team] = max(counts.items(), key=lambda x: x[1])[0]
    for t in set(df.home_team) | set(df.away_team):
        out.setdefault(t, "OTHER")
    return out


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_results() -> pd.DataFrame:
    df = pd.read_csv(config.INTL_RESULTS_CSV).dropna(subset=["home_score", "away_score"])
    fn_path = config.ROOT / "data" / "former_names.csv"
    if fn_path.exists():
        fn = pd.read_csv(fn_path)
        mp = dict(zip(fn.former, fn.current))
        df["home_team"] = df["home_team"].replace(mp)
        df["away_team"] = df["away_team"].replace(mp)
    df["date"] = pd.to_datetime(df["date"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    return df.sort_values("date").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Single-pass Elo + feature builder
# --------------------------------------------------------------------------- #
def build(save: bool = True):
    df = load_results()
    conf = infer_confederations(df)

    elo: dict[str, float] = defaultdict(lambda: BASE_RATING)
    form: dict[str, deque] = defaultdict(lambda: deque(maxlen=FORM_WINDOW))  # (gf, ga, pts)
    elo_hist: dict[str, deque] = defaultdict(lambda: deque(maxlen=FORM_WINDOW + 1))
    last_date: dict[str, pd.Timestamp] = {}
    h2h: dict[tuple, deque] = defaultdict(lambda: deque(maxlen=10))          # result for home team

    def momentum(team, cur):
        q = elo_hist[team]
        return round(cur - q[0], 1) if q else 0.0

    rows = []
    for r in df.itertuples():
        h, a = r.home_team, r.away_team
        rh, ra = elo[h], elo[a]
        ha = 0.0 if r.neutral else HOME_ADV
        elo_diff = (rh + ha) - ra

        def form_avg(team, idx, default):
            q = form[team]
            return float(np.mean([x[idx] for x in q])) if q else default

        pair = tuple(sorted((h, a)))
        hist = h2h[pair]
        h2h_played = len(hist)
        # rate from the current home team's perspective
        h2h_home_rate = (float(np.mean([res if pair[0] == h else 1 - res for res in hist]))
                         if hist else 0.5)

        rows.append({
            "date": r.date, "home_team": h, "away_team": a,
            "elo_home": rh, "elo_away": ra, "elo_diff": elo_diff,
            "home_gf_avg": form_avg(h, 0, 1.2), "home_ga_avg": form_avg(h, 1, 1.2),
            "home_pts_avg": form_avg(h, 2, 1.3),
            "away_gf_avg": form_avg(a, 0, 1.2), "away_ga_avg": form_avg(a, 1, 1.2),
            "away_pts_avg": form_avg(a, 2, 1.3),
            "home_rest_days": min((r.date - last_date[h]).days, REST_CAP) if h in last_date else REST_CAP,
            "away_rest_days": min((r.date - last_date[a]).days, REST_CAP) if a in last_date else REST_CAP,
            "h2h_home_rate": h2h_home_rate, "h2h_played": h2h_played,
            "neutral": int(bool(r.neutral)),
            "tournament_importance": tournament_importance(r.tournament),
            "home_confederation": CONF_CODES[conf[h]], "away_confederation": CONF_CODES[conf[a]],
            "home_elo_momentum": momentum(h, rh), "away_elo_momentum": momentum(a, ra),
            "home_goals": r.home_score, "away_goals": r.away_score,
            "year": r.date.year,
        })
        elo_hist[h].append(rh); elo_hist[a].append(ra)

        # ---- update state after recording pre-match features ----
        gd = r.home_score - r.away_score
        w = 1.0 if gd > 0 else (0.5 if gd == 0 else 0.0)
        k = k_factor(r.tournament) * _gd_multiplier(gd)
        we = 1.0 / (1.0 + 10 ** (-elo_diff / 400.0))
        delta = k * (w - we)
        elo[h] = rh + delta
        elo[a] = ra - delta
        form[h].append((r.home_score, r.away_score, 3 if gd > 0 else (1 if gd == 0 else 0)))
        form[a].append((r.away_score, r.home_score, 3 if gd < 0 else (1 if gd == 0 else 0)))
        last_date[h] = last_date[a] = r.date
        hist.append(1.0 if gd > 0 else (0.5 if gd == 0 else 0.0)) if pair[0] == h else \
            hist.append(0.0 if gd > 0 else (0.5 if gd == 0 else 1.0))

    feat = pd.DataFrame(rows)

    # current team state for serving
    current = {}
    for team in set(df.home_team) | set(df.away_team):
        q = form[team]
        current[team] = {
            "elo": round(float(elo[team]), 1),
            "confederation": conf[team],
            "gf_avg": float(np.mean([x[0] for x in q])) if q else 1.2,
            "ga_avg": float(np.mean([x[1] for x in q])) if q else 1.2,
            "pts_avg": float(np.mean([x[2] for x in q])) if q else 1.3,
            "elo_momentum": momentum(team, elo[team]),
            "last_date": str(last_date[team].date()) if team in last_date else None,
            "matches": int(sum(1 for _ in q)),
        }

    if save:
        feat.to_csv(config.PROCESSED_DIR / "intl_training_data.csv", index=False)
        (config.PROCESSED_DIR / "current_ratings.json").write_text(json.dumps(current, indent=2))
        (config.PROCESSED_DIR / "team_confederation.json").write_text(json.dumps(conf, indent=2))
        print(f"[elo] wrote {len(feat)} matches, {len(current)} team ratings")
    return feat, current, conf


if __name__ == "__main__":
    feat, current, conf = build()
    top = sorted(current.items(), key=lambda x: -x[1]["elo"])[:15]
    for t, v in top:
        print(f"  {t:<18}{v['elo']:7.0f}  {v['confederation']}")
