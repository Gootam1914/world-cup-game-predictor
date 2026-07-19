"""Feature engineering: turn StatsBomb events into the 20 model features.

Pipeline
--------
1. ``build_team_match_table`` reduces every match's event stream to two rows
   (one per team) with seven per-match performance metrics.
2. ``team_profile`` averages those metrics across a team's matches to form a
   team-strength profile (with optional leave-one-match-out to avoid leakage).
3. ``build_training_dataset`` assembles one row per match containing exactly the
   20 ``FEATURE_COLS`` plus the two goal targets.

StatsBomb pitch coordinates are normalised so the team in possession always
attacks towards x = 120, which makes "forward progress" simply end_x > start_x.
"""
from __future__ import annotations

import math
import warnings
from typing import Optional

import numpy as np
import pandas as pd

import config
from src import statsbomb_loader as L

warnings.filterwarnings("ignore")

# Shot outcomes counted as "on target" for shot accuracy.
ON_TARGET = {"Goal", "Saved", "Saved to Post", "Saved Off T"}
# A completed pass has to move the ball at least this many units closer to the
# opponent goal (and forward) to count as "progressive".
PROG_THRESHOLD = 10.0
GOAL_XY = (120.0, 40.0)

# Host nation per tournament -> that team's matches are NOT neutral.
HOSTS = {
    (43, 3): "Russia",             # World Cup 2018
    (43, 106): "Qatar",            # World Cup 2022
    (55, 43): None,                # Euro 2020 (pan-European, treat as neutral)
    (55, 282): "Germany",          # Euro 2024
    (223, 282): "United States",   # Copa America 2024
    (1267, 107): "Côte d'Ivoire",  # AFCON 2023
}


def _dist_to_goal(x: float, y: float) -> float:
    return math.hypot(GOAL_XY[0] - x, GOAL_XY[1] - y)


def _team_metrics(ev: pd.DataFrame, team: str) -> dict:
    """Seven within-match performance metrics for one team."""
    shots = ev[(ev.type_name == "Shot") & (ev.team_name == team)]
    xg_for = float(shots["shot_statsbomb_xg"].fillna(0).sum())
    n_shots = len(shots)
    on_target = int(shots["shot_outcome_name"].isin(ON_TARGET).sum())
    shot_acc = (on_target / n_shots * 100.0) if n_shots else 0.0

    passes = ev[(ev.type_name == "Pass") & (ev.team_name == team)]
    total_passes_both = int((ev.type_name == "Pass").sum())
    possession = (len(passes) / total_passes_both * 100.0) if total_passes_both else 50.0

    completed = passes[passes["pass_outcome_name"].isna()]
    prog = 0
    for loc, end in zip(completed["location"], completed["pass_end_location"]):
        if not isinstance(loc, list) or not isinstance(end, list):
            continue
        sx, sy = loc[0], loc[1]
        ex, ey = end[0], end[1]
        if ex > sx and (_dist_to_goal(sx, sy) - _dist_to_goal(ex, ey)) >= PROG_THRESHOLD:
            prog += 1
    prog_pct = (prog / len(completed) * 100.0) if len(completed) else 0.0

    return {
        "xg_for": xg_for,
        "shot_accuracy_pct": shot_acc,
        "possession_pct": possession,
        "progressive_passes_pct": prog_pct,
    }


def build_team_match_table(save: bool = True) -> pd.DataFrame:
    """One row per (match, team) with the seven metrics + goals."""
    matches = L.load_all_matches()
    rows = []
    for _, mt in matches.iterrows():
        ev = L.load_events(mt.match_id)
        metrics = {mt.home_team: _team_metrics(ev, mt.home_team),
                   mt.away_team: _team_metrics(ev, mt.away_team)}
        for team, opp, gf, ga, is_home in (
            (mt.home_team, mt.away_team, mt.home_score, mt.away_score, 1),
            (mt.away_team, mt.home_team, mt.away_score, mt.home_score, 0),
        ):
            m = metrics[team]
            rows.append(
                {
                    "match_id": mt.match_id,
                    "competition_id": mt.competition_id,
                    "season_id": mt.season_id,
                    "match_date": mt.match_date,
                    "team": team,
                    "opponent": opp,
                    "is_home": is_home,
                    "goals_for": int(gf),
                    "goals_against": int(ga),
                    "xg_for": m["xg_for"],
                    "xg_against": metrics[opp]["xg_for"],
                    "shot_accuracy_pct": m["shot_accuracy_pct"],
                    "possession_pct": m["possession_pct"],
                    "progressive_passes_pct": m["progressive_passes_pct"],
                }
            )
    df = pd.DataFrame(rows)
    if save:
        out = config.PROCESSED_DIR / "team_match_stats.csv"
        df.to_csv(out, index=False)
        print(f"[features] wrote {out} ({len(df)} team-match rows)")
    return df


# --------------------------------------------------------------------------- #
# Profiles / head-to-head
# --------------------------------------------------------------------------- #
_PROFILE_COLS = ["xg_for", "xg_against", "goals_for", "goals_against",
                 "shot_accuracy_pct", "possession_pct", "progressive_passes_pct"]


def team_profile(stats: pd.DataFrame, team: str,
                 exclude_match_id: Optional[int] = None) -> dict:
    """Average a team's seven metrics (optionally leaving one match out)."""
    sub = stats[stats.team == team]
    if exclude_match_id is not None:
        sub = sub[sub.match_id != exclude_match_id]
    if sub.empty:  # unseen team -> neutral league-average fallback
        sub = stats
    return {c: float(sub[c].mean()) for c in _PROFILE_COLS}


def head_to_head(matches: pd.DataFrame, home: str, away: str,
                 exclude_match_id: Optional[int] = None) -> tuple[int, int, int]:
    """Historical (home_wins, draws, away_wins) between the two teams."""
    pair = matches[
        ((matches.home_team == home) & (matches.away_team == away))
        | ((matches.home_team == away) & (matches.away_team == home))
    ]
    if exclude_match_id is not None:
        pair = pair[pair.match_id != exclude_match_id]
    hw = dr = aw = 0
    for _, m in pair.iterrows():
        # normalise result to the requested home/away orientation
        if m.home_team == home:
            hs, as_ = m.home_score, m.away_score
        else:
            hs, as_ = m.away_score, m.home_score
        if hs > as_:
            hw += 1
        elif hs < as_:
            aw += 1
        else:
            dr += 1
    return hw, dr, aw


def _is_neutral(competition_id: int, season_id: int, home_team: str) -> int:
    host = HOSTS.get((competition_id, season_id))
    return 0 if (host is not None and home_team == host) else 1


def _feature_row(home_prof: dict, away_prof: dict, home: str, away: str,
                 h2h: tuple[int, int, int], is_neutral: int) -> dict:
    return {
        "home_xg_for_avg": home_prof["xg_for"],
        "home_xg_against_avg": home_prof["xg_against"],
        "home_goals_for_avg": home_prof["goals_for"],
        "home_goals_against_avg": home_prof["goals_against"],
        "home_shot_accuracy_pct": home_prof["shot_accuracy_pct"],
        "home_possession_pct": home_prof["possession_pct"],
        "home_progressive_passes_pct": home_prof["progressive_passes_pct"],
        "home_confederation": config.get_confederation_code(home),
        "away_xg_for_avg": away_prof["xg_for"],
        "away_xg_against_avg": away_prof["xg_against"],
        "away_goals_for_avg": away_prof["goals_for"],
        "away_goals_against_avg": away_prof["goals_against"],
        "away_shot_accuracy_pct": away_prof["shot_accuracy_pct"],
        "away_possession_pct": away_prof["possession_pct"],
        "away_progressive_passes_pct": away_prof["progressive_passes_pct"],
        "away_confederation": config.get_confederation_code(away),
        "h2h_home_wins": h2h[0],
        "h2h_draws": h2h[1],
        "h2h_away_wins": h2h[2],
        "is_neutral": is_neutral,
    }


def build_training_dataset(save: bool = True) -> pd.DataFrame:
    """One row per match: the 20 features (leakage-free) + goal targets."""
    stats = build_team_match_table(save=save)
    matches = L.load_all_matches()

    rows = []
    for _, mt in matches.iterrows():
        hp = team_profile(stats, mt.home_team, exclude_match_id=mt.match_id)
        ap = team_profile(stats, mt.away_team, exclude_match_id=mt.match_id)
        h2h = head_to_head(matches, mt.home_team, mt.away_team, exclude_match_id=mt.match_id)
        isn = _is_neutral(mt.competition_id, mt.season_id, mt.home_team)
        feat = _feature_row(hp, ap, mt.home_team, mt.away_team, h2h, isn)
        feat.update(
            {
                "match_id": mt.match_id,
                "competition_id": mt.competition_id,
                "season_id": mt.season_id,
                "home_team": mt.home_team,
                "away_team": mt.away_team,
                "home_goals": int(mt.home_score),
                "away_goals": int(mt.away_score),
            }
        )
        rows.append(feat)

    df = pd.DataFrame(rows)
    df = df[[*_meta_cols(), *config.FEATURE_COLS, "home_goals", "away_goals"]]
    if save:
        out = config.PROCESSED_DIR / "training_data.csv"
        df.to_csv(out, index=False)
        # Cache the raw match list so serving never needs the multi-GB clone.
        matches.to_csv(config.PROCESSED_DIR / "matches.csv", index=False)
        print(f"[features] wrote {out} ({len(df)} matches, {len(config.FEATURE_COLS)} features)")
    return df


def load_matches_cached() -> pd.DataFrame:
    """Prefer the cached match list; fall back to the StatsBomb clone."""
    cached = config.PROCESSED_DIR / "matches.csv"
    if cached.exists():
        df = pd.read_csv(cached)
        df["match_date"] = pd.to_datetime(df["match_date"])
        return df
    return L.load_all_matches()


def _meta_cols() -> list[str]:
    return ["match_id", "competition_id", "season_id", "home_team", "away_team"]


def build_matchup_features(home: str, away: str, is_neutral: int = 1,
                           stats: Optional[pd.DataFrame] = None,
                           matches: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Serving-time feature vector using each team's FULL profile."""
    if stats is None:
        stats = pd.read_csv(config.PROCESSED_DIR / "team_match_stats.csv")
    if matches is None:
        matches = load_matches_cached()
    hp = team_profile(stats, home)
    ap = team_profile(stats, away)
    h2h = head_to_head(matches, home, away)
    feat = _feature_row(hp, ap, home, away, h2h, is_neutral)
    return pd.DataFrame([feat])[config.FEATURE_COLS]


if __name__ == "__main__":
    df = build_training_dataset()
    print(df.head())
    print("\nfeature columns match config:",
          list(df.columns[5:25]) == config.FEATURE_COLS)
