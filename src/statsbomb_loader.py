"""Load StatsBomb Open Data efficiently.

StatsBomb's open-data repo is several GB, so we use a *partial* clone
(``--filter=blob:none``) combined with a *sparse* checkout so that only the
JSON files we actually need are downloaded:

* ``data/competitions.json``
* ``data/matches/<comp>/<season>.json`` for our target tournaments
* ``data/events/<match_id>.json`` for every match in those tournaments

The functions here are idempotent and re-runnable: calling :func:`sync` again
after StatsBomb publishes new data will fetch only the new blobs, which is how
we "keep the data as updated as possible".
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Iterable

import pandas as pd

from config import STATSBOMB_CLONE_DIR, STATSBOMB_REPO, TOURNAMENTS


# --------------------------------------------------------------------------- #
# git helpers
# --------------------------------------------------------------------------- #
def _git(*args: str, cwd: Path | None = None, timeout: int = 600) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed:\n{result.stderr.strip()}"
        )
    return result.stdout


def _ensure_clone() -> None:
    """Create a blobless, no-checkout clone if it does not exist yet."""
    repo = STATSBOMB_CLONE_DIR
    if (repo / ".git").exists():
        return
    repo.parent.mkdir(parents=True, exist_ok=True)
    print(f"[statsbomb] cloning {STATSBOMB_REPO} (blobless) -> {repo}")
    _git(
        "clone", "--filter=blob:none", "--no-checkout", "--depth", "1",
        STATSBOMB_REPO, str(repo),
    )
    _git("sparse-checkout", "init", "--no-cone", cwd=repo)


def _sparse_set(paths: Iterable[str]) -> None:
    _git("sparse-checkout", "set", "--no-cone", *paths, cwd=STATSBOMB_CLONE_DIR)
    _git("checkout", cwd=STATSBOMB_CLONE_DIR)


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def sync(refresh: bool = False) -> None:
    """Ensure competitions, target match lists, and all event files are present.

    Parameters
    ----------
    refresh : if True, run ``git pull`` first to pick up newly published data.
    """
    _ensure_clone()
    repo = STATSBOMB_CLONE_DIR

    if refresh:
        print("[statsbomb] git pull for latest open data ...")
        try:
            _git("pull", "--depth", "1", cwd=repo)
        except RuntimeError as exc:  # non-fatal; keep working with cached data
            print(f"[statsbomb] pull skipped: {exc}")

    # Step 1: competitions + the six match-list files.
    match_files = [f"data/matches/{cid}/{sid}.json" for cid, sid, _ in TOURNAMENTS]
    _sparse_set(["data/competitions.json", *match_files])

    # Step 2: collect every match_id, then sparse-checkout their event files.
    match_ids: list[int] = []
    for cid, sid, _ in TOURNAMENTS:
        mf = repo / f"data/matches/{cid}/{sid}.json"
        for m in json.loads(mf.read_text()):
            match_ids.append(m["match_id"])

    event_files = [f"data/events/{mid}.json" for mid in match_ids]
    print(f"[statsbomb] checking out {len(event_files)} event files ...")
    _sparse_set(["data/competitions.json", *match_files, *event_files])
    print("[statsbomb] sync complete.")


def load_matches(competition_id: int, season_id: int) -> pd.DataFrame:
    """Return the match list for one tournament as a tidy DataFrame."""
    mf = STATSBOMB_CLONE_DIR / f"data/matches/{competition_id}/{season_id}.json"
    raw = json.loads(mf.read_text())
    rows = []
    for m in raw:
        rows.append(
            {
                "match_id": m["match_id"],
                "competition_id": competition_id,
                "season_id": season_id,
                "match_date": m["match_date"],
                "home_team": m["home_team"]["home_team_name"],
                "away_team": m["away_team"]["away_team_name"],
                "home_score": m["home_score"],
                "away_score": m["away_score"],
                "stage": m.get("competition_stage", {}).get("name", ""),
            }
        )
    return pd.DataFrame(rows)


def load_all_matches() -> pd.DataFrame:
    frames = [load_matches(cid, sid) for cid, sid, _ in TOURNAMENTS]
    df = pd.concat(frames, ignore_index=True)
    df["match_date"] = pd.to_datetime(df["match_date"])
    return df.sort_values("match_date").reset_index(drop=True)


def load_events(match_id: int) -> pd.DataFrame:
    """Return the raw event stream for a single match."""
    ef = STATSBOMB_CLONE_DIR / f"data/events/{match_id}.json"
    return pd.json_normalize(json.loads(ef.read_text()), sep="_")


if __name__ == "__main__":
    sync()
    m = load_all_matches()
    print(f"Loaded {len(m)} matches across {m['competition_id'].nunique()} competitions")
    print(m.head())
