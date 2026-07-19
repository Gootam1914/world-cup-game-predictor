"""One-command pipeline for the (upgraded) Elo + form model.

    python scripts/build_all.py            # rebuild ratings, train, backtest, flags
    python scripts/build_all.py --refresh  # pull the latest results first

"Keep the data as updated as possible": --refresh re-downloads the international
results dataset (which is updated within days of every match) before rebuilding.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config                                   # noqa: E402
from src import elo, train_intl                 # noqa: E402


def refresh_results() -> None:
    """Re-download the latest international results into the repo."""
    with tempfile.TemporaryDirectory() as tmp:
        print("[refresh] cloning latest international results ...")
        subprocess.run(
            ["git", "clone", "--depth", "1", config.INTL_RESULTS_REPO, tmp + "/r"],
            check=True, capture_output=True, text=True,
        )
        src = Path(tmp) / "r"
        shutil.copy(src / "results.csv", config.INTL_RESULTS_CSV)
        if (src / "former_names.csv").exists():
            shutil.copy(src / "former_names.csv", config.ROOT / "data" / "former_names.csv")
    print("[refresh] updated data/international_results.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="download latest results first")
    args = ap.parse_args()

    if args.refresh:
        refresh_results()

    print("1/3  Building Elo ratings + features ...")
    elo.build()

    print("2/3  Training model + chronological backtest ...")
    train_intl.train_and_save()

    print("3/3  Building flag map ...")
    subprocess.run([sys.executable, str(config.ROOT / "scripts" / "build_flags.py")], check=True)

    print("\nDone. Launch with:  npm run dev   (or)   uvicorn app.api:app --reload")


if __name__ == "__main__":
    main()
