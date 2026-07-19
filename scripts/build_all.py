"""One-command pipeline: sync data -> features -> train -> evaluate.

Run from the project root:
    python scripts/build_all.py            # use cached StatsBomb data
    python scripts/build_all.py --refresh  # git pull latest StatsBomb data first

This is how you "keep the data as updated as possible": re-run it whenever
StatsBomb publishes new matches (and, if you have keys, after running the
football-data.org gap-fill).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import statsbomb_loader as L      # noqa: E402
from src import features as F               # noqa: E402
from src import train, evaluate            # noqa: E402
import config                              # noqa: E402
import json                                # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true",
                    help="git pull latest StatsBomb open data before building")
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()

    print("1/4  Syncing StatsBomb open data ...")
    L.sync(refresh=args.refresh)

    print("2/4  Building the 20-feature dataset ...")
    F.build_training_dataset()

    print("3/4  Training XGBoost models ...")
    train.train_and_save(rebuild=False)

    if not args.skip_eval:
        print("4/4  Cross-validating (leave-one-tournament-out) ...")
        metrics = evaluate.cross_validate()
        (config.MODELS_DIR / "cv_metrics.json").write_text(json.dumps(metrics, indent=2))
        print(json.dumps(metrics, indent=2))

    print("\nDone. Launch the app with:  uvicorn app.api:app --reload")


if __name__ == "__main__":
    main()
