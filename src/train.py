"""Train the two XGBoost expected-goals regressors and save them.

Run:  python -m src.train        (rebuilds features if needed, then trains)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xgboost as xgb

import config
from src import features as F

# Conservative defaults: the dataset is ~300 matches, so we keep the trees
# shallow and well-regularised to avoid overfitting. These are overridden by
# models/best_params.json when the hyperparameter search has been run.
DEFAULT_XGB_PARAMS = dict(
    objective="count:poisson",
    n_estimators=300,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=4,
    reg_lambda=1.5,
    reg_alpha=0.2,
    random_state=config.RANDOM_SEED,
    n_jobs=4,
)

BEST_PARAMS_PATH = config.MODELS_DIR / "best_params.json"


def get_xgb_params() -> dict:
    """Default params, overridden by a tuned set if scripts/tune.py has run."""
    params = dict(DEFAULT_XGB_PARAMS)
    if BEST_PARAMS_PATH.exists():
        params.update(json.loads(BEST_PARAMS_PATH.read_text()))
    return params


# Kept for backwards compatibility with existing imports.
XGB_PARAMS = get_xgb_params()


def make_models() -> tuple[xgb.XGBRegressor, xgb.XGBRegressor]:
    params = get_xgb_params()
    return xgb.XGBRegressor(**params), xgb.XGBRegressor(**params)


def load_training_frame(rebuild: bool = False) -> pd.DataFrame:
    path = config.PROCESSED_DIR / "training_data.csv"
    if rebuild or not path.exists():
        return F.build_training_dataset()
    return pd.read_csv(path)


def train_and_save(rebuild: bool = False) -> dict:
    df = load_training_frame(rebuild=rebuild)
    X = df[config.FEATURE_COLS]
    yh, ya = df["home_goals"], df["away_goals"]

    home_model, away_model = make_models()
    home_model.fit(X, yh)
    away_model.fit(X, ya)

    home_model.save_model(config.MODELS_DIR / "home_goals_model.json")
    away_model.save_model(config.MODELS_DIR / "away_goals_model.json")

    # feature importances (gain) for transparency
    imp = {c: float(v) for c, v in zip(config.FEATURE_COLS, home_model.feature_importances_)}
    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_matches": int(len(df)),
        "feature_cols": config.FEATURE_COLS,
        "tournaments": [t[2] for t in config.TOURNAMENTS],
        "xgb_params": XGB_PARAMS,
        "target": "expected goals per team (Poisson)",
        "home_feature_importance_gain": imp,
        "data_source": "StatsBomb Open Data",
    }
    (config.MODELS_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"[train] saved 2 models + metadata to {config.MODELS_DIR}")
    print(f"[train] top features: "
          + ", ".join(k for k, _ in sorted(imp.items(), key=lambda x: -x[1])[:5]))
    return meta


if __name__ == "__main__":
    train_and_save(rebuild=False)
