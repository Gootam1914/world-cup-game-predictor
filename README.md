# Soccer Game Predictor

Predict the result of any international men's football matchup with an
**XGBoost expected-goals model** trained on **StatsBomb Open Data**. Pick two
national teams and get a projected scoreline, win / draw / loss probabilities,
and a confidence score — through a clean web UI.

![prediction](https://img.shields.io/badge/model-XGBoost-blue) ![data](https://img.shields.io/badge/data-StatsBomb-orange) ![python](https://img.shields.io/badge/python-3.10%2B-green)

---

## What it does

You select **Team A** and **Team B** (with flags). The model builds a 20-feature
matchup vector from each team's historical performance and predicts the expected
goals for each side. Those two rates are treated as independent Poisson
distributions, from which we derive:

- the **most likely scoreline** (e.g. `Argentina 1 – 1 Spain`)
- **win / draw / loss** probabilities
- a **confidence** score and band (Low / Medium / High) — the share of
  probability sitting on the most likely result

## The 20 features

Each is built directly from StatsBomb event data (per-team season-to-date
averages), matching the model's `FEATURE_COLS`:

| Home | Away | Context |
|------|------|---------|
| `home_xg_for_avg` | `away_xg_for_avg` | `h2h_home_wins` |
| `home_xg_against_avg` | `away_xg_against_avg` | `h2h_draws` |
| `home_goals_for_avg` | `away_goals_for_avg` | `h2h_away_wins` |
| `home_goals_against_avg` | `away_goals_against_avg` | `is_neutral` |
| `home_shot_accuracy_pct` | `away_shot_accuracy_pct` | |
| `home_possession_pct` | `away_possession_pct` | |
| `home_progressive_passes_pct` | `away_progressive_passes_pct` | |
| `home_confederation` | `away_confederation` | |

## Training data

StatsBomb's free open data covers these modern international men's tournaments
with full event detail (~314 matches, 76 national teams):

FIFA World Cup 2018 · FIFA World Cup 2022 · UEFA Euro 2020 · UEFA Euro 2024 ·
Copa América 2024 · Africa Cup of Nations 2023

## Model performance

Evaluated with **leave-one-tournament-out** cross-validation — the model is
tested on a whole tournament it never saw during training, which mirrors the
World-Cup use case.

| Metric | Value | Baseline |
|--------|-------|----------|
| Outcome accuracy (3-way) | **51.9%** | 40.1% (always pick home) |
| Multiclass log-loss | **0.999** | 1.099 (random) |
| Multiclass Brier | 0.598 | — |
| Goals MAE | 0.86 | — |
| Goals RMSE | 1.13 | — |

Top features by gain: `home_xg_for_avg`, `away_xg_for_avg`, confederation, and
shot accuracy — exactly what you'd expect to drive results.

---

## Quick start

```bash
git clone <your-repo-url>
cd "Soccer Game Predictor"
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

The repo ships with pre-built features and trained models, so you can run the
app immediately:

```bash
uvicorn app.api:app --reload
# open http://127.0.0.1:8000
```

### Rebuild from scratch / keep data fresh

```bash
python scripts/build_all.py            # sync data -> features -> train -> evaluate
python scripts/build_all.py --refresh  # git pull the latest StatsBomb data first
```

Re-running this is how you "keep the data as updated as possible": it pulls any
newly published StatsBomb matches, rebuilds the features, and retrains.

### Improve / retune the model

```bash
python scripts/train_multi.py 10   # train 10 seeds, keep the most stable
python scripts/tune.py 40          # randomised hyperparameter search (CV-scored)
```

`tune.py` writes `models/best_params.json`, which training then uses
automatically. On this dataset tuning yields a small calibration gain
(log-loss 0.999 → 0.991, Brier 0.598 → 0.593) with accuracy flat — the model is
near its ceiling, so the bigger levers are more data and richer features.

### Command-line prediction

```bash
python -m src.predict Brazil France
```

---

## Optional live modules (current rosters + gap-fill)

These need an API key and internet access, so they run on your machine. The
model and UI work fully **without** them. Copy `.env.example` to `.env` and add
any keys you have:

```
API_FOOTBALL_KEY=...        # api-football.com  -> current national-team squads
FOOTBALL_DATA_ORG_KEY=...   # football-data.org -> recent international results
```

- `python -m src.roster_fetch Argentina` — fetch the current squad (shown in the
  app when a key is present).
- `python -m src.football_data_api` — append recent international results that
  StatsBomb hasn't published yet, keeping team profiles current.

Both degrade gracefully to a no-op when the key is missing.

---

## Project structure

```
Soccer Game Predictor/
├── config.py                 # paths, 20-feature schema, team metadata
├── src/
│   ├── statsbomb_loader.py    # partial + sparse git clone of StatsBomb data
│   ├── features.py            # build the 20 features from events
│   ├── model.py               # Poisson scoreline maths + confidence
│   ├── train.py               # train + save the two XGBoost regressors
│   ├── evaluate.py            # leave-one-tournament-out cross-validation
│   ├── predict.py             # serving-time prediction
│   ├── roster_fetch.py        # (optional) current squads via API-Football
│   └── football_data_api.py   # (optional) gap-fill via football-data.org
├── app/
│   ├── api.py                 # FastAPI backend (/api/teams, /api/predict)
│   └── web/index.html         # single-page UI
├── scripts/build_all.py       # one-command pipeline
├── models/                    # trained models + metrics (committed)
├── data/processed/            # cached features + match list (committed)
├── data/raw/                  # StatsBomb clone cache (git-ignored, regenerable)
└── tests/test_model.py
```

## How it works

1. **Load** — `statsbomb_loader.py` does a blobless, sparse git clone so only the
   JSON we need (matches + events for the six tournaments) is downloaded.
2. **Engineer** — `features.py` reduces each match to per-team metrics (xG,
   goals, shot accuracy, possession share, progressive-pass rate), then averages
   them into team-strength profiles. Profiles are computed leave-one-match-out to
   avoid target leakage. Confederation, head-to-head record, and a neutral-venue
   flag complete the 20 features.
3. **Train** — two well-regularised XGBoost regressors (`objective=count:poisson`)
   predict expected goals for the home and away side.
4. **Predict** — expected goals become independent Poisson rates; enumerating the
   score matrix yields the scoreline, outcome probabilities, and confidence.

## Notes & limitations

- ~314 international matches is a modest dataset, so probabilities are
  deliberately calibrated rather than overconfident — football is high-variance.
- Possession is approximated by share of passes; progressive passes use a
  distance-to-goal threshold on StatsBomb's direction-normalised coordinates.
- The cross-validation caveat: team profiles are built across all tournaments, so
  a small amount of cross-tournament information is shared. Same-match leakage is
  fully prevented.

## License & attribution

MIT (see `LICENSE`). This project uses
[StatsBomb Open Data](https://github.com/statsbomb/open-data); any public use of
derived work must credit StatsBomb per their user agreement.
