# Soccer Game Predictor

Predict the result of any international football matchup. Pick two national
teams and get a projected scoreline, win / draw / loss probabilities, and a
confidence score — through a clean web UI, powered by an **Elo + form XGBoost
model** trained on **150 years of international results**.

![model](https://img.shields.io/badge/model-XGBoost%20%2B%20Elo-blue) ![data](https://img.shields.io/badge/matches-49k-orange) ![python](https://img.shields.io/badge/python-3.10%2B-green)

---

## What it does

Select **Team A** and **Team B**. The model reads each team's current strength
(a live Elo rating), recent form, momentum, head-to-head record and the match
context, and predicts expected goals for each side. Those become a full
scoreline distribution (with a Dixon-Coles low-score correction), yielding:

- the **most likely scoreline** (e.g. `France 2 – 1 Morocco`)
- **win / draw / loss** probabilities
- a **confidence** score and band (Low / Medium / High)

Every one of **237 currently-active national teams** is selectable, each shown
with its flag and up-to-date Elo rating.

## Model performance

Evaluated with an honest **chronological backtest**: trained on matches before
2023, tested on the **3,708 matches from 2023 to mid-2026** the model never saw.

| Model | Accuracy | Log-loss | Brier | RPS |
|-------|----------|----------|-------|-----|
| **This model (ensemble)** | **60.8%** | **0.853** | **0.502** | **0.332** |
| Goals model only | 60.8% | 0.855 | 0.502 | 0.332 |
| Elo-only baseline | 60.5% | 0.865 | 0.509 | 0.337 |
| Always predict home win | 47.2% | — | — | — |

The final model is an **ensemble**: 60% the Poisson goals model + 40% a direct
XGBoost outcome classifier, trained on all results since 1994 (~26k matches)
with recency weighting.

RPS (ranked probability score) of ~0.33 is competitive with published
international-football forecasting models. For context, 3-way football outcomes
top out in the mid-50s%/low-60s% even for bookmakers — the sport is genuinely
high-variance — so the gains here come mainly from a strong, current
team-strength signal and good calibration.

## How the model works

1. **Elo ratings** — a single chronological pass over ~49,000 international
   matches (1872–2026) computes each team's rating using the World Football Elo
   conventions: a margin-of-victory multiplier and a tournament-importance
   weight (World Cup K=60 … friendly K=20), plus home advantage. Ratings are
   always current, so predictions reflect today's strengths.
2. **Features** — for every match the model uses only pre-kickoff information
   (no leakage): Elo of each side and their difference, rolling form (goals
   for/against and points over the last 10 games), rest days, head-to-head rate,
   neutral-venue flag, tournament importance, confederation, and Elo momentum.
3. **Goals model** — two XGBoost regressors (`objective=count:poisson`) predict
   expected goals for each side, trained with recency weighting so modern
   football counts more.
4. **Scoreline** — the two expected-goal rates form a score matrix (independent
   Poisson + a Dixon-Coles correction, ρ=−0.12, fitted on validation) that gives
   the projected scoreline.
5. **Ensemble** — the goals model's implied win/draw/loss probabilities are
   blended (60/40) with a direct XGBoost outcome classifier for the final
   probabilities and confidence, which improves calibration.

## Data

- **Primary:** [martj42/international_results](https://github.com/martj42/international_results)
  — ~49k international matches, updated within days of every game. Committed to
  the repo so the model works out of the box; refresh anytime.
- **Optional (legacy):** StatsBomb Open Data powers an earlier event-level model
  (xG, possession, progressive passes) kept in the repo under `src/statsbomb_loader.py`,
  `src/features.py`, `src/train.py`. See "Legacy model" below.

---

## Quick start

```bash
git clone <your-repo-url>
cd "Soccer Game Predictor"
npm run dev          # sets up Python, installs deps, launches the app
```

`npm run dev` opens the app at http://127.0.0.1:8000. (The model runs on Python,
so Python 3 must be installed — the launcher tells you if it isn't. Not a Node
user? Double-click `Start Match Predictor.command`, or run
`uvicorn app.api:app --reload`.)

The repo ships with trained models and current ratings, so it runs immediately.

### Rebuild / keep data fresh

```bash
python scripts/build_all.py            # rebuild ratings, retrain, backtest, flags
python scripts/build_all.py --refresh  # download the latest results first
```

### Improve / experiment

```bash
python -m src.elo             # rebuild Elo ratings + features only
python -m src.train_intl      # retrain + print the chronological backtest
python scripts/tune_intl.py 24   # hyperparameter search (time-honest split)
python scripts/build_flags.py    # regenerate the flag map
```

### Command-line prediction

```bash
python -m src.predict Brazil France
```

---

## Higher, justified confidence: live market odds

The model is already **well-calibrated** — when it says 62%, it's right ~61% of
the time — so artificially inflating the confidence number would just make it
lie (the research term is "sharpness *subject to* calibration"). The only honest
way to raise confidence is a sharper *input signal*, and the strongest one in
football is the **betting market**: de-vigged odds are ~perfectly calibrated
(r≈0.995 vs actual frequencies) and beat rating models head-to-head.

So `src/market_odds.py` blends live market odds into the prediction when you
supply a free key. It:

1. fetches live 1X2 odds for a real fixture from **The Odds API** (free tier,
   500 requests/month; covers World Cup, qualifiers, Euro, Nations League, Copa
   América, AFCON, Gold Cup);
2. removes the bookmaker margin with **Shin's method**;
3. blends market + model probabilities via a **logarithmic opinion pool**.

Enable it by adding `ODDS_API_KEY` to `.env`, then toggle "Blend live odds" in
the app. It only affects **scheduled fixtures** (odds don't exist for
hypothetical matchups) and falls back to the pure model otherwise.

## Optional live modules (current rosters + gap-fill)

`.env`-keyed helpers you run locally (see `.env.example`). The model works fully
without them.

- `python -m src.roster_fetch Argentina` — current squad via API-Football.
- `python -m src.football_data_api` — recent results via football-data.org.

---

## Project structure

```
Soccer Game Predictor/
├── config.py                 # paths, schemas, team metadata
├── src/
│   ├── elo.py                 # Elo ratings + feature builder (primary)
│   ├── train_intl.py          # train + chronological backtest (primary)
│   ├── predict.py             # serving-time prediction (Elo model)
│   ├── model.py               # Poisson + Dixon-Coles scoreline maths
│   ├── statsbomb_loader.py    # legacy: StatsBomb data loader
│   ├── features.py / train.py # legacy: 20-feature event model
│   ├── roster_fetch.py        # optional current squads
│   └── football_data_api.py   # optional recent-results gap-fill
├── app/
│   ├── api.py                 # FastAPI backend
│   └── web/index.html         # single-page UI
├── scripts/
│   ├── build_all.py           # one-command pipeline
│   ├── tune_intl.py           # hyperparameter search
│   └── build_flags.py         # team -> flag map
├── models/                    # trained models + backtest metrics
├── data/
│   ├── international_results.csv   # ~49k matches (committed)
│   └── processed/                 # current_ratings.json, flags, etc.
└── tests/
```

## Legacy model

The original model (exactly 20 StatsBomb event features — xG, possession,
progressive passes, etc. — across six modern tournaments) is preserved in the
repo. It reached ~52% outcome accuracy on ~314 matches; the Elo model
superseded it by training on ~150× more data. Rebuild it with
`python scripts/train_multi.py` / `python scripts/tune.py` (see `src/train.py`).

## Notes & limitations

- International football is high-variance; even a strong model is right on the
  outcome ~60% of the time. Treat probabilities as probabilities, not certainties.
- Elo captures most of the signal; the extra features add small, consistent
  calibration gains.
- The backtest is a true forward-in-time split, so the reported numbers reflect
  genuine out-of-sample performance.

## License & attribution

MIT (see `LICENSE`). Match data from
[martj42/international_results](https://github.com/martj42/international_results)
and [StatsBomb Open Data](https://github.com/statsbomb/open-data); please credit
both in any public use.
