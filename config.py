"""Central configuration for the Soccer Game Predictor.

Defines paths, the training tournament set, the exact 20-feature schema used by
the model, and national-team metadata (confederation + flag code).
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
# Large regenerable cache (git clone of StatsBomb open-data + downloaded events).
# Override with SGP_DATA_DIR so a sandbox/CI can point it at fast local storage.
DATA_DIR = Path(os.environ.get("SGP_DATA_DIR") or (ROOT / "data" / "raw"))
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
ASSETS_DIR = ROOT / "data" / "assets"

STATSBOMB_REPO = "https://github.com/statsbomb/open-data"
STATSBOMB_CLONE_DIR = DATA_DIR / "statsbomb_open_data"

# Large historical international results dataset (martj42/international_results),
# committed to the repo so the upgraded Elo model works out of the box.
INTL_RESULTS_CSV = ROOT / "data" / "international_results.csv"
INTL_RESULTS_REPO = "https://github.com/martj42/international_results"

for _p in (PROCESSED_DIR, MODELS_DIR, ASSETS_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Training tournaments (modern international men's competitions with full events)
# (competition_id, season_id, label)
# --------------------------------------------------------------------------- #
TOURNAMENTS = [
    (43, 3, "FIFA World Cup 2018"),
    (43, 106, "FIFA World Cup 2022"),
    (55, 43, "UEFA Euro 2020"),
    (55, 282, "UEFA Euro 2024"),
    (223, 282, "Copa America 2024"),
    (1267, 107, "African Cup of Nations 2023"),
]

# --------------------------------------------------------------------------- #
# The 20 model features (order matters; must match training + serving)
# --------------------------------------------------------------------------- #
FEATURE_COLS = [
    "home_xg_for_avg", "home_xg_against_avg", "home_goals_for_avg",
    "home_goals_against_avg", "home_shot_accuracy_pct", "home_possession_pct",
    "home_progressive_passes_pct", "home_confederation",
    "away_xg_for_avg", "away_xg_against_avg",
    "away_goals_for_avg", "away_goals_against_avg", "away_shot_accuracy_pct",
    "away_possession_pct", "away_progressive_passes_pct", "away_confederation",
    "h2h_home_wins", "h2h_draws", "h2h_away_wins", "is_neutral",
]

# Per-team profile stats (the 7 metrics we average across a team's matches).
TEAM_STAT_COLS = [
    "xg_for", "xg_against", "goals_for", "goals_against",
    "shot_accuracy_pct", "possession_pct", "progressive_passes_pct",
]

# --------------------------------------------------------------------------- #
# Confederations (categorical feature -> integer code)
# --------------------------------------------------------------------------- #
CONFEDERATION_CODES = {"UEFA": 0, "CONMEBOL": 1, "CONCACAF": 2, "CAF": 3, "AFC": 4, "OFC": 5}

TEAM_CONFEDERATION = {
    # UEFA
    "Albania": "UEFA", "Austria": "UEFA", "Belgium": "UEFA", "Croatia": "UEFA",
    "Czech Republic": "UEFA", "Denmark": "UEFA", "England": "UEFA", "Finland": "UEFA",
    "France": "UEFA", "Georgia": "UEFA", "Germany": "UEFA", "Hungary": "UEFA",
    "Iceland": "UEFA", "Italy": "UEFA", "Netherlands": "UEFA", "North Macedonia": "UEFA",
    "Poland": "UEFA", "Portugal": "UEFA", "Romania": "UEFA", "Russia": "UEFA",
    "Scotland": "UEFA", "Serbia": "UEFA", "Slovakia": "UEFA", "Slovenia": "UEFA",
    "Spain": "UEFA", "Sweden": "UEFA", "Switzerland": "UEFA", "Turkey": "UEFA",
    "Ukraine": "UEFA", "Wales": "UEFA",
    # CONMEBOL
    "Argentina": "CONMEBOL", "Bolivia": "CONMEBOL", "Brazil": "CONMEBOL",
    "Chile": "CONMEBOL", "Colombia": "CONMEBOL", "Ecuador": "CONMEBOL",
    "Paraguay": "CONMEBOL", "Peru": "CONMEBOL", "Uruguay": "CONMEBOL",
    "Venezuela": "CONMEBOL",
    # CONCACAF
    "Canada": "CONCACAF", "Costa Rica": "CONCACAF", "Jamaica": "CONCACAF",
    "Mexico": "CONCACAF", "Panama": "CONCACAF", "United States": "CONCACAF",
    # CAF
    "Algeria": "CAF", "Angola": "CAF", "Burkina Faso": "CAF", "Cameroon": "CAF",
    "Cape Verde Islands": "CAF", "Congo DR": "CAF", "Côte d'Ivoire": "CAF",
    "Egypt": "CAF", "Equatorial Guinea": "CAF", "Gambia": "CAF", "Ghana": "CAF",
    "Guinea": "CAF", "Guinea-Bissau": "CAF", "Mali": "CAF", "Mauritania": "CAF",
    "Morocco": "CAF", "Mozambique": "CAF", "Namibia": "CAF", "Nigeria": "CAF",
    "Senegal": "CAF", "South Africa": "CAF", "Tanzania": "CAF", "Tunisia": "CAF",
    "Zambia": "CAF",
    # AFC
    "Australia": "AFC", "Iran": "AFC", "Japan": "AFC", "Qatar": "AFC",
    "Saudi Arabia": "AFC", "South Korea": "AFC",
}

# ISO flag codes for https://flagcdn.com (lowercase). Sub-national UK flags use
# the gb-eng / gb-sct / gb-wls codes that flagcdn supports.
TEAM_FLAG_CODE = {
    "Albania": "al", "Algeria": "dz", "Angola": "ao", "Argentina": "ar",
    "Australia": "au", "Austria": "at", "Belgium": "be", "Bolivia": "bo",
    "Brazil": "br", "Burkina Faso": "bf", "Cameroon": "cm", "Canada": "ca",
    "Cape Verde Islands": "cv", "Chile": "cl", "Colombia": "co", "Congo DR": "cd",
    "Costa Rica": "cr", "Croatia": "hr", "Czech Republic": "cz", "Côte d'Ivoire": "ci",
    "Denmark": "dk", "Ecuador": "ec", "Egypt": "eg", "England": "gb-eng",
    "Equatorial Guinea": "gq", "Finland": "fi", "France": "fr", "Gambia": "gm",
    "Georgia": "ge", "Germany": "de", "Ghana": "gh", "Guinea": "gn",
    "Guinea-Bissau": "gw", "Hungary": "hu", "Iceland": "is", "Iran": "ir",
    "Italy": "it", "Jamaica": "jm", "Japan": "jp", "Mali": "ml",
    "Mauritania": "mr", "Mexico": "mx", "Morocco": "ma", "Mozambique": "mz",
    "Namibia": "na", "Netherlands": "nl", "Nigeria": "ng", "North Macedonia": "mk",
    "Panama": "pa", "Paraguay": "py", "Peru": "pe", "Poland": "pl",
    "Portugal": "pt", "Qatar": "qa", "Romania": "ro", "Russia": "ru",
    "Saudi Arabia": "sa", "Scotland": "gb-sct", "Senegal": "sn", "Serbia": "rs",
    "Slovakia": "sk", "Slovenia": "si", "South Africa": "za", "South Korea": "kr",
    "Spain": "es", "Sweden": "se", "Switzerland": "ch", "Tanzania": "tz",
    "Tunisia": "tn", "Turkey": "tr", "Ukraine": "ua", "United States": "us",
    "Uruguay": "uy", "Venezuela": "ve", "Wales": "gb-wls", "Zambia": "zm",
}


def get_confederation(team: str) -> str:
    return TEAM_CONFEDERATION.get(team, "UEFA")


def get_confederation_code(team: str) -> int:
    return CONFEDERATION_CODES[get_confederation(team)]


def get_flag_code(team: str) -> str:
    return TEAM_FLAG_CODE.get(team, "un")  # 'un' = generic UN flag fallback


# Model / simulation hyper-parameters
MAX_GOALS = 10          # cap when enumerating the scoreline distribution
RANDOM_SEED = 42
