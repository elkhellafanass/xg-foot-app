"""Configuration centrale du projet (chemins, competitions, graine aleatoire)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

SHOTS_PATH = PROCESSED_DIR / "shots.parquet"

RAW_BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

RANDOM_STATE = 42

# Competitions masculines retenues : (competition_id, season_id).
# Toutes sont des tournois ou des saisons couverts integralement par StatsBomb.
COMPETITIONS: list[tuple[int, int]] = [
    (43, 106),  # FIFA World Cup 2022 (point de depart)
    (43, 3),  # FIFA World Cup 2018
    (55, 43),  # UEFA Euro 2020
    (55, 282),  # UEFA Euro 2024
    (223, 282),  # Copa America 2024
    (1267, 107),  # African Cup of Nations 2023
    (2, 27),  # Premier League 2015/2016
    (11, 27),  # La Liga 2015/2016
    (12, 27),  # Serie A 2015/2016
    (9, 27),  # 1. Bundesliga 2015/2016
    (7, 27),  # Ligue 1 2015/2016
]

# Sous-ensemble telecharge par l'application en ligne quand les donnees locales
# sont absentes (la licence StatsBomb interdit de redistribuer les donnees dans Git).
APP_COMPETITIONS: list[tuple[int, int]] = [
    (43, 106),  # FIFA World Cup 2022
    (55, 282),  # UEFA Euro 2024
]

ATTRIBUTION = "Données : StatsBomb Open Data (https://github.com/statsbomb/open-data)"
