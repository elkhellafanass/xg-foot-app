"""Variables explicatives des modeles xG.

Repere StatsBomb : terrain de 120 x 80, l'equipe qui tire attaque vers x = 120.
Le but est centre en (120, 40), poteaux en y = 36 et y = 44 (largeur 8 yards).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0
GOAL_X = 120.0
GOAL_Y = 40.0
POST_LEFT_Y = 36.0
POST_RIGHT_Y = 44.0

CATEGORICAL_FEATURES = ["body_part", "shot_type", "technique", "play_pattern"]
BOOLEAN_FEATURES = ["under_pressure", "first_time", "is_volley"]
NUMERIC_FEATURES = ["distance", "angle", "minute", "score_diff_before"]
FREEZE_FRAME_FEATURES = [
    "gk_distance",
    "gk_to_goal",
    "gk_in_triangle",
    "n_defenders_triangle",
    "nearest_defender_dist",
]

FEATURE_SETS: dict[str, list[str]] = {
    "simple": NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES,
    "enrichi": NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES + FREEZE_FRAME_FEATURES,
}

# Libelles francais (affichage uniquement ; les modeles utilisent les valeurs StatsBomb).
FEATURE_LABELS = {
    "distance": "Distance au but",
    "angle": "Angle d'ouverture",
    "minute": "Minute",
    "score_diff_before": "Écart au score avant le tir",
    "under_pressure": "Sous pression",
    "first_time": "Premier contact",
    "is_volley": "Volée / demi-volée",
    "body_part": "Partie du corps",
    "shot_type": "Type de tir",
    "technique": "Technique",
    "play_pattern": "Phase de jeu",
    "gk_distance": "Distance tireur-gardien",
    "gk_to_goal": "Distance gardien-but",
    "gk_in_triangle": "Gardien dans le triangle",
    "n_defenders_triangle": "Défenseurs dans le triangle",
    "nearest_defender_dist": "Défenseur le plus proche",
}

VALUE_LABELS = {
    "Right Foot": "Pied droit",
    "Left Foot": "Pied gauche",
    "Head": "Tête",
    "Other": "Autre",
    "Open Play": "Jeu ouvert",
    "Free Kick": "Coup franc direct",
    "Corner": "Corner direct",
    "Kick Off": "Engagement",
    "Normal": "Normale",
    "Volley": "Volée",
    "Half Volley": "Demi-volée",
    "Lob": "Lob",
    "Backheel": "Talonnade",
    "Diving Header": "Tête plongeante",
    "Overhead Kick": "Retourné",
    "Regular Play": "Jeu placé",
    "From Corner": "Sur corner",
    "From Free Kick": "Sur coup franc",
    "From Throw In": "Sur touche",
    "From Counter": "Contre-attaque",
    "From Goal Kick": "Sur dégagement de but",
    "From Keeper": "Relance du gardien",
    "From Kick Off": "Sur engagement",
    "Goal": "But",
    "Saved": "Arrêté",
    "Saved Off Target": "Arrêté (non cadré)",
    "Saved to Post": "Arrêté sur le poteau",
    "Off T": "Non cadré",
    "Blocked": "Contré",
    "Post": "Poteau",
    "Wayward": "Raté",
}


def fr(value: object) -> str:
    """Traduit une modalite StatsBomb en francais (identite si inconnue)."""
    return VALUE_LABELS.get(str(value), str(value))


def shot_distance(x, y):
    """Distance (yards) entre le point de tir et le centre du but : sqrt((120-x)^2 + (40-y)^2)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.sqrt((GOAL_X - x) ** 2 + (GOAL_Y - y) ** 2)


def shot_angle(x, y):
    """Angle d'ouverture du but (radians) : angle entre les vecteurs tireur->poteaux.

    Vaut 0 si le tireur est sur un poteau (vecteur nul) et pi sur la ligne de but entre les poteaux.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ax, ay = GOAL_X - x, POST_LEFT_Y - y
    bx, by = GOAL_X - x, POST_RIGHT_Y - y
    norm = np.sqrt(ax**2 + ay**2) * np.sqrt(bx**2 + by**2)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.where(norm > 0, (ax * bx + ay * by) / np.where(norm > 0, norm, 1.0), 1.0)
    return np.arccos(np.clip(cos, -1.0, 1.0))


def _in_triangle(px: float, py: float, x: float, y: float) -> bool:
    """Le point (px, py) est-il dans le triangle tireur (x, y) - poteau gauche - poteau droit ?"""

    def cross(ax, ay, bx, by, cx, cy):
        return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

    d1 = cross(x, y, GOAL_X, POST_LEFT_Y, px, py)
    d2 = cross(GOAL_X, POST_LEFT_Y, GOAL_X, POST_RIGHT_Y, px, py)
    d3 = cross(GOAL_X, POST_RIGHT_Y, x, y, px, py)
    has_neg = d1 < 0 or d2 < 0 or d3 < 0
    has_pos = d1 > 0 or d2 > 0 or d3 > 0
    return not (has_neg and has_pos)


def freeze_frame_features(x: float | None, y: float | None, freeze_frame: list | None) -> dict:
    """Variables issues du freeze_frame (positions des joueurs au moment du tir).

    Seuls les adversaires (teammate = False) sont consideres. Valeurs manquantes (NaN)
    si le freeze_frame est absent.
    """
    out = {
        "has_freeze_frame": False,
        "gk_distance": math.nan,
        "gk_to_goal": math.nan,
        "gk_in_triangle": math.nan,
        "n_defenders_triangle": math.nan,
        "nearest_defender_dist": math.nan,
    }
    if not freeze_frame or x is None or y is None:
        return out
    out["has_freeze_frame"] = True
    n_def = 0
    nearest = math.inf
    for player in freeze_frame:
        if player.get("teammate", True):
            continue
        px, py = player["location"][:2]
        position = (player.get("position") or {}).get("name", "")
        if position == "Goalkeeper":
            out["gk_distance"] = math.hypot(px - x, py - y)
            out["gk_to_goal"] = math.hypot(GOAL_X - px, GOAL_Y - py)
            out["gk_in_triangle"] = float(_in_triangle(px, py, x, y))
            continue
        nearest = min(nearest, math.hypot(px - x, py - y))
        if _in_triangle(px, py, x, y):
            n_def += 1
    out["n_defenders_triangle"] = float(n_def)
    out["nearest_defender_dist"] = nearest if math.isfinite(nearest) else math.nan
    return out


def add_features(shots: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les variables geometriques et derivees a une table de tirs."""
    df = shots.copy()
    df["distance"] = shot_distance(df["x"], df["y"])
    df["angle"] = shot_angle(df["x"], df["y"])
    df["is_volley"] = df["technique"].isin(["Volley", "Half Volley"])
    for col in ("under_pressure", "first_time", "is_volley"):
        df[col] = df[col].astype(bool)
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("Other").astype(str)
    if "score_diff_before" not in df:
        df["score_diff_before"] = df["goals_for_before"] - df["goals_against_before"]
    return df


def make_shot(
    x: float,
    y: float,
    body_part: str = "Right Foot",
    shot_type: str = "Open Play",
    technique: str = "Normal",
    play_pattern: str = "Regular Play",
    under_pressure: bool = False,
    first_time: bool = False,
    minute: int = 45,
    score_diff_before: int = 0,
    freeze: dict | None = None,
) -> pd.DataFrame:
    """Construit une ligne de tir (pour le simulateur) avec toutes les variables des modeles."""
    row = {
        "x": x,
        "y": y,
        "body_part": body_part,
        "shot_type": shot_type,
        "technique": technique,
        "play_pattern": play_pattern,
        "under_pressure": under_pressure,
        "first_time": first_time,
        "minute": minute,
        "score_diff_before": score_diff_before,
    }
    for col in FREEZE_FRAME_FEATURES:
        row[col] = (freeze or {}).get(col, math.nan)
    return add_features(pd.DataFrame([row]))
