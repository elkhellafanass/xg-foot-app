"""Fonctions partagees par les pages : chargement (avec cache) des donnees, modeles et rapports."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    APP_COMPETITIONS,
    ATTRIBUTION,
    DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    SHOTS_PATH,
)
from src.features import add_features  # noqa: E402

APP_CACHE_DIR = DATA_DIR / "app_cache"
APP_SHOTS_PATH = APP_CACHE_DIR / "shots_app.parquet"
LOGO_PATH = Path(__file__).resolve().parent / "assets" / "statsbomb_logo.png"
FIGURES_DIR = REPORTS_DIR / "figures"

FS_LABELS = {"simple": "Jeu simple", "enrichi": "Jeu enrichi (freeze frame)"}


class MissingFileError(RuntimeError):
    """Un fichier produit par le pipeline est absent."""


def show_missing(exc: Exception) -> None:
    st.error(
        f"**Fichier manquant** : {exc}\n\n"
        "Produisez-le avec le pipeline (voir README) : `python scripts/download_data.py`, "
        "`python scripts/build_dataset.py`, puis `python scripts/train.py`."
    )
    st.stop()


# ----------------------------------------------------------------------------- rapports
@st.cache_data(show_spinner=False)
def load_csv(name: str) -> pd.DataFrame:
    path = REPORTS_DIR / name
    if not path.exists():
        raise MissingFileError(f"reports/{name}")
    return pd.read_csv(path, encoding="utf-8")


@st.cache_data(show_spinner=False)
def load_metrics_json() -> dict:
    path = REPORTS_DIR / "metrics.json"
    if not path.exists():
        raise MissingFileError("reports/metrics.json")
    return json.loads(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------ modeles
@st.cache_resource(show_spinner="Chargement des modèles...")
def load_models() -> tuple[dict, dict]:
    """Charge les modeles deja entraines (jamais de re-entrainement dans l'application).

    Les modeles dont la bibliotheque n'est pas installee (ex. XGBoost en ligne) sont ignores.
    """
    manifest_path = MODELS_DIR / "manifest.json"
    if not manifest_path.exists():
        raise MissingFileError("models/manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    models = {}
    for info in manifest["models"]:
        try:
            models[info["key"]] = joblib.load(MODELS_DIR / info["file"])
        except (ImportError, ModuleNotFoundError, FileNotFoundError):
            continue
    if not models:
        raise MissingFileError("models/*.joblib")
    return models, manifest


def predict(models: dict, manifest: dict, shots: pd.DataFrame) -> dict[str, np.ndarray]:
    out = {}
    for info in manifest["models"]:
        if info["key"] in models:
            out[info["key"]] = models[info["key"]].predict_proba(shots[info["features"]])[:, 1]
    return out


# ------------------------------------------------------------------------------ donnees
def ensure_shots() -> tuple[Path, str]:
    """Retourne le chemin de la table des tirs et sa provenance.

    - En local : table complete produite par le pipeline (data/processed/shots.parquet).
    - En ligne : la licence StatsBomb interdit de redistribuer les donnees dans le depot ;
      l'application telecharge donc un sous-ensemble au premier lancement (cache disque).
    """
    if SHOTS_PATH.exists():
        return SHOTS_PATH, "complet"
    if APP_SHOTS_PATH.exists():
        return APP_SHOTS_PATH, "en_ligne"
    from src.data import build_shots_table

    with st.status("Premier lancement : téléchargement des données StatsBomb...", expanded=True) as status:
        st.write(
            "Les données ne sont pas stockées dans le dépôt (licence) : elles sont "
            "téléchargées une seule fois depuis StatsBomb Open Data puis mises en cache."
        )
        bar = st.progress(0.0)

        def progress(done: int, total: int) -> None:
            bar.progress(done / max(total, 1), text=f"{done} / {total} matchs")

        try:
            df = build_shots_table(APP_COMPETITIONS, raw_dir=APP_CACHE_DIR / "raw", progress=progress)
        except Exception as exc:  # reseau indisponible, etc.
            status.update(label="Échec du téléchargement", state="error")
            st.error(f"Impossible de télécharger les données StatsBomb : {exc}")
            st.stop()
        APP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        add_features(df).to_parquet(APP_SHOTS_PATH, index=False)
        status.update(label="Données prêtes", state="complete", expanded=False)
    return APP_SHOTS_PATH, "en_ligne"


@st.cache_data(show_spinner="Préparation des tirs et calcul des xG...")
def _load_shots_cached(path_str: str, mtime: float) -> pd.DataFrame:
    df = pd.read_parquet(path_str)
    models, manifest = load_models()
    preds = predict(models, manifest, df)
    for fs, name in manifest.get("main_models", {}).items():
        key = f"{fs}__{name}"
        if key in preds:
            df[f"xg_{fs}"] = preds[key]
    split_path = MODELS_DIR / "split.json"
    if split_path.exists():
        split = json.loads(split_path.read_text(encoding="utf-8"))
        test_ids = set(split["test"]["match_ids"])
        df["jeu_test"] = df["match_id"].isin(test_ids)
    else:
        df["jeu_test"] = False
    return df


def load_shots() -> tuple[pd.DataFrame, str]:
    try:
        path, source = ensure_shots()
        return _load_shots_cached(str(path), path.stat().st_mtime), source
    except MissingFileError as exc:
        show_missing(exc)
        raise


def source_note(source: str) -> None:
    if source == "en_ligne":
        st.info(
            "Version en ligne : les tirs affichés proviennent de la Coupe du monde 2022 et de "
            "l'Euro 2024, téléchargés au démarrage (la licence StatsBomb interdit de les "
            "stocker dans le dépôt). Les modèles et les métriques portent, eux, sur le jeu "
            "complet (11 compétitions).",
            icon=":material/info:",
        )


# ------------------------------------------------------------------------------ terrain
LINE = "#9a9993"
PITCH_BG = "#f4f8f1"


def half_pitch(height: int = 520) -> go.Figure:
    """Demi-terrain vertical (but en haut). Axe horizontal = y StatsBomb, vertical = x."""
    fig = go.Figure()
    shapes = [
        dict(type="rect", x0=0, y0=60, x1=80, y1=120),
        dict(type="rect", x0=18, y0=102, x1=62, y1=120),
        dict(type="rect", x0=30, y0=114, x1=50, y1=120),
        dict(type="rect", x0=36, y0=120, x1=44, y1=121.5),
    ]
    for s in shapes:
        fig.add_shape(**s, line=dict(color=LINE, width=1.2), layer="below")
    t = np.linspace(np.deg2rad(217), np.deg2rad(323), 40)
    fig.add_trace(
        go.Scatter(
            x=40 + 10 * np.cos(t),
            y=108 + 10 * np.sin(t),
            mode="lines",
            line=dict(color=LINE, width=1.2),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    t = np.linspace(0, np.pi, 40)
    fig.add_trace(
        go.Scatter(
            x=40 + 10 * np.cos(t),
            y=60 + 10 * np.sin(t),
            mode="lines",
            line=dict(color=LINE, width=1.2),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[40],
            y=[108],
            mode="markers",
            marker=dict(color=LINE, size=4),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.update_xaxes(range=[-1, 81], visible=False, fixedrange=True, constrain="domain")
    # Terrain colle en haut (sous la legende) quand la hauteur disponible est plus grande.
    fig.update_yaxes(
        range=[59, 123],
        visible=False,
        scaleanchor="x",
        scaleratio=1,
        fixedrange=True,
        constrain="domain",
        constraintoward="top",
    )
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor=PITCH_BG,
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.0, xanchor="center", x=0.5, bgcolor="rgba(0,0,0,0)"
        ),
    )
    return fig


def sidebar() -> None:
    with st.sidebar:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), use_container_width=True)
        st.caption(ATTRIBUTION)
