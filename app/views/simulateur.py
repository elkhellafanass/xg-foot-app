"""Simulateur xG : l'utilisateur place un tir et compare les probabilites des modeles."""

import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from common import FS_LABELS, MissingFileError, half_pitch, load_models, show_missing

from src.features import GOAL_X, GOAL_Y, POST_LEFT_Y, POST_RIGHT_Y, fr, make_shot, shot_angle, shot_distance
from src.viz import GOAL_COLOR, MODEL_COLORS, TEXT_SECONDARY

st.title("Simulateur xG")
st.markdown(
    "Placez un tir en **cliquant sur le terrain** ou avec les curseurs, choisissez son contexte : "
    "chaque modèle déjà entraîné calcule sa probabilité de but."
)

try:
    models, manifest = load_models()
except MissingFileError as exc:
    show_missing(exc)

ss = st.session_state
ss.setdefault("sim_x", 108.0)
ss.setdefault("sim_y", 40.0)
ss.setdefault("sim_last_click", None)

# --- Clic sur le terrain : met a jour la position avant l'affichage des curseurs
event = ss.get("sim_pitch")
if event and event.get("selection", {}).get("points"):
    pt = event["selection"]["points"][0]
    click = (float(pt["y"]), float(pt["x"]))
    if click != ss.sim_last_click:
        ss.sim_last_click = click
        ss.sim_x, ss.sim_y = click

controls, pitch_col, result_col = st.columns([1.0, 1.7, 1.3], gap="medium")

with controls:
    st.subheader("Tir")
    x = st.slider("x : profondeur (120 = ligne de but)", 60.0, 120.0, step=0.5, key="sim_x")
    y = st.slider("y : largeur (40 = axe du but)", 0.0, 80.0, step=0.5, key="sim_y")
    body = st.selectbox("Partie du corps", ["Right Foot", "Left Foot", "Head", "Other"], format_func=fr)
    shot_type = st.selectbox("Type de tir", ["Open Play", "Free Kick", "Corner", "Kick Off"], format_func=fr)
    technique = st.selectbox(
        "Technique",
        ["Normal", "Half Volley", "Volley", "Lob", "Backheel", "Diving Header", "Overhead Kick"],
        format_func=fr,
    )
    pattern = st.selectbox(
        "Phase de jeu",
        [
            "Regular Play",
            "From Counter",
            "From Corner",
            "From Free Kick",
            "From Throw In",
            "From Goal Kick",
            "From Keeper",
            "From Kick Off",
            "Other",
        ],
        format_func=fr,
    )
    c1, c2 = st.columns(2)
    pressure = c1.checkbox("Sous pression")
    first_time = c2.checkbox("Premier contact")
    minute = st.slider("Minute", 1, 120, 45)
    score = st.slider("Écart au score avant le tir (équipe qui tire)", -4, 4, 0)

    defaults = manifest.get("freeze_defaults", {})
    with st.expander("Défenseurs et gardien (modèles enrichis)", expanded=False):
        st.caption(
            "Valeurs par défaut : médianes observées à l'entraînement. Le gardien est "
            "placé sur la droite tireur - centre du but."
        )
        gk_to_goal = st.slider(
            "Distance gardien - centre du but (yards)",
            0.0,
            15.0,
            float(round(defaults.get("gk_to_goal", 2.5), 1)),
            0.5,
        )
        n_def = st.slider(
            "Défenseurs dans le triangle de tir", 0, 6, int(round(defaults.get("n_defenders_triangle", 1)))
        )
        nearest = st.slider(
            "Distance du défenseur le plus proche (yards)",
            0.5,
            20.0,
            float(round(defaults.get("nearest_defender_dist", 3.0), 1)),
            0.5,
        )

dist = float(shot_distance(x, y))
angle = float(shot_angle(x, y))
# Gardien sur le segment centre du but -> tireur, a gk_to_goal yards de son but.
ratio = min(gk_to_goal / dist, 1.0) if dist > 0 else 0.0
gk_x, gk_y = GOAL_X + (x - GOAL_X) * ratio, GOAL_Y + (y - GOAL_Y) * ratio
freeze = {
    "gk_distance": max(dist - gk_to_goal, 0.0),
    "gk_to_goal": gk_to_goal,
    "gk_in_triangle": 1.0,
    "n_defenders_triangle": float(n_def),
    "nearest_defender_dist": nearest,
}
shot = make_shot(x, y, body, shot_type, technique, pattern, pressure, first_time, minute, score, freeze)

with pitch_col:
    fig = half_pitch(height=420)
    gx, gy = np.meshgrid(np.arange(0, 80.5, 1.0), np.arange(60, 120.5, 1.0))
    fig.add_trace(
        go.Scatter(
            x=gx.ravel(),
            y=gy.ravel(),
            mode="markers",
            marker=dict(size=9, color="rgba(255,255,255,0.02)"),
            hovertemplate="x = %{y}, y = %{x}<extra>Cliquer pour tirer d'ici</extra>",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[y, POST_LEFT_Y, POST_RIGHT_Y, y],
            y=[x, GOAL_X, GOAL_X, x],
            mode="lines",
            fill="toself",
            fillcolor="rgba(235,104,52,0.18)",
            line=dict(color=GOAL_COLOR, width=1.5),
            name="Triangle d'ouverture",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[gk_y],
            y=[gk_x],
            mode="markers",
            name="Gardien (hypothèse)",
            hoverinfo="skip",
            marker=dict(size=11, color="#4a3aa7", symbol="diamond", line=dict(color="white", width=1)),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[y],
            y=[x],
            mode="markers",
            name="Tireur",
            hoverinfo="skip",
            marker=dict(size=14, color=GOAL_COLOR, line=dict(color="white", width=2)),
        )
    )
    fig.update_layout(clickmode="event+select", dragmode=False)
    st.plotly_chart(
        fig,
        use_container_width=True,
        key="sim_pitch",
        on_select="rerun",
        selection_mode="points",
        config={"displayModeBar": False},
    )
    m1, m2 = st.columns(2)
    m1.metric("Distance au but", f"{dist:.1f} yd".replace(".", ","))
    m2.metric("Angle d'ouverture", f"{math.degrees(angle):.1f}°".replace(".", ","))

with result_col:
    st.subheader("Probabilité de but")
    rows = []
    for info in manifest["models"]:
        if info["key"] not in models:
            continue
        p = float(models[info["key"]].predict_proba(shot[info["features"]])[:, 1][0])
        rows.append((info, p))
    main = manifest.get("main_models", {})
    main_key = f"enrichi__{main.get('enrichi', 'hgb')}"
    main_p = next((p for info, p in rows if info["key"] == main_key), None)
    if main_p is not None:
        st.metric("Modèle principal (HistGB, jeu enrichi)", f"{main_p:.1%}".replace(".", ","))
    # Echelle commune aux deux graphiques, adaptee aux probabilites affichees.
    x_max = min(1.0, max(0.3, 1.3 * max((p for _, p in rows), default=0.3)))
    for fs in ("simple", "enrichi"):
        sub = [(i, p) for i, p in rows if i["feature_set"] == fs]
        if not sub:
            continue
        bar = go.Figure(
            go.Bar(
                x=[p for _, p in sub],
                y=[i["label"] for i, _ in sub],
                orientation="h",
                marker_color=[MODEL_COLORS[i["name"]] for i, _ in sub],
                text=[f"{p:.1%}".replace(".", ",") for _, p in sub],
                textposition="outside",
                hovertemplate="%{y} : %{x:.1%}<extra></extra>",
                cliponaxis=False,
            )
        )
        bar.update_layout(
            title=dict(text=FS_LABELS[fs], font=dict(size=14)),
            height=60 + 42 * len(sub),
            margin=dict(l=10, r=50, t=40, b=10),
            xaxis=dict(range=[0, x_max], tickformat=".0%", gridcolor="#e4e3df"),
            yaxis=dict(autorange="reversed"),
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT_SECONDARY),
        )
        st.plotly_chart(bar, use_container_width=True, config={"displayModeBar": False})
    st.caption(
        "La baseline prédit toujours la fréquence moyenne de but de l'entraînement. "
        "XGBoost n'apparaît que si la bibliothèque est installée (non déployée en ligne)."
    )
