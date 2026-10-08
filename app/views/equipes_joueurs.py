"""Equipes et joueurs : buts reels contre xG cumules, avec mise en garde sur les petits echantillons."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import load_shots, source_note

from src.viz import GOAL_COLOR, MISS_COLOR, TEXT_SECONDARY

st.title("Équipes et joueurs")
shots, source = load_shots()
source_note(source)

st.warning(
    "**Prudence avec les petits échantillons.** Un écart entre buts et xG sur quelques dizaines "
    "de tirs relève souvent du hasard, pas du talent. Sous l'hypothèse que les xG sont exacts, "
    "le nombre de buts a une variance égale à la somme des p(1 - p) : la colonne « Écart "
    "significatif » indique si l'écart dépasse 1,96 écart-type (seuil de 95 %). Avec de "
    "nombreuses lignes comparées, quelques écarts « significatifs » sont attendus par hasard.",
    icon=":material/warning:",
)

xg_col = "xg_enrichi" if "xg_enrichi" in shots else "statsbomb_xg"
c1, c2, c3 = st.columns([2, 1, 1])
comps = sorted((shots["competition"] + " " + shots["season"]).unique())
sel = c1.multiselect("Compétitions", comps, placeholder="Toutes les compétitions")
test_only = c2.toggle(
    "Matchs du test uniquement",
    value=False,
    help="Restreint aux matchs jamais vus par les modèles pendant l'entraînement.",
)
min_shots = c3.number_input("Tirs minimum (joueurs)", min_value=1, max_value=300, value=30, step=5)

df = shots
if sel:
    df = df[(df["competition"] + " " + df["season"]).isin(sel)]
if test_only:
    df = df[df["jeu_test"]]
if df.empty:
    st.warning("Aucun tir pour cette sélection.")
    st.stop()
if not test_only:
    st.caption(
        "Par défaut, la majorité de ces tirs ont servi à entraîner les modèles (prédictions "
        "« dans l'échantillon ») ; activez « Matchs du test uniquement » pour une lecture hors échantillon."
    )


def aggregate(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    g = (
        frame.assign(var=frame[xg_col] * (1 - frame[xg_col]))
        .groupby(key)
        .agg(
            Tirs=("is_goal", "size"),
            Buts=("is_goal", "sum"),
            xG=(xg_col, "sum"),
            xG_StatsBomb=("statsbomb_xg", "sum"),
            var=("var", "sum"),
        )
    )
    g["Buts - xG"] = g["Buts"] - g["xG"]
    z = g["Buts - xG"] / np.sqrt(g["var"].clip(lower=1e-9))
    g["Écart significatif"] = np.where(z.abs() > 1.96, np.where(z > 0, "oui (+)", "oui (-)"), "non")
    return g.drop(columns="var").reset_index()


num_cfg = {c: st.column_config.NumberColumn(format="%.1f") for c in ["xG", "xG_StatsBomb", "Buts - xG"]}
tab_t, tab_p = st.tabs(["Équipes", "Joueurs"])

with tab_t:
    teams = aggregate(df, "team").sort_values("xG", ascending=False)
    fig = go.Figure()
    lim = float(max(teams["xG"].max(), teams["Buts"].max()) * 1.08)
    fig.add_trace(
        go.Scatter(
            x=[0, lim],
            y=[0, lim],
            mode="lines",
            line=dict(color="#c3c2b7", width=1),
            name="Buts = xG",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=teams["xG"],
            y=teams["Buts"],
            mode="markers",
            name="Équipes",
            text=teams["team"],
            marker=dict(
                size=9,
                color=np.where(teams["Buts - xG"] >= 0, GOAL_COLOR, MISS_COLOR),
                line=dict(color="white", width=1),
            ),
            hovertemplate="<b>%{text}</b><br>xG %{x:.1f} - buts %{y}<extra></extra>",
        )
    )
    fig.update_layout(
        height=480,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="xG cumulés (modèle enrichi)", gridcolor="#e4e3df", range=[0, lim]),
        yaxis=dict(title="Buts marqués (hors penalties)", gridcolor="#e4e3df", range=[0, lim]),
        font=dict(color=TEXT_SECONDARY),
        showlegend=False,
    )
    left, right = st.columns([2, 3], gap="large")
    with left:
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.caption("Au-dessus de la diagonale (orange) : plus de buts que d'xG.")
    with right:
        st.dataframe(
            teams.rename(columns={"team": "Équipe"}),
            hide_index=True,
            use_container_width=True,
            height=480,
            column_config=num_cfg,
        )

with tab_p:
    players = aggregate(df.dropna(subset=["player"]), "player")
    players = players[players["Tirs"] >= min_shots].sort_values("Buts - xG", ascending=False)
    if players.empty:
        st.info("Aucun joueur n'atteint le nombre minimal de tirs.")
    else:
        st.markdown(f"**{len(players)} joueurs** avec au moins {min_shots} tirs.")
        st.dataframe(
            players.rename(columns={"player": "Joueur"}),
            hide_index=True,
            use_container_width=True,
            height=520,
            column_config=num_cfg,
        )
