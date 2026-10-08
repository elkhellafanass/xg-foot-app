"""Explorateur de tirs : carte des tirs filtrable sur un demi-terrain."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from common import half_pitch, load_shots, source_note

from src.features import fr
from src.viz import GOAL_COLOR, MISS_COLOR

st.title("Explorateur de tirs")
shots, source = load_shots()
source_note(source)

ALL = "Toutes"
with st.container(border=True):
    c1, c2, c3 = st.columns(3)
    comps = sorted((shots["competition"] + " " + shots["season"]).unique())
    sel_comps = c1.multiselect(
        "Compétition",
        comps,
        default=[c for c in comps if "World Cup 2022" in c] or comps[:1],
        placeholder="Toutes les compétitions",
    )
    df = shots
    if sel_comps:
        df = df[(df["competition"] + " " + df["season"]).isin(sel_comps)]
    team = c2.selectbox("Équipe", [ALL] + sorted(df["team"].unique()))
    if team != ALL:
        df = df[df["team"] == team]
    player = c3.selectbox("Joueur", ["Tous"] + sorted(df["player"].dropna().unique()))
    if player != "Tous":
        df = df[df["player"] == player]
    c4, c5, c6 = st.columns(3)
    bodies = sorted(df["body_part"].unique())
    sel_body = c4.multiselect("Partie du corps", bodies, format_func=fr, placeholder="Toutes")
    types = sorted(df["shot_type"].unique())
    sel_type = c5.multiselect("Type de tir", types, format_func=fr, placeholder="Tous")
    goals_only = c6.toggle("Buts uniquement", value=False)
    if sel_body:
        df = df[df["body_part"].isin(sel_body)]
    if sel_type:
        df = df[df["shot_type"].isin(sel_type)]
    if goals_only:
        df = df[df["is_goal"] == 1]

xg_col = "xg_enrichi" if "xg_enrichi" in df else None
n, g = len(df), int(df["is_goal"].sum())
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Tirs", f"{n:,}".replace(",", " "))
m2.metric("Buts", g)
m3.metric("Conversion", f"{g / n:.1%}".replace(".", ",") if n else "-")
m4.metric("xG (modèle enrichi)", f"{df[xg_col].sum():.1f}".replace(".", ",") if xg_col and n else "-")
m5.metric("xG StatsBomb", f"{df['statsbomb_xg'].sum():.1f}".replace(".", ",") if n else "-")

if n == 0:
    st.warning("Aucun tir ne correspond aux filtres.")
    st.stop()

fig = half_pitch(height=560)
xg = df[xg_col] if xg_col else df["statsbomb_xg"]
custom = np.stack(
    [
        df["player"].fillna("?"),
        df["team"],
        df["opponent"],
        df["minute"],
        df["body_part"].map(fr),
        df["shot_type"].map(fr),
        xg,
        df["statsbomb_xg"],
        df["outcome"].fillna("?"),
    ],
    axis=1,
)
hover = (
    "<b>%{customdata[0]}</b> (%{customdata[1]})<br>contre %{customdata[2]}, %{customdata[3]}e min<br>"
    "%{customdata[4]} - %{customdata[5]}<br>xG modèle : %{customdata[6]:.2f} | "
    "xG StatsBomb : %{customdata[7]:.2f}<br>Issue : %{customdata[8]}<extra></extra>"
)
for is_goal, name, color in [(0, "Tir non converti", MISS_COLOR), (1, "But", GOAL_COLOR)]:
    mask = (df["is_goal"] == is_goal).to_numpy()
    if not mask.any():
        continue
    fig.add_trace(
        go.Scattergl(
            x=df["y"][mask],
            y=df["x"][mask],
            mode="markers",
            name=name,
            customdata=custom[mask],
            hovertemplate=hover,
            marker=dict(
                size=6 + 22 * xg[mask].to_numpy() if is_goal else 5 + 14 * xg[mask].to_numpy(),
                color=color,
                opacity=0.95 if is_goal else 0.45,
                line=dict(color="white", width=1 if is_goal else 0.3),
            ),
        )
    )

left, right = st.columns([3, 2], gap="large")
with left:
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption("Taille des points proportionnelle au xG du modèle enrichi. Buts en orange.")
with right:
    st.subheader("Tirs les plus dangereux")
    cols = [
        "player",
        "team",
        "outcome",
        xg_col or "statsbomb_xg",
        "statsbomb_xg",
    ]
    top = df.nlargest(15, xg_col or "statsbomb_xg")[cols].copy()
    top["outcome"] = top["outcome"].map(fr)
    st.dataframe(
        top.rename(
            columns={
                "player": "Joueur",
                "team": "Équipe",
                "opponent": "Adversaire",
                "minute": "Min.",
                "body_part": "Corps",
                "outcome": "Issue",
                "xg_enrichi": "xG modèle",
                "statsbomb_xg": "xG StatsBomb",
            }
        ),
        hide_index=True,
        use_container_width=True,
        column_config={
            "xG modèle": st.column_config.NumberColumn(format="%.2f"),
            "xG StatsBomb": st.column_config.NumberColumn(format="%.2f"),
        },
    )
