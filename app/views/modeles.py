"""Comparaison des modeles : metriques, courbes ROC et de calibration, importance des variables."""

import plotly.graph_objects as go
import streamlit as st
from common import FS_LABELS, MissingFileError, load_csv, load_metrics_json, show_missing

from src.features import FEATURE_LABELS
from src.models import MODEL_LABELS
from src.viz import MODEL_COLORS, TEXT_SECONDARY

DASH = {
    "baseline": "dot",
    "logreg": "solid",
    "rf": "dash",
    "hgb": "dashdot",
    "xgb": "longdashdot",
    "statsbomb": "dot",
}

st.title("Modèles")
try:
    metrics = load_csv("metrics.csv")
    mjson = load_metrics_json()
    importance = load_csv("permutation_importance.csv")
    errors = load_csv("error_analysis.csv")
except MissingFileError as exc:
    show_missing(exc)

split = mjson["split"]
st.markdown(
    f"Découpage **par match** (aucun match partagé) : entraînement {split['train']['n_matchs']} matchs "
    f"({split['train']['n_tirs']} tirs), calibration {split['calib']['n_matchs']} matchs, "
    f"test {split['test']['n_matchs']} matchs ({split['test']['n_tirs']} tirs, "
    f"{split['test']['n_buts']} buts). Intervalles de confiance à 95 % par bootstrap sur les matchs du test."
)


def layout(fig: go.Figure, xt: str, yt: str, height: int = 460) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title=xt, gridcolor="#e4e3df", zeroline=False),
        yaxis=dict(title=yt, gridcolor="#e4e3df", zeroline=False),
        legend=dict(orientation="v", x=1.02, y=1, xanchor="left"),
        font=dict(color=TEXT_SECONDARY),
    )
    return fig


fs = st.radio(
    "Jeu de variables pour les courbes et l'importance",
    ["enrichi", "simple"],
    format_func=FS_LABELS.get,
    horizontal=True,
)

tab_table, tab_roc, tab_cal, tab_imp, tab_err = st.tabs(
    ["Tableau comparatif", "Courbes ROC", "Calibration", "Importance des variables", "Analyse d'erreurs"]
)

with tab_table:
    t = metrics.copy()
    t["Jeu"] = t["jeu"].map({"simple": "Simple", "enrichi": "Enrichi", "reference": "Référence"})

    def ci(v, lo, hi, d):
        return f"{v:.{d}f} [{lo:.{d}f} ; {hi:.{d}f}]"

    t["ROC-AUC [IC 95 %]"] = [ci(r.roc_auc, r.roc_auc_ic_bas, r.roc_auc_ic_haut, 3) for r in t.itertuples()]
    t["Brier [IC 95 %]"] = [ci(r.brier, r.brier_ic_bas, r.brier_ic_haut, 4) for r in t.itertuples()]
    t["Log-loss [IC 95 %]"] = [
        ci(r.log_loss, r.log_loss_ic_bas, r.log_loss_ic_haut, 4) for r in t.itertuples()
    ]
    t["Gain Brier vs baseline"] = t["brier_skill_vs_baseline"].map(lambda v: f"{v:.1%}")
    t = t.drop_duplicates(subset=["modele", "Jeu"])
    t = t[~((t["modele"] == "baseline") & (t["jeu"] == "enrichi"))]
    st.dataframe(
        t[
            [
                "Jeu",
                "libelle",
                "ROC-AUC [IC 95 %]",
                "Brier [IC 95 %]",
                "Log-loss [IC 95 %]",
                "Gain Brier vs baseline",
            ]
        ].rename(columns={"libelle": "Modèle"}),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "ROC-AUC : capacité à classer les buts au-dessus des non-buts (plus haut = mieux). "
        "Brier et log-loss : erreur des probabilités (plus bas = mieux). La baseline prédit "
        "toujours le taux de but moyen. Le xG StatsBomb sert uniquement de référence externe."
    )
    plot_df = metrics[metrics["modele"] != "baseline"]
    fig = go.Figure()
    for r in plot_df.itertuples():
        name = f"{MODEL_LABELS[r.modele]}" + ("" if r.jeu == "reference" else f" - {r.jeu}")
        fig.add_trace(
            go.Scatter(
                x=[r.roc_auc],
                y=[name],
                mode="markers",
                showlegend=False,
                marker=dict(
                    size=11,
                    color=MODEL_COLORS[r.modele],
                    symbol="circle"
                    if r.jeu == "enrichi"
                    else "circle-open"
                    if r.jeu == "simple"
                    else "diamond",
                    line=dict(width=2, color=MODEL_COLORS[r.modele]),
                ),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=[r.roc_auc_ic_haut - r.roc_auc],
                    arrayminus=[r.roc_auc - r.roc_auc_ic_bas],
                    color=MODEL_COLORS[r.modele],
                ),
                hovertemplate=f"{name}<br>AUC = {r.roc_auc:.3f} [{r.roc_auc_ic_bas:.3f} ; "
                f"{r.roc_auc_ic_haut:.3f}]<extra></extra>",
            )
        )
    layout(fig, "ROC-AUC sur le test (IC 95 %)", "", height=420)
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with tab_roc:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color="#c3c2b7", width=1),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    for name, c in mjson["courbes_roc"][fs].items():
        auc = metrics[(metrics["modele"] == name) & (metrics["jeu"].isin([fs, "reference"]))]["roc_auc"]
        fig.add_trace(
            go.Scatter(
                x=c["fpr"],
                y=c["tpr"],
                mode="lines",
                name=f"{MODEL_LABELS[name]} ({auc.iloc[0]:.3f})" if len(auc) else MODEL_LABELS[name],
                line=dict(color=MODEL_COLORS[name], dash=DASH[name], width=2),
                hovertemplate="FPR %{x:.2f} - TPR %{y:.2f}<extra></extra>",
            )
        )
    st.plotly_chart(
        layout(fig, "Taux de faux positifs", "Taux de vrais positifs"),
        use_container_width=True,
        config={"displayModeBar": False},
    )

with tab_cal:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 0.6],
            y=[0, 0.6],
            mode="lines",
            name="Calibration parfaite",
            line=dict(color="#c3c2b7", width=1),
            hoverinfo="skip",
        )
    )
    for name, c in mjson["courbes_calibration"][fs].items():
        fig.add_trace(
            go.Scatter(
                x=c["pred_moyenne"],
                y=c["frequence_observee"],
                mode="lines+markers",
                name=MODEL_LABELS[name],
                line=dict(color=MODEL_COLORS[name], dash=DASH[name], width=2),
                marker=dict(size=7),
                hovertemplate="prédit %{x:.3f} - observé %{y:.3f}<extra></extra>",
            )
        )
    layout(fig, "Probabilité prédite (moyenne par classe)", "Fréquence observée de but")
    fig.update_xaxes(range=[0, 0.6])
    fig.update_yaxes(range=[0, 0.6])
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    calib = {k: v.get("calibration") for k, v in mjson["modeles"].items() if v.get("calibration")}
    st.caption(
        "Recalibrage : pour chaque modèle, la méthode (aucune, Platt, isotonique) est choisie par "
        "validation croisée groupée sur l'ensemble de calibration, distinct du test. Méthodes "
        "retenues : " + ", ".join(f"{k.replace('__', ' / ')} = {v}" for k, v in calib.items()) + "."
    )

with tab_imp:
    imp = importance[importance["jeu"] == fs].sort_values("importance_moyenne")
    model_name = MODEL_LABELS[imp["modele"].iloc[0]]
    fig = go.Figure(
        go.Bar(
            x=imp["importance_moyenne"],
            y=[FEATURE_LABELS.get(v, v) for v in imp["variable"]],
            orientation="h",
            marker_color=MODEL_COLORS["logreg"],
            error_x=dict(type="data", array=imp["ecart_type"], color="#52514e"),
            hovertemplate="%{y} : %{x:.4f}<extra></extra>",
        )
    )
    layout(fig, "Hausse de la log-loss quand la variable est permutée", "", height=40 + 28 * len(imp))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption(f"Importance par permutation sur le test ({model_name}, 10 répétitions).")

with tab_err:
    st.markdown(
        f"Modèle analysé : **{mjson['analyse_erreurs']['modele'].replace('__', ' / ')}** "
        "(test). Écart = buts réels - xG prédits : positif = le modèle sous-estime."
    )
    dims = errors["dimension"].unique().tolist()
    dim = st.selectbox("Dimension", dims)
    e = errors[errors["dimension"] == dim].copy()
    e["Taux observé"] = e["buts"] / e["n_tirs"]
    e["xG moyen modèle"] = e["xg_modele"] / e["n_tirs"]
    st.dataframe(
        e[
            [
                "modalite",
                "n_tirs",
                "buts",
                "xg_modele",
                "xg_statsbomb",
                "ecart_buts_moins_xg",
                "Taux observé",
                "xG moyen modèle",
            ]
        ].rename(
            columns={
                "modalite": "Modalité",
                "n_tirs": "Tirs",
                "buts": "Buts",
                "xg_modele": "xG modèle",
                "xg_statsbomb": "xG StatsBomb",
                "ecart_buts_moins_xg": "Écart buts - xG",
            }
        ),
        hide_index=True,
        use_container_width=True,
        column_config={
            c: st.column_config.NumberColumn(format="%.1f")
            for c in ["xG modèle", "xG StatsBomb", "Écart buts - xG"]
        }
        | {c: st.column_config.NumberColumn(format="%.3f") for c in ["Taux observé", "xG moyen modèle"]},
    )
    corr = mjson["analyse_erreurs"]["correlation_xg_modele_statsbomb"]
    st.caption(f"Corrélation entre le xG du modèle et le xG StatsBomb sur le test : {corr:.3f}.")
    try:
        ex = load_csv("error_examples.csv")
        st.markdown("**Exemples : buts les plus improbables et grosses occasions manquées**")
        st.dataframe(ex, hide_index=True, use_container_width=True)
    except MissingFileError:
        pass
