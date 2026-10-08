"""Page d'accueil : contexte, question de recherche, chiffres cles."""

import streamlit as st
from common import MissingFileError, load_csv, load_metrics_json, show_missing

st.title("Estimer les buts attendus (xG) par apprentissage automatique")
st.markdown(
    "Mémoire de Master 2 (IPSSI) - modélisation de la probabilité qu'un tir devienne un but, "
    "à partir des données événementielles **StatsBomb Open Data**."
)

try:
    summary = load_csv("dataset_summary.csv")
    metrics = load_csv("metrics.csv")
    mjson = load_metrics_json()
except MissingFileError as exc:
    show_missing(exc)

total = summary[summary["competition"] == "TOTAL"].iloc[0]
comps = summary[summary["competition"] != "TOTAL"]


def fmt_int(v) -> str:
    return f"{int(v):,}".replace(",", " ")


def dec(v: float, d: int) -> str:
    return f"{v:.{d}f}".replace(".", ",")


c1, c2, c3, c4 = st.columns(4)
c1.metric("Matchs", fmt_int(total["n_matchs"]))
c2.metric("Tirs (hors penalties)", fmt_int(total["n_tirs"]))
c3.metric("Buts", fmt_int(total["n_buts"]))
c4.metric("Taux de conversion", f"{total['taux_conversion']:.1%}".replace(".", ","))

main = mjson["modele_principal"]["enrichi"]
row = metrics[(metrics["jeu"] == "enrichi") & (metrics["modele"] == main)].iloc[0]
ref = metrics[metrics["modele"] == "statsbomb"].iloc[0]
c1, c2, c3, c4 = st.columns(4)
c1.metric("ROC-AUC - modèle principal", dec(row["roc_auc"], 3), help=f"{row['libelle']}, jeu enrichi")
c2.metric("ROC-AUC - xG StatsBomb", dec(ref["roc_auc"], 3))
c3.metric("Score de Brier - modèle principal", dec(row["brier"], 4))
c4.metric("Score de Brier - xG StatsBomb", dec(ref["brier"], 4))
st.caption(
    f"Évaluation sur {fmt_int(mjson['split']['test']['n_tirs'])} tirs de "
    f"{mjson['split']['test']['n_matchs']} matchs jamais vus à l'entraînement."
)

left, right = st.columns([3, 2], gap="large")
with left:
    st.subheader("Contexte")
    st.markdown(
        "Le nombre de buts est une statistique rare et bruitée : une équipe peut dominer un "
        "match sans marquer. Les **buts attendus (expected goals, xG)** attribuent à chaque tir "
        "la probabilité qu'il soit converti, compte tenu de sa position, de la partie du corps, "
        "du contexte de jeu et, quand c'est disponible, de la position des défenseurs et du "
        "gardien. Additionnés, ces xG mesurent la qualité des occasions créées ou concédées."
    )
    st.subheader("Question de recherche")
    st.markdown(
        "> Dans quelle mesure des modèles d'apprentissage automatique entraînés sur des données "
        "ouvertes permettent-ils d'estimer de façon **discriminante et bien calibrée** la "
        "probabilité de but d'un tir, et quel est l'apport des informations de position des "
        "joueurs (*freeze frame*) par rapport aux seules caractéristiques du tir ?"
    )
    st.subheader("Contenu de l'application")
    st.markdown(
        "- **Explorateur de tirs** : carte interactive filtrable.\n"
        "- **Simulateur xG** : placez un tir et comparez les probabilités des modèles.\n"
        "- **Modèles** : métriques, courbes ROC, calibration, importance des variables.\n"
        "- **Équipes et joueurs** : buts réels contre xG cumulés.\n"
        "- **Méthodologie et sources** : pipeline, limites, licence."
    )
with right:
    st.subheader("Jeu de données")
    table = comps[["competition", "season", "n_matchs", "n_tirs", "n_buts"]].rename(
        columns={
            "competition": "Compétition",
            "season": "Saison",
            "n_matchs": "Matchs",
            "n_tirs": "Tirs",
            "n_buts": "Buts",
        }
    )
    st.dataframe(
        table,
        hide_index=True,
        use_container_width=True,
        column_config={c: st.column_config.NumberColumn(format="%d") for c in ["Matchs", "Tirs", "Buts"]},
    )
    st.caption("Penalties et tirs au but exclus. Bundesliga 2015/2016 : seuls 34 matchs sont publiés.")
