"""Methodologie, limites et sources (attribution StatsBomb obligatoire)."""

import streamlit as st
from common import LOGO_PATH, MissingFileError, load_metrics_json

from src.config import ATTRIBUTION

st.title("Méthodologie et sources")

with st.container(border=True):
    lc, rc = st.columns([1, 3], vertical_alignment="center")
    with lc:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), use_container_width=True)
        else:
            st.caption("[Emplacement du logo StatsBomb : app/assets/statsbomb_logo.png]")
    with rc:
        st.markdown(f"**{ATTRIBUTION}**")
        st.caption("Les analyses présentées ici n'engagent pas StatsBomb.")

st.header("Pipeline")
st.markdown(
    """
1. **Téléchargement** (`scripts/download_data.py`) : `competitions.json`, matchs et événements
   de 11 compétitions masculines, depuis le dépôt GitHub StatsBomb, avec cache local compressé
   et reprise après interruption.
2. **Table des tirs** (`scripts/build_dataset.py`) : un tir = une ligne. Penalties et séances de
   tirs au but exclus ; cible `but = 1` si l'issue est *Goal*. Le score avant chaque tir est
   reconstitué en parcourant les événements (buts et contre-son-camp).
3. **Variables** (`src/features.py`) :
   - *jeu simple* : distance au centre du but `√((120 − x)² + (40 − y)²)`, angle d'ouverture
     entre les vecteurs tireur → poteaux (y = 36 et y = 44), partie du corps, type de tir,
     technique, phase de jeu, sous pression, premier contact, volée, minute, écart au score ;
   - *jeu enrichi* : + variables du *freeze frame* (distance tireur-gardien, distance gardien-but,
     gardien dans le triangle, défenseurs dans le triangle tireur-poteaux, défenseur le plus proche).
   - Le xG StatsBomb n'est **jamais** une variable explicative : c'est une référence externe.
4. **Modèles** (`src/models.py`, `scripts/train.py`) : baseline constante, régression logistique
   normalisée, forêt aléatoire, gradient boosting (HistGradientBoosting) et XGBoost (comparaison).
5. **Protocole** : découpage par match (test 20 % des matchs, puis calibration 20 % du reste),
   réglage des hyperparamètres par validation croisée groupée (GroupKFold à 5 plis, log-loss),
   choix du recalibrage (aucun, Platt ou isotonique) sur l'ensemble de calibration,
   évaluation unique sur le test avec intervalles de confiance par bootstrap sur les matchs.
6. **Application** : charge les modèles déjà entraînés (`models/*.joblib`) et les rapports
   (`reports/`) ; elle ne ré-entraîne jamais rien.
"""
)

try:
    m = load_metrics_json()
    st.caption(
        f"Graine aléatoire : {m['protocole']['graine']} - scikit-learn {m['protocole']['scikit_learn']}."
    )
except MissingFileError:
    pass

st.header("Limites")
st.markdown(
    """
- **Données** : compétitions publiées librement par StatsBomb, donc non représentatives de tout
  le football (forte part de la saison 2015/2016, Bundesliga partielle, uniquement masculin).
- **Variables** : pas de suivi continu des joueurs (seulement une photo au moment du tir),
  pas de vitesse du ballon ni de qualité du tireur ; le contexte « sous pression » est annoté
  manuellement.
- **Événements rares** : environ 9 % de buts ; les tirs très dangereux sont peu nombreux,
  ce qui rend la calibration des fortes probabilités moins précise.
- **Interprétation** : un xG est une probabilité moyenne pour un tir « typique » de ce
  type, pas une note de la finition d'un joueur ; les écarts buts - xG sur de petits
  échantillons sont dominés par le hasard.
- **Version en ligne** : la licence interdisant la redistribution des données, l'application
  déployée télécharge seulement la Coupe du monde 2022 et l'Euro 2024 pour les cartes.
"""
)

st.header("Licence et conditions d'utilisation")
st.markdown(
    """
Les données proviennent de [StatsBomb Open Data](https://github.com/statsbomb/open-data),
mises à disposition pour la recherche selon le *StatsBomb Public Data User Agreement* :

- toute publication d'analyses doit citer **StatsBomb** comme source et afficher son **logo** ;
- les données elles-mêmes ne doivent pas être redistribuées ni exploitées commercialement :
  ce dépôt ne contient donc **aucune donnée brute ni table de tirs**, seulement le code, les
  modèles entraînés et des résultats agrégés.
"""
)
