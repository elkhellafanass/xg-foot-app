"""Point d'entree de l'application Streamlit : streamlit run app/streamlit_app.py"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(APP_DIR.parent))

from common import sidebar  # noqa: E402

st.set_page_config(
    page_title="xG Foot - Buts attendus",
    page_icon=":material/sports_soccer:",
    layout="wide",
)

PAGES = [
    st.Page("views/accueil.py", title="Accueil", icon=":material/home:", default=True),
    st.Page("views/explorateur.py", title="Explorateur de tirs", icon=":material/scatter_plot:"),
    st.Page("views/simulateur.py", title="Simulateur xG", icon=":material/target:"),
    st.Page("views/modeles.py", title="Modèles", icon=":material/monitoring:"),
    st.Page("views/equipes_joueurs.py", title="Équipes et joueurs", icon=":material/groups:"),
    st.Page("views/methodologie.py", title="Méthodologie et sources", icon=":material/menu_book:"),
]

nav = st.navigation(PAGES)
sidebar()
nav.run()
