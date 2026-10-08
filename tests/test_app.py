"""Tests de fumee de l'application : chaque page s'execute sans exception."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.config import MODELS_DIR, SHOTS_PATH

APP = Path(__file__).resolve().parents[1] / "app"
PAGES = ["accueil", "explorateur", "simulateur", "modeles", "equipes_joueurs", "methodologie"]

needs_artifacts = pytest.mark.skipif(
    not (MODELS_DIR / "manifest.json").exists() or not SHOTS_PATH.exists(),
    reason="modeles ou table des tirs absents",
)


@needs_artifacts
def test_application_principale():
    # AppTest (Streamlit 1.40) n'execute pas les pages de st.navigation : on verifie ici le
    # script principal (navigation + barre laterale) ; chaque page est testee ci-dessous.
    at = AppTest.from_file(str(APP / "streamlit_app.py"), default_timeout=120).run()
    assert not at.exception
    assert any("StatsBomb Open Data" in c.value for c in at.sidebar.caption)


@needs_artifacts
@pytest.mark.parametrize("page", PAGES)
def test_page_sans_exception(page):
    at = AppTest.from_file(str(APP / "views" / f"{page}.py"), default_timeout=120).run()
    assert not at.exception, at.exception
    assert len(at.title) == 1


@needs_artifacts
def test_simulateur_reagit_aux_curseurs():
    at = AppTest.from_file(str(APP / "views" / "simulateur.py"), default_timeout=120).run()
    at.slider(key="sim_x").set_value(119.0).run()
    assert not at.exception
    main = [m for m in at.metric if "principal" in m.label]
    assert main and main[0].value.endswith("%")


@needs_artifacts
def test_equipes_test_uniquement():
    at = AppTest.from_file(str(APP / "views" / "equipes_joueurs.py"), default_timeout=120).run()
    at.toggle[0].set_value(True).run()
    assert not at.exception
