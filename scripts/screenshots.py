"""Captures d'ecran de chaque page de l'application (Playwright + Chromium).

Usage :
    streamlit run app/streamlit_app.py --server.port 8599   (dans un autre terminal)
    python scripts/screenshots.py [--url http://localhost:8599]

Sorties : docs/screenshots/*.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "screenshots"

PAGES = [
    ("", "01_accueil"),
    ("explorateur", "02_explorateur_tirs"),
    ("simulateur", "03_simulateur_xg"),
    ("modeles", "04_modeles"),
    ("equipes_joueurs", "05_equipes_joueurs"),
    ("methodologie", "06_methodologie_sources"),
]


def wait_ready(page: Page) -> None:
    page.wait_for_selector('[data-testid="stMainBlockContainer"] h1', timeout=120_000)
    page.wait_for_function(
        "() => !document.querySelector('[data-testid=\"stStatusWidget\"]')", timeout=120_000
    )
    page.wait_for_timeout(2500)  # rendu Plotly


def full_screenshot(page: Page, path: Path) -> None:
    """Streamlit fait defiler un conteneur interne : on agrandit la fenetre a la hauteur du contenu."""
    height = page.evaluate("() => document.querySelector('[data-testid=\"stMain\"]').scrollHeight")
    page.set_viewport_size({"width": 1440, "height": max(900, int(height) + 40)})
    page.wait_for_timeout(1500)
    page.screenshot(path=str(path), full_page=True)
    page.set_viewport_size({"width": 1440, "height": 900})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8599")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1.5)
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        for path, name in PAGES:
            page.goto(f"{args.url}/{path}")
            wait_ready(page)
            if page.locator('[data-testid="stException"]').count():
                errors.append(f"Exception Streamlit sur la page {name}")
            full_screenshot(page, OUT / f"{name}.png")
            print(f"OK {name}")

        # Simulateur : clic sur le terrain (doit deplacer le tir) puis capture.
        page.goto(f"{args.url}/simulateur")
        wait_ready(page)
        before = page.locator('[data-testid="stMetricValue"]').first.inner_text()
        # Zone de trace du terrain : axe horizontal y de -1 a 81, axe vertical x de 59 a 123.
        area = page.locator('[data-testid="stPlotlyChart"]').first.locator(".nsewdrag").first
        box = area.bounding_box()
        if box:  # clic en (x = 110, y = 30) dans le repere StatsBomb
            px = box["x"] + box["width"] * (30 + 1) / 82
            py = box["y"] + box["height"] * (123 - 110) / 64
            page.mouse.click(px, py)
            page.wait_for_timeout(3000)
            wait_ready(page)
        after = page.locator('[data-testid="stMetricValue"]').first.inner_text()
        print(f"Clic sur le terrain : distance {before} -> {after}")
        if before == after:
            errors.append("Le clic sur le terrain n'a pas modifie la position du tir")
        full_screenshot(page, OUT / "03b_simulateur_apres_clic.png")
        browser.close()
    for e in errors:
        print(f"ERREUR : {e}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
