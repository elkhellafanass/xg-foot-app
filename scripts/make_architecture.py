"""Genere le schema d'architecture du projet : docs/architecture.png

Usage : python scripts/make_architecture.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from src.viz import TEXT_PRIMARY, TEXT_SECONDARY  # noqa: E402

BLUE_BG, BLUE = "#e6f0fc", "#2a78d6"
ORANGE_BG, ORANGE = "#fdeee7", "#eb6834"
GREEN_BG, GREEN = "#e3f6ef", "#1baf7a"
GREY_BG, GREY = "#f0efec", "#8a8984"

BOXES = {
    # cle : (x, y, largeur, hauteur, titre, detail, fond, bord)
    "sb": (0.2, 4.6, 2.6, 1.1, "StatsBomb Open Data", "GitHub (JSON brut)", GREY_BG, GREY),
    "dl": (3.4, 4.6, 2.9, 1.1, "download_data.py", "cache data/raw/ (.json.gz)", BLUE_BG, BLUE),
    "build": (6.9, 4.6, 2.9, 1.1, "build_dataset.py", "src/data.py + src/features.py", BLUE_BG, BLUE),
    "shots": (10.4, 4.6, 2.6, 1.1, "shots.parquet", "45 827 tirs (local, hors Git)", GREY_BG, GREY),
    "train": (10.4, 2.6, 2.6, 1.1, "train.py", "src/models.py (CV par match)", BLUE_BG, BLUE),
    "models": (6.9, 2.6, 2.9, 1.1, "models/*.joblib", "modèles + manifeste + split", GREEN_BG, GREEN),
    "reports": (3.4, 2.6, 2.9, 1.1, "reports/", "métriques, figures PNG, CSV", GREEN_BG, GREEN),
    "app": (5.15, 0.4, 3.0, 1.2, "Application Streamlit", "app/ : 6 pages, Plotly", ORANGE_BG, ORANGE),
    "cloud": (10.2, 0.45, 2.95, 1.1, "Streamlit Cloud", "CdM 2022 + Euro 2024 au démarrage", GREY_BG, GREY),
}
ARROWS = [
    ("sb", "dl", "r", "l"),
    ("dl", "build", "r", "l"),
    ("build", "shots", "r", "l"),
    ("shots", "train", "b", "t"),
    ("train", "models", "l", "r"),
    ("models", "reports", "l", "r"),
    ("models", "app", "b", "t"),
    ("reports", "app", "b", "t"),
    ("cloud", "app", "l", "r"),
]


def anchor(key: str, side: str) -> tuple[float, float]:
    x, y, w, h = BOXES[key][:4]
    return {"r": (x + w, y + h / 2), "l": (x, y + h / 2), "t": (x + w / 2, y + h), "b": (x + w / 2, y)}[side]


def main() -> None:
    fig, ax = plt.subplots(figsize=(13, 6.6))
    for x, y, w, h, title, detail, bg, edge in BOXES.values():
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=bg, ec=edge, lw=1.4))
        ax.text(x + w / 2, y + h * 0.63, title, ha="center", va="center", fontsize=11,
                fontweight="bold", color=TEXT_PRIMARY)
        ax.text(x + w / 2, y + h * 0.28, detail, ha="center", va="center", fontsize=8.8,
                color=TEXT_SECONDARY)
    for a, b, sa, sb in ARROWS:
        ax.add_patch(FancyArrowPatch(anchor(a, sa), anchor(b, sb), arrowstyle="-|>", mutation_scale=14,
                                     color="#52514e", lw=1.2, connectionstyle="arc3,rad=0.0"))
    ax.text(0.2, 6.25, "Architecture du projet xG", fontsize=14, fontweight="bold", color=TEXT_PRIMARY)
    ax.text(0.2, 5.95, "Bleu : scripts et modules Python - Vert : artefacts versionnés dans Git - "
            "Gris : données (non versionnées, licence StatsBomb) - Orange : application",
            fontsize=9, color=TEXT_SECONDARY)
    ax.text(0.2, 1.0, "Tests : pytest (variables,\nfuite train/test, modèles)\n+ AppTest par page\n"
            "+ captures Playwright", fontsize=9, color=TEXT_SECONDARY, va="center")
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 6.5)
    ax.axis("off")
    out = ROOT / "docs" / "architecture.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Schema enregistre : {out}")


if __name__ == "__main__":
    main()
