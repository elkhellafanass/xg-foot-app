"""Palette et style graphique communs (figures du memoire et application)."""

from __future__ import annotations

# Ordre categoriel fixe (palette validee daltonisme), une couleur par modele.
MODEL_COLORS = {
    "baseline": "#8a8984",
    "logreg": "#2a78d6",
    "rf": "#eb6834",
    "hgb": "#1baf7a",
    "xgb": "#c98500",
    "statsbomb": "#0b0b0b",
}
MODEL_LINESTYLES = {
    "baseline": ":",
    "logreg": "-",
    "rf": "--",
    "hgb": "-.",
    "xgb": (0, (5, 1, 1, 1, 1, 1)),
    "statsbomb": (0, (2, 1)),
}

GOAL_COLOR = "#eb6834"
MISS_COLOR = "#2a78d6"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e4e3df"
SEQUENTIAL_BLUES = ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]

MPL_STYLE = {
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.edgecolor": "#b5b4ae",
    "axes.labelcolor": TEXT_PRIMARY,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID_COLOR,
    "grid.linewidth": 0.6,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "lines.linewidth": 1.8,
}
