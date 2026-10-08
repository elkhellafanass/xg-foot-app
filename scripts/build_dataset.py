"""Construit la table des tirs (data/processed/shots.parquet) et le resume du jeu de donnees.

Usage : python scripts/build_dataset.py
Prerequis : python scripts/download_data.py (sinon les fichiers manquants sont telecharges).

Sorties :
- data/processed/shots.parquet   (non versionne : licence StatsBomb)
- reports/dataset_summary.csv    (comptages agreges par competition/saison)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.config import COMPETITIONS, PROCESSED_DIR, REPORTS_DIR, SHOTS_PATH  # noqa: E402
from src.data import build_shots_table  # noqa: E402
from src.features import add_features  # noqa: E402


def summarize(all_shots: pd.DataFrame) -> pd.DataFrame:
    kept = all_shots[all_shots["exclusion"].isna()]
    keys = ["competition", "season"]
    summary = (
        all_shots.groupby(keys, sort=False)
        .agg(
            n_matchs=("match_id", "nunique"),
            penalties_exclus=("exclusion", lambda s: int((s == "penalty").sum())),
            tirs_au_but_exclus=("exclusion", lambda s: int((s == "tir_au_but").sum())),
        )
        .join(
            kept.groupby(keys, sort=False).agg(
                n_tirs=("event_id", "size"),
                n_buts=("is_goal", "sum"),
                taux_conversion=("is_goal", "mean"),
                part_freeze_frame=("has_freeze_frame", "mean"),
                xg_statsbomb_total=("statsbomb_xg", "sum"),
            )
        )
        .reset_index()
    )
    total = {
        "competition": "TOTAL",
        "season": "",
        "n_matchs": int(all_shots["match_id"].nunique()),
        "penalties_exclus": int((all_shots["exclusion"] == "penalty").sum()),
        "tirs_au_but_exclus": int((all_shots["exclusion"] == "tir_au_but").sum()),
        "n_tirs": len(kept),
        "n_buts": int(kept["is_goal"].sum()),
        "taux_conversion": float(kept["is_goal"].mean()),
        "part_freeze_frame": float(kept["has_freeze_frame"].mean()),
        "xg_statsbomb_total": float(kept["statsbomb_xg"].sum()),
    }
    summary = pd.concat([summary, pd.DataFrame([total])], ignore_index=True)
    cols = ["n_tirs", "n_buts", "penalties_exclus", "tirs_au_but_exclus"]
    summary[cols] = summary[cols].fillna(0).astype(int)
    return summary.round({"taux_conversion": 4, "part_freeze_frame": 4, "xg_statsbomb_total": 1})


def main() -> int:
    bar = tqdm(total=0, desc="Evenements", unit="match", ascii=True)

    def progress(done: int, total: int) -> None:
        bar.total = total
        bar.n = done
        bar.refresh()

    all_shots = build_shots_table(COMPETITIONS, progress=progress, keep_excluded=True)
    bar.close()

    shots = add_features(all_shots[all_shots["exclusion"].isna()].drop(columns="exclusion"))
    shots = shots.reset_index(drop=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    shots.to_parquet(SHOTS_PATH, index=False)

    summary = summarize(all_shots)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(REPORTS_DIR / "dataset_summary.csv", index=False, encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"\nTable des tirs : {SHOTS_PATH} ({len(shots)} tirs, {shots['match_id'].nunique()} matchs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
