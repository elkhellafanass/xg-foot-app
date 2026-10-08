"""Telecharge les donnees StatsBomb Open Data (competitions, matchs, evenements) dans data/raw/.

Usage : python scripts/download_data.py [--only-wc2022]

Cache local compresse, reprise automatique apres interruption (les fichiers deja
presents ne sont pas retelecharges), barre de progression.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm  # noqa: E402

from src.config import COMPETITIONS, RAW_DIR  # noqa: E402
from src.data import download_events, load_competitions, load_matches  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only-wc2022", action="store_true", help="Coupe du monde 2022 uniquement")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    competitions = [(43, 106)] if args.only_wc2022 else COMPETITIONS
    comps = load_competitions()
    print(f"competitions.json : {len(comps)} couples competition/saison disponibles")

    for comp_id, season_id in competitions:
        meta = comps[(comps.competition_id == comp_id) & (comps.season_id == season_id)]
        label = (
            f"{meta.competition_name.iloc[0]} {meta.season_name.iloc[0]}"
            if len(meta)
            else f"{comp_id}/{season_id}"
        )
        matches = load_matches(comp_id, season_id)
        bar = tqdm(total=len(matches), desc=label[:32].ljust(32), unit="match", ascii=True)

        def progress(done: int, total: int, bar=bar) -> None:
            bar.n = done
            bar.refresh()

        failed = download_events([m["match_id"] for m in matches], RAW_DIR, args.workers, progress)
        bar.close()
        if failed:
            print(f"ERREUR : {len(failed)} matchs en echec ({failed[:5]}...). Relancez le script.")
            return 1
    size = sum(p.stat().st_size for p in RAW_DIR.rglob("*.gz")) / 1e6
    print(f"Termine. Cache : {RAW_DIR} ({size:.0f} Mo compresses)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
