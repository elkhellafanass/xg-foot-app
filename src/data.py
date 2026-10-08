"""Telechargement (avec cache) des donnees StatsBomb Open Data et construction de la table des tirs.

Les fichiers bruts sont mis en cache dans ``data/raw/`` sous forme compressee (``.json.gz``).
L'ecriture est atomique (fichier temporaire puis renommage) : un telechargement interrompu
ne laisse jamais de fichier corrompu, et une relance reprend la ou elle s'etait arretee.
"""

from __future__ import annotations

import gzip
import json
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

from src.config import RAW_BASE_URL, RAW_DIR
from src.features import freeze_frame_features

ProgressCallback = Callable[[int, int], None]

_SESSION: requests.Session | None = None


def _session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()
        _SESSION.headers.update({"Accept-Encoding": "gzip", "User-Agent": "xg-foot-app"})
    return _SESSION


def _cache_path(relative: str, raw_dir: Path) -> Path:
    return raw_dir / (relative + ".gz")


def fetch_json(relative: str, raw_dir: Path = RAW_DIR, retries: int = 4) -> object:
    """Retourne le JSON ``data/<relative>`` du depot StatsBomb, depuis le cache si possible."""
    path = _cache_path(relative, raw_dir)
    if path.exists():
        try:
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, EOFError, json.JSONDecodeError):
            path.unlink(missing_ok=True)  # cache corrompu : on retelecharge

    url = f"{RAW_BASE_URL}/{relative}"
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            resp = _session().get(url, timeout=60)
            resp.raise_for_status()
            content = resp.content
            data = json.loads(content.decode("utf-8"))
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            with gzip.open(tmp, "wb", compresslevel=6) as fh:
                fh.write(content)
            tmp.replace(path)
            return data
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Echec du telechargement de {url} : {last_error}")


def load_competitions(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    return pd.DataFrame(fetch_json("competitions.json", raw_dir))


def load_matches(competition_id: int, season_id: int, raw_dir: Path = RAW_DIR) -> list[dict]:
    return fetch_json(f"matches/{competition_id}/{season_id}.json", raw_dir)  # type: ignore[return-value]


def download_events(
    match_ids: Iterable[int],
    raw_dir: Path = RAW_DIR,
    workers: int = 8,
    progress: ProgressCallback | None = None,
) -> list[int]:
    """Telecharge les evenements des matchs (en parallele). Retourne les identifiants en echec."""
    match_ids = list(match_ids)
    todo = [m for m in match_ids if not _cache_path(f"events/{m}.json", raw_dir).exists()]
    done = len(match_ids) - len(todo)
    if progress:
        progress(done, len(match_ids))
    failed: list[int] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_json, f"events/{m}.json", raw_dir): m for m in todo}
        for fut in as_completed(futures):
            try:
                fut.result()
            except RuntimeError:
                failed.append(futures[fut])
            done += 1
            if progress:
                progress(done, len(match_ids))
    return failed


def _name(obj: dict | None) -> str | None:
    return obj.get("name") if isinstance(obj, dict) else None


def extract_shots(events: list[dict], match: dict, keep_excluded: bool = False) -> list[dict]:
    """Extrait les tirs d'un match (hors penalties et seance de tirs au but).

    Avec ``keep_excluded=True``, les tirs exclus sont conserves et marques dans la colonne
    ``exclusion`` ("penalty" ou "tir_au_but"), pour documenter le jeu de donnees.
    Le score avant chaque tir est reconstitue en parcourant les evenements dans l'ordre.
    """
    home = match["home_team"]["home_team_name"]
    away = match["away_team"]["away_team_name"]
    goals = {home: 0, away: 0}
    rows: list[dict] = []
    for ev in sorted(events, key=lambda e: e.get("index", 0)):
        etype = _name(ev.get("type"))
        team = _name(ev.get("team"))
        period = ev.get("period")
        if etype == "Own Goal For" and period != 5 and team in goals:
            goals[team] += 1
            continue
        if etype != "Shot":
            continue
        shot = ev.get("shot", {})
        is_goal = _name(shot.get("outcome")) == "Goal"
        shot_type = _name(shot.get("type"))
        opponent = away if team == home else home
        exclusion = "tir_au_but" if period == 5 else ("penalty" if shot_type == "Penalty" else None)
        if exclusion is None or keep_excluded:
            loc = ev.get("location") or [None, None]
            row = {
                "event_id": ev.get("id"),
                "match_id": match["match_id"],
                "competition": match["competition"]["competition_name"],
                "season": match["season"]["season_name"],
                "match_date": match.get("match_date"),
                "home_team": home,
                "away_team": away,
                "team": team,
                "opponent": opponent,
                "player": _name(ev.get("player")),
                "period": period,
                "minute": ev.get("minute"),
                "second": ev.get("second"),
                "x": loc[0],
                "y": loc[1],
                "body_part": _name(shot.get("body_part")),
                "shot_type": shot_type,
                "technique": _name(shot.get("technique")),
                "play_pattern": _name(ev.get("play_pattern")),
                "under_pressure": bool(ev.get("under_pressure", False)),
                "first_time": bool(shot.get("first_time", False)),
                "goals_for_before": goals.get(team, 0),
                "goals_against_before": goals.get(opponent, 0),
                "outcome": _name(shot.get("outcome")),
                "is_goal": int(is_goal),
                "statsbomb_xg": shot.get("statsbomb_xg"),
                "exclusion": exclusion,
            }
            row.update(freeze_frame_features(loc[0], loc[1], shot.get("freeze_frame")))
            rows.append(row)
        if is_goal and period != 5 and team in goals:
            goals[team] += 1
    return rows


def build_shots_table(
    competitions: list[tuple[int, int]],
    raw_dir: Path = RAW_DIR,
    progress: ProgressCallback | None = None,
    workers: int = 8,
    keep_excluded: bool = False,
) -> pd.DataFrame:
    """Telecharge (si besoin) puis construit la table des tirs pour les competitions donnees."""
    matches: list[dict] = []
    for comp_id, season_id in competitions:
        matches.extend(load_matches(comp_id, season_id, raw_dir))
    failed = download_events([m["match_id"] for m in matches], raw_dir, workers, progress)
    if failed:
        raise RuntimeError(f"{len(failed)} matchs n'ont pas pu etre telecharges : {failed[:10]}")
    rows: list[dict] = []
    for match in matches:
        events = fetch_json(f"events/{match['match_id']}.json", raw_dir)
        rows.extend(extract_shots(events, match, keep_excluded))  # type: ignore[arg-type]
    df = pd.DataFrame(rows)
    df["score_diff_before"] = df["goals_for_before"] - df["goals_against_before"]
    return df
