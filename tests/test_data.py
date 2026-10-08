import pandas as pd
import pytest

from src.config import SHOTS_PATH
from src.data import extract_shots
from src.features import FEATURE_SETS

MATCH = {
    "match_id": 1,
    "competition": {"competition_name": "Test Cup"},
    "season": {"season_name": "2022"},
    "match_date": "2022-12-18",
    "home_team": {"home_team_name": "A"},
    "away_team": {"away_team_name": "B"},
}


def _shot(index, team, outcome="Saved", shot_type="Open Play", period=1, x=108.0, y=40.0):
    return {
        "id": f"e{index}",
        "index": index,
        "period": period,
        "minute": index,
        "second": 0,
        "type": {"name": "Shot"},
        "team": {"name": team},
        "player": {"name": "Joueur"},
        "location": [x, y],
        "play_pattern": {"name": "Regular Play"},
        "shot": {
            "outcome": {"name": outcome},
            "type": {"name": shot_type},
            "body_part": {"name": "Right Foot"},
            "technique": {"name": "Normal"},
            "statsbomb_xg": 0.1,
        },
    }


def test_extract_shots_exclusions_et_score():
    events = [
        _shot(1, "A", outcome="Goal"),
        _shot(2, "B", shot_type="Penalty", outcome="Goal"),  # penalty : exclu mais compte au score
        {"id": "og", "index": 3, "period": 1, "type": {"name": "Own Goal For"}, "team": {"name": "A"}},
        _shot(4, "B"),
        _shot(5, "A", period=5, outcome="Goal"),  # seance de tirs au but : exclu
    ]
    rows = extract_shots(events, MATCH)
    assert [r["event_id"] for r in rows] == ["e1", "e4"]
    assert rows[0]["is_goal"] == 1 and rows[1]["is_goal"] == 0
    # Avant le tir e4 (equipe B) : A a 2 buts (dont csc), B a 1 but (penalty).
    assert (rows[1]["goals_for_before"], rows[1]["goals_against_before"]) == (1, 2)
    all_rows = extract_shots(events, MATCH, keep_excluded=True)
    assert [r["exclusion"] for r in all_rows] == [None, "penalty", None, "tir_au_but"]


@pytest.mark.skipif(not SHOTS_PATH.exists(), reason="table des tirs non construite")
def test_forme_de_la_table_des_tirs():
    df = pd.read_parquet(SHOTS_PATH)
    assert len(df) > 10_000
    assert df["event_id"].is_unique
    for feats in FEATURE_SETS.values():
        assert set(feats) <= set(df.columns)
    assert set(df["is_goal"].unique()) == {0, 1}
    assert (df["shot_type"] != "Penalty").all()
    assert (df["period"] != 5).all()
    # Le repere StatsBomb deborde parfois tres legerement (ex. x = 120.2).
    assert df["x"].between(0, 121).all() and df["y"].between(0, 80).all()
    assert df[FEATURE_SETS["simple"]].notna().all().all()
