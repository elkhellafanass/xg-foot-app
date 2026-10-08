import math

import numpy as np
import pytest

from src.features import (
    FEATURE_SETS,
    add_features,
    freeze_frame_features,
    make_shot,
    shot_angle,
    shot_distance,
)


@pytest.mark.parametrize(
    ("x", "y", "expected"),
    [(120, 40, 0.0), (108, 40, 12.0), (114, 32, 10.0), (0, 40, 120.0), (120, 0, 40.0)],
)
def test_distance_valeurs_connues(x, y, expected):
    assert shot_distance(x, y) == pytest.approx(expected)


def test_distance_vectorisee():
    d = shot_distance(np.array([120, 108]), np.array([40, 40]))
    assert d.tolist() == pytest.approx([0.0, 12.0])


def test_angle_point_de_penalty():
    # Point de penalty (108, 40) : demi-angle atan(4/12), angle total 2*atan(1/3).
    assert shot_angle(108, 40) == pytest.approx(2 * math.atan(4 / 12))


def test_angle_sur_la_ligne_entre_les_poteaux():
    assert shot_angle(120, 40) == pytest.approx(math.pi)


def test_angle_sur_un_poteau_et_ligne_de_but_hors_cadre():
    assert shot_angle(120, 36) == pytest.approx(0.0)
    assert shot_angle(120, 20) == pytest.approx(0.0)


def test_angle_decroit_avec_la_distance_et_symetrique():
    assert shot_angle(110, 40) > shot_angle(100, 40) > shot_angle(80, 40)
    assert shot_angle(105, 30) == pytest.approx(shot_angle(105, 50))
    assert 0 <= shot_angle(60, 10) <= math.pi


def _player(x, y, teammate=False, position="Center Back"):
    return {"location": [x, y], "teammate": teammate, "position": {"name": position}}


def test_freeze_frame_triangle_et_gardien():
    frame = [
        _player(118, 40, position="Goalkeeper"),
        _player(110, 40),  # dans le triangle
        _player(110, 60),  # hors triangle
        _player(105, 40, teammate=True),  # coequipier : ignore
    ]
    f = freeze_frame_features(100, 40, frame)
    assert f["has_freeze_frame"]
    assert f["gk_distance"] == pytest.approx(18.0)
    assert f["gk_to_goal"] == pytest.approx(2.0)
    assert f["gk_in_triangle"] == 1.0
    assert f["n_defenders_triangle"] == 1.0
    assert f["nearest_defender_dist"] == pytest.approx(10.0)


def test_freeze_frame_absent():
    f = freeze_frame_features(100, 40, None)
    assert not f["has_freeze_frame"]
    assert math.isnan(f["gk_distance"])


def test_make_shot_contient_toutes_les_variables():
    df = make_shot(108, 40, body_part="Head", technique="Volley")
    for feats in FEATURE_SETS.values():
        assert set(feats) <= set(df.columns)
    assert df.loc[0, "distance"] == pytest.approx(12.0)
    assert bool(df.loc[0, "is_volley"])


def test_statsbomb_xg_n_est_pas_une_variable():
    for feats in FEATURE_SETS.values():
        assert "statsbomb_xg" not in feats
        assert "is_goal" not in feats and "outcome" not in feats


def test_add_features_remplit_les_categories():
    df = make_shot(100, 30)
    df.loc[0, "technique"] = None
    out = add_features(df.drop(columns=["distance", "angle", "is_volley"]))
    assert out.loc[0, "technique"] == "Other"
