import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src.config import MODELS_DIR, REPORTS_DIR
from src.features import make_shot
from src.models import bootstrap_metrics, compute_metrics, split_by_match

MANIFEST = MODELS_DIR / "manifest.json"


def _toy(n_matches=50, shots_per_match=20, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "match_id": np.repeat(np.arange(n_matches), shots_per_match),
            "is_goal": rng.integers(0, 2, n_matches * shots_per_match),
        }
    )


def test_split_par_match_sans_fuite():
    df = _toy()
    split = split_by_match(df)
    ids = {k: set(df.iloc[v]["match_id"]) for k, v in split.items()}
    assert not ids["train"] & ids["test"]
    assert not ids["train"] & ids["calib"]
    assert not ids["calib"] & ids["test"]
    assert sum(len(v) for v in split.values()) == len(df)


@pytest.mark.skipif(not (MODELS_DIR / "split.json").exists(), reason="modeles non entraines")
def test_split_reel_sans_match_commun():
    split = json.loads((MODELS_DIR / "split.json").read_text(encoding="utf-8"))
    tr, ca, te = (set(split[k]["match_ids"]) for k in ("train", "calib", "test"))
    assert not tr & te and not tr & ca and not ca & te
    assert len(te) > 100


def test_metriques_et_bootstrap():
    y = np.array([0, 0, 1, 1, 0, 1, 0, 0])
    m = compute_metrics(y, np.array([0.1, 0.2, 0.8, 0.7, 0.3, 0.9, 0.2, 0.1]))
    assert m["roc_auc"] == pytest.approx(1.0)
    assert 0 <= m["brier"] <= 1
    groups = np.array([1, 1, 2, 2, 3, 3, 4, 4])
    ci = bootstrap_metrics(y, {"a": np.full(8, 0.375), "b": y * 0.8 + 0.1}, groups, n_boot=50, reference="a")
    lo, hi = ci["b"]["brier_ci95"]
    assert lo <= hi
    assert "brier_diff_vs_a_ci95" in ci["b"]


@pytest.mark.skipif(not MANIFEST.exists(), reason="modeles non entraines")
def test_chargement_des_modeles_et_probabilites():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    shots = pd.concat(
        [
            make_shot(108, 40),
            make_shot(119, 40, body_part="Head"),
            make_shot(70, 5, shot_type="Free Kick"),
        ],
        ignore_index=True,
    )
    loaded = 0
    for info in manifest["models"]:
        try:
            model = joblib.load(MODELS_DIR / info["file"])
        except ImportError:  # XGBoost absent
            continue
        p = model.predict_proba(shots[info["features"]])[:, 1]
        assert np.all((p >= 0) & (p <= 1))
        if info["name"] != "baseline":
            assert p[1] > p[2], info["key"]  # tir a 1 yard > coup franc lointain excentre
        loaded += 1
    assert loaded >= 8


@pytest.mark.skipif(not (REPORTS_DIR / "metrics.json").exists(), reason="metriques absentes")
def test_metriques_sauvegardees_coherentes():
    m = json.loads((REPORTS_DIR / "metrics.json").read_text(encoding="utf-8"))
    base = m["modeles"]["enrichi__baseline"]["test"]
    for key, entry in m["modeles"].items():
        t = entry["test"]
        assert 0 <= t["brier"] <= 1 and 0 <= t["roc_auc"] <= 1
        if "baseline" not in key:
            assert t["brier"] < base["brier"], key
