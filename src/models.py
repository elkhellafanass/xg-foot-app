"""Modeles xG : pipelines scikit-learn, grilles d'hyperparametres, metriques et bootstrap."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import RANDOM_STATE
from src.features import BOOLEAN_FEATURES, CATEGORICAL_FEATURES

MODEL_LABELS = {
    "baseline": "Baseline (fréquence moyenne)",
    "logreg": "Régression logistique",
    "rf": "Forêt aléatoire",
    "hgb": "Gradient boosting (HistGB)",
    "xgb": "XGBoost",
    "statsbomb": "xG StatsBomb (référence)",
}


def _to_float(x):
    return np.asarray(x, dtype=float)


def make_preprocessor(features: list[str], scale: bool) -> ColumnTransformer:
    """Encodage des variables : one-hot pour les categories, imputation (+ normalisation) sinon."""
    cats = [f for f in features if f in CATEGORICAL_FEATURES]
    bools = [f for f in features if f in BOOLEAN_FEATURES]
    nums = [f for f in features if f not in cats and f not in bools]
    num_steps: list = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [
            ("num", Pipeline(num_steps), nums),
            ("bool", FunctionTransformer(_to_float, feature_names_out="one-to-one"), bools),
            (
                "cat",
                OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20, sparse_output=False),
                cats,
            ),
        ],
        verbose_feature_names_out=False,
    )


def available_models() -> list[str]:
    names = ["baseline", "logreg", "rf", "hgb"]
    try:
        import xgboost  # noqa: F401

        names.append("xgb")
    except ImportError:
        pass
    return names


def make_model(name: str, features: list[str]) -> tuple[Pipeline, dict]:
    """Retourne (pipeline, grille d'hyperparametres) pour un modele."""
    if name == "baseline":
        return Pipeline([("clf", DummyClassifier(strategy="prior"))]), {}
    if name == "logreg":
        clf = LogisticRegression(max_iter=2000)
        grid = {"clf__C": [0.01, 0.1, 1.0, 10.0, 100.0]}
        return Pipeline([("prep", make_preprocessor(features, scale=True)), ("clf", clf)]), grid
    if name == "rf":
        clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE)
        grid = {
            "clf__max_depth": [6, 8, 10, 14],
            "clf__min_samples_leaf": [10, 20, 50],
            "clf__max_features": ["sqrt", 0.5],
        }
        return Pipeline([("prep", make_preprocessor(features, scale=False)), ("clf", clf)]), grid
    if name == "hgb":
        clf = HistGradientBoostingClassifier(early_stopping=False, random_state=RANDOM_STATE)
        grid = {
            "clf__learning_rate": [0.03, 0.1],
            "clf__max_iter": [100, 200, 400],
            "clf__max_leaf_nodes": [7, 15, 31],
            "clf__min_samples_leaf": [20, 80],
        }
        return Pipeline([("prep", make_preprocessor(features, scale=False)), ("clf", clf)]), grid
    if name == "xgb":
        from xgboost import XGBClassifier

        clf = XGBClassifier(
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            tree_method="hist",
        )
        grid = {
            "clf__n_estimators": [150, 300, 600],
            "clf__max_depth": [2, 3, 5],
            "clf__learning_rate": [0.03, 0.1],
            "clf__min_child_weight": [1, 10],
        }
        return Pipeline([("prep", make_preprocessor(features, scale=False)), ("clf", clf)]), grid
    raise ValueError(f"Modele inconnu : {name}")


def choose_calibration(
    model, X_cal: pd.DataFrame, y_cal: np.ndarray, groups: np.ndarray, n_splits: int = 5
) -> tuple[str, dict[str, float]]:
    """Choisit la calibration (aucune, Platt ou isotonique) par validation croisee groupee
    sur l'ensemble de calibration uniquement (le test n'est jamais utilise).

    Retourne la methode retenue et la log-loss moyenne de chaque option.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import GroupKFold

    raw = model.predict_proba(X_cal)[:, 1]
    scores: dict[str, list[float]] = {"aucune": [], "sigmoid": [], "isotonic": []}
    for fit_idx, val_idx in GroupKFold(n_splits=n_splits).split(X_cal, y_cal, groups):
        scores["aucune"].append(log_loss(y_cal[val_idx], np.clip(raw[val_idx], 1e-6, 1 - 1e-6)))
        for method in ("sigmoid", "isotonic"):
            cal = CalibratedClassifierCV(model, method=method, cv="prefit")
            cal.fit(X_cal.iloc[fit_idx], y_cal[fit_idx])
            p = np.clip(cal.predict_proba(X_cal.iloc[val_idx])[:, 1], 1e-6, 1 - 1e-6)
            scores[method].append(log_loss(y_cal[val_idx], p, labels=[0, 1]))
    means = {k: float(np.mean(v)) for k, v in scores.items()}
    return min(means, key=means.get), means


def split_by_match(
    df: pd.DataFrame, test_size: float = 0.2, calib_size: float = 0.2, seed: int = RANDOM_STATE
) -> dict[str, np.ndarray]:
    """Decoupe par match : entrainement / calibration / test (aucun match partage).

    ``calib_size`` est la part de l'ensemble hors-test reservee a la calibration.
    Retourne les indices positionnels de chaque ensemble.
    """
    groups = df["match_id"].to_numpy()
    idx = np.arange(len(df))
    outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    dev, test = next(outer.split(idx, groups=groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=calib_size, random_state=seed + 1)
    tr, cal = next(inner.split(dev, groups=groups[dev]))
    return {"train": dev[tr], "calib": dev[cal], "test": test}


def compute_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "mean_pred": float(p.mean()),
        "goal_rate": float(np.mean(y)),
    }


def bootstrap_metrics(
    y: np.ndarray,
    preds: dict[str, np.ndarray],
    groups: np.ndarray,
    n_boot: int = 1000,
    seed: int = RANDOM_STATE,
    reference: str | None = None,
) -> dict[str, dict[str, list[float]]]:
    """Intervalles de confiance a 95 % par bootstrap sur les matchs (tirage avec remise).

    Si ``reference`` est donne, calcule aussi l'IC de la difference (modele - reference).
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    samples: dict[str, dict[str, list[float]]] = {
        m: {"roc_auc": [], "brier": [], "log_loss": []} for m in preds
    }
    diffs: dict[str, dict[str, list[float]]] = {
        m: {"roc_auc": [], "brier": [], "log_loss": []} for m in preds if m != reference
    }
    for _ in range(n_boot):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by_group[g] for g in draw])
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        res = {m: compute_metrics(yb, p[idx]) for m, p in preds.items()}
        for m, r in res.items():
            for k in samples[m]:
                samples[m][k].append(r[k])
            if reference and m != reference:
                for k in diffs[m]:
                    diffs[m][k].append(r[k] - res[reference][k])

    def ci(values: list[float]) -> list[float]:
        return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]

    out: dict[str, dict[str, list[float]]] = {}
    for m in preds:
        out[m] = {f"{k}_ci95": ci(v) for k, v in samples[m].items()}
        if reference and m != reference:
            out[m].update({f"{k}_diff_vs_{reference}_ci95": ci(v) for k, v in diffs[m].items()})
    return out
