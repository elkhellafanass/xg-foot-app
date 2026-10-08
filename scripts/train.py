"""Entraine, calibre et evalue les modeles xG ; produit metriques, figures et modeles.

Usage : python scripts/train.py [--n-boot 1000] [--no-xgb]

Protocole (toutes les decoupes sont faites par MATCH, graine fixe) :
1. 20 % des matchs -> ensemble de TEST (evaluation finale uniquement).
2. Le reste est decoupe en ENTRAINEMENT (80 %) et CALIBRATION (20 %) par match.
3. Reglage des hyperparametres par GridSearchCV + GroupKFold(5) sur l'entrainement (log-loss).
4. Recalibrage sur l'ensemble de calibration : la methode (aucune, Platt ou isotonique)
   est choisie par validation croisee groupee sur ce seul ensemble.
5. Evaluation sur le test : ROC-AUC, Brier, log-loss + IC 95 % par bootstrap sur les matchs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from matplotlib.patches import Arc, Rectangle  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV, calibration_curve  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402
from sklearn.model_selection import GridSearchCV, GroupKFold  # noqa: E402

from src.config import FIGURES_DIR, MODELS_DIR, RANDOM_STATE, REPORTS_DIR, SHOTS_PATH  # noqa: E402
from src.features import FEATURE_LABELS, FEATURE_SETS, FREEZE_FRAME_FEATURES, fr  # noqa: E402
from src.models import (  # noqa: E402
    MODEL_LABELS,
    available_models,
    bootstrap_metrics,
    choose_calibration,
    compute_metrics,
    make_model,
    split_by_match,
)
from src.viz import (  # noqa: E402
    GOAL_COLOR,
    MISS_COLOR,
    MODEL_COLORS,
    MODEL_LINESTYLES,
    MPL_STYLE,
    SEQUENTIAL_BLUES,
)

plt.rcParams.update(MPL_STYLE)
FS_LABELS = {"simple": "jeu simple", "enrichi": "jeu enrichi (freeze frame)"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- figures
def draw_half_pitch(ax) -> None:
    """Demi-terrain StatsBomb (x de 60 a 120), vue verticale : but en haut."""
    lc, lw = "#9a9993", 1.0
    # Coordonnees du trace : abscisse = y StatsBomb, ordonnee = x StatsBomb.
    ax.add_patch(Rectangle((0, 60), 80, 60, fill=False, ec=lc, lw=lw))
    ax.add_patch(Rectangle((18, 102), 44, 18, fill=False, ec=lc, lw=lw))
    ax.add_patch(Rectangle((30, 114), 20, 6, fill=False, ec=lc, lw=lw))
    ax.add_patch(Rectangle((36, 120), 8, 1.5, fill=False, ec=lc, lw=lw))
    ax.add_patch(Arc((40, 108), 20, 20, theta1=217, theta2=323, ec=lc, lw=lw))
    ax.add_patch(Arc((40, 60), 20, 20, theta1=0, theta2=180, ec=lc, lw=lw))
    ax.plot([40], [108], marker="o", ms=2, color=lc)
    ax.set_xlim(-1, 81)
    ax.set_ylim(59, 122.5)
    ax.set_aspect("equal")
    ax.axis("off")


def fig_shot_map(df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4), layout="constrained")
    for ax, (goal, title, color) in zip(
        axes, [(0, "Tirs non convertis", MISS_COLOR), (1, "Buts", GOAL_COLOR)], strict=True
    ):
        sub = df[df["is_goal"] == goal]
        draw_half_pitch(ax)
        hb = ax.hexbin(
            sub["y"],
            sub["x"],
            gridsize=(26, 20),
            extent=(0, 80, 60, 120),
            mincnt=1,
            cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", SEQUENTIAL_BLUES)
            if goal == 0
            else matplotlib.colors.LinearSegmentedColormap.from_list(
                "o", ["#fde3d6", "#f5a27f", color, "#a43a10"]
            ),
            linewidths=0.2,
        )
        ax.set_title(f"{title} (n = {len(sub):,})".replace(",", " "))
        fig.colorbar(hb, ax=ax, shrink=0.7, label="Nombre de tirs")
    fig.suptitle("Localisation des tirs (hors penalties)")
    fig.savefig(path)
    plt.close(fig)


def fig_goal_rate_distance(df: pd.DataFrame, path: Path) -> None:
    bins = np.arange(0, 42, 2)
    d = df.assign(bin=pd.cut(df["distance"], bins, right=False))
    g = d.groupby("bin", observed=True).agg(n=("is_goal", "size"), taux=("is_goal", "mean"))
    g = g[g["n"] >= 30]
    centers = [b.left + 1 for b in g.index]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar(centers, g["taux"], width=1.7, color=MODEL_COLORS["logreg"])
    ax.set_xlabel("Distance au centre du but (yards, classes de 2 yards)")
    ax.set_ylabel("Taux de conversion observé")
    ax.set_title("Taux de but selon la distance de tir")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    fig.savefig(path)
    plt.close(fig)


def fig_roc(y, preds: dict[str, np.ndarray], fs: str, path: Path) -> dict:
    curves = {}
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    for name, p in preds.items():
        fpr, tpr, _ = roc_curve(y, p)
        keep = np.unique(np.linspace(0, len(fpr) - 1, min(len(fpr), 200)).astype(int))
        curves[name] = {"fpr": fpr[keep].round(4).tolist(), "tpr": tpr[keep].round(4).tolist()}
        auc = compute_metrics(y, p)["roc_auc"]
        ax.plot(
            fpr,
            tpr,
            color=MODEL_COLORS[name],
            ls=MODEL_LINESTYLES[name],
            label=f"{MODEL_LABELS[name]} (AUC = {auc:.3f})",
        )
    ax.plot([0, 1], [0, 1], color="#c3c2b7", lw=0.8)
    ax.set_xlabel("Taux de faux positifs")
    ax.set_ylabel("Taux de vrais positifs")
    ax.set_title(f"Courbes ROC sur le test – {FS_LABELS[fs]}")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    fig.savefig(path)
    plt.close(fig)
    return curves


def fig_calibration(y, preds: dict[str, np.ndarray], fs: str, path: Path) -> dict:
    curves = {}
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    ax.plot([0, 1], [0, 1], color="#c3c2b7", lw=0.8, label="Calibration parfaite")
    for name, p in preds.items():
        if name == "baseline":
            continue
        frac, mean = calibration_curve(y, p, n_bins=15, strategy="quantile")
        curves[name] = {"pred_moyenne": mean.tolist(), "frequence_observee": frac.tolist()}
        ax.plot(
            mean,
            frac,
            color=MODEL_COLORS[name],
            ls=MODEL_LINESTYLES[name],
            marker="o",
            ms=3.5,
            label=MODEL_LABELS[name],
        )
    ax.set_xlabel("Probabilité prédite (moyenne par classe de 1/15 des tirs)")
    ax.set_ylabel("Fréquence observée de but")
    ax.set_title(f"Calibration sur le test – {FS_LABELS[fs]}")
    ax.set_xlim(0, 0.6)
    ax.set_ylim(0, 0.6)
    ax.legend(loc="upper left")
    fig.savefig(path)
    plt.close(fig)
    return curves


def fig_importance(imp: pd.DataFrame, title: str, path: Path) -> None:
    imp = imp.sort_values("importance_moyenne")
    fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(imp) + 1.0))
    ax.barh(
        [FEATURE_LABELS.get(f, f) for f in imp["variable"]],
        imp["importance_moyenne"],
        xerr=imp["ecart_type"],
        color=MODEL_COLORS["logreg"],
        error_kw={"lw": 0.8},
    )
    ax.set_xlabel("Hausse de la log-loss quand la variable est permutée")
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------------ pipeline
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--no-xgb", action="store_true")
    args = parser.parse_args()

    if not SHOTS_PATH.exists():
        print(f"ERREUR : {SHOTS_PATH} introuvable. Lancez d'abord scripts/build_dataset.py")
        return 1
    for d in (MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(SHOTS_PATH)
    log(f"{len(df)} tirs, {df['match_id'].nunique()} matchs, taux de but {df['is_goal'].mean():.4f}")
    if df["statsbomb_xg"].isna().any():
        log(f"Attention : {df['statsbomb_xg'].isna().sum()} tirs sans xG StatsBomb")

    fig_shot_map(df, FIGURES_DIR / "fig_carte_tirs.png")
    fig_goal_rate_distance(df, FIGURES_DIR / "fig_taux_but_distance.png")

    split = split_by_match(df)
    sets = {k: df.iloc[v].reset_index(drop=True) for k, v in split.items()}
    for a, b in [("train", "calib"), ("train", "test"), ("calib", "test")]:
        assert not set(sets[a]["match_id"]) & set(sets[b]["match_id"]), "fuite entre ensembles"
    split_info = {
        k: {
            "n_matchs": int(v["match_id"].nunique()),
            "n_tirs": len(v),
            "n_buts": int(v["is_goal"].sum()),
            "match_ids": sorted(int(m) for m in v["match_id"].unique()),
        }
        for k, v in sets.items()
    }
    (MODELS_DIR / "split.json").write_text(json.dumps(split_info), encoding="utf-8")
    for k, v in split_info.items():
        log(f"Ensemble {k}: {v['n_matchs']} matchs, {v['n_tirs']} tirs, {v['n_buts']} buts")

    train, calib, test = sets["train"], sets["calib"], sets["test"]
    y_tr, y_cal, y_te = (s["is_goal"].to_numpy() for s in (train, calib, test))
    groups_te = test["match_id"].to_numpy()

    model_names = [m for m in available_models() if not (args.no_xgb and m == "xgb")]
    results: dict = {
        "protocole": {
            "graine": RANDOM_STATE,
            "decoupe": "par match (GroupShuffleSplit) : test 20 % ; calibration 20 % du reste",
            "validation_croisee": "GroupKFold(5) sur l'entrainement, critere log-loss",
            "calibration": "aucune / Platt / isotonique, choisie par GroupKFold(5) sur l'ensemble "
            "de calibration (critere log-loss)",
            "bootstrap": f"{args.n_boot} tirages de matchs avec remise sur le test",
            "scikit_learn": sklearn.__version__,
        },
        "split": {k: {kk: vv for kk, vv in v.items() if kk != "match_ids"} for k, v in split_info.items()},
        "feature_sets": FEATURE_SETS,
        "modeles": {},
    }
    rows = []
    manifest = {"models": [], "freeze_defaults": {}}
    manifest["freeze_defaults"] = {c: float(train[c].median()) for c in FREEZE_FRAME_FEATURES}
    best_by_fs: dict[str, tuple[str, float]] = {}
    all_test_preds: dict[str, dict[str, np.ndarray]] = {}

    for fs, feats in FEATURE_SETS.items():
        X_tr, X_cal, X_te = train[feats], calib[feats], test[feats]
        preds: dict[str, np.ndarray] = {}
        for name in model_names:
            t0 = time.time()
            pipe, grid = make_model(name, feats)
            entry: dict = {"modele": name, "jeu": fs, "libelle": MODEL_LABELS[name]}
            if grid:
                search = GridSearchCV(
                    pipe,
                    grid,
                    scoring="neg_log_loss",
                    cv=GroupKFold(n_splits=5),
                    n_jobs=-1,
                    refit=True,
                )
                search.fit(X_tr, y_tr, groups=train["match_id"])
                best = search.best_estimator_
                entry["meilleurs_hyperparametres"] = {
                    k.replace("clf__", ""): v for k, v in search.best_params_.items()
                }
                entry["cv_log_loss"] = float(-search.best_score_)
                entry["cv_log_loss_std"] = float(search.cv_results_["std_test_score"][search.best_index_])
                raw_p = best.predict_proba(X_te)[:, 1]
                entry["test_non_calibre"] = compute_metrics(y_te, raw_p)
                method, cal_scores = choose_calibration(best, X_cal, y_cal, calib["match_id"].to_numpy())
                entry["calibration"] = method
                entry["calibration_log_loss_cv"] = cal_scores
                if method == "aucune":
                    model = best
                else:
                    model = CalibratedClassifierCV(best, method=method, cv="prefit")
                    model.fit(X_cal, y_cal)
            else:
                model = pipe.fit(X_tr, y_tr)
            p = model.predict_proba(X_te)[:, 1]
            preds[name] = p
            entry["test"] = compute_metrics(y_te, p)
            entry["duree_s"] = round(time.time() - t0, 1)
            fname = f"{fs}__{name}.joblib"
            joblib.dump(model, MODELS_DIR / fname, compress=3)
            entry["fichier"] = f"models/{fname}"
            results["modeles"][f"{fs}__{name}"] = entry
            manifest["models"].append(
                {
                    "key": f"{fs}__{name}",
                    "name": name,
                    "feature_set": fs,
                    "file": fname,
                    "label": MODEL_LABELS[name],
                    "features": feats,
                }
            )
            log(
                f"{fs:8s} {name:9s} AUC={entry['test']['roc_auc']:.4f} "
                f"Brier={entry['test']['brier']:.4f} LL={entry['test']['log_loss']:.4f} "
                f"calib={entry.get('calibration', '-')} ({entry['duree_s']} s)"
            )
            # Modele principal : meilleur en CV parmi les modeles scikit-learn (deployables).
            if grid and name != "xgb" and (fs not in best_by_fs or entry["cv_log_loss"] < best_by_fs[fs][1]):
                best_by_fs[fs] = (name, entry["cv_log_loss"])

        preds["statsbomb"] = test["statsbomb_xg"].to_numpy()
        all_test_preds[fs] = preds
        results.setdefault("courbes_roc", {})[fs] = fig_roc(
            y_te, preds, fs, FIGURES_DIR / f"fig_roc_{fs}.png"
        )
        results.setdefault("courbes_calibration", {})[fs] = fig_calibration(
            y_te, preds, fs, FIGURES_DIR / f"fig_calibration_{fs}.png"
        )

    # Reference externe StatsBomb
    results["modeles"]["reference__statsbomb"] = {
        "modele": "statsbomb",
        "jeu": "reference",
        "libelle": MODEL_LABELS["statsbomb"],
        "test": compute_metrics(y_te, test["statsbomb_xg"].to_numpy()),
    }

    # Bootstrap sur les matchs (tous les modeles, avec difference vs StatsBomb)
    log(f"Bootstrap ({args.n_boot} tirages)...")
    flat = {f"{fs}__{m}": p for fs, pr in all_test_preds.items() for m, p in pr.items() if m != "statsbomb"}
    flat["reference__statsbomb"] = test["statsbomb_xg"].to_numpy()
    boot = bootstrap_metrics(y_te, flat, groups_te, n_boot=args.n_boot, reference="reference__statsbomb")
    base_brier = {fs: results["modeles"][f"{fs}__baseline"]["test"]["brier"] for fs in FEATURE_SETS}
    for key, entry in results["modeles"].items():
        entry["test_ic95"] = boot[key]
        fs = entry["jeu"] if entry["jeu"] in FEATURE_SETS else "simple"
        entry["test"]["brier_skill_vs_baseline"] = 1 - entry["test"]["brier"] / base_brier[fs]
        t, ci = entry["test"], boot[key]
        rows.append(
            {
                "jeu": entry["jeu"],
                "modele": entry["modele"],
                "libelle": entry["libelle"],
                "roc_auc": t["roc_auc"],
                "roc_auc_ic_bas": ci["roc_auc_ci95"][0],
                "roc_auc_ic_haut": ci["roc_auc_ci95"][1],
                "brier": t["brier"],
                "brier_ic_bas": ci["brier_ci95"][0],
                "brier_ic_haut": ci["brier_ci95"][1],
                "log_loss": t["log_loss"],
                "log_loss_ic_bas": ci["log_loss_ci95"][0],
                "log_loss_ic_haut": ci["log_loss_ci95"][1],
                "brier_skill_vs_baseline": t["brier_skill_vs_baseline"],
                "xg_moyen": t["mean_pred"],
                "taux_but": t["goal_rate"],
                "cv_log_loss": entry.get("cv_log_loss"),
                "delta_auc_vs_statsbomb_ic": ci.get("roc_auc_diff_vs_reference__statsbomb_ci95"),
                "delta_brier_vs_statsbomb_ic": ci.get("brier_diff_vs_reference__statsbomb_ci95"),
            }
        )
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(REPORTS_DIR / "metrics.csv", index=False, encoding="utf-8", float_format="%.5f")

    # Modele principal par jeu de variables : meilleure log-loss en validation croisee (pas le test)
    results["modele_principal"] = {fs: v[0] for fs, v in best_by_fs.items()}
    manifest["main_models"] = results["modele_principal"]
    log(f"Modeles principaux (choisis en CV) : {results['modele_principal']}")

    # Importance par permutation (sur le test, modele principal de chaque jeu)
    imp_all = []
    for fs, (name, _) in best_by_fs.items():
        model = joblib.load(MODELS_DIR / f"{fs}__{name}.joblib")
        feats = FEATURE_SETS[fs]
        r = permutation_importance(
            model,
            test[feats],
            y_te,
            scoring="neg_log_loss",
            n_repeats=10,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        imp = pd.DataFrame(
            {"variable": feats, "importance_moyenne": r.importances_mean, "ecart_type": r.importances_std}
        )
        imp.insert(0, "modele", name)
        imp.insert(0, "jeu", fs)
        imp_all.append(imp)
        fig_importance(
            imp,
            f"Importance par permutation – {MODEL_LABELS[name]}, {FS_LABELS[fs]}",
            FIGURES_DIR / f"fig_importance_{fs}.png",
        )
    pd.concat(imp_all).to_csv(
        REPORTS_DIR / "permutation_importance.csv", index=False, encoding="utf-8", float_format="%.5f"
    )

    # Analyse d'erreurs (modele principal enrichi vs StatsBomb, sur le test)
    fs_main = "enrichi"
    main_name = best_by_fs[fs_main][0]
    ea = test.assign(xg_modele=all_test_preds[fs_main][main_name])
    ea["classe_distance"] = pd.cut(
        ea["distance"],
        [0, 6, 12, 18, 24, 30, 200],
        right=False,
        labels=["0-6", "6-12", "12-18", "18-24", "24-30", "30+"],
    )
    err_tables = []
    for col, lab in [
        ("classe_distance", "distance (yards)"),
        ("body_part", "partie du corps"),
        ("shot_type", "type de tir"),
        ("play_pattern", "phase de jeu"),
        ("competition", "competition"),
    ]:
        g = (
            ea.groupby(col, observed=True)
            .agg(
                n_tirs=("is_goal", "size"),
                buts=("is_goal", "sum"),
                xg_modele=("xg_modele", "sum"),
                xg_statsbomb=("statsbomb_xg", "sum"),
            )
            .reset_index()
            .rename(columns={col: "modalite"})
        )
        g.insert(0, "dimension", lab)
        g["modalite"] = g["modalite"].astype(str).map(fr)
        g["ecart_buts_moins_xg"] = g["buts"] - g["xg_modele"]
        err_tables.append(g)
    err = pd.concat(err_tables)
    err.to_csv(REPORTS_DIR / "error_analysis.csv", index=False, encoding="utf-8", float_format="%.2f")
    cols = [
        "competition",
        "season",
        "team",
        "opponent",
        "player",
        "minute",
        "distance",
        "body_part",
        "shot_type",
        "xg_modele",
        "statsbomb_xg",
    ]
    surprises = pd.concat(
        [
            ea[ea.is_goal == 1].nsmallest(10, "xg_modele")[cols].assign(cas="but improbable"),
            ea[ea.is_goal == 0].nlargest(10, "xg_modele")[cols].assign(cas="grosse occasion manquee"),
        ]
    )
    surprises.to_csv(REPORTS_DIR / "error_examples.csv", index=False, encoding="utf-8", float_format="%.3f")
    results["analyse_erreurs"] = {
        "modele": f"{fs_main}__{main_name}",
        "correlation_xg_modele_statsbomb": float(np.corrcoef(ea["xg_modele"], ea["statsbomb_xg"])[0, 1]),
        "tirs_sans_freeze_frame_test": int((~ea["has_freeze_frame"]).sum()),
    }

    (REPORTS_DIR / "metrics.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (MODELS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        metrics_df[["jeu", "modele", "roc_auc", "brier", "log_loss", "brier_skill_vs_baseline"]].to_string(
            index=False, float_format="%.4f"
        )
    )
    log("Termine.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
