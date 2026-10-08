# Correspondance résultats ↔ sections du mémoire

Toutes les valeurs ci-dessous sont **calculées par le code** et lisibles dans les fichiers cités :
`reports/dataset_summary.csv`, `reports/metrics.csv`, `reports/metrics.json`,
`reports/permutation_importance.csv`, `reports/error_analysis.csv` et `reports/error_examples.csv`.
Elles proviennent du run de `scripts/train.py` (graine 42, scikit-learn 1.5.2). Deux exécutions
successives ont donné des valeurs identiques.

Notation : IC 95 % = intervalle de confiance à 95 % obtenu par bootstrap (1 000 tirages de matchs
avec remise sur le jeu de test).

---

## 3.3 Données

| Élément | Valeur exacte | Source |
|---|---|---|
| Source | StatsBomb Open Data, dépôt GitHub `statsbomb/open-data`, fichiers JSON | `src/config.py`, `src/data.py` |
| Compétitions | 11 couples compétition/saison masculins (tableau ci-dessous) | `reports/dataset_summary.csv` |
| Matchs | **1 865** | idem |
| Tirs retenus | **45 827** | idem |
| Buts | **4 319**, soit un taux de conversion de **9,42 %** | idem |
| Penalties exclus | 523 | idem |
| Tirs au but exclus (période 5) | 241 | idem |
| Tirs avec freeze frame | 100 % (35 tirs sans gardien dans le freeze frame, 1 sans défenseur) | `shots.parquet` |
| Somme des xG StatsBomb | 4 202,5 | `reports/dataset_summary.csv` |
| Volume brut (cache compressé) | environ 539 Mo (.json.gz), non versionné | `data/raw/` |

| Compétition | Saison | Matchs | Tirs | Buts | Conversion |
|---|---|---|---|---|---|
| FIFA World Cup | 2022 | 64 | 1 430 | 152 | 10,63 % |
| FIFA World Cup | 2018 | 64 | 1 638 | 135 | 8,24 % |
| UEFA Euro | 2020 | 51 | 1 234 | 122 | 9,89 % |
| UEFA Euro | 2024 | 51 | 1 304 | 98 | 7,52 % |
| Copa America | 2024 | 32 | 741 | 63 | 8,50 % |
| African Cup of Nations | 2023 | 52 | 1 162 | 98 | 8,43 % |
| Premier League | 2015/2016 | 380 | 9 817 | 914 | 9,31 % |
| La Liga | 2015/2016 | 380 | 9 071 | 945 | 10,42 % |
| Serie A | 2015/2016 | 380 | 9 877 | 858 | 8,69 % |
| 1. Bundesliga | 2015/2016 | 34 (partielle) | 830 | 82 | 9,88 % |
| Ligue 1 | 2015/2016 | 377 | 8 723 | 852 | 9,77 % |

Figures : `reports/figures/fig_carte_tirs.png` (localisation des tirs et des buts) et
`reports/figures/fig_taux_but_distance.png` (taux de but selon la distance).

Licence : voir le README, section « Licence des données ». La redistribution est interdite, donc
aucune donnée n'est versionnée dans le dépôt.

## 3.5 Variables

| Jeu | Variables | Fichier |
|---|---|---|
| Simple (11) | distance, angle, minute, écart au score avant le tir, sous pression, premier contact, volée/demi-volée, partie du corps, type de tir, technique, phase de jeu | `src/features.py` (`FEATURE_SETS`) |
| Enrichi (16) | simple + distance tireur-gardien, distance gardien-but, gardien dans le triangle, nombre de défenseurs dans le triangle tireur-poteaux, distance du défenseur le plus proche | idem |

- Distance = √((120 − x)² + (40 − y)²). Angle = arccos de l'angle entre les vecteurs tireur → (120, 36)
  et tireur → (120, 44). Les valeurs connues sont testées dans `tests/test_features.py` :
  (120, 40) donne une distance de 0, (108, 40) une distance de 12, et l'angle au point de penalty
  vaut 2·atan(1/3).
- Statistiques descriptives (jeu complet) : distance médiane **19,04 yards** (moyenne 19,51),
  angle médian **0,34 rad**, défenseurs dans le triangle : médiane 1 (moyenne 0,78), défenseur le
  plus proche : médiane **2,35 yards**, distance tireur-gardien : médiane **15,34 yards**.
- Répartition : pied droit 23 710, pied gauche 14 348, tête 7 634, autre 135 ; jeu ouvert 43 692,
  coup franc direct 2 120, corner direct 15. Tirs sous pression : 26,0 % ; premier contact : 31,2 %.
- Le xG StatsBomb n'est **jamais** une variable explicative : il sert de référence externe, ce que
  vérifie le test `test_statsbomb_xg_n_est_pas_une_variable`.

## 3.6 Modèles

| Modèle | Prétraitement | Meilleurs hyperparamètres (simple / enrichi) |
|---|---|---|
| Baseline | aucun (fréquence moyenne de but à l'entraînement) | - |
| Régression logistique | imputation par la médiane, normalisation, one-hot | C = 1 / C = 1 |
| Forêt aléatoire (200 arbres) | imputation, one-hot | profondeur 8, feuille ≥ 10, max_features 0,5 / profondeur 10, feuille ≥ 20, max_features 0,5 |
| HistGradientBoosting (**modèle principal**) | imputation, one-hot | lr 0,03, 200 itérations, 7 feuilles, feuille ≥ 20 / lr 0,03, 200 itérations, 15 feuilles, feuille ≥ 20 |
| XGBoost (comparaison) | imputation, one-hot | profondeur 2, lr 0,03, 600 arbres, min_child_weight 1 (identiques sur les deux jeux) |

Le modèle principal est choisi sur la meilleure log-loss en validation croisée : 0,2606 (simple) et
**0,2499** (enrichi). Le choix se fait parmi les modèles scikit-learn, déployables en ligne.
Source : `reports/metrics.json` (`meilleurs_hyperparametres`, `cv_log_loss`, `modele_principal`).
Fichiers des modèles : `models/<jeu>__<modele>.joblib`.

## 3.7 Protocole expérimental

| Ensemble | Matchs | Tirs | Buts |
|---|---|---|---|
| Entraînement | 1 193 | 29 321 | 2 757 |
| Calibration | 299 | 7 332 | 670 |
| Test | 373 | 9 174 | 892 |

- Découpage par **match** (GroupShuffleSplit, graine 42) : test 20 % des matchs, calibration 20 %
  du reste. L'absence de match commun est vérifiée dans le script et par
  `tests/test_models.py::test_split_reel_sans_match_commun` (identifiants dans `models/split.json`).
- Réglage : GridSearchCV avec GroupKFold(5) sur l'entraînement, critère log-loss.
- Recalibrage : pour chaque modèle, la méthode (aucune, Platt ou isotonique) est choisie par
  GroupKFold(5) **sur l'ensemble de calibration uniquement**. Résultat : **aucune** pour tous les
  modèles. Exemple pour HistGB enrichi : log-loss en CV de 0,2446 sans calibration, 0,2446 avec
  Platt, 0,2486 en isotonique.
- Métriques sur le test : ROC-AUC, score de Brier, log-loss, IC 95 % par bootstrap sur les matchs
  et IC de la différence avec le xG StatsBomb.

## 4.2 Architecture

- Schéma : `docs/architecture.png` (généré par `scripts/make_architecture.py`).
- Chaîne : `download_data.py` → `build_dataset.py` → `train.py` → `models/` + `reports/` →
  application `app/streamlit_app.py` (6 pages dans `app/views/`).
- Qualité : 31 tests pytest (variables, données, modèles, fuite, pages Streamlit avec AppTest),
  ruff sans erreur, captures Playwright.

## 4.3 Résultats

| Modèle | Jeu | ROC-AUC [IC 95 %] | Brier [IC 95 %] | Log-loss [IC 95 %] | Gain de Brier vs baseline |
|---|---|---|---|---|---|
| Baseline | - | 0,500 | 0,0878 [0,0825 ; 0,0928] | 0,3190 [0,3042 ; 0,3330] | 0 % |
| Régression logistique | simple | 0,793 [0,778 ; 0,809] | 0,0751 [0,0707 ; 0,0793] | 0,2641 [0,2506 ; 0,2768] | 14,4 % |
| Forêt aléatoire | simple | 0,795 [0,780 ; 0,811] | 0,0754 [0,0710 ; 0,0797] | 0,2636 [0,2505 ; 0,2767] | 14,1 % |
| HistGB | simple | 0,796 [0,782 ; 0,812] | 0,0754 [0,0710 ; 0,0796] | 0,2636 [0,2505 ; 0,2763] | 14,1 % |
| XGBoost | simple | 0,798 [0,783 ; 0,814] | 0,0751 [0,0707 ; 0,0792] | 0,2626 [0,2492 ; 0,2753] | 14,5 % |
| Régression logistique | enrichi | 0,810 [0,795 ; 0,825] | 0,0734 [0,0691 ; 0,0775] | 0,2567 [0,2440 ; 0,2695] | 16,4 % |
| Forêt aléatoire | enrichi | 0,815 [0,801 ; 0,830] | 0,0721 [0,0680 ; 0,0763] | 0,2528 [0,2402 ; 0,2658] | 17,8 % |
| **HistGB (principal)** | **enrichi** | **0,819 [0,805 ; 0,834]** | **0,0719 [0,0677 ; 0,0761]** | **0,2517 [0,2391 ; 0,2647]** | **18,1 %** |
| XGBoost | enrichi | 0,820 [0,806 ; 0,835] | 0,0720 [0,0677 ; 0,0761] | 0,2515 [0,2388 ; 0,2644] | 18,0 % |
| xG StatsBomb | référence | 0,821 [0,807 ; 0,836] | 0,0714 [0,0670 ; 0,0756] | 0,2509 [0,2380 ; 0,2640] | 18,7 % |

Points saillants, tous lisibles dans `reports/metrics.json` :

- **Apport du freeze frame** : avec HistGB, l'AUC passe de 0,796 à 0,819 (+0,023) et le Brier de
  0,0754 à 0,0719.
- **Comparaison avec StatsBomb** : pour HistGB enrichi, l'IC 95 % de la différence d'AUC (modèle −
  StatsBomb) vaut [−0,0090 ; +0,0044], celui du Brier [−0,0004 ; +0,0016] et celui de la log-loss
  [−0,0023 ; +0,0039]. Tous contiennent 0 : **pas de différence significative** avec le xG StatsBomb.
  À l'inverse, tous les modèles du jeu simple, ainsi que la régression logistique enrichie
  (AUC [−0,0184 ; −0,0030]), sont significativement moins discriminants que StatsBomb.
- **Calibration globale sur le test** : 892 buts réels, contre 865,0 xG pour HistGB enrichi et
  843,6 xG pour StatsBomb.
- **Importance par permutation** (hausse de la log-loss, HistGB enrichi) : angle 0,0514, distance
  tireur-gardien 0,0199, distance 0,0105, défenseurs dans le triangle 0,0100, défenseur le plus
  proche 0,0089, partie du corps 0,0078. Minute, sous pression et premier contact ont une
  importance d'environ 0.
- **Analyse d'erreurs (test)** : corrélation de **0,911** entre le xG du modèle et celui de StatsBomb.
  Par distance : à 6-12 yards, 358 buts pour 340,3 xG (sous-estimation) ; au-delà de 30 yards,
  15 buts pour 20,2 xG (surestimation). Pied gauche : 283 buts pour 257,0 xG.

Figures (300 dpi) :

| Figure | Fichier |
|---|---|
| Courbes ROC, jeu simple / enrichi | `reports/figures/fig_roc_simple.png`, `reports/figures/fig_roc_enrichi.png` |
| Calibration, jeu simple / enrichi | `reports/figures/fig_calibration_simple.png`, `reports/figures/fig_calibration_enrichi.png` |
| Importance par permutation | `reports/figures/fig_importance_simple.png`, `reports/figures/fig_importance_enrichi.png` |
| Carte des tirs, taux de but par distance | `reports/figures/fig_carte_tirs.png`, `reports/figures/fig_taux_but_distance.png` |

## 4.4 Application

| Page | Capture |
|---|---|
| Accueil | `docs/screenshots/01_accueil.png` |
| Explorateur de tirs | `docs/screenshots/02_explorateur_tirs.png` |
| Simulateur xG | `docs/screenshots/03_simulateur_xg.png` et `docs/screenshots/03b_simulateur_apres_clic.png` (après un clic sur le terrain) |
| Modèles | `docs/screenshots/04_modeles.png` |
| Équipes et joueurs | `docs/screenshots/05_equipes_joueurs.png` |
| Méthodologie et sources | `docs/screenshots/06_methodologie_sources.png` |

- Exemple du simulateur (modèle principal) : un tir du pied droit en jeu ouvert au point de
  penalty (108, 40), avec les valeurs par défaut, vaut **19,7 %** ; en (111, 30), il vaut **8,8 %**.
- Version en ligne : téléchargement de 115 matchs (CdM 2022 et Euro 2024) au premier lancement, en
  environ 18 s lors du test en environnement vierge. Sur la CdM 2022, on retrouve 1 430 tirs,
  152 buts, 141,5 xG pour le modèle et 137,9 xG pour StatsBomb.
- Les captures ont été réalisées avec les données locales complètes (11 compétitions).

## 4.5 Discussion et limites constatées

1. **Les performances atteignent un plafond proche du xG StatsBomb.** Les modèles à arbres
   enrichis ne s'en distinguent pas statistiquement. Mais StatsBomb a peut-être entraîné son
   modèle sur ces mêmes matchs : la comparaison est indicative, pas un « match » équitable.
2. **Les modèles simples se valent tous** (AUC de 0,793 à 0,798, IC très recouvrants). Le gain vient
   des variables (freeze frame), pas de l'algorithme.
3. **Recalibrage** : aucun n'améliorait la log-loss sur l'ensemble de calibration. Les modèles
   optimisés en log-loss étaient déjà bien calibrés. Sur le test, ils sous-estiment légèrement le
   total (865,0 xG contre 892 buts), en partie parce que le taux de but du test (9,72 %) dépasse
   celui de l'entraînement (9,40 %).
4. **Biais d'équipe** (page Équipes, sur toutes les données, donc en partie dans l'échantillon) :
   les meilleures équipes dépassent leurs xG (Real Madrid +21,0, AS Roma +17,6, PSG +14,9). Le
   modèle ignore la qualité du tireur.
5. **Données** : la saison 2015/2016 est surreprésentée (38 318 tirs sur 45 827) et la Bundesliga
   est partielle. Il n'y a que des compétitions masculines, et l'annotation « sous pression »
   est manuelle.
6. **Variables absentes** : vitesse et hauteur du ballon, qualité du tireur, suivi continu (le
   freeze frame n'est qu'une image fixe). Le simulateur fixe des hypothèses pour le gardien et les
   défenseurs (médianes de l'entraînement).
7. **Hyperparamètres** : l'optimum de XGBoost est en bord de grille (profondeur 2, 600 arbres) ;
   étendre la grille pourrait gagner quelques millièmes.
8. **Version en ligne** : la licence interdit de redistribuer les données. Les cartes en ligne se
   limitent donc à deux tournois, tandis que les métriques portent sur le jeu complet.
