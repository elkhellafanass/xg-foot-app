# Déployer l'application sur Streamlit Community Cloud

Le dépôt est prêt : `requirements.txt` est à la racine, les modèles (`models/`) et les résultats
(`reports/`) sont versionnés, et l'application télécharge elle-même les données nécessaires au
premier lancement. Aucun secret n'est requis.

## Pas à pas

1. Ouvrez https://share.streamlit.io et cliquez sur **Continue with GitHub**. Connectez-vous avec le
   compte `elkhellafanass`, puis autorisez Streamlit à accéder à vos dépôts.
2. Cliquez sur **Create app**, puis choisissez **Deploy a public app from GitHub**.
3. Renseignez :
   - **Repository** : `elkhellafanass/xg-foot-app`
   - **Branch** : `main`
   - **Main file path** : `app/streamlit_app.py`
   - **App URL** (facultatif) : par exemple `xg-foot-memoire`
4. Ouvrez **Advanced settings** :
   - **Python version** : **3.12** (version utilisée pour le développement et les tests) ;
   - **Secrets** : laissez vide.
5. Cliquez sur **Deploy**. La première installation prend 2 à 4 minutes.
6. Au premier affichage d'une page qui utilise les tirs (Explorateur, Équipes et joueurs), l'application
   télécharge 115 matchs depuis StatsBomb Open Data, ce qui prend environ 10 à 30 secondes avec une barre
   de progression. Les visites suivantes utilisent le cache.

## Ce qui est téléchargé en ligne, et pourquoi

La licence StatsBomb interdit de redistribuer les données. Le dépôt ne contient donc pas la table
des tirs. En ligne, les pages « Explorateur de tirs » et « Équipes et joueurs » portent sur la
**Coupe du monde 2022** et l'**Euro 2024**, une liste modifiable dans `src/config.py`
(`APP_COMPETITIONS`). Les pages « Accueil », « Modèles » et « Simulateur » utilisent les résultats et
modèles calculés sur le jeu complet (11 compétitions, 45 827 tirs).

## Vérification effectuée avant livraison

Le démarrage a été testé dans un environnement vierge, comme sur Streamlit Cloud :

```powershell
git clone https://github.com/elkhellafanass/xg-foot-app.git xg-test
cd xg-test
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

Dans ce cas, sans `data/processed/shots.parquet`, l'application déclenche le téléchargement en ligne
décrit plus haut, exactement comme sur le cloud.

## En cas de problème

| Symptôme | Cause probable | Solution |
|---|---|---|
| `ModuleNotFoundError` au démarrage | Mauvais fichier principal | Vérifiez *Main file path* = `app/streamlit_app.py` |
| Erreur à l'installation d'un paquet | Version de Python différente | Dans *Settings > General*, choisissez Python 3.12, puis *Reboot app* |
| « Impossible de télécharger les données StatsBomb » | GitHub momentanément indisponible | Rechargez la page plus tard (*Reboot app* si besoin) |
| L'application « dort » | Inactivité prolongée (comportement normal du plan gratuit) | Cliquez sur *Yes, get this app back up!*. Le cache est reconstruit au besoin. |
| Logo absent | Fichier non ajouté | Ajoutez `app/assets/statsbomb_logo.png` (voir README), puis commitez et poussez |

Chaque `git push` sur `main` redéploie l'application automatiquement.
