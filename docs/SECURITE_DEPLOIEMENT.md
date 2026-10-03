# Déploiement durci de l'API EuroMillions

Ce guide explique comment activer les protections ajoutées (toutes **opt-in**,
donc sans impact tant qu'on ne les active pas) et comment réaliser les étapes
d'infrastructure dans le bon ordre pour **ne pas couper le site en production**.

Voir l'inventaire des vulnérabilités dans [`AUDIT_SECURITE.md`](AUDIT_SECURITE.md).

## 1. Serveur WSGI de production (remplace le serveur de dev Flask)

Le serveur de développement Flask (`app.run`) ne doit jamais être exposé. Un
point d'entrée WSGI est fourni : [`wsgi.py`](../wsgi.py).

```bash
# Linux
gunicorn -w 2 -k gthread -b 127.0.0.1:5002 wsgi:app

# Windows
waitress-serve --listen=127.0.0.1:5002 wsgi:app
```

`gunicorn` / `waitress` sont dans `requirements_api.txt`. Lier à `127.0.0.1`
et n'exposer au public que **via le reverse-proxy** (IIS/nginx) en HTTPS.

> ⚠️ **Mise à jour automatique des tirages.** Lancé en dev (`python
> api_server.py`), le serveur exécute `initialize_system()` (scraping FDJ +
> scheduler mardis/vendredis 22h). Sous gunicorn/waitress, `wsgi:app`
> n'exécute cette initialisation **que si** `WSGI_RUN_INIT=1`. Pour conserver
> la MAJ auto, définir `WSGI_RUN_INIT=1` **avec un seul worker** (`-w 1`) afin
> de ne pas dupliquer le scheduler, ou — recommandé en multi-worker —
> externaliser la MAJ dans un cron / systemd-timer appelant
> `python3 check_and_train.py` indépendamment du serveur web.

## 2. Leviers de sécurité applicatifs (variables d'environnement)

| Variable | Défaut | Effet |
|----------|--------|-------|
| `API_KEY` | vide | Si définie, exige l'en-tête `X-API-Key` sur `/api/predict`. |
| `CORS_ORIGINS` | domaine + localhost | Origines autorisées (CSV). Remplace le wildcard `*`. |
| `RATE_LIMIT_PER_MIN` | vide (désactivé) | Limite de requêtes/min/IP sur les routes de prédiction. |
| `TRUSTED_PROXIES` | vide | IP des proxies dont on accepte le `X-Forwarded-For`. |
| `EUROMILLIONS_STRICT_MODELS` | vide | `1` = refuse tout modèle non listé au manifeste (anti-RCE). |

Notes :
- L'authentification ne s'applique **pas** à `/api/predict/simple` (route
  appelée par le site public sans clé), qui reste protégée par le rate-limiting.
- `debug` est désormais forcé à `False` ; ne l'activez jamais en production.
- **Posture par défaut (choix assumé, non cassant)** : sans `API_KEY` ni
  `RATE_LIMIT_PER_MIN`, `/api/predict` reste ouvert et non limité (la surface
  DoS de la pile lourde est inchangée). En production, définir au minimum
  `RATE_LIMIT_PER_MIN` (avec `TRUSTED_PROXIES`), et `API_KEY` si le front peut
  transmettre la clé. La route lourde `/api/predict` et la route rapide
  `/api/predict/simple` partagent le même compteur ; prévoir un plafond adapté.

## 3. Rate-limiting derrière un reverse-proxy (IMPORTANT)

Derrière IIS/nginx, `request.remote_addr` est l'IP **du proxy**, pas du client.
Activer le rate-limiting sans IP cliente distincte étranglerait **tout le
trafic ensemble**. Procédure sûre :

1. Faire transmettre l'IP réelle par le proxy. Exemple IIS (`web.config`) :
   ```xml
   <set name="HTTP_X_FORWARDED_FOR" value="{REMOTE_ADDR}" />
   ```
   (nginx : `proxy_set_header X-Forwarded-For $remote_addr;`)
2. Déclarer l'IP du proxy dans `TRUSTED_PROXIES` (sinon le `X-Forwarded-For`
   est ignoré, car un client pourrait l'usurper).
3. Seulement ensuite, définir `RATE_LIMIT_PER_MIN` (ex. `30`).

> Le rate-limiting est en mémoire (un worker). Pour plusieurs workers/instances,
> utiliser un store partagé (Redis) — non inclus ici.

## 4. Chargement de modèles sécurisé (anti-RCE pickle) — déploiement en 2 temps

Par défaut (sans manifeste), le chargement reste permissif (avertissement) pour
ne rien casser. Pour activer la protection :

```bash
# 1) Générer le manifeste SHA-256 des modèles de CONFIANCE (sur la machine sûre)
python3 script/generate_model_manifest.py

# 2) Activer le mode strict sur le serveur
export EUROMILLIONS_STRICT_MODELS=1
```

En mode strict, tout `.pkl`/`.joblib` absent du manifeste ou altéré est refusé
(`ModelIntegrityError`) au lieu d'être désérialisé. Régénérer le manifeste après
chaque réentraînement légitime.

## 5. Étapes d'infrastructure (hors de cette PR — à faire au déploiement)

Ces changements sont **cassants si mal ordonnés** ; ils relèvent d'un runbook ops
séquencé, pas d'un commit de code. Ordre recommandé :

1. **Mettre le reverse-proxy en HTTPS de bout en bout** (certificat valide côté
   proxy, puis chiffrer le lien proxy → API ou co-localiser).
2. Basculer le proxy pour cibler l'API en interne (`127.0.0.1:5002`).
3. **Seulement après**, lier l'API à `127.0.0.1` (via gunicorn/waitress) et
   **fermer le port 5002 au pare-feu** côté public. (Binder en loopback avant
   que le proxy ne cible l'interne couperait le site.)
4. Windows : exécuter le service sous un **compte dédié à privilèges réduits**,
   jamais `SYSTEM/HIGHEST` (cf. `install_windows_service.bat`). Linux : le
   service systemd est déjà durci (`User=www-data`, `NoNewPrivileges`,
   `PrivateTmp`).
5. Externaliser l'IP publique (`SERVER_IP`) via l'environnement, ne pas la coder
   en dur.

## 6. Ne rien retirer de risqué dans cette PR

`torch`/`pennylane` restent dans `requirements_api.txt` : le code force
`use_quantum=True`, les retirer dégraderait silencieusement les prédictions.
Leur isolation est un chantier séparé.
