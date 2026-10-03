# Audit de sécurité — Projet EuroMillions

Audit en lecture seule (aucune modification de code applicatif). Portée : API
Flask, déploiement, secrets, fichiers versionnés, chargement de modèles, qualité.

## Synthèse

Application Flask exposée publiquement (VPS `107.189.17.46:5002`, reverse-proxy
IIS). Problèmes majeurs **structurels** : API 100 % non authentifiée sans
rate-limiting, serveur de développement Flask en production, désérialisation
`pickle`/`joblib` de ~198 fichiers `.pkl` versionnés (vecteur RCE), CORS en
wildcard, service Windows en `SYSTEM`. **Aucun secret réel en dur** n'a été
trouvé. Le dépôt est alourdi par ~118 Mo de vidéos `.webm` (`.git` ≈ 123 Mo).

## Top 5 priorisé

| # | Correctif | Emplacement | Sévérité |
|---|-----------|-------------|----------|
| 1 | Remplacer le serveur de dev Flask par gunicorn/waitress | `api_server.py:582`, `euromillions-api.service:20` | ÉLEVÉE |
| 2 | Authentifier l'API + rate-limiting + CORS restreint | `api_server.py:46,219,331` | ÉLEVÉE |
| 3 | Ne pas exécuter le service Windows en `SYSTEM/HIGHEST` | `install_windows_service.bat:121` | ÉLEVÉE |
| 4 | Sortir les `.pkl`/vidéos de git + sécuriser le chargement des modèles | `.gitignore:18-23`, `script/*` | ÉLEVÉE |
| 5 | Chiffrer le trafic (fin du HTTP clair backend + fallback client) | `web.config:14`, `euromillion.html:316` | MOYENNE |

## Détail des constats

### API Flask (`api_server.py`)
- **F1 — Aucune authentification (ÉLEVÉE)** : tous les `@app.route` sont ouverts
  (`/api/predict` l.219, `/api/predict/simple` l.331, `/api/status` l.153).
  *Correctif* : clé d'API comparée en temps constant, ou auth au reverse-proxy ;
  a minima, pare-feu sur le port 5002.
- **F2 — Pas de rate-limiting + calcul lourd par requête = DoS (ÉLEVÉE)** :
  `/api/predict` instancie le prédicteur avec quantique activé (l.266-277) à
  chaque POST anonyme. *Correctif* : `flask-limiter`, timeout, quantique off par
  défaut.
- **F3 — CORS wildcard `*` (ÉLEVÉE)** : `CORS(app)` (l.46) + fallback
  `Access-Control-Allow-Origin: *` (l.51-53). *Correctif* : restreindre aux
  origines légitimes.
- **F4 — Serveur de dev en production (ÉLEVÉE)** : `app.run(..., debug=...)`
  (l.582). *Correctif* : gunicorn/waitress derrière le proxy.
- **F5 — `debug` dérivé d'une variable d'env (MOYENNE)** : `FLASK_ENV=development`
  exposerait la console Werkzeug (RCE via PIN) (l.562). *Correctif* : forcer
  `debug=False`.
- **F6 — Fuite de chemins/erreurs internes (MOYENNE)** : `/api/status` renvoie
  des chemins absolus et l'IP (l.161-172) ; `str(e)` renvoyé au client (l.97,
  131, 247, 327). *Correctif* : messages génériques, détails en logs.
- **F7 — `GAME_TYPE` interpolé dans un chemin (FAIBLE)** : non exploitable à
  distance (vient de l'env), à valider par liste blanche (l.61-63).
- *Point positif* : la validation du payload `predict` est correcte
  (`combinations` 1-20, `method` sur liste blanche, l.230-241) ; pas d'`eval`/
  `exec`/`subprocess`.

### Secrets & déploiement
- **F8 — Service Windows en `SYSTEM/HIGHEST` (ÉLEVÉE)** :
  `install_windows_service.bat:121`. *Correctif* : compte de service dédié à
  privilèges minimaux. (Le service Linux `euromillions-api.service` est correct :
  `User=www-data`, `NoNewPrivileges=true`, `PrivateTmp=true`.)
- **F9 — HTTP clair backend + `X-Forwarded-Proto` forcé (MOYENNE)** :
  `web.config:14-16`. *Correctif* : TLS de bout en bout ou backend sur
  `127.0.0.1`.
- **F10 — Fallback HTTP non chiffré côté client (MOYENNE)** :
  `euromillion.html:315-318` (`http://107.189.17.46:5002`), risque mixed-content.
- **F11 — IP publique codée en dur partout (MOYENNE)** : `api_server.py:64`,
  `.env.example:12`, `web.config`, `euromillion.html`, `test_reverse_proxy.ps1`,
  `train_local.bat`. *Correctif* : centraliser en configuration.
- **F12 — `OPENAI_API_KEY` déclaré mais jamais lu (FAIBLE/MOYENNE)** :
  `ai_reflection_encoder.py` initialise `api_key: ''` et ne lit jamais
  `os.environ`. *Correctif* : charger via `os.environ.get`, jamais en dur.
- *Point positif* : aucun secret réel versionné ; `.env` absent et ignoré.

### Fichiers volumineux versionnés
- **F13 — 118 Mo de `.webm` + 198 `.pkl` + 3 `.joblib` committés (MOYENNE)** :
  `.gitignore:18-23` désactive explicitement leur exclusion. *Correctif* : Git
  LFS/stockage objet, purge d'historique (`git filter-repo`).

### Chargement de modèles — RCE par désérialisation (ÉLEVÉE)
`pickle.load`/`joblib.load` exécutent du code arbitraire :
`script/video_embeddings_loader.py:49`, `script/patch_video_integration.py:220`,
`script/meta_model_fusion.py:553`, `script/euromillions_analyzer.py:302,311,3362`,
`script/incremental_learning.py:74,93,104,121`. Les modèles sont transférés par
scp depuis un PC local : tout `.pkl` malveillant déposé dans ces dossiers = RCE.
*Correctif* : vérification d'intégrité (hash/signature), permissions d'écriture
restreintes, formats sûrs (`safetensors`, ONNX, JSON). `torch.load` n'est pas
utilisé.

### Qualité / dépendances
- **F14 — `requirements.txt` surdimensionné, versions non figées (MOYENNE)** :
  mélange runtime ML lourd, outils de dev et Jupyter en dépendances d'exécution.
  *Correctif* : séparer `requirements-dev.txt`, épingler les versions.
- **F15 — Code mort / artefacts (FAIBLE)** : `command.txt` (fragments de session)
  à supprimer du dépôt.

---

*Les correctifs ci-dessus n'ont pas été appliqués dans ce lot de modifications
(hors périmètre « analyse honnête »). Ils sont documentés ici pour être traités
séparément, idéalement par ordre du top 5.*
