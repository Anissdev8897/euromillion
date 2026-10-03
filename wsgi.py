#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Point d'entrée WSGI pour serveur de PRODUCTION (gunicorn / waitress).

Le serveur de développement Flask (`app.run`) ne doit pas être exposé en
production. Utiliser un serveur WSGI robuste :

    # Linux (recommandé)
    gunicorn -w 2 -k gthread -b 127.0.0.1:5002 wsgi:app

    # Windows
    waitress-serve --listen=127.0.0.1:5002 wsgi:app

Lier le serveur à 127.0.0.1 et exposer au public uniquement via le
reverse-proxy (IIS/nginx) en HTTPS.

Initialisation lourde (scraping FDJ, scheduler) : désactivée par défaut ici
pour ne pas la dupliquer dans chaque worker. Pour la lancer dans un worker
unique, définir WSGI_RUN_INIT=1.
"""

import logging
import os

from api_server import app  # l'application Flask déjà configurée

logger = logging.getLogger("wsgi")

if os.environ.get("WSGI_RUN_INIT", "").strip().lower() in ("1", "true", "yes", "on"):
    try:
        from api_server import initialize_system

        initialize_system()
    except Exception as exc:  # non bloquant : le service doit rester debout
        logger.warning("initialize_system a échoué (non bloquant): %s", exc)

# `app` est l'objet WSGI exposé aux serveurs gunicorn/waitress.
if __name__ == "__main__":
    # Repli de confort pour un test local ; en prod, utiliser gunicorn/waitress.
    port = int(os.environ.get("PORT", 5002))
    app.run(host="127.0.0.1", port=port, debug=False)
