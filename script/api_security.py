#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Helpers de sécurité pour l'API EuroMillions — SANS dépendance Flask.

Logique pure et testable (bibliothèque standard uniquement) :
  - comparaison de clé d'API en temps constant (anti timing-attack) ;
  - limiteur de débit en mémoire par IP (fenêtre glissante, thread-safe) ;
  - parsing/validation des origines CORS (avec motifs `*`) ;
  - réponses d'erreur assainies (message générique + identifiant de corrélation).

Le câblage Flask (décorateurs, after_request, app.run) vit dans
`api_server.py` et consomme ces helpers. Ce découpage garde ce module
importable et testable même quand Flask n'est pas installé.

Toutes les protections « cassables » sont OPT-IN par variable d'environnement
pour ne pas interrompre un déploiement existant :
  - API_KEY (ou EUROMILLIONS_API_KEY) : si définie, l'auth est exigée.
  - CORS_ORIGINS : liste CSV d'origines autorisées (défaut : domaine + localhost).
  - RATE_LIMIT_PER_MIN : nombre de requêtes/minute/IP (défaut 30).
"""

from __future__ import annotations

import hmac
import os
import threading
import time
import uuid
from collections import deque
from typing import Callable, Deque, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Authentification par clé d'API (opt-in)
# ---------------------------------------------------------------------------
API_KEY_HEADER = "X-API-Key"


def get_expected_api_key() -> Optional[str]:
    """Clé attendue, ou None si l'auth n'est pas configurée (prod inchangée)."""
    key = os.environ.get("API_KEY") or os.environ.get("EUROMILLIONS_API_KEY")
    key = (key or "").strip()
    return key or None


def auth_enabled() -> bool:
    return get_expected_api_key() is not None


def constant_time_compare(provided: Optional[str], expected: Optional[str]) -> bool:
    """Comparaison en temps constant (hmac.compare_digest), robuste à None."""
    if not provided or not expected:
        return False
    try:
        return hmac.compare_digest(str(provided), str(expected))
    except (TypeError, ValueError):
        return False


def check_api_key(provided: Optional[str]) -> Tuple[bool, str]:
    """Vérifie la clé fournie.

    Retourne (autorisé, raison). Si aucune clé n'est configurée, autorise mais
    signale "auth-disabled" pour que l'appelant journalise un avertissement.
    """
    expected = get_expected_api_key()
    if expected is None:
        return True, "auth-disabled"
    if constant_time_compare(provided, expected):
        return True, "ok"
    return False, "invalid-or-missing-key"


# ---------------------------------------------------------------------------
# CORS : origines autorisées
# ---------------------------------------------------------------------------
DEFAULT_CORS_ORIGINS = [
    "https://kenopredictionia.fr",
    "https://www.kenopredictionia.fr",
    "http://localhost:*",
    "http://127.0.0.1:*",
]


def parse_cors_origins(raw: Optional[str] = None) -> List[str]:
    """Parse la liste CSV d'origines ; défaut sûr si non définie."""
    if raw is None:
        raw = os.environ.get("CORS_ORIGINS", "")
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    return origins or list(DEFAULT_CORS_ORIGINS)


def _origin_matches(origin: str, pattern: str) -> bool:
    """Match exact, ou avec un unique joker `*` (ex. http://localhost:*)."""
    if pattern == "*":
        return True
    if "*" not in pattern:
        return origin == pattern
    prefix, _, suffix = pattern.partition("*")
    return (
        origin.startswith(prefix)
        and origin.endswith(suffix)
        and len(origin) >= len(prefix) + len(suffix)
    )


def is_origin_allowed(origin: Optional[str], allowed: Optional[List[str]] = None) -> bool:
    if not origin:
        return False
    if allowed is None:
        allowed = parse_cors_origins()
    return any(_origin_matches(origin, p) for p in allowed)


# ---------------------------------------------------------------------------
# Limiteur de débit en mémoire (fenêtre glissante, thread-safe)
# ---------------------------------------------------------------------------
class RateLimiter:
    """Limite `max_requests` par `window_seconds` et par clé (IP).

    En mémoire, sans dépendance externe. `clock` est injectable pour les tests.
    Convient à un worker unique ; pour plusieurs workers/instances, préférer
    un store partagé (Redis). Documenté comme tel.
    """

    def __init__(
        self,
        max_requests: int = 30,
        window_seconds: float = 60.0,
        clock: Callable[[], float] = time.time,
        max_keys: int = 10000,
    ) -> None:
        self.max_requests = max(1, int(max_requests))
        self.window = float(window_seconds)
        self._clock = clock
        self._max_keys = max_keys
        self._hits: Dict[str, Deque[float]] = {}
        self._lock = threading.Lock()

    def _prune(self, dq: Deque[float], now: float) -> None:
        threshold = now - self.window
        while dq and dq[0] <= threshold:
            dq.popleft()

    def allow(self, key: str) -> bool:
        """True si la requête est autorisée, en l'enregistrant le cas échéant."""
        now = self._clock()
        with self._lock:
            dq = self._hits.get(key)
            if dq is None:
                # Garde-fou mémoire : éviction grossière si trop de clés.
                if len(self._hits) >= self._max_keys:
                    self._hits.clear()
                dq = deque()
                self._hits[key] = dq
            self._prune(dq, now)
            if len(dq) >= self.max_requests:
                return False
            dq.append(now)
            return True

    def retry_after(self, key: str) -> int:
        """Secondes à attendre avant une nouvelle tentative (approximatif)."""
        now = self._clock()
        with self._lock:
            dq = self._hits.get(key)
            if not dq:
                return 0
            self._prune(dq, now)
            if len(dq) < self.max_requests:
                return 0
            return max(1, int(self.window - (now - dq[0])) + 1)


# ---------------------------------------------------------------------------
# Réponses d'erreur assainies
# ---------------------------------------------------------------------------
def new_correlation_id() -> str:
    """Identifiant court pour relier une erreur client à la trace serveur."""
    return uuid.uuid4().hex[:12]


def sanitized_error(
    message: str = "Une erreur interne est survenue.",
    error_id: Optional[str] = None,
) -> Dict[str, str]:
    """Corps d'erreur JSON générique : aucun détail interne ni chemin exposé."""
    return {
        "status": "error",
        "message": message,
        "error_id": error_id or new_correlation_id(),
    }
