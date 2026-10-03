#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chargement de modèles avec vérification d'intégrité (atténuation RCE).

`pickle.load` / `joblib.load` exécutent du code arbitraire à la
désérialisation : un fichier `.pkl`/`.joblib` malveillant déposé dans un
répertoire de modèles = exécution de code sur le serveur. Ce module calcule le
SHA-256 du fichier et le compare à un **manifeste de confiance** (JSON
`{chemin_relatif: sha256}`) AVANT de charger.

Modes (opt-in, pour ne pas casser un déploiement existant) :
  - Manifeste présent + hash correspondant  -> chargement autorisé.
  - Manifeste présent + hash absent/différent -> refus (ou warning en non strict).
  - Pas de manifeste -> mode non strict : avertissement + chargement
    (comportement actuel) ; mode strict (EUROMILLIONS_STRICT_MODELS=1) : refus.

Générer le manifeste des modèles de confiance :
    python3 script/generate_model_manifest.py
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

logger = logging.getLogger("SafeModelLoader")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "config" / "model_manifest.json"


class ModelIntegrityError(Exception):
    """Levée quand l'intégrité d'un modèle ne peut pas être garantie."""


def strict_mode() -> bool:
    return os.environ.get("EUROMILLIONS_STRICT_MODELS", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 hexadécimal d'un fichier, lu par blocs (gros fichiers OK)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


def load_manifest(manifest_path: str | Path | None = None) -> Optional[Dict[str, str]]:
    """Charge le manifeste JSON de hashes, ou None s'il est absent/illisible."""
    path = Path(manifest_path) if manifest_path else DEFAULT_MANIFEST
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        files = data.get("files", data) if isinstance(data, dict) else {}
        return {str(k): str(v).lower() for k, v in files.items()}
    except (json.JSONDecodeError, OSError, AttributeError) as exc:
        logger.warning("Manifeste illisible (%s): %s", path, exc)
        return None


def _manifest_key(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return path.name


def _expected_hash(
    path: Path, manifest: Optional[Dict[str, str]]
) -> Tuple[Optional[str], str]:
    """Hash attendu pour ce fichier d'après le manifeste, ou (None, raison).

    La clé est le chemin relatif au dépôt (identique côté génération). Pas de
    repli sur le nom de base seul, pour ne pas accepter un homonyme par hasard.
    """
    if manifest is None:
        return None, "manifeste-absent"
    expected = manifest.get(_manifest_key(path))
    if not expected:
        return None, "hash-non-reference"
    return expected.lower(), "ok"


def verify_file(
    path: str | Path, manifest: Optional[Dict[str, str]]
) -> Tuple[bool, str]:
    """Vérifie un fichier contre le manifeste. Retourne (ok, raison)."""
    p = Path(path)
    if not p.exists():
        return False, "fichier-absent"
    expected, reason = _expected_hash(p, manifest)
    if expected is None:
        return False, reason
    if sha256_file(p).lower() != expected:
        return False, "hash-different"
    return True, "ok"


def safe_load(
    path: str | Path,
    loader: Callable[[bytes], object],
    manifest_path: str | Path | None = None,
    strict: Optional[bool] = None,
):
    """Charge `path` via `loader` seulement après vérification d'intégrité.

    Anti-TOCTOU : le fichier est lu UNE seule fois en mémoire ; l'octet haché
    est exactement l'octet désérialisé (`loader` reçoit les octets, pas un
    chemin ré-ouvert après la vérification).

    - strict=True  : refuse tout fichier non vérifié (lève ModelIntegrityError).
    - strict=False : avertit et charge quand même (rétrocompatible).
    - strict=None  : valeur depuis EUROMILLIONS_STRICT_MODELS.
    """
    if strict is None:
        strict = strict_mode()
    manifest = load_manifest(manifest_path)
    p = Path(path)

    data: Optional[bytes] = None
    if not p.exists():
        ok, reason = False, "fichier-absent"
    else:
        data = p.read_bytes()
        expected, reason = _expected_hash(p, manifest)
        if expected is None:
            ok = False
        else:
            ok = hashlib.sha256(data).hexdigest().lower() == expected
            reason = "ok" if ok else "hash-different"

    if ok and data is not None:
        logger.info("Intégrité OK: %s", p)
        return loader(data)

    msg = f"Intégrité non vérifiée pour {p} (raison={reason})"
    if strict:
        raise ModelIntegrityError(
            msg + ". Mode strict actif : chargement refusé. "
            "Régénérez le manifeste avec generate_model_manifest.py."
        )
    logger.warning(
        "%s. Mode non strict : chargement malgré tout. "
        "Définissez EUROMILLIONS_STRICT_MODELS=1 pour refuser.", msg
    )
    if data is None:
        data = p.read_bytes()
    return loader(data)


def safe_joblib_load(path, manifest_path=None, strict: Optional[bool] = None):
    """joblib.load après vérification d'intégrité (import paresseux de joblib)."""
    import joblib  # import paresseux : évite la dépendance à l'import du module

    return safe_load(
        path, lambda data: joblib.load(io.BytesIO(data)), manifest_path, strict
    )


def safe_pickle_load(path, manifest_path=None, strict: Optional[bool] = None):
    """pickle.load après vérification d'intégrité."""
    import pickle

    return safe_load(path, lambda data: pickle.loads(data), manifest_path, strict)
