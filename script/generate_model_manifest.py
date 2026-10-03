#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Génère le manifeste d'intégrité des modèles de confiance (SHA-256).

À exécuter après avoir entraîné/validé des modèles de confiance, pour que
`safe_model_loader` puisse ensuite refuser tout fichier altéré :

    python3 script/generate_model_manifest.py
    python3 script/generate_model_manifest.py --dirs models_euromillions encoded_videos

Écrit `config/model_manifest.json` (clés = chemins relatifs au dépôt).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from safe_model_loader import REPO_ROOT, sha256_file  # type: ignore

DEFAULT_DIRS = [
    "models_euromillions",
    "encoded_videos",
    "resultats_euromillions",
]
PATTERNS = ("*.pkl", "*.joblib")


def _rel(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT)).replace("\\", "/")


def generate(dirs: list[str]) -> dict:
    files: dict[str, str] = {}
    for d in dirs:
        base = (REPO_ROOT / d)
        if not base.exists():
            continue
        for pattern in PATTERNS:
            for f in sorted(base.rglob(pattern)):
                if f.is_file():
                    files[_rel(f)] = sha256_file(f)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": "sha256",
        "count": len(files),
        "files": files,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Générer le manifeste SHA-256 des modèles")
    parser.add_argument("--dirs", nargs="*", default=DEFAULT_DIRS, help="Répertoires à scanner")
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "config" / "model_manifest.json"),
        help="Chemin du manifeste de sortie",
    )
    args = parser.parse_args(argv)

    manifest = generate(args.dirs)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2, sort_keys=True)

    print(f"✅ Manifeste écrit : {out}")
    print(f"   {manifest['count']} fichier(s) de confiance référencé(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
