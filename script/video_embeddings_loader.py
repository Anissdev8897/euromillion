#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper pour charger les embeddings vidéo
À utiliser dans votre script d'entraînement principal
"""

import logging
from pathlib import Path
from typing import Dict, Any

try:
    from safe_model_loader import safe_pickle_load  # type: ignore
except ImportError:  # pragma: no cover - selon le chemin d'import
    from script.safe_model_loader import safe_pickle_load  # type: ignore

logger = logging.getLogger("VideoEmbeddingsLoader")


def load_video_embeddings(encoded_videos_dir: str = "encoded_videos") -> Dict[str, Any]:
    """
    Charge tous les embeddings vidéo depuis le répertoire.
    
    Args:
        encoded_videos_dir: Répertoire contenant les fichiers *_embedding.pkl
        
    Returns:
        Dict {video_name: embedding_dict}
    """
    embeddings = {}
    encoded_path = Path(encoded_videos_dir)
    
    if not encoded_path.exists():
        logger.warning(f"Répertoire {encoded_videos_dir} n'existe pas")
        return embeddings
    
    # Chercher tous les fichiers d'embeddings
    pkl_files = list(encoded_path.glob("*_embedding.pkl"))
    
    if not pkl_files:
        logger.warning(f"Aucun fichier d'embedding trouvé dans {encoded_videos_dir}")
        return embeddings
    
    logger.info(f"🎥 Chargement des embeddings vidéo depuis {encoded_videos_dir}...")
    
    for pkl_file in pkl_files:
        try:
            # Extraire le nom de la vidéo
            video_name = pkl_file.stem.replace("_embedding", "")
            
            # Charger l'embedding après vérification d'intégrité (anti-RCE).
            # En mode strict, un fichier non vérifié lève et est ignoré ci-dessous.
            embedding = safe_pickle_load(pkl_file)

            embeddings[video_name] = embedding
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement de {pkl_file.name}: {e}")
            continue
    
    logger.info(f"✅ {len(embeddings)} embeddings vidéo chargés")
    
    return embeddings


def get_embedding_for_date(embeddings: Dict[str, Any], date_str: str) -> Any:
    """
    Récupère l'embedding correspondant à une date.
    
    Args:
        embeddings: Dict des embeddings
        date_str: Date au format YYYYMMDD
        
    Returns:
        Embedding ou None si non trouvé
    """
    for video_name, embedding in embeddings.items():
        if date_str in video_name:
            return embedding
    
    return None


# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================

if __name__ == "__main__":
    # Configuration du logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Charger les embeddings
    embeddings = load_video_embeddings("encoded_videos")
    
    print(f"\n{'='*80}")
    print(f"EMBEDDINGS VIDÉO CHARGÉS: {len(embeddings)}")
    print(f"{'='*80}\n")
    
    # Afficher quelques exemples
    for i, (video_name, embedding) in enumerate(list(embeddings.items())[:5]):
        print(f"{i+1}. {video_name}")
        print(f"   - Dimension: {embedding['feature_dim']}D")
        print(f"   - Frames: {embedding['num_frames']}")
        print(f"   - Features extraites: {embedding['num_features']}")
        print()
    
    # Test de récupération par date
    test_date = "20251115"
    embedding = get_embedding_for_date(embeddings, test_date)
    
    if embedding:
        print(f"✅ Embedding trouvé pour la date {test_date}")
        print(f"   Video: {embedding['video_name']}")
    else:
        print(f"❌ Aucun embedding trouvé pour la date {test_date}")
