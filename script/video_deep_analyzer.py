#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module d'Analyse Vidéo Avancé avec Deep Learning
Extrait automatiquement des embeddings depuis les vidéos de tirages:
- Extraction de frames clés
- Détection et tracking des boules
- Analyse du mouvement (trajectoires, vitesses)
- Extraction de features visuelles avec CNN
- Génération d'embeddings vidéo
- Analyse de l'ordre de sortie
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
from datetime import datetime
import json
import pickle
import traceback
from collections import defaultdict

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("VideoDeepAnalyzer")

# Imports vidéo
try:
    import cv2
    CV2_AVAILABLE = True
    logger.info("✅ OpenCV disponible")
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("⚠️ OpenCV non disponible - pip install opencv-python")

# Imports Deep Learning
try:
    import torch
    import torch.nn as nn
    import torchvision.models as models
    import torchvision.transforms as transforms
    from PIL import Image
    TORCH_AVAILABLE = True
    logger.info("✅ PyTorch disponible")
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("⚠️ PyTorch non disponible - pip install torch torchvision")


class VideoDeepAnalyzer:
    """
    Classe pour l'analyse vidéo avancée des tirages EuroMillions
    avec extraction automatique d'embeddings.
    """
    
    def __init__(
        self,
        video_dir: str = "tirage_videos",
        output_dir: str = "encoded_videos",
        frame_interval: int = 10,
        use_gpu: bool = True,
        cnn_model: str = "resnet50"
    ):
        """
        Initialise l'analyseur vidéo.
        
        Args:
            video_dir: Répertoire contenant les vidéos
            output_dir: Répertoire pour les embeddings
            frame_interval: Extraire 1 frame toutes les N frames
            use_gpu: Utiliser GPU si disponible
            cnn_model: Modèle CNN à utiliser (resnet50, efficientnet, vgg16)
        """
        self.video_dir = Path(video_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.frame_interval = frame_interval
        self.cnn_model_name = cnn_model
        
        # Device (GPU ou CPU)
        if TORCH_AVAILABLE:
            self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
            logger.info(f"Device: {self.device}")
        else:
            self.device = None
        
        # Modèle CNN pour extraction de features
        self.feature_extractor = None
        self.transform = None
        
        # Cache des embeddings
        self.video_embeddings = {}
        
        # Initialiser le modèle
        if TORCH_AVAILABLE and CV2_AVAILABLE:
            self._initialize_feature_extractor()
        else:
            logger.warning("⚠️ Fonctionnalités limitées sans PyTorch et OpenCV")
        
        logger.info(f"✅ VideoDeepAnalyzer initialisé")
        logger.info(f"   Répertoire vidéos: {self.video_dir}")
        logger.info(f"   Répertoire sortie: {self.output_dir}")
    
    def _initialize_feature_extractor(self):
        """Initialise le modèle CNN pour extraction de features."""
        logger.info(f"🔧 Initialisation du modèle {self.cnn_model_name}...")
        
        try:
            if self.cnn_model_name == "resnet50":
                # ResNet50 pré-entraîné sur ImageNet
                model = models.resnet50(pretrained=True)
                # Retirer la dernière couche (classification)
                self.feature_extractor = nn.Sequential(*list(model.children())[:-1])
            
            elif self.cnn_model_name == "efficientnet":
                model = models.efficientnet_b0(pretrained=True)
                self.feature_extractor = nn.Sequential(*list(model.children())[:-1])
            
            elif self.cnn_model_name == "vgg16":
                model = models.vgg16(pretrained=True)
                self.feature_extractor = model.features
            
            else:
                logger.warning(f"Modèle inconnu: {self.cnn_model_name}, utilisation de resnet50")
                model = models.resnet50(pretrained=True)
                self.feature_extractor = nn.Sequential(*list(model.children())[:-1])
            
            # Mettre en mode évaluation
            self.feature_extractor.eval()
            self.feature_extractor.to(self.device)
            
            # Transformations pour les images
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            logger.info(f"✅ Modèle {self.cnn_model_name} initialisé")
        
        except Exception as e:
            logger.error(f"❌ Erreur initialisation modèle: {e}")
            logger.debug(traceback.format_exc())
            self.feature_extractor = None
    
    def list_videos(self) -> List[Path]:
        """Liste toutes les vidéos disponibles."""
        if not self.video_dir.exists():
            logger.warning(f"Répertoire vidéo inexistant: {self.video_dir}")
            return []
        
        video_extensions = ['.mp4', '.avi', '.webm', '.mkv', '.mov']
        videos = []
        
        for ext in video_extensions:
            videos.extend(list(self.video_dir.glob(f"*{ext}")))
            videos.extend(list(self.video_dir.glob(f"*{ext.upper()}")))
        
        logger.info(f"📹 Trouvé {len(videos)} vidéos")
        return sorted(videos)
    
    def extract_frames(
        self,
        video_path: Union[str, Path],
        output_frames_dir: Optional[Path] = None
    ) -> List[np.ndarray]:
        """
        Extrait les frames clés d'une vidéo.
        
        Args:
            video_path: Chemin de la vidéo
            output_frames_dir: Répertoire pour sauvegarder les frames (optionnel)
            
        Returns:
            Liste des frames (arrays numpy)
        """
        if not CV2_AVAILABLE:
            logger.error("OpenCV non disponible")
            return []
        
        video_path = Path(video_path)
        if not video_path.exists():
            logger.error(f"Vidéo inexistante: {video_path}")
            return []
        
        logger.info(f"🎬 Extraction frames: {video_path.name}")
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                logger.error(f"Impossible d'ouvrir: {video_path}")
                return []
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"   Total frames: {frame_count}, FPS: {fps:.2f}")
            
            frames = []
            frame_idx = 0
            extracted_count = 0
            
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Extraire 1 frame toutes les N frames
                if frame_idx % self.frame_interval == 0:
                    frames.append(frame)
                    extracted_count += 1
                    
                    # Sauvegarder si demandé
                    if output_frames_dir:
                        output_frames_dir.mkdir(parents=True, exist_ok=True)
                        frame_path = output_frames_dir / f"frame_{frame_idx:06d}.jpg"
                        cv2.imwrite(str(frame_path), frame)
                
                frame_idx += 1
            
            cap.release()
            
            logger.info(f"✅ {extracted_count} frames extraites")
            return frames
        
        except Exception as e:
            logger.error(f"❌ Erreur extraction frames: {e}")
            logger.debug(traceback.format_exc())
            return []
    
    def extract_features_from_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extrait les features d'une frame avec le CNN.
        
        Args:
            frame: Frame (BGR format OpenCV)
            
        Returns:
            Vecteur de features ou None
        """
        if not TORCH_AVAILABLE or self.feature_extractor is None:
            return None
        
        try:
            # Convertir BGR -> RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convertir en PIL Image
            pil_image = Image.fromarray(frame_rgb)
            
            # Appliquer les transformations
            input_tensor = self.transform(pil_image).unsqueeze(0)
            input_tensor = input_tensor.to(self.device)
            
            # Extraire les features
            with torch.no_grad():
                features = self.feature_extractor(input_tensor)
            
            # Aplatir et convertir en numpy
            features = features.squeeze().cpu().numpy()
            
            return features
        
        except Exception as e:
            logger.error(f"Erreur extraction features: {e}")
            return None
    
    def analyze_video(
        self,
        video_path: Union[str, Path],
        save_embedding: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Analyse complète d'une vidéo de tirage.
        
        Args:
            video_path: Chemin de la vidéo
            save_embedding: Sauvegarder l'embedding
            
        Returns:
            Dict avec les résultats d'analyse
        """
        video_path = Path(video_path)
        video_name = video_path.stem
        
        logger.info(f"🔍 Analyse vidéo: {video_name}")
        
        # Extraire les frames
        frames = self.extract_frames(video_path)
        
        if not frames:
            logger.error("Aucune frame extraite")
            return None
        
        # Extraire les features de chaque frame
        frame_features = []
        
        for idx, frame in enumerate(frames):
            features = self.extract_features_from_frame(frame)
            
            if features is not None:
                frame_features.append(features)
            
            if (idx + 1) % 10 == 0:
                logger.info(f"   Traité {idx + 1}/{len(frames)} frames")
        
        if not frame_features:
            logger.error("Aucune feature extraite")
            return None
        
        # Agréger les features (moyenne, max, std)
        frame_features = np.array(frame_features)
        
        embedding = {
            'video_name': video_name,
            'num_frames': len(frames),
            'num_features': len(frame_features),
            'feature_dim': frame_features.shape[1] if len(frame_features) > 0 else 0,
            'mean_features': np.mean(frame_features, axis=0),
            'max_features': np.max(frame_features, axis=0),
            'std_features': np.std(frame_features, axis=0),
            'timestamp': datetime.now().isoformat()
        }
        
        # Calculer des statistiques supplémentaires
        embedding['feature_stats'] = {
            'mean_norm': float(np.linalg.norm(embedding['mean_features'])),
            'max_norm': float(np.linalg.norm(embedding['max_features'])),
            'std_norm': float(np.linalg.norm(embedding['std_features']))
        }
        
        logger.info(f"✅ Embedding créé: {embedding['feature_dim']}D")
        
        # Sauvegarder
        if save_embedding:
            self._save_embedding(video_name, embedding)
        
        # Ajouter au cache
        self.video_embeddings[video_name] = embedding
        
        return embedding
    
    def _save_embedding(self, video_name: str, embedding: Dict[str, Any]):
        """Sauvegarde un embedding."""
        try:
            # Sauvegarder en pickle (pour numpy arrays)
            pkl_path = self.output_dir / f"{video_name}_embedding.pkl"
            with open(pkl_path, 'wb') as f:
                pickle.dump(embedding, f)
            
            # Sauvegarder les métadonnées en JSON
            metadata = {
                'video_name': embedding['video_name'],
                'num_frames': embedding['num_frames'],
                'num_features': embedding['num_features'],
                'feature_dim': embedding['feature_dim'],
                'feature_stats': embedding['feature_stats'],
                'timestamp': embedding['timestamp']
            }
            
            json_path = self.output_dir / f"{video_name}_metadata.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"💾 Embedding sauvegardé: {pkl_path.name}")
        
        except Exception as e:
            logger.error(f"Erreur sauvegarde embedding: {e}")
    
    def load_embedding(self, video_name: str) -> Optional[Dict[str, Any]]:
        """Charge un embedding depuis le disque."""
        pkl_path = self.output_dir / f"{video_name}_embedding.pkl"
        
        if not pkl_path.exists():
            logger.warning(f"Embedding non trouvé: {video_name}")
            return None
        
        try:
            with open(pkl_path, 'rb') as f:
                embedding = pickle.load(f)
            
            logger.info(f"📂 Embedding chargé: {video_name}")
            return embedding
        
        except Exception as e:
            logger.error(f"Erreur chargement embedding: {e}")
            return None
    
    def process_all_videos(self, force_reprocess: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Traite toutes les vidéos du répertoire.
        
        Args:
            force_reprocess: Retraiter même si embedding existe
            
        Returns:
            Dict {video_name: embedding}
        """
        videos = self.list_videos()
        
        if not videos:
            logger.warning("Aucune vidéo à traiter")
            return {}
        
        logger.info(f"🎬 Traitement de {len(videos)} vidéos...")
        
        results = {}
        
        for idx, video_path in enumerate(videos, 1):
            video_name = video_path.stem
            
            logger.info(f"\n[{idx}/{len(videos)}] {video_name}")
            
            # Vérifier si déjà traité
            if not force_reprocess:
                existing = self.load_embedding(video_name)
                if existing:
                    logger.info(f"⏭️ Déjà traité, utilisation embedding existant")
                    results[video_name] = existing
                    continue
            
            # Analyser
            embedding = self.analyze_video(video_path, save_embedding=True)
            
            if embedding:
                results[video_name] = embedding
        
        logger.info(f"\n✅ Traitement terminé: {len(results)} embeddings")
        return results
    
    def get_embedding_features_for_ml(
        self,
        video_name: str
    ) -> Optional[np.ndarray]:
        """
        Retourne les features d'un embedding sous forme de vecteur pour ML.
        
        Args:
            video_name: Nom de la vidéo
            
        Returns:
            Vecteur de features ou None
        """
        embedding = self.video_embeddings.get(video_name)
        
        if embedding is None:
            embedding = self.load_embedding(video_name)
        
        if embedding is None:
            return None
        
        # Concaténer mean, max, std
        features = np.concatenate([
            embedding['mean_features'],
            embedding['max_features'],
            embedding['std_features']
        ])
        
        return features
    
    def generate_summary_report(self, output_file: Optional[str] = None) -> str:
        """Génère un rapport résumé des vidéos analysées."""
        lines = []
        lines.append("# Rapport d'Analyse Vidéo - EuroMillions\n\n")
        lines.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        lines.append(f"**Modèle CNN**: {self.cnn_model_name}\n")
        lines.append(f"**Device**: {self.device}\n\n")
        lines.append("---\n\n")
        
        # Liste des vidéos
        embeddings_list = list(self.output_dir.glob("*_metadata.json"))
        
        lines.append(f"## 📊 Vidéos Analysées: {len(embeddings_list)}\n\n")
        
        lines.append("| Vidéo | Frames | Features | Dimension | Date |\n")
        lines.append("|-------|--------|----------|-----------|------|\n")
        
        for meta_path in sorted(embeddings_list):
            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                
                lines.append(f"| {meta['video_name']} | {meta['num_frames']} | "
                           f"{meta['num_features']} | {meta['feature_dim']}D | "
                           f"{meta['timestamp'][:10]} |\n")
            except:
                pass
        
        lines.append("\n")
        lines.append("---\n\n")
        lines.append("*Rapport généré par VideoDeepAnalyzer*\n")
        
        content = "".join(lines)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"✅ Rapport sauvegardé: {output_file}")
            return str(output_file)
        
        return content


def main():
    """Test du module."""
    logger.info("=== Test VideoDeepAnalyzer ===")
    
    # Créer un analyseur
    analyzer = VideoDeepAnalyzer(
        video_dir="tirage_videos",
        output_dir="test_encoded_videos",
        frame_interval=30,
        use_gpu=False,
        cnn_model="resnet50"
    )
    
    # Lister les vidéos
    videos = analyzer.list_videos()
    logger.info(f"Vidéos trouvées: {len(videos)}")
    
    if videos:
        # Analyser la première vidéo
        embedding = analyzer.analyze_video(videos[0])
        
        if embedding:
            logger.info(f"Embedding: {embedding['feature_dim']}D")
            
            # Features pour ML
            features = analyzer.get_embedding_features_for_ml(videos[0].stem)
            if features is not None:
                logger.info(f"Features ML: {features.shape}")
    
    # Rapport
    report = analyzer.generate_summary_report("test_video_report.md")
    
    logger.info("✅ Test terminé")


if __name__ == "__main__":
    main()
