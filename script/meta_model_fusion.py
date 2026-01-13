#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module de Fusion Multi-Modèles Dynamique
Ce module fusionne intelligemment les prédictions de tous les modèles avec:
- Stacking avancé avec méta-modèle
- Blending pondéré dynamique
- Voting intelligent avec poids adaptatifs
- Auto-ajustement des poids après chaque tirage
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
logger = logging.getLogger("MetaModelFusion")

# Imports ML
try:
    from sklearn.ensemble import StackingClassifier, VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    import xgboost as xgb
    import lightgbm as lgb
    SKLEARN_AVAILABLE = True
    logger.info("✅ Bibliothèques ML disponibles")
except ImportError as e:
    SKLEARN_AVAILABLE = False
    logger.error(f"❌ Erreur import ML: {e}")


class MetaModelFusion:
    """
    Classe pour la fusion intelligente de multiples modèles de prédiction.
    Supporte stacking, blending, et voting avec ajustement dynamique des poids.
    """
    
    def __init__(
        self,
        fusion_method: str = "stacking",
        meta_model_type: str = "xgboost",
        auto_adjust: bool = True,
        config_file: Optional[str] = None
    ):
        """
        Initialise le système de fusion multi-modèles.
        
        Args:
            fusion_method: Méthode de fusion ("stacking", "blending", "voting")
            meta_model_type: Type de méta-modèle ("xgboost", "lightgbm", "logistic")
            auto_adjust: Activer l'ajustement automatique des poids
            config_file: Fichier de configuration YAML/JSON
        """
        self.fusion_method = fusion_method
        self.meta_model_type = meta_model_type
        self.auto_adjust = auto_adjust
        
        # Modèles et leurs poids
        self.models = {}
        self.model_weights = {}
        self.model_scores = defaultdict(list)
        
        # Méta-modèle
        self.meta_model = None
        self.scaler = StandardScaler()
        
        # Historique des performances
        self.performance_history = []
        
        # Configuration
        self.config = self._load_config(config_file) if config_file else {}
        
        logger.info(f"✅ MetaModelFusion initialisé: {fusion_method} + {meta_model_type}")
    
    def _load_config(self, config_file: str) -> Dict:
        """Charge la configuration depuis un fichier."""
        try:
            config_path = Path(config_file)
            if not config_path.exists():
                logger.warning(f"Fichier config non trouvé: {config_file}")
                return {}
            
            if config_file.endswith('.json'):
                with open(config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            elif config_file.endswith('.yaml') or config_file.endswith('.yml'):
                import yaml
                with open(config_file, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            else:
                logger.error(f"Format config non supporté: {config_file}")
                return {}
        except Exception as e:
            logger.error(f"Erreur chargement config: {e}")
            return {}
    
    def register_model(
        self,
        model_name: str,
        model: Any,
        initial_weight: float = 1.0,
        enabled: bool = True
    ):
        """
        Enregistre un modèle dans le système de fusion.
        
        Args:
            model_name: Nom unique du modèle
            model: Instance du modèle (doit avoir predict/predict_proba)
            initial_weight: Poids initial (défaut: 1.0)
            enabled: Activer le modèle
        """
        if not enabled:
            logger.info(f"⏭️ Modèle {model_name} désactivé")
            return
        
        self.models[model_name] = model
        self.model_weights[model_name] = initial_weight
        logger.info(f"✅ Modèle enregistré: {model_name} (poids: {initial_weight})")
    
    def _create_meta_model(self):
        """Crée le méta-modèle selon la configuration."""
        if self.meta_model_type == "xgboost":
            self.meta_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        elif self.meta_model_type == "lightgbm":
            self.meta_model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        elif self.meta_model_type == "logistic":
            self.meta_model = LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        else:
            logger.warning(f"Type méta-modèle inconnu: {self.meta_model_type}, utilisation logistic")
            self.meta_model = LogisticRegression(max_iter=1000, random_state=42)
        
        logger.info(f"✅ Méta-modèle créé: {self.meta_model_type}")
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Entraîne le système de fusion sur les données.
        
        Args:
            X: Features d'entraînement
            y: Labels d'entraînement
        """
        if not self.models:
            logger.error("❌ Aucun modèle enregistré")
            return
        
        logger.info(f"🔧 Entraînement du système de fusion ({len(self.models)} modèles)...")
        
        try:
            if self.fusion_method == "stacking":
                self._fit_stacking(X, y)
            elif self.fusion_method == "blending":
                self._fit_blending(X, y)
            elif self.fusion_method == "voting":
                self._fit_voting(X, y)
            else:
                logger.error(f"Méthode de fusion inconnue: {self.fusion_method}")
                return
            
            logger.info("✅ Entraînement du système de fusion terminé")
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'entraînement: {e}")
            logger.debug(traceback.format_exc())
    
    def _fit_stacking(self, X: np.ndarray, y: np.ndarray):
        """Entraîne avec la méthode stacking."""
        # Créer le méta-modèle
        self._create_meta_model()
        
        # Collecter les prédictions de tous les modèles
        meta_features = []
        
        for model_name, model in self.models.items():
            try:
                # Entraîner le modèle de base
                logger.info(f"  Entraînement de {model_name}...")
                model.fit(X, y)
                
                # Obtenir les prédictions (probabilities si disponible)
                if hasattr(model, 'predict_proba'):
                    preds = model.predict_proba(X)
                else:
                    preds = model.predict(X).reshape(-1, 1)
                
                meta_features.append(preds)
                logger.info(f"  ✅ {model_name} entraîné")
            except Exception as e:
                logger.error(f"  ❌ Erreur avec {model_name}: {e}")
        
        # Concaténer les features pour le méta-modèle
        if meta_features:
            X_meta = np.hstack(meta_features)
            
            # Normaliser
            X_meta = self.scaler.fit_transform(X_meta)
            
            # Entraîner le méta-modèle
            logger.info("  Entraînement du méta-modèle...")
            self.meta_model.fit(X_meta, y)
            logger.info("  ✅ Méta-modèle entraîné")
    
    def _fit_blending(self, X: np.ndarray, y: np.ndarray):
        """Entraîne avec la méthode blending (similaire à stacking mais plus simple)."""
        # Pour blending, on entraîne simplement tous les modèles
        for model_name, model in self.models.items():
            try:
                logger.info(f"  Entraînement de {model_name}...")
                model.fit(X, y)
                logger.info(f"  ✅ {model_name} entraîné")
            except Exception as e:
                logger.error(f"  ❌ Erreur avec {model_name}: {e}")
    
    def _fit_voting(self, X: np.ndarray, y: np.ndarray):
        """Entraîne avec la méthode voting."""
        # Entraîner tous les modèles
        for model_name, model in self.models.items():
            try:
                logger.info(f"  Entraînement de {model_name}...")
                model.fit(X, y)
                logger.info(f"  ✅ {model_name} entraîné")
            except Exception as e:
                logger.error(f"  ❌ Erreur avec {model_name}: {e}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fait une prédiction en fusionnant tous les modèles.
        
        Args:
            X: Features pour la prédiction
            
        Returns:
            Prédictions fusionnées
        """
        if not self.models:
            logger.error("❌ Aucun modèle disponible")
            return np.array([])
        
        try:
            if self.fusion_method == "stacking":
                return self._predict_stacking(X)
            elif self.fusion_method == "blending":
                return self._predict_blending(X)
            elif self.fusion_method == "voting":
                return self._predict_voting(X)
            else:
                logger.error(f"Méthode de fusion inconnue: {self.fusion_method}")
                return np.array([])
        except Exception as e:
            logger.error(f"❌ Erreur lors de la prédiction: {e}")
            logger.debug(traceback.format_exc())
            return np.array([])
    
    def _predict_stacking(self, X: np.ndarray) -> np.ndarray:
        """Prédiction avec stacking."""
        if self.meta_model is None:
            logger.error("❌ Méta-modèle non entraîné")
            return np.array([])
        
        # Collecter les prédictions de tous les modèles
        meta_features = []
        
        for model_name, model in self.models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    preds = model.predict_proba(X)
                else:
                    preds = model.predict(X).reshape(-1, 1)
                meta_features.append(preds)
            except Exception as e:
                logger.error(f"Erreur prédiction {model_name}: {e}")
        
        if not meta_features:
            return np.array([])
        
        # Concaténer et normaliser
        X_meta = np.hstack(meta_features)
        X_meta = self.scaler.transform(X_meta)
        
        # Prédiction finale avec méta-modèle
        return self.meta_model.predict(X_meta)
    
    def _predict_blending(self, X: np.ndarray) -> np.ndarray:
        """Prédiction avec blending (moyenne pondérée)."""
        predictions = []
        weights = []
        
        for model_name, model in self.models.items():
            try:
                pred = model.predict(X)
                predictions.append(pred)
                weights.append(self.model_weights.get(model_name, 1.0))
            except Exception as e:
                logger.error(f"Erreur prédiction {model_name}: {e}")
        
        if not predictions:
            return np.array([])
        
        # Normaliser les poids
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Moyenne pondérée
        predictions = np.array(predictions)
        weighted_pred = np.average(predictions, axis=0, weights=weights)
        
        # Arrondir pour classification
        return np.round(weighted_pred).astype(int)
    
    def _predict_voting(self, X: np.ndarray) -> np.ndarray:
        """Prédiction avec voting (vote majoritaire pondéré)."""
        all_predictions = []
        weights = []
        
        for model_name, model in self.models.items():
            try:
                pred = model.predict(X)
                all_predictions.append(pred)
                weights.append(self.model_weights.get(model_name, 1.0))
            except Exception as e:
                logger.error(f"Erreur prédiction {model_name}: {e}")
        
        if not all_predictions:
            return np.array([])
        
        # Vote pondéré
        all_predictions = np.array(all_predictions)
        weights = np.array(weights)
        
        # Pour chaque échantillon, compter les votes pondérés
        final_predictions = []
        for i in range(all_predictions.shape[1]):
            votes = all_predictions[:, i]
            # Vote majoritaire pondéré
            unique_votes = np.unique(votes)
            vote_counts = {}
            for vote in unique_votes:
                mask = votes == vote
                vote_counts[vote] = np.sum(weights[mask])
            
            # Sélectionner le vote avec le plus grand poids
            winner = max(vote_counts.items(), key=lambda x: x[1])[0]
            final_predictions.append(winner)
        
        return np.array(final_predictions)
    
    def predict_numbers_and_stars(
        self,
        X: np.ndarray,
        num_numbers: int = 5,
        num_stars: int = 2,
        max_number: int = 50,
        max_star: int = 12
    ) -> Dict[str, List[int]]:
        """
        Prédit des numéros et étoiles EuroMillions.
        
        Args:
            X: Features pour la prédiction
            num_numbers: Nombre de numéros à prédire (5)
            num_stars: Nombre d'étoiles à prédire (2)
            max_number: Numéro maximum (50)
            max_star: Étoile maximum (12)
            
        Returns:
            Dict avec 'numbers' et 'stars'
        """
        try:
            # Obtenir les scores de probabilité pour chaque numéro
            number_scores = self._get_number_scores(X, max_number)
            star_scores = self._get_star_scores(X, max_star)
            
            # Sélectionner les top numéros et étoiles
            top_numbers = sorted(
                range(1, max_number + 1),
                key=lambda x: number_scores.get(x, 0),
                reverse=True
            )[:num_numbers]
            
            top_stars = sorted(
                range(1, max_star + 1),
                key=lambda x: star_scores.get(x, 0),
                reverse=True
            )[:num_stars]
            
            return {
                'numbers': sorted(top_numbers),
                'stars': sorted(top_stars)
            }
        except Exception as e:
            logger.error(f"Erreur prédiction numéros/étoiles: {e}")
            # Fallback: sélection aléatoire
            import random
            return {
                'numbers': sorted(random.sample(range(1, max_number + 1), num_numbers)),
                'stars': sorted(random.sample(range(1, max_star + 1), num_stars))
            }
    
    def _get_number_scores(self, X: np.ndarray, max_number: int) -> Dict[int, float]:
        """Calcule les scores pour chaque numéro."""
        scores = {}
        
        for model_name, model in self.models.items():
            try:
                weight = self.model_weights.get(model_name, 1.0)
                
                # Obtenir les scores du modèle (à adapter selon l'interface du modèle)
                if hasattr(model, 'get_number_scores'):
                    model_scores = model.get_number_scores(X)
                    for num, score in model_scores.items():
                        scores[num] = scores.get(num, 0) + score * weight
            except Exception as e:
                logger.debug(f"Impossible d'obtenir scores de {model_name}: {e}")
        
        # Normaliser les scores
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                scores = {k: v / max_score for k, v in scores.items()}
        
        return scores
    
    def _get_star_scores(self, X: np.ndarray, max_star: int) -> Dict[int, float]:
        """Calcule les scores pour chaque étoile."""
        scores = {}
        
        for model_name, model in self.models.items():
            try:
                weight = self.model_weights.get(model_name, 1.0)
                
                if hasattr(model, 'get_star_scores'):
                    model_scores = model.get_star_scores(X)
                    for star, score in model_scores.items():
                        scores[star] = scores.get(star, 0) + score * weight
            except Exception as e:
                logger.debug(f"Impossible d'obtenir scores étoiles de {model_name}: {e}")
        
        # Normaliser
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                scores = {k: v / max_score for k, v in scores.items()}
        
        return scores
    
    def update_weights(self, model_scores: Dict[str, float]):
        """
        Met à jour les poids des modèles basé sur leurs performances.
        
        Args:
            model_scores: Dict {model_name: score} avec scores entre 0 et 1
        """
        if not self.auto_adjust:
            logger.info("Auto-ajustement désactivé")
            return
        
        logger.info("🔄 Mise à jour des poids des modèles...")
        
        for model_name, score in model_scores.items():
            if model_name not in self.model_weights:
                continue
            
            old_weight = self.model_weights[model_name]
            
            # Ajuster le poids en fonction du score
            if score > 0.7:
                # Bon score: augmenter le poids
                new_weight = old_weight * 1.1
            elif score < 0.4:
                # Mauvais score: diminuer le poids
                new_weight = old_weight * 0.9
            else:
                # Score moyen: légère augmentation
                new_weight = old_weight * 1.02
            
            # Limiter les poids entre 0.1 et 2.0
            new_weight = max(0.1, min(2.0, new_weight))
            
            self.model_weights[model_name] = new_weight
            
            logger.info(f"  {model_name}: {old_weight:.3f} → {new_weight:.3f} (score: {score:.3f})")
            
            # Enregistrer dans l'historique
            self.model_scores[model_name].append(score)
        
        # Normaliser les poids
        self._normalize_weights()
    
    def _normalize_weights(self):
        """Normalise les poids pour qu'ils somment à 1."""
        total = sum(self.model_weights.values())
        if total > 0:
            for model_name in self.model_weights:
                self.model_weights[model_name] /= total
    
    def save(self, output_dir: str):
        """Sauvegarde le système de fusion."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder le méta-modèle
        if self.meta_model is not None:
            meta_model_path = output_path / "meta_model.pkl"
            with open(meta_model_path, 'wb') as f:
                pickle.dump(self.meta_model, f)
            logger.info(f"✅ Méta-modèle sauvegardé: {meta_model_path}")
        
        # Sauvegarder les poids et scores
        weights_path = output_path / "model_weights.json"
        with open(weights_path, 'w', encoding='utf-8') as f:
            json.dump({
                'weights': self.model_weights,
                'scores': {k: list(v) for k, v in self.model_scores.items()}
            }, f, indent=2)
        logger.info(f"✅ Poids sauvegardés: {weights_path}")
    
    def load(self, input_dir: str):
        """Charge le système de fusion."""
        input_path = Path(input_dir)
        
        # Charger le méta-modèle
        meta_model_path = input_path / "meta_model.pkl"
        if meta_model_path.exists():
            with open(meta_model_path, 'rb') as f:
                self.meta_model = pickle.load(f)
            logger.info(f"✅ Méta-modèle chargé: {meta_model_path}")
        
        # Charger les poids
        weights_path = input_path / "model_weights.json"
        if weights_path.exists():
            with open(weights_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.model_weights = data.get('weights', {})
                scores = data.get('scores', {})
                self.model_scores = defaultdict(list, {k: v for k, v in scores.items()})
            logger.info(f"✅ Poids chargés: {weights_path}")


def main():
    """Fonction de test."""
    logger.info("=== Test MetaModelFusion ===")
    
    # Créer une instance
    fusion = MetaModelFusion(
        fusion_method="stacking",
        meta_model_type="xgboost",
        auto_adjust=True
    )
    
    # Simuler des modèles (à remplacer par de vrais modèles)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    
    model1 = RandomForestClassifier(n_estimators=10, random_state=42)
    model2 = LogisticRegression(random_state=42)
    
    fusion.register_model("random_forest", model1, initial_weight=1.0)
    fusion.register_model("logistic", model2, initial_weight=0.8)
    
    # Données de test
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    
    # Entraîner
    fusion.fit(X, y)
    
    # Prédire
    X_test = np.random.rand(10, 10)
    predictions = fusion.predict(X_test)
    logger.info(f"Prédictions: {predictions}")
    
    # Mettre à jour les poids
    fusion.update_weights({
        "random_forest": 0.75,
        "logistic": 0.85
    })
    
    logger.info("✅ Test terminé")


if __name__ == "__main__":
    main()

