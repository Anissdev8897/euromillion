#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module d'apprentissage incrémental pour l'analyseur Euromillions.
Ce module implémente une structure minimale pour permettre l'exécution du script principal.

Améliorations:
- Ajout de la méthode load_models() pour charger les modèles entraînés incrémentalement.
- Amélioration de l'initialisation et de la gestion des modèles MLP pour l'apprentissage incrémental.
- Correction de l'AttributeError: 'EuromillionsIncrementalLearning' object has no attribute 'backtest_results'
  en initialisant un historique de performance incrémentale et en le visualisant.
- Correction de l'erreur "accuracy_score" non définie.
- Correction de l'erreur TypeError: MLPClassifier.__init__() got an unexpected keyword argument 'hidden_layers'.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import joblib
import traceback
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score # Ajout de l'importation pour accuracy_score
from typing import Tuple, Optional, Any # Ajouter Any pour les modèles sklearn

logger = logging.getLogger("EuromillionsIncrementalLearning")

class EuromillionsIncrementalLearning:
    """Classe pour l'apprentissage incrémental des modèles Euromillions."""
    
    def __init__(self, analyzer):
        """
        Initialise l'apprentissage incrémental avec l'analyseur fourni.
        
        Args:
            analyzer: Instance de EuromillionsAdvancedAnalyzer
        """
        self.analyzer = analyzer
        self.output_dir = Path(analyzer.output_dir) / "incremental"
        
        # Créer le répertoire de sortie s'il n'existe pas
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)
            logger.info(f"Répertoire créé: {self.output_dir}")
        self.update_counter = 0 # Compteur pour les mises à jour
        self.incremental_performance_history = [] # Initialiser l'historique de performance incrémentale

        # Initialiser les modèles et scalers (ou charger s'ils existent)
        self._initialize_mlp_models()
    
    def _initialize_mlp_models(self):
        """
        Initialise les modèles MLPClassifier et leurs scalers pour l'apprentissage incrémental.
        Charge les modèles/scalers existants si disponibles.
        """
        logger.info("Initialisation/Vérification des modèles pour l'apprentissage incrémental (MLPClassifier)...")
        
        models_dir = self.output_dir / "models" # Utiliser le répertoire models sous incremental
        if not models_dir.exists():
            models_dir.mkdir(parents=True)

        # Tenter de charger les modèles et scalers existants
        number_model_path = models_dir / "number_model.joblib"
        star_model_path = models_dir / "star_model.joblib"
        scaler_numbers_path = models_dir / "scaler_numbers.joblib"
        scaler_stars_path = models_dir / "scaler_stars.joblib"

        # Charger le modèle des numéros
        if number_model_path.exists():
            try:
                self.analyzer.number_model = joblib.load(number_model_path)
                logger.info(f"Modèle de numéros chargé depuis {number_model_path}")
                self.first_fit_done_numbers = True # Marquer comme déjà fit si chargé
            except Exception as e:
                logger.warning(f"Impossible de charger le modèle de numéros existant: {e}. Création d'un nouveau modèle.")
                self.analyzer.number_model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', 
                                                          random_state=42, warm_start=True, max_iter=1)
                self.first_fit_done_numbers = False
        else:
            self.analyzer.number_model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', 
                                                      random_state=42, warm_start=True, max_iter=1)
            self.first_fit_done_numbers = False

        # ⚠️ CRITIQUE : Créer des scalers SÉPARÉS pour l'apprentissage incrémental
        # L'apprentissage incrémental utilise 35 features (5 tirages * 7 valeurs)
        # alors que l'analyseur principal utilise 2073 features (encodeur avancé)
        # Charger le scaler des numéros (incrémental)
        if scaler_numbers_path.exists():
            try:
                self.incremental_scaler_numbers = joblib.load(scaler_numbers_path)
                logger.info(f"Scaler incrémental de numéros chargé depuis {scaler_numbers_path}")
            except Exception as e:
                logger.warning(f"Impossible de charger le scaler incrémental de numéros existant: {e}. Création d'un nouveau scaler.")
                self.incremental_scaler_numbers = StandardScaler()
        else:
            self.incremental_scaler_numbers = StandardScaler()

        # Charger le modèle des étoiles
        if star_model_path.exists():
            try:
                self.analyzer.star_model = joblib.load(star_model_path)
                logger.info(f"Modèle d'étoiles chargé depuis {star_model_path}")
                self.first_fit_done_stars = True # Marquer comme déjà fit si chargé
            except Exception as e:
                logger.warning(f"Impossible de charger le modèle d'étoiles existant: {e}. Création d'un nouveau modèle.")
                self.analyzer.star_model = MLPClassifier(hidden_layer_sizes=(50,), activation='relu', solver='adam', 
                                                        random_state=42, warm_start=True, max_iter=1)
                self.first_fit_done_stars = False
        else:
            # Correction ici : hidden_layers -> hidden_layer_sizes
            self.analyzer.star_model = MLPClassifier(hidden_layer_sizes=(50,), activation='relu', solver='adam', 
                                                    random_state=42, warm_start=True, max_iter=1)
            self.first_fit_done_stars = False

        # Charger le scaler des étoiles (incrémental)
        if scaler_stars_path.exists():
            try:
                self.incremental_scaler_stars = joblib.load(scaler_stars_path)
                logger.info(f"Scaler incrémental d'étoiles chargé depuis {scaler_stars_path}")
            except Exception as e:
                logger.warning(f"Impossible de charger le scaler incrémental d'étoiles existant: {e}. Création d'un nouveau scaler.")
                self.incremental_scaler_stars = StandardScaler()
        else:
            self.incremental_scaler_stars = StandardScaler()

    def load_models(self) -> None:
        """
        Charge les modèles et scalers pré-entraînés depuis le répertoire de modèles.
        Cette méthode appelle _initialize_mlp_models, qui gère le chargement.
        """
        logger.info("Tentative de chargement des modèles incrémentaux existants.")
        self._initialize_mlp_models()


    def _prepare_incremental_data(self, draw_index: int, window_size: int = 5) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Prépare les données (features et targets) pour un tirage spécifique pour l'entraînement incrémental.

        Args:
            draw_index: Index du tirage actuel dans le DataFrame de l'analyseur.
            window_size: Nombre de tirages précédents à utiliser comme features.

        Returns:
            Tuple contenant (features, targets_numbers, targets_stars) ou (None, None, None) si impossible.
        """
        if draw_index < window_size:
            # Pas assez de données précédentes pour créer les features
            return None, None, None

        try:
            # Features: les window_size derniers tirages avant le tirage actuel
            features = []
            for j in range(window_size):
                row_idx = draw_index - window_size + j
                row = self.analyzer.df.iloc[row_idx]
                numbers = row[self.analyzer.number_cols].dropna().astype(int).tolist()
                stars = row[self.analyzer.star_cols].dropna().astype(int).tolist()
                features.extend(numbers)
                features.extend(stars)
            
            X = np.array([features]) # Une seule ligne de features pour ce tirage

            # Targets: le tirage actuel (celui à l'index draw_index)
            target_row = self.analyzer.df.iloc[draw_index]
            target_numbers_list = target_row[self.analyzer.number_cols].dropna().astype(int).tolist()
            target_stars_list = target_row[self.analyzer.star_cols].dropna().astype(int).tolist()

            # Transformation en format multi-label
            y_numbers_multi = np.zeros((1, self.analyzer.config["max_number"]))
            for num in target_numbers_list:
                if 1 <= num <= self.analyzer.config["max_number"]:
                    y_numbers_multi[0, num - 1] = 1
            
            y_stars_multi = np.zeros((1, self.analyzer.config["max_star"]))
            for star in target_stars_list:
                if 1 <= star <= self.analyzer.config["max_star"]:
                    y_stars_multi[0, star - 1] = 1

            return X, y_numbers_multi, y_stars_multi

        except Exception as e:
            logger.error(f"Erreur lors de la préparation des données incrémentales pour l'index {draw_index}: {str(e)}")
            logger.debug(traceback.format_exc())
            return None, None, None

    def update_models(self, draw_index: int):
        """Met à jour les modèles MLP avec les données du tirage spécifié."""
        
        X, y_numbers, y_stars = self._prepare_incremental_data(draw_index)
        
        if X is None or y_numbers is None or y_stars is None:
            logger.warning(f"Impossible de préparer les données pour le tirage {draw_index}, mise à jour ignorée.")
            return

        try:
            # ⚠️ CRITIQUE : Utiliser les scalers incrémentaux séparés (pas ceux de l'analyseur principal)
            # Mise à jour du scaler et transformation des features pour les numéros
            # partial_fit sur le scaler avant transform
            self.incremental_scaler_numbers.partial_fit(X)
            X_scaled_numbers = self.incremental_scaler_numbers.transform(X)
            
            # Mise à jour du modèle des numéros avec MLPClassifier
            # Le paramètre classes n'est nécessaire que lors du premier appel à partial_fit
            # pour définir l'espace de sortie
            if not self.first_fit_done_numbers:
                 self.analyzer.number_model.partial_fit(X_scaled_numbers, y_numbers, classes=np.arange(self.analyzer.config["max_number"]))
                 self.first_fit_done_numbers = True
            else:
                 self.analyzer.number_model.partial_fit(X_scaled_numbers, y_numbers)
            logger.debug(f"Modèle de numéros mis à jour avec le tirage {draw_index}")

            # Mise à jour du scaler et transformation des features pour les étoiles
            self.incremental_scaler_stars.partial_fit(X)
            X_scaled_stars = self.incremental_scaler_stars.transform(X)
            
            # Mise à jour du modèle des étoiles avec MLPClassifier
            if not self.first_fit_done_stars:
                 self.analyzer.star_model.partial_fit(X_scaled_stars, y_stars, classes=np.arange(self.analyzer.config["max_star"]))
                 self.first_fit_done_stars = True
            else:
                 self.analyzer.star_model.partial_fit(X_scaled_stars, y_stars)
            logger.debug(f"Modèle d'étoiles mis à jour avec le tirage {draw_index}")

        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour incrémentale des modèles pour le tirage {draw_index}: {str(e)}")
            logger.debug(traceback.format_exc())

    def run_incremental_learning(self, start_idx: int = 50, step_size: int = 1):
        """
        Exécute l'apprentissage incrémental.

        Args:
            start_idx: Indice de départ pour l'apprentissage (doit être >= window_size)
            step_size: Pas pour l'incrémentation
            
        Returns:
            Résultats de l'apprentissage incrémental (placeholder)
        """
        logger.info(f"Exécution de l'apprentissage incrémental (départ: {start_idx}, pas: {step_size})...")
        
        # Vérifier si les données sont chargées
        if self.analyzer.df is None or self.analyzer.df.empty:
            logger.error("Les données n'ont pas été chargées dans l'analyseur.")
            return {"accuracy": [], "predictions": []}
            
        # S'assurer que start_idx est valide par rapport à la taille de la fenêtre
        # window_size utilisée pour les features
        window_size = 5 # Correspond à la window_size dans _prepare_incremental_data et train_ml_models
        if start_idx < window_size:
            logger.warning(f"L'indice de départ ({start_idx}) est inférieur à la taille de la fenêtre ({window_size}). Ajustement à {window_size}.")
            start_idx = window_size
            
        # all_results = {"accuracy": [], "predictions": []} # Placeholder pour les résultats
        # Collecter des métriques simples pour la performance incrémentale
        self.incremental_performance_history = [] # Réinitialiser pour chaque exécution
        
        save_frequency = 100 # Sauvegarder tous les 100 tirages
        
        # Boucle sur les tirages pour l'apprentissage incrémental
        for i in range(start_idx, len(self.analyzer.df), step_size):
            logger.debug(f"Traitement du tirage {i} pour la mise à jour incrémentale")
            
            # Mettre à jour les modèles avec les données du tirage i
            self.update_models(i)
            
            # Incrémenter le compteur de mises à jour
            self.update_counter += 1
            
            # --- Évaluation et collecte des résultats pour plot_performance --- 
            # Pour une évaluation simple, nous pouvons prédire le tirage actuel
            # et comparer avec la réalité.
            try:
                # Préparer les features pour la prédiction du tirage actuel (i)
                X_current, y_numbers_current, y_stars_current = self._prepare_incremental_data(i)
                
                if X_current is not None and y_numbers_current is not None and y_stars_current is not None:
                    # ⚠️ CRITIQUE : Utiliser les scalers incrémentaux séparés
                    # Transformer les features avec les scalers mis à jour
                    X_scaled_numbers_current = self.incremental_scaler_numbers.transform(X_current)
                    X_scaled_stars_current = self.incremental_scaler_stars.transform(X_current)

                    # Prédire avec les modèles mis à jour
                    # Utiliser predict_proba pour obtenir des "confiances" ou des "scores"
                    number_probs = self.analyzer.number_model.predict_proba(X_scaled_numbers_current)[0]
                    star_probs = self.analyzer.star_model.predict_proba(X_scaled_stars_current)[0]

                    # Convertir les probabilités en prédictions binaires (par ex., seuil 0.5)
                    predicted_numbers_binary = (number_probs >= 0.5).astype(int)
                    predicted_stars_binary = (star_probs >= 0.5).astype(int)

                    # Calculer l'accuracy pour les numéros et étoiles
                    # Comparer les prédictions binaires avec les cibles réelles (y_numbers_current, y_stars_current)
                    accuracy_num = accuracy_score(y_numbers_current[0], predicted_numbers_binary)
                    accuracy_star = accuracy_score(y_stars_current[0], predicted_stars_binary)

                    self.incremental_performance_history.append({
                        'draw_index': i,
                        'accuracy_numbers': accuracy_num,
                        'accuracy_stars': accuracy_star,
                        'date': self.analyzer.df.iloc[i]['Date'].strftime('%Y-%m-%d') if 'Date' in self.analyzer.df.columns else None
                    })
                else:
                    logger.warning(f"Données de prédiction/évaluation non disponibles pour le tirage {i}.")

            except Exception as e:
                logger.error(f"Erreur lors de l'évaluation incrémentale pour le tirage {i}: {e}")
                logger.debug(traceback.format_exc())
            # --------------------------------------------------------------
            
            # Sauvegarder les modèles périodiquement
            if self.update_counter % save_frequency == 0:
                logger.info(f"Sauvegarde périodique des modèles (tirage {i}, compteur {self.update_counter})")
                self.save_models()
                

        # Sauvegarder les modèles une dernière fois à la fin du processus
        logger.info(f"Sauvegarde finale des modèles après {self.update_counter} mises à jour.")
        self.save_models()
        
        # Retourner les résultats collectés (maintenant stockés dans incremental_performance_history)
        logger.info("Apprentissage incrémental terminé.")
        return self.incremental_performance_history # Retourne l'historique pour une utilisation externe si besoin
    
    def save_models(self) -> bool:
        """
        Sauvegarde les modèles entraînés et scalers.
        
        Returns:
            bool: True si la sauvegarde a réussi, False sinon
        """
        try:
            models_dir = self.output_dir / "models"
            if not models_dir.exists():
                models_dir.mkdir(parents=True)
                logger.info(f"Répertoire des modèles créé: {models_dir}")
                
            # Sauvegarder le modèle principal des numéros (MLPClassifier)
            if hasattr(self.analyzer, 'number_model') and isinstance(self.analyzer.number_model, MLPClassifier):
                model_path = models_dir / "number_model.joblib"
                joblib.dump(self.analyzer.number_model, model_path)
                logger.info(f"Modèle de numéros sauvegardé dans {model_path}")
            else:
                logger.warning("Modèle de numéros non trouvé ou n'est pas un MLPClassifier pour la sauvegarde.")

            # Sauvegarder le modèle principal des étoiles (MLPClassifier)
            if hasattr(self.analyzer, 'star_model') and isinstance(self.analyzer.star_model, MLPClassifier):
                model_path = models_dir / "star_model.joblib"
                joblib.dump(self.analyzer.star_model, model_path)
                logger.info(f"Modèle d'étoiles sauvegardé dans {model_path}")
            else:
                logger.warning("Modèle d'étoiles non trouvé ou n'est pas un MLPClassifier pour la sauvegarde.")
            
            # Sauvegarder le scaler des numéros (incrémental)
            if hasattr(self, 'incremental_scaler_numbers') and self.incremental_scaler_numbers:
                scaler_path = models_dir / "scaler_numbers.joblib"
                joblib.dump(self.incremental_scaler_numbers, scaler_path)
                logger.info(f"Scaler incrémental de numéros sauvegardé dans {scaler_path}")
            else:
                logger.warning("Scaler incrémental de numéros non trouvé pour la sauvegarde.")

            # Sauvegarder le scaler des étoiles (incrémental)
            if hasattr(self, 'incremental_scaler_stars') and self.incremental_scaler_stars:
                scaler_path = models_dir / "scaler_stars.joblib"
                joblib.dump(self.incremental_scaler_stars, scaler_path)
                logger.info(f"Scaler incrémental d'étoiles sauvegardé dans {scaler_path}")
            else:
                logger.warning("Scaler incrémental d'étoiles non trouvé pour la sauvegarde.")
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des modèles: {str(e)}")
            logger.debug(traceback.format_exc())
            return False
    
    def plot_performance(self):
        """
        Génère des graphiques de performance pour l'apprentissage incrémental.
        Visualise l'évolution de l'accuracy des modèles de numéros et d'étoiles.
        """
        logger.info("Génération des graphiques de performance de l'apprentissage incrémental...")
        
        if not self.incremental_performance_history:
            logger.warning("Aucun résultat de performance incrémentale à visualiser.")
            return

        try:
            # Préparer les données pour le plotting
            draw_indices = [entry['draw_index'] for entry in self.incremental_performance_history]
            accuracy_numbers = [entry['accuracy_numbers'] for entry in self.incremental_performance_history]
            accuracy_stars = [entry['accuracy_stars'] for entry in self.incremental_performance_history]
            dates = [entry['date'] for entry in self.incremental_performance_history]

            # Créer le répertoire de visualisations
            plots_dir = self.output_dir / "plots"
            if not plots_dir.exists():
                plots_dir.mkdir(parents=True)
                logger.info(f"Répertoire de plots créé: {plots_dir}")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Graphique 1: Évolution de l'accuracy des numéros
            plt.figure(figsize=(12, 6))
            plt.plot(draw_indices, accuracy_numbers, marker='o', linestyle='-', color='skyblue', label='Accuracy Numéros')
            plt.title('Évolution de l\'Accuracy des Modèles de Numéros (Apprentissage Incrémental)')
            plt.xlabel('Index du Tirage')
            plt.ylabel('Accuracy')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.ylim(0, 1) # L'accuracy est entre 0 et 1
            plot_path_num = plots_dir / f"incremental_accuracy_numbers_{timestamp}.png"
            plt.savefig(plot_path_num)
            plt.close()
            logger.info(f"Graphique d'accuracy des numéros sauvegardé: {plot_path_num}")

            # Graphique 2: Évolution de l'accuracy des étoiles
            plt.figure(figsize=(12, 6))
            plt.plot(draw_indices, accuracy_stars, marker='s', linestyle='-', color='lightcoral', label='Accuracy Étoiles')
            plt.title('Évolution de l\'Accuracy des Modèles d\'Étoiles (Apprentissage Incrémental)')
            plt.xlabel('Index du Tirage')
            plt.ylabel('Accuracy')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.ylim(0, 1) # L'accuracy est entre 0 et 1
            plot_path_star = plots_dir / f"incremental_accuracy_stars_{timestamp}.png"
            plt.savefig(plot_path_star)
            plt.close()
            logger.info(f"Graphique d'accuracy des étoiles sauvegardé: {plot_path_star}")

            # Optionnel: Graphique combiné
            plt.figure(figsize=(14, 7))
            plt.plot(draw_indices, accuracy_numbers, marker='o', linestyle='-', color='skyblue', label='Accuracy Numéros')
            plt.plot(draw_indices, accuracy_stars, marker='s', linestyle='-', color='lightcoral', label='Accuracy Étoiles')
            plt.title('Évolution de l\'Accuracy des Modèles (Apprentissage Incrémental)')
            plt.xlabel('Index du Tirage')
            plt.ylabel('Accuracy')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.ylim(0, 1)
            plot_path_combined = plots_dir / f"incremental_accuracy_combined_{timestamp}.png"
            plt.savefig(plot_path_combined)
            plt.close()
            logger.info(f"Graphique d'accuracy combiné sauvegardé: {plot_path_combined}")

        except Exception as e:
            logger.error(f"Erreur lors de la génération des graphiques de performance incrémentale: {str(e)}")
            logger.debug(traceback.format_exc())
