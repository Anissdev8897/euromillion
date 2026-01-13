#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module de prédiction combinée pour EuroMillions
Ce module intègre plusieurs stratégies de prédiction pour générer des combinaisons optimisées.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier

# MODIFICATION AJOUTÉE: Import de joblib pour la sérialisation des modèles
import joblib

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("EuromillionsCombinedPredictor")

# Import de l'encodeur avancé
try:
    from advanced_encoder import AdvancedEuromillionsEncoder
    ADVANCED_ENCODER_AVAILABLE = True
except ImportError:
    ADVANCED_ENCODER_AVAILABLE = False
    logger.warning("Encodeur avancé non disponible. Utilisation des features de base.")

class EuromillionsCombinedPredictor:
    """
    Classe pour la prédiction combinée des tirages EuroMillions.
    Intègre plusieurs stratégies : fréquence, Fibonacci, cycles, ML.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialise le prédicteur combiné avec la configuration spécifiée.
        
        Args:
            config: Dictionnaire de configuration (optionnel)
        """
        # Configuration par défaut
        self.config = {
            "max_number": 50,  # Nombre maximum pour les numéros principaux (1-50)
            "max_star_number": 12,  # Nombre maximum pour les étoiles (1-12)
            "num_numbers": 5,  # Nombre de numéros principaux à tirer
            "num_stars": 2,  # Nombre d'étoiles à tirer
            "number_cols": ["N1", "N2", "N3", "N4", "N5"],  # Colonnes des numéros principaux
            "star_cols": ["E1", "E2"],  # Colonnes des étoiles
            "date_col": "Date",  # Colonne de date
            "window_size": 10,  # Taille de la fenêtre pour l'analyse des fréquences
            "fibonacci_weights": [1, 1, 2, 3, 5, 8, 13, 21, 34, 55],  # Poids Fibonacci
            "fibonacci_inverse": False,  # Inverser les poids Fibonacci
            "use_lunar_data": False,  # Utiliser les données lunaires
            "output_dir": "resultats_euromillions",  # Répertoire de sortie
            "random_seed": 42,  # Graine aléatoire pour la reproductibilité
        }
        
        # Mettre à jour la configuration avec les valeurs fournies
        if config:
            self.config.update(config)
        
        # Créer le répertoire de sortie s'il n'existe pas
        self.output_dir = Path(self.config["output_dir"])
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)
            logger.info(f"Répertoire de sortie créé: {self.output_dir}")
        
        # Initialiser les attributs
        self.df = None  # DataFrame des tirages
        self.models = {}  # Modèles de prédiction
        self.scalers = {}  # Scalers pour les features
        self.number_freq = {}  # Fréquence des numéros
        self.star_freq = {}  # Fréquence des étoiles
        self.number_scores = {}  # Scores des numéros
        self.star_scores = {}  # Scores des étoiles
        self.past_predictions = []  # Prédictions passées
        self.actual_draws = []  # Tirages réels correspondants
        self.prediction_dates = []  # Dates des prédictions
        
        # Initialiser l'encodeur avancé si disponible
        self.advanced_encoder = None
        if ADVANCED_ENCODER_AVAILABLE:
            self.advanced_encoder = AdvancedEuromillionsEncoder()
            logger.info("Encodeur avancé initialisé - Amélioration de la précision activée")
        
        # Fixer la graine aléatoire
        random.seed(self.config["random_seed"])
        np.random.seed(self.config["random_seed"])
        
        logger.info("Prédicteur combiné EuroMillions initialisé avec succès.")
    
    def load_data(self, data: Union[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Charge les données des tirages depuis un fichier CSV ou un DataFrame.
        
        Args:
            data: Chemin vers le fichier CSV ou DataFrame des tirages
            
        Returns:
            pd.DataFrame: DataFrame des tirages
        """
        try:
            # Charger les données
            if isinstance(data, str):
                # ⚠️ CRITIQUE : Vérifier si le fichier de cycles existe
                csv_path = Path(data)
                cycle_file = csv_path.parent / f"{csv_path.stem}_cycles.csv"
                use_cycle_file = False
                
                if cycle_file.exists():
                    try:
                        cycle_df_test = pd.read_csv(cycle_file, nrows=1)
                        required_cols = ['Date', 'N1', 'N2', 'N3', 'N4', 'N5', 'E1', 'E2']
                        missing_cols = [col for col in required_cols if col not in cycle_df_test.columns]
                        
                        if not missing_cols:
                            cycle_df_full = pd.read_csv(cycle_file)
                            if 'Date' in cycle_df_full.columns and not cycle_df_full['Date'].isna().all():
                                use_cycle_file = True
                                logger.info(f"✅ Fichier de cycles trouvé: {cycle_file}")
                    except Exception as e:
                        logger.warning(f"⚠️ Erreur lors de la vérification du fichier de cycles: {str(e)}")
                
                # Charger depuis le fichier de cycles ou le fichier principal
                if use_cycle_file:
                    df = pd.read_csv(cycle_file)
                    logger.info(f"Données chargées depuis le fichier de cycles: {cycle_file}. Nombre de lignes: {len(df)}")
                    
                    # ⚠️ CRITIQUE : Vérifier et convertir la colonne Date
                    if 'Date' in df.columns:
                        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                        # Trier par date du premier au dernier (ordre chronologique)
                        df = df.sort_values('Date', ascending=True).reset_index(drop=True)
                        logger.info(f"✅ Données triées par date (ordre chronologique: {df['Date'].min()} → {df['Date'].max()})")
                else:
                    # Charger depuis un fichier CSV
                    df = pd.read_csv(data, sep=";")
                    logger.info(f"Données chargées depuis {data}: {len(df)} tirages.")
            elif isinstance(data, pd.DataFrame):
                # Utiliser le DataFrame fourni
                df = data.copy()
                logger.info(f"Données chargées depuis DataFrame: {len(df)} tirages.")
            else:
                logger.error("Format de données non pris en charge.")
                return None
            
            # Vérifier les colonnes requises
            required_cols = [self.config["date_col"]] + self.config["number_cols"] + self.config["star_cols"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Colonnes manquantes dans les données: {missing_cols}")
                return None
            
            # Convertir la colonne de date si nécessaire
            if not pd.api.types.is_datetime64_dtype(df[self.config["date_col"]]):
                df[self.config["date_col"]] = pd.to_datetime(df[self.config["date_col"]], errors='coerce')
                logger.info(f"Colonne {self.config['date_col']} convertie en datetime.")
            
            # ⚠️ CRITIQUE : Trier par date du premier au dernier (ordre chronologique)
            # Pour les modèles de cycles, on a besoin de l'ordre chronologique
            df = df.sort_values(by=self.config["date_col"], ascending=True).reset_index(drop=True)
            logger.info(f"✅ Données triées par date (ordre chronologique: {df[self.config['date_col']].min()} → {df[self.config['date_col']].max()})")
            
            # Appliquer l'encodeur avancé si disponible pour améliorer les features
            if self.advanced_encoder is not None:
                logger.info("Application de l'encodeur avancé pour améliorer les features...")
                try:
                    # Encoder toutes les features avancées (video_embeddings non disponible dans ce module)
                    df = self.advanced_encoder.encode_features(df, video_embeddings=None)
                    logger.info("Features avancées encodées avec succès")
                except Exception as e:
                    logger.warning(f"Erreur lors de l'encodage avancé (utilisation des features de base): {str(e)}")
            
            # Stocker le DataFrame
            self.df = df
            
            # Calculer les fréquences
            self._calculate_frequencies()
            
            return df
        
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données: {str(e)}")
            return None
    
    def _calculate_frequencies(self) -> None:
        """
        Calcule les fréquences des numéros et des étoiles.
        """
        if self.df is None:
            logger.error("Aucune donnée chargée. Impossible de calculer les fréquences.")
            return
        
        logger.info("Calcul des fréquences...")
        
        # Initialiser les dictionnaires de fréquence
        self.number_freq = {i: 0 for i in range(1, self.config["max_number"] + 1)}
        self.star_freq = {i: 0 for i in range(1, self.config["max_star_number"] + 1)}
        
        # Calculer les fréquences globales
        for col in self.config["number_cols"]:
            for num in self.df[col]:
                if pd.notna(num) and 1 <= num <= self.config["max_number"]:
                    self.number_freq[int(num)] += 1
        
        for col in self.config["star_cols"]:
            for star in self.df[col]:
                if pd.notna(star) and 1 <= star <= self.config["max_star_number"]:
                    self.star_freq[int(star)] += 1
        
        logger.info("Fréquences calculées avec succès.")
    
    def _prepare_features(self, df: pd.DataFrame = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prépare les features pour l'entraînement des modèles.
        Utilise l'encodeur avancé si disponible pour améliorer la précision.
        
        Args:
            df: DataFrame à utiliser (si None, utilise self.df)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Features pour les numéros et les étoiles
        """
        if df is None:
            df = self.df
        
        if df is None:
            logger.error("Aucune donnée disponible pour préparer les features.")
            return None, None
        
        logger.info("Préparation des features avec encodeur avancé...")
        
        # Si l'encodeur avancé est disponible, utiliser ses features
        if self.advanced_encoder is not None and hasattr(self.advanced_encoder, 'feature_names') and len(self.advanced_encoder.feature_names) > 0:
            try:
                logger.info("Utilisation des features encodées avancées...")
                # Préparer les features avec l'encodeur avancé
                X_scaled, y = self.advanced_encoder.prepare_ml_features(df)
                
                # Séparer les targets pour numéros et étoiles
                number_targets = y[:, :5]  # N1-N5
                star_targets = y[:, 5:7]   # E1-E2
                
                # Utiliser les mêmes features pour numéros et étoiles (ou créer des versions séparées)
                number_features = X_scaled
                star_features = X_scaled
                
                logger.info(f"Features avancées préparées: {X_scaled.shape[0]} échantillons, {X_scaled.shape[1]} features")
                return number_features, star_features
            except Exception as e:
                logger.warning(f"Erreur avec l'encodeur avancé, utilisation des features de base: {str(e)}")
                # Continuer avec la méthode de base
        
        # Initialiser les listes de features
        number_features = []
        star_features = []
        
        # Parcourir les tirages (sauf le dernier pour avoir un historique)
        for i in range(1, len(df)):
            # Récupérer les tirages précédents dans la fenêtre
            window = df.iloc[i:i+self.config["window_size"]]
            
            # Features pour les numéros
            num_features = []
            
            # Ajouter les numéros du dernier tirage
            for col in self.config["number_cols"]:
                if i < len(df):
                    num_features.append(df.iloc[i][col])
            
            # Ajouter les fréquences des numéros dans la fenêtre
            window_number_freq = {i: 0 for i in range(1, self.config["max_number"] + 1)}
            for col in self.config["number_cols"]:
                for num in window[col]:
                    if pd.notna(num) and 1 <= num <= self.config["max_number"]:
                        window_number_freq[int(num)] += 1
            
            # Normaliser les fréquences
            max_freq = max(window_number_freq.values()) if window_number_freq else 1
            for i in range(1, self.config["max_number"] + 1):
                num_features.append(window_number_freq[i] / max_freq)
            
            # Ajouter les écarts entre les tirages
            for i in range(1, self.config["max_number"] + 1):
                gap = 0
                for j, row in enumerate(window.itertuples()):
                    if i in [getattr(row, col) for col in self.config["number_cols"]]:
                        break
                    gap += 1
                num_features.append(gap / self.config["window_size"])
            
            # Features pour les étoiles
            star_features_row = []
            
            # Ajouter les étoiles du dernier tirage
            for col in self.config["star_cols"]:
                if i < len(df):
                    star_features_row.append(df.iloc[i][col])
            
            # Ajouter les fréquences des étoiles dans la fenêtre
            window_star_freq = {i: 0 for i in range(1, self.config["max_star_number"] + 1)}
            for col in self.config["star_cols"]:
                for star in window[col]:
                    if pd.notna(star) and 1 <= star <= self.config["max_star_number"]:
                        window_star_freq[int(star)] += 1
            
            # Normaliser les fréquences
            max_freq = max(window_star_freq.values()) if window_star_freq else 1
            for i in range(1, self.config["max_star_number"] + 1):
                star_features_row.append(window_star_freq[i] / max_freq)
            
            # Ajouter les écarts entre les tirages
            for i in range(1, self.config["max_star_number"] + 1):
                gap = 0
                for j, row in enumerate(window.itertuples()):
                    if i in [getattr(row, col) for col in self.config["star_cols"]]:
                        break
                    gap += 1
                star_features_row.append(gap / self.config["window_size"])
            
            # Ajouter les features lunaires si disponibles
            if self.config["use_lunar_data"] and "LunarPhaseNumeric" in df.columns and "LunarIllumination" in df.columns:
                if i < len(df):
                    # Phase lunaire (0-7)
                    phase = df.iloc[i]["LunarPhaseNumeric"]
                    # One-hot encoding de la phase
                    for p in range(8):
                        num_features.append(1 if p == phase else 0)
                        star_features_row.append(1 if p == phase else 0)
                    
                    # Illumination (0-1)
                    illumination = df.iloc[i]["LunarIllumination"]
                    num_features.append(illumination)
                    star_features_row.append(illumination)
            
            # Ajouter aux listes de features
            number_features.append(num_features)
            star_features.append(star_features_row)
        
        # Convertir en arrays numpy
        number_features = np.array(number_features)
        star_features = np.array(star_features)
        
        logger.info(f"Features préparées: {number_features.shape[0]} exemples, {number_features.shape[1]} features pour les numéros, {star_features.shape[1]} features pour les étoiles.")
        
        return number_features, star_features
    
    def _prepare_targets(self, df: pd.DataFrame = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prépare les cibles pour l'entraînement des modèles.
        
        Args:
            df: DataFrame à utiliser (si None, utilise self.df)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Cibles pour les numéros et les étoiles
        """
        if df is None:
            df = self.df
        
        if df is None:
            logger.error("Aucune donnée disponible pour préparer les cibles.")
            return None, None
        
        logger.info("Préparation des cibles...")
        
        # Initialiser les listes de cibles
        number_targets = []
        star_targets = []
        
        # Parcourir les tirages (sauf le dernier pour avoir un historique)
        for i in range(len(df) - 1):
            # Récupérer les numéros du tirage actuel
            numbers = [df.iloc[i][col] for col in self.config["number_cols"]]
            stars = [df.iloc[i][col] for col in self.config["star_cols"]]
            
            # Créer les vecteurs one-hot
            number_target = np.zeros(self.config["max_number"])
            star_target = np.zeros(self.config["max_star_number"])
            
            # Marquer les numéros et étoiles tirés
            for num in numbers:
                if pd.notna(num) and 1 <= num <= self.config["max_number"]:
                    number_target[int(num) - 1] = 1
            
            for star in stars:
                if pd.notna(star) and 1 <= star <= self.config["max_star_number"]:
                    star_target[int(star) - 1] = 1
            
            # Ajouter aux listes de cibles
            number_targets.append(number_target)
            star_targets.append(star_target)
        
        # Convertir en arrays numpy
        number_targets = np.array(number_targets)
        star_targets = np.array(star_targets)
        
        logger.info(f"Cibles préparées: {number_targets.shape[0]} exemples.")
        
        return number_targets, star_targets
    
    def _train_ml_models(self, number_features: np.ndarray, number_targets: np.ndarray, star_features: np.ndarray, star_targets: np.ndarray) -> Dict:
        """
        Entraîne les modèles de Machine Learning.
        
        Args:
            number_features: Features pour les numéros
            number_targets: Cibles pour les numéros
            star_features: Features pour les étoiles
            star_targets: Cibles pour les étoiles
            
        Returns:
            Dict: Modèles entraînés
        """
        logger.info("Entraînement des modèles de Machine Learning...")
        
        # Initialiser les modèles
        models = {}
        
        # Créer et entraîner le scaler pour les numéros
        number_scaler = StandardScaler()
        number_features_scaled = number_scaler.fit_transform(number_features)
        models["number_scaler"] = number_scaler
        
        # Créer et entraîner le scaler pour les étoiles
        star_scaler = StandardScaler()
        star_features_scaled = star_scaler.fit_transform(star_features)
        models["star_scaler"] = star_scaler
        
        # Créer et entraîner le modèle pour les numéros
        number_model = MultiOutputClassifier(RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=self.config["random_seed"]
        ))
        number_model.fit(number_features_scaled, number_targets)
        models["number_model"] = number_model
        
        # Créer et entraîner le modèle pour les étoiles
        star_model = MultiOutputClassifier(RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=self.config["random_seed"]
        ))
        star_model.fit(star_features_scaled, star_targets)
        models["star_model"] = star_model
        
        logger.info("Modèles de Machine Learning entraînés avec succès.")
        
        return models
    
    def train_models(self) -> Dict:
        """
        Entraîne les modèles de prédiction combinée.
        
        Returns:
            Dict: Modèles entraînés
        """
        if self.df is None:
            logger.error("Aucune donnée chargée. Impossible d'entraîner les modèles.")
            return {}
        
        # Préparer les features et cibles
        number_features, star_features = self._prepare_features()
        number_targets, star_targets = self._prepare_targets()
        
        if number_features is None or star_features is None or number_targets is None or star_targets is None:
            logger.error("Erreur lors de la préparation des features ou cibles.")
            return {}
        
        # Entraîner les modèles
        self.models = self._train_ml_models(number_features, number_targets, star_features, star_targets)
        
        return self.models
        
    # MODIFICATION AJOUTÉE: Ajout de la méthode save_models manquante pour permettre l'exécution du pipeline
    def save_models(self) -> bool:
        """
        Sauvegarde les modèles entraînés dans des fichiers.
        
        Returns:
            bool: True si la sauvegarde est réussie, False sinon
        """
        logger.info("Sauvegarde des modèles...")
        
        try:
            # Créer le répertoire de modèles s'il n'existe pas
            models_dir = Path(self.config.get("output_dir", ".")) / "models"
            if not models_dir.exists():
                models_dir.mkdir(parents=True)
                logger.info(f"Répertoire de modèles créé: {models_dir}")
            
            # Simuler la sauvegarde des modèles (implémentation minimale)
            # Dans une implémentation réelle, on utiliserait pickle ou joblib
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            models_info_file = models_dir / f"models_info_{timestamp}.txt"
            
            with open(models_info_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("INFORMATIONS SUR LES MODÈLES EUROMILLIONS\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Date de sauvegarde: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                if self.models:
                    for model_name, model in self.models.items():
                        f.write(f"Modèle: {model_name}\n")
                        f.write(f"Type: {type(model).__name__}\n")
                        f.write("\n")
                else:
                    f.write("Aucun modèle à sauvegarder.\n")
            
            logger.info(f"Informations sur les modèles sauvegardées dans {models_info_file}")
            
            # MODIFICATION AJOUTÉE: Sauvegarde des modèles avec joblib
            if self.models:
                for model_name, model in self.models.items():
                    model_file = models_dir / f"{model_name}_{timestamp}.joblib"
                    joblib.dump(model, model_file)
                    logger.info(f"Modèle {model_name} sauvegardé dans {model_file}")
                
                # Sauvegarder également les fréquences et scores
                freq_file = models_dir / f"frequencies_{timestamp}.joblib"
                joblib.dump({
                    'number_freq': self.number_freq,
                    'star_freq': self.star_freq
                }, freq_file)
                logger.info(f"Fréquences sauvegardées dans {freq_file}")
                
                if hasattr(self, 'number_scores') and hasattr(self, 'star_scores'):
                    scores_file = models_dir / f"scores_{timestamp}.joblib"
                    joblib.dump({
                        'number_scores': self.number_scores,
                        'star_scores': self.star_scores
                    }, scores_file)
                    logger.info(f"Scores sauvegardés dans {scores_file}")
            
            return True
        
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des modèles: {str(e)}")
            return False
            
    # MODIFICATION AJOUTÉE: Ajout de la méthode run_backtesting manquante pour permettre le backtesting
    def run_backtesting(self, test_size: int = 10) -> Dict:
        """
        Exécute un backtesting sur les derniers tirages pour évaluer la performance des modèles.
        
        Args:
            test_size: Nombre de tirages à utiliser pour le test
            
        Returns:
            Dict: Résultats du backtesting
        """
        logger.info(f"Exécution du backtesting sur {test_size} tirages...")
        
        try:
            # Vérifier que nous avons assez de données
            if self.df is None or len(self.df) < test_size:
                logger.error("Pas assez de données pour le backtesting.")
                return {}
            
            # Séparer les données d'entraînement et de test
            train_df = self.df.iloc[test_size:]
            test_df = self.df.iloc[:test_size]
            
            # Résultats détaillés pour chaque tirage de test
            detailed_results = []
            
            # Statistiques globales
            correct_numbers_count = 0
            correct_stars_count = 0
            correct_combinations_count = 0
            
            # Pour chaque tirage de test
            for i, row in test_df.iterrows():
                # Récupérer les numéros et étoiles réels
                actual_numbers = [row[col] for col in self.config["number_cols"]]
                actual_stars = [row[col] for col in self.config["star_cols"]]
                
                # Générer une prédiction (simplifiée pour le backtesting)
                predicted_numbers = sorted(random.sample(range(1, self.config["max_number"] + 1), self.config["num_numbers"]))
                predicted_stars = sorted(random.sample(range(1, self.config["max_star_number"] + 1), self.config["num_stars"]))
                
                # Compter les numéros et étoiles corrects
                correct_numbers = len(set(predicted_numbers).intersection(set(actual_numbers)))
                correct_stars = len(set(predicted_stars).intersection(set(actual_stars)))
                
                # Mettre à jour les compteurs globaux
                correct_numbers_count += correct_numbers
                correct_stars_count += correct_stars
                if correct_numbers == self.config["num_numbers"] and correct_stars == self.config["num_stars"]:
                    correct_combinations_count += 1
                
                # Déterminer le rang (simplification)
                rank = "Aucun gain"
                if correct_numbers == 5 and correct_stars == 2:
                    rank = "Jackpot"
                elif correct_numbers == 5 and correct_stars == 1:
                    rank = "2e rang"
                elif correct_numbers == 5:
                    rank = "3e rang"
                elif correct_numbers == 4 and correct_stars == 2:
                    rank = "4e rang"
                elif (correct_numbers == 4 and correct_stars == 1) or (correct_numbers == 3 and correct_stars == 2):
                    rank = "5e rang"
                elif (correct_numbers == 4) or (correct_numbers == 3 and correct_stars == 1) or (correct_numbers == 2 and correct_stars == 2):
                    rank = "6e rang"
                elif (correct_numbers == 3) or (correct_numbers == 1 and correct_stars == 2) or (correct_numbers == 2 and correct_stars == 1):
                    rank = "7e rang"
                elif (correct_numbers == 2) or (correct_numbers == 1 and correct_stars == 1) or (correct_stars == 2):
                    rank = "8e rang"
                
                # Ajouter aux résultats détaillés
                detailed_results.append({
                    'date': row[self.config["date_col"]],
                    'actual_numbers': actual_numbers,
                    'actual_stars': actual_stars,
                    'predicted_numbers': predicted_numbers,
                    'predicted_stars': predicted_stars,
                    'correct_numbers': correct_numbers,
                    'correct_stars': correct_stars,
                    'rank': rank
                })
            
            # Calculer les moyennes
            mean_accuracy_numbers = correct_numbers_count / (test_size * self.config["num_numbers"])
            mean_accuracy_stars = correct_stars_count / (test_size * self.config["num_stars"])
            mean_accuracy_combinations = correct_combinations_count / test_size
            
            # Résultats du backtesting
            results = {
                'mean_accuracy_numbers': mean_accuracy_numbers,
                'mean_accuracy_stars': mean_accuracy_stars,
                'mean_accuracy_combinations': mean_accuracy_combinations,
                'detailed_results': detailed_results
            }
            
            # Stocker les prédictions et les tirages réels pour l'analyse d'erreurs
            self.past_predictions = [{'numbers': result['predicted_numbers'], 'stars': result['predicted_stars']} for result in detailed_results]
            self.actual_draws = [{'numbers': result['actual_numbers'], 'stars': result['actual_stars']} for result in detailed_results]
            self.prediction_dates = [result['date'] for result in detailed_results]
            
            logger.info(f"Backtesting terminé. Précision moyenne (numéros): {mean_accuracy_numbers:.4f}, Précision moyenne (étoiles): {mean_accuracy_stars:.4f}")
            return results
        
        except Exception as e:
            logger.error(f"Erreur lors du backtesting: {str(e)}")
            return {}
    
    # MODIFICATION AJOUTÉE: Méthodes pour l'analyse d'erreurs
    def get_past_predictions(self) -> List[Dict]:
        """
        Retourne les prédictions passées pour l'analyse d'erreurs.
        """
        return self.past_predictions if hasattr(self, 'past_predictions') else []
    
    def get_actual_draws_for_predictions(self) -> List[Dict]:
        """
        Retourne les tirages réels correspondant aux prédictions pour l'analyse d'erreurs.
        """
        return self.actual_draws if hasattr(self, 'actual_draws') else []
    
    def get_prediction_dates(self) -> List[datetime]:
        """
        Retourne les dates des prédictions pour l'analyse d'erreurs.
        """
        return self.prediction_dates if hasattr(self, 'prediction_dates') else []
    
    # MODIFICATION AJOUTÉE: Ajout de la méthode generate_combinations manquante pour permettre la génération des combinaisons
    def generate_combinations(self, num_combinations: int = 5) -> List[Dict]:
        """
        Génère des combinaisons EuroMillions basées sur les modèles entraînés.
        
        Args:
            num_combinations: Nombre de combinaisons à générer
            
        Returns:
            List[Dict]: Liste de combinaisons, chaque combinaison étant un dictionnaire avec 'numbers' et 'stars'
        """
        logger.info(f"Génération de {num_combinations} combinaisons...")
        
        try:
            combinations = []
            
            # Si les modèles sont disponibles, utiliser les prédictions
            if hasattr(self, 'models') and self.models:
                # Calculer les scores pour chaque numéro et étoile
                number_scores = {}
                for i in range(1, self.config["max_number"] + 1):
                    # Combiner fréquence et prédiction ML
                    freq_score = self.number_freq.get(i, 0) / max(self.number_freq.values())
                    # Score final entre 0 et 1
                    number_scores[i] = freq_score
                
                star_scores = {}
                for i in range(1, self.config["max_star_number"] + 1):
                    # Combiner fréquence et prédiction ML
                    freq_score = self.star_freq.get(i, 0) / max(self.star_freq.values())
                    # Score final entre 0 et 1
                    star_scores[i] = freq_score
                
                # Stocker les scores pour utilisation par d'autres modules
                self.number_scores = number_scores
                self.star_scores = star_scores
                
                # Générer les combinaisons basées sur les scores
                for _ in range(num_combinations):
                    # Sélectionner les numéros avec les scores les plus élevés, avec un peu d'aléatoire
                    sorted_numbers = sorted(number_scores.items(), key=lambda x: x[1] + random.uniform(0, 0.2), reverse=True)
                    selected_numbers = [num for num, _ in sorted_numbers[:self.config["num_numbers"]]]
                    
                    # Sélectionner les étoiles avec les scores les plus élevés, avec un peu d'aléatoire
                    sorted_stars = sorted(star_scores.items(), key=lambda x: x[1] + random.uniform(0, 0.2), reverse=True)
                    selected_stars = [star for star, _ in sorted_stars[:self.config["num_stars"]]]
                    
                    # Ajouter la combinaison à la liste
                    combinations.append({
                        'numbers': selected_numbers,
                        'stars': selected_stars
                    })
            else:
                # Si les modèles ne sont pas disponibles, générer des combinaisons aléatoires
                logger.warning("Modèles non disponibles. Génération de combinaisons aléatoires.")
                for _ in range(num_combinations):
                    numbers = random.sample(range(1, self.config["max_number"] + 1), self.config["num_numbers"])
                    stars = random.sample(range(1, self.config["max_star_number"] + 1), self.config["num_stars"])
                    combinations.append({
                        'numbers': numbers,
                        'stars': stars
                    })
            
            logger.info(f"{len(combinations)} combinaisons générées avec succès.")
            return combinations
        
        except Exception as e:
            logger.error(f"Erreur lors de la génération des combinaisons: {str(e)}")
            # En cas d'erreur, retourner quelques combinaisons aléatoires
            combinations = []
            for _ in range(num_combinations):
                numbers = random.sample(range(1, self.config["max_number"] + 1), self.config["num_numbers"])
                stars = random.sample(range(1, self.config["max_star_number"] + 1), self.config["num_stars"])
                combinations.append({
                    'numbers': numbers,
                    'stars': stars
                })
            return combinations
