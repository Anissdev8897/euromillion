#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module de prédiction unifié pour l'analyseur Euromillions.
Ce script centralise toutes les fonctions de prédiction des différents modèles
d'analyse Euromillions et permet de générer des combinaisons optimisées.

Version: 1.0.0
Date: 2025-05-26
"""

import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any
import traceback
import warnings
import json

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("EuromillionsPredictor")

# Ignorer les avertissements
warnings.filterwarnings("ignore")

# Assurer que le répertoire courant est dans sys.path pour l'importation des modules locaux
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importation conditionnelle des modules
MODULES = {
    "main_analyzer": False,
    "fibonacci_analyzer": False,
    "lunar_analyzer": False,
    "combination_optimizer": False,
    "super_optimizer": False,
    "video_analyzer": False
}

# Essayer d'importer le module d'analyse principal
try:
    from euromillions_analyzer import EuromillionsAnalyzer
    MODULES["main_analyzer"] = True
except ImportError:
    logger.warning("Module euromillions_analyzer non trouvé. Certaines fonctionnalités seront limitées.")

# Essayer d'importer le module d'analyse Fibonacci
try:
    from euromillions_fibonacci_analyzer import EuromillionsFibonacciAnalyzer
    MODULES["fibonacci_analyzer"] = True
except ImportError:
    logger.warning("Module euromillions_fibonacci_analyzer non trouvé. Certaines fonctionnalités seront limitées.")

# Essayer d'importer le module d'analyse du cycle lunaire
try:
    from lunar_cycle_analyzer import LunarCycleAnalyzer
    MODULES["lunar_analyzer"] = True
except ImportError:
    logger.warning("Module lunar_cycle_analyzer non trouvé. Certaines fonctionnalités seront limitées.")

# Essayer d'importer le module d'optimisation des combinaisons
try:
    from combination_optimizer import CombinationOptimizer
    MODULES["combination_optimizer"] = True
except ImportError:
    logger.warning("Module combination_optimizer non trouvé. Certaines fonctionnalités seront limitées.")

# Essayer d'importer le module super-optimiseur
try:
    from integrator_simple import EuromillionsSuperOptimizer
    MODULES["super_optimizer"] = True
except ImportError:
    logger.warning("Module integrator_simple non trouvé. Certaines fonctionnalités seront limitées.")

# Essayer d'importer le module d'analyse vidéo
try:
    from euromillions_video_analyzer import EuromillionsVideoAnalyzer
    MODULES["video_analyzer"] = True
except ImportError:
    logger.warning("Module euromillions_video_analyzer non trouvé. Certaines fonctionnalités seront limitées.")

class EuromillionsPredictor:
    """Classe pour la prédiction unifiée des combinaisons Euromillions."""
    
    def __init__(self, config):
        """
        Initialise le prédicteur avec la configuration spécifiée.
        
        Args:
            config: Dictionnaire de configuration
        """
        self.config = config
        self.csv_file = config.get("csv_file", "tirage_euromillions.csv")
        self.output_dir = Path(config.get("output_dir", "resultats_euromillions"))
        self.model_dir = Path(config.get("model_dir", "models_euromillions"))
        self.method = config.get("method", "all")
        self.combinations_to_generate = config.get("combinations", 5)
        self.video_dir = Path(config.get("video_dir", "tirage_video"))
        
        # Créer les répertoires nécessaires
        for directory in [self.output_dir, self.model_dir]:
            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)
                logger.info(f"Répertoire créé: {directory}")
        
        # Charger les données
        self.df = None
        self.load_data()
        
        # Initialiser les prédicteurs disponibles
        self.predictors = {}
        self.initialize_predictors()
        
        # Résultats
        self.predictions = []
        
        logger.info(f"EuromillionsPredictor initialisé avec method: {self.method}, output_dir: {self.output_dir}")
    
    def load_data(self):
        """
        Charge les données depuis le fichier CSV.
        Utilise le fichier de cycles s'il existe (tirage_euromillions_complet_cycles.csv).
        Respecte l'ordre chronologique du premier au dernier tirage.
        """
        try:
            csv_path = Path(self.csv_file)
            
            # ⚠️ CRITIQUE : Vérifier si le fichier de cycles existe
            cycle_file = csv_path.parent / f"{csv_path.stem}_cycles.csv"
            use_cycle_file = False
            
            if cycle_file.exists():
                logger.info(f"Fichier de cycles trouvé: {cycle_file}")
                try:
                    cycle_df_test = pd.read_csv(cycle_file, nrows=1)
                    required_cols = ['Date', 'N1', 'N2', 'N3', 'N4', 'N5', 'E1', 'E2']
                    missing_cols = [col for col in required_cols if col not in cycle_df_test.columns]
                    
                    if not missing_cols:
                        cycle_df_full = pd.read_csv(cycle_file)
                        if 'Date' in cycle_df_full.columns and not cycle_df_full['Date'].isna().all():
                            use_cycle_file = True
                            logger.info("✅ Utilisation du fichier de cycles avec dates")
                except Exception as e:
                    logger.warning(f"⚠️ Erreur lors de la vérification du fichier de cycles: {str(e)}")
            
            # Charger les données
            if use_cycle_file:
                self.df = pd.read_csv(cycle_file)
                logger.info(f"Données chargées depuis le fichier de cycles: {cycle_file}. Nombre de lignes: {len(self.df)}")
                
                # ⚠️ CRITIQUE : Vérifier et convertir la colonne Date
                if 'Date' in self.df.columns:
                    self.df['Date'] = pd.to_datetime(self.df['Date'], errors='coerce')
                    # Trier par date du premier au dernier (ordre chronologique)
                    self.df = self.df.sort_values('Date', ascending=True).reset_index(drop=True)
                    logger.info(f"✅ Données triées par date (ordre chronologique: {self.df['Date'].min()} → {self.df['Date'].max()})")
                elif 'Index' in self.df.columns:
                    self.df = self.df.sort_values('Index', ascending=True).reset_index(drop=True)
                    logger.info("✅ Données triées par Index (ordre chronologique)")
            else:
                if not csv_path.exists():
                    logger.error(f"Fichier CSV {self.csv_file} non trouvé.")
                    return False
                
                self.df = pd.read_csv(self.csv_file)
                logger.info(f"Données chargées depuis {self.csv_file}. Nombre de lignes: {len(self.df)}")
                
                # ⚠️ CRITIQUE : Vérifier et créer la colonne Date si manquante
                if 'Date' not in self.df.columns:
                    logger.warning("Colonne 'Date' non trouvée. Création de dates automatiques...")
                    from datetime import datetime, timedelta
                    first_draw_date = datetime(2004, 2, 13)
                    for i in range(len(self.df)):
                        weeks = i // 2
                        day_in_week = (i % 2) * 3
                        date = first_draw_date + timedelta(weeks=weeks, days=day_in_week)
                        self.df.loc[i, 'Date'] = date
                    logger.info("✅ Dates automatiques créées")
                
                # Convertir et trier par date
                if 'Date' in self.df.columns:
                    self.df['Date'] = pd.to_datetime(self.df['Date'], errors='coerce')
                    if not self.df['Date'].isna().all():
                        self.df = self.df.sort_values('Date', ascending=True).reset_index(drop=True)
                        logger.info(f"✅ Données triées par date (ordre chronologique: {self.df['Date'].min()} → {self.df['Date'].max()})")
            
            return True
        except FileNotFoundError:
            logger.error(f"Fichier CSV {self.csv_file} non trouvé.")
            return False
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
            return False
    
    def initialize_predictors(self):
        """Initialise les prédicteurs disponibles selon la méthode choisie."""
        # Analyseur principal
        if self.method in ["all", "main"] and MODULES["main_analyzer"]:
            try:
                self.predictors["main"] = EuromillionsAnalyzer(self.config)
                logger.info("Prédicteur principal initialisé.")
            except Exception as e:
                logger.error(f"Erreur lors de l'initialisation du prédicteur principal: {str(e)}")
        
        # Analyseur Fibonacci
        if self.method in ["all", "fibonacci"] and MODULES["fibonacci_analyzer"]:
            try:
                self.predictors["fibonacci"] = EuromillionsFibonacciAnalyzer(self.csv_file, self.output_dir / "fibonacci")
                logger.info("Prédicteur Fibonacci initialisé.")
            except Exception as e:
                logger.error(f"Erreur lors de l'initialisation du prédicteur Fibonacci: {str(e)}")
        
        # Super-optimiseur
        if self.method in ["all", "super"] and MODULES["super_optimizer"]:
            try:
                self.predictors["super"] = EuromillionsSuperOptimizer(
                    csv_file=self.csv_file,
                    output_dir=self.output_dir / "super",
                    combinations=self.combinations_to_generate,
                    show_all=True
                )
                logger.info("Super-optimiseur initialisé.")
            except Exception as e:
                logger.error(f"Erreur lors de l'initialisation du super-optimiseur: {str(e)}")
        
        # Optimiseur de combinaisons (toujours initialisé pour l'optimisation finale)
        if MODULES["combination_optimizer"]:
            try:
                self.predictors["optimizer"] = CombinationOptimizer(
                    num_main_numbers=5,
                    num_stars=2,
                    output_dir=self.output_dir / "optimizer"
                )
                logger.info("Optimiseur de combinaisons initialisé.")
            except Exception as e:
                logger.error(f"Erreur lors de l'initialisation de l'optimiseur de combinaisons: {str(e)}")
        
        # Analyseur vidéo
        if self.method in ["all", "video"] and MODULES["video_analyzer"]:
            try:
                self.predictors["video"] = EuromillionsVideoAnalyzer(
                    video_dir=self.video_dir,
                    output_dir=self.output_dir / "video"
                )
                logger.info("Analyseur vidéo initialisé.")
            except Exception as e:
                logger.error(f"Erreur lors de l'initialisation de l'analyseur vidéo: {str(e)}")
    
    def predict_with_main_analyzer(self):
        """Génère des prédictions avec l'analyseur principal."""
        if "main" not in self.predictors:
            logger.error("Prédicteur principal non disponible.")
            return []
        
        try:
            logger.info("Génération de prédictions avec l'analyseur principal...")
            
            # Exécuter l'analyse si nécessaire
            if not hasattr(self.predictors["main"], "final_main_scores") or not self.predictors["main"].final_main_scores:
                self.predictors["main"].run_analysis()
            
            # Générer les combinaisons
            combinations = self.predictors["main"].generate_euro_combinations(count=self.combinations_to_generate)
            
            logger.info(f"{len(combinations)} combinaisons générées avec l'analyseur principal.")
            return combinations
        except Exception as e:
            logger.error(f"Erreur lors de la génération de prédictions avec l'analyseur principal: {str(e)}")
            logger.debug(traceback.format_exc())
            return []
    
    def predict_with_fibonacci_analyzer(self):
        """Génère des prédictions avec l'analyseur Fibonacci."""
        if "fibonacci" not in self.predictors:
            logger.error("Prédicteur Fibonacci non disponible.")
            return []
        
        try:
            logger.info("Génération de prédictions avec l'analyseur Fibonacci...")
            
            # Analyser les fréquences et appliquer la pondération Fibonacci
            self.predictors["fibonacci"].analyze_frequencies()
            self.predictors["fibonacci"].apply_fibonacci_weighting()
            
            # Générer les combinaisons
            combinations_tuples = self.predictors["fibonacci"].generate_combinations(num_combinations=self.combinations_to_generate)
            
            # Convertir les tuples en listes
            combinations = []
            for numbers, stars in combinations_tuples:
                combinations.append(numbers + stars)
            
            logger.info(f"{len(combinations)} combinaisons générées avec l'analyseur Fibonacci.")
            return combinations
        except Exception as e:
            logger.error(f"Erreur lors de la génération de prédictions avec l'analyseur Fibonacci: {str(e)}")
            logger.debug(traceback.format_exc())
            return []
    
    def predict_with_super_optimizer(self):
        """Génère des prédictions avec le super-optimiseur."""
        if "super" not in self.predictors:
            logger.error("Super-optimiseur non disponible.")
            return []
        
        try:
            logger.info("Génération de prédictions avec le super-optimiseur...")
            
            # Charger les données
            self.predictors["super"].load_data()
            
            # Générer les combinaisons
            self.predictors["super"].generate_combinations()
            
            # Créer la combinaison super-optimisée
            self.predictors["super"].create_super_optimized_combination()
            
            # Récupérer les combinaisons
            combinations = []
            for numbers, stars in self.predictors["super"].normal_combinations:
                combinations.append(numbers + stars)
            
            # Ajouter la combinaison super-optimisée
            if self.predictors["super"].super_optimized_combination:
                numbers, stars = self.predictors["super"].super_optimized_combination
                combinations.append(numbers + stars)
            
            logger.info(f"{len(combinations)} combinaisons générées avec le super-optimiseur.")
            return combinations
        except Exception as e:
            logger.error(f"Erreur lors de la génération de prédictions avec le super-optimiseur: {str(e)}")
            logger.debug(traceback.format_exc())
            return []
    
    def predict_with_video_analyzer(self):
        """Génère des prédictions basées sur l'analyse vidéo."""
        if "video" not in self.predictors:
            logger.error("Analyseur vidéo non disponible.")
            return []
        
        try:
            logger.info("Génération de prédictions avec l'analyseur vidéo...")
            
            # Lister les vidéos disponibles
            videos = self.predictors["video"].list_videos()
            
            if not videos:
                logger.warning("Aucune vidéo disponible pour analyse.")
                return []
            
            # Analyser la dernière vidéo
            latest_video = videos[-1]
            logger.info(f"Analyse de la vidéo: {latest_video}")
            
            video_results = self.predictors["video"].analyze_video(latest_video)
            
            if not video_results or not video_results.get("success", False):
                logger.error("Échec de l'analyse vidéo.")
                return []
            
            # Générer des prédictions basées sur les résultats vidéo
            combinations = self.generate_predictions_from_video(video_results)
            
            logger.info(f"{len(combinations)} combinaisons générées avec l'analyseur vidéo.")
            return combinations
        except Exception as e:
            logger.error(f"Erreur lors de la génération de prédictions avec l'analyseur vidéo: {str(e)}")
            logger.debug(traceback.format_exc())
            return []
    
    def generate_predictions_from_video(self, video_results):
        """
        Génère des prédictions basées sur les résultats de l'analyse vidéo.
        
        Args:
            video_results: Résultats de l'analyse vidéo
            
        Returns:
            Liste de combinaisons
        """
        try:
            # Extraire les numéros et étoiles détectés
            main_numbers = video_results.get("main_numbers", [])
            stars = video_results.get("stars", [])
            
            if not main_numbers or not stars:
                logger.warning("Aucun numéro ou étoile détecté dans la vidéo.")
                return []
            
            # Générer des variations autour des numéros et étoiles détectés
            combinations = []
            
            # Ajouter la combinaison détectée
            if len(main_numbers) >= 5 and len(stars) >= 2:
                combinations.append(main_numbers[:5] + stars[:2])
            
            # Générer des variations
            import random
            
            # Charger tous les numéros et étoiles possibles
            all_numbers = list(range(1, 51))
            all_stars = list(range(1, 13))
            
            for _ in range(self.combinations_to_generate - 1):
                # Conserver certains numéros détectés et en remplacer d'autres
                num_to_keep = random.randint(2, 4)
                kept_numbers = random.sample(main_numbers[:5], min(num_to_keep, len(main_numbers[:5])))
                
                # Compléter avec des numéros aléatoires
                remaining_numbers = [n for n in all_numbers if n not in kept_numbers]
                additional_numbers = random.sample(remaining_numbers, 5 - len(kept_numbers))
                
                new_numbers = sorted(kept_numbers + additional_numbers)
                
                # Faire de même pour les étoiles
                num_stars_to_keep = random.randint(0, 1)
                kept_stars = random.sample(stars[:2], min(num_stars_to_keep, len(stars[:2])))
                
                remaining_stars = [s for s in all_stars if s not in kept_stars]
                additional_stars = random.sample(remaining_stars, 2 - len(kept_stars))
                
                new_stars = sorted(kept_stars + additional_stars)
                
                combinations.append(new_numbers + new_stars)
            
            return combinations
        except Exception as e:
            logger.error(f"Erreur lors de la génération de prédictions à partir des résultats vidéo: {str(e)}")
            logger.debug(traceback.format_exc())
            return []
    
    def generate_fallback_predictions(self):
        """
        Génère des prédictions de secours en cas d'échec des autres méthodes.
        
        Returns:
            Liste de combinaisons
        """
        logger.info("Génération de prédictions de secours...")
        
        try:
            import random
            
            combinations = []
            
            for _ in range(self.combinations_to_generate):
                # Générer 5 numéros aléatoires entre 1 et 50
                numbers = sorted(random.sample(range(1, 51), 5))
                
                # Générer 2 étoiles aléatoires entre 1 et 12
                stars = sorted(random.sample(range(1, 13), 2))
                
                combinations.append(numbers + stars)
            
            logger.info(f"{len(combinations)} combinaisons de secours générées.")
            return combinations
        except Exception as e:
            logger.error(f"Erreur lors de la génération de prédictions de secours: {str(e)}")
            logger.debug(traceback.format_exc())
            return []
    
    def optimize_combinations(self, combinations):
        """
        Optimise les combinaisons générées.
        
        Args:
            combinations: Liste de combinaisons à optimiser
            
        Returns:
            Liste de combinaisons optimisées
        """
        if "optimizer" not in self.predictors or not combinations:
            return combinations
        
        try:
            logger.info("Optimisation des combinaisons...")
            
            # Récupérer les scores des numéros et étoiles si disponibles
            number_scores = None
            star_scores = None
            
            if "main" in self.predictors and hasattr(self.predictors["main"], "final_main_scores"):
                number_scores = self.predictors["main"].final_main_scores
                star_scores = self.predictors["main"].final_star_scores
            elif "fibonacci" in self.predictors and hasattr(self.predictors["fibonacci"], "number_weights"):
                number_scores = self.predictors["fibonacci"].number_weights
                star_scores = self.predictors["fibonacci"].star_weights
            
            # Optimiser les combinaisons
            optimized_combinations = self.predictors["optimizer"].optimize_combinations(
                combinations=combinations,
                number_scores=number_scores,
                star_scores=star_scores
            )
            
            # Sauvegarder les combinaisons optimisées
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.predictors["optimizer"].save_optimized_combinations(
                original_combinations=combinations,
                optimized_combinations=optimized_combinations,
                timestamp=timestamp
            )
            
            logger.info(f"{len(optimized_combinations)} combinaisons optimisées.")
            return optimized_combinations
        except Exception as e:
            logger.error(f"Erreur lors de l'optimisation des combinaisons: {str(e)}")
            logger.debug(traceback.format_exc())
            return combinations
    
    def generate_predictions(self):
        """
        Génère des prédictions selon la méthode choisie.
        
        Returns:
            Liste de combinaisons prédites
        """
        logger.info(f"Génération de prédictions avec la méthode: {self.method}")
        
        combinations = []
        
        # Générer les prédictions selon la méthode choisie
        if self.method == "all":
            # Utiliser toutes les méthodes disponibles
            all_combinations = []
            
            # Prédictions avec l'analyseur principal
            if "main" in self.predictors:
                main_combinations = self.predict_with_main_analyzer()
                all_combinations.extend(main_combinations)
            
            # Prédictions avec l'analyseur Fibonacci
            if "fibonacci" in self.predictors:
                fibonacci_combinations = self.predict_with_fibonacci_analyzer()
                all_combinations.extend(fibonacci_combinations)
            
            # Prédictions avec le super-optimiseur
            if "super" in self.predictors:
                super_combinations = self.predict_with_super_optimizer()
                all_combinations.extend(super_combinations)
            
            # Prédictions avec l'analyseur vidéo
            if "video" in self.predictors:
                video_combinations = self.predict_with_video_analyzer()
                all_combinations.extend(video_combinations)
            
            # Si aucune prédiction n'a été générée, utiliser la méthode de secours
            if not all_combinations:
                all_combinations = self.generate_fallback_predictions()
            
            # Optimiser les combinaisons
            combinations = self.optimize_combinations(all_combinations)
            
        elif self.method == "main":
            # Prédictions avec l'analyseur principal
            combinations = self.predict_with_main_analyzer()
            
            # Si aucune prédiction n'a été générée, utiliser la méthode de secours
            if not combinations:
                combinations = self.generate_fallback_predictions()
            
            # Optimiser les combinaisons
            combinations = self.optimize_combinations(combinations)
            
        elif self.method == "fibonacci":
            # Prédictions avec l'analyseur Fibonacci
            combinations = self.predict_with_fibonacci_analyzer()
            
            # Si aucune prédiction n'a été générée, utiliser la méthode de secours
            if not combinations:
                combinations = self.generate_fallback_predictions()
            
            # Optimiser les combinaisons
            combinations = self.optimize_combinations(combinations)
            
        elif self.method == "super":
            # Prédictions avec le super-optimiseur
            combinations = self.predict_with_super_optimizer()
            
            # Si aucune prédiction n'a été générée, utiliser la méthode de secours
            if not combinations:
                combinations = self.generate_fallback_predictions()
            
        elif self.method == "video":
            # Prédictions avec l'analyseur vidéo
            combinations = self.predict_with_video_analyzer()
            
            # Si aucune prédiction n'a été générée, utiliser la méthode de secours
            if not combinations:
                combinations = self.generate_fallback_predictions()
            
            # Optimiser les combinaisons
            combinations = self.optimize_combinations(combinations)
            
        elif self.method == "fallback":
            # Prédictions de secours
            combinations = self.generate_fallback_predictions()
        
        # Limiter le nombre de combinaisons si nécessaire
        if len(combinations) > self.combinations_to_generate:
            combinations = combinations[:self.combinations_to_generate]
        
        self.predictions = combinations
        return combinations
    
    def save_predictions(self):
        """
        Sauvegarde les prédictions dans un fichier texte.
        
        Returns:
            Chemin du fichier sauvegardé
        """
        if not self.predictions:
            logger.warning("Aucune prédiction à sauvegarder.")
            return None
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = self.output_dir / f"predictions_{self.method}_{timestamp}.txt"
            
            with open(results_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write(f"PRÉDICTIONS EUROMILLIONS ({self.method.upper()})\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Date de génération: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Méthode: {self.method}\n")
                f.write(f"Fichier CSV: {self.csv_file}\n\n")
                
                f.write("-" * 80 + "\n")
                f.write("COMBINAISONS PRÉDITES\n")
                f.write("-" * 80 + "\n\n")
                
                for i, combo in enumerate(self.predictions, 1):
                    if len(combo) >= 7:
                        main_nums = combo[:5]
                        stars = combo[5:7]
                        f.write(f"Combinaison {i}: {' - '.join(map(str, main_nums))} | Étoiles: {' - '.join(map(str, stars))}\n")
                    else:
                        f.write(f"Combinaison {i}: Format incorrect - {combo}\n")
                
                f.write("\n")
                f.write("-" * 80 + "\n")
                f.write("INFORMATIONS SUR LA PRÉDICTION\n")
                f.write("-" * 80 + "\n")
                f.write(f"Nombre de combinaisons: {len(self.predictions)}\n")
                f.write(f"Modules disponibles: {', '.join([k for k, v in MODULES.items() if v])}\n")
                f.write(f"Méthode utilisée: {self.method}\n")
            
            logger.info(f"Prédictions sauvegardées dans: {results_file}")
            return str(results_file)
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des prédictions: {str(e)}")
            logger.debug(traceback.format_exc())
            return None
    
    def save_predictions_json(self):
        """
        Sauvegarde les prédictions dans un fichier JSON.
        
        Returns:
            Chemin du fichier sauvegardé
        """
        if not self.predictions:
            logger.warning("Aucune prédiction à sauvegarder en JSON.")
            return None
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_file = self.output_dir / f"predictions_{self.method}_{timestamp}.json"
            
            # Préparer les données JSON
            predictions_data = []
            
            for i, combo in enumerate(self.predictions, 1):
                if len(combo) >= 7:
                    prediction = {
                        "id": i,
                        "main_numbers": combo[:5],
                        "stars": combo[5:7],
                        "method": self.method,
                        "timestamp": timestamp
                    }
                    predictions_data.append(prediction)
            
            # Créer l'objet JSON complet
            json_data = {
                "metadata": {
                    "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "method": self.method,
                    "csv_file": self.csv_file,
                    "modules_available": {k: v for k, v in MODULES.items()},
                    "combinations_count": len(self.predictions)
                },
                "predictions": predictions_data
            }
            
            # Sauvegarder en JSON
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2)
            
            logger.info(f"Prédictions sauvegardées en JSON dans: {json_file}")
            return str(json_file)
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des prédictions en JSON: {str(e)}")
            logger.debug(traceback.format_exc())
            return None
    
    def run(self):
        """
        Exécute le processus complet de prédiction.
        
        Returns:
            Tuple (succès, chemin du fichier de résultats)
        """
        try:
            # Générer les prédictions
            self.generate_predictions()
            
            if not self.predictions:
                logger.error("Échec de la génération des prédictions.")
                return False, None
            
            # Sauvegarder les prédictions
            results_file = self.save_predictions()
            
            # Sauvegarder en JSON si demandé
            if self.config.get("save_json", False):
                self.save_predictions_json()
            
            logger.info("Processus de prédiction terminé avec succès.")
            return True, results_file
        except Exception as e:
            logger.error(f"Erreur lors du processus de prédiction: {str(e)}")
            logger.debug(traceback.format_exc())
            return False, None

def parse_arguments():
    """
    Parse les arguments de ligne de commande pour la prédiction Euromillions.
    """
    parser = argparse.ArgumentParser(description="Prédicteur unifié pour les combinaisons Euromillions")
    
    parser.add_argument("--csv", type=str, default="tirage_euromillions.csv",
                        help="Chemin vers le fichier CSV des tirages EuroMillions")
    parser.add_argument("--output", type=str, default="resultats_euromillions",
                        help="Répertoire de sortie pour les résultats")
    parser.add_argument("--model-dir", type=str, default="models_euromillions",
                        help="Répertoire des modèles entraînés")
    parser.add_argument("--video-dir", type=str, default="tirage_video",
                        help="Répertoire des vidéos de tirage")
    
    parser.add_argument("--method", type=str, 
                        choices=["all", "main", "fibonacci", "super", "video", "fallback"],
                        default="all", help="Méthode de prédiction à utiliser")
    
    parser.add_argument("--combinations", type=int, default=5,
                        help="Nombre de combinaisons à générer")
    
    parser.add_argument("--run", action="store_true",
                        help="Exécuter le processus de prédiction")
    
    parser.add_argument("--save-json", action="store_true",
                        help="Sauvegarder les prédictions en format JSON")
    
    parser.add_argument("--check-deps", action="store_true",
                        help="Vérifier les dépendances disponibles")
    
    return parser.parse_args()

def main():
    """
    Fonction principale pour la prédiction des combinaisons Euromillions.
    """
    args = parse_arguments()
    
    # Vérifier les dépendances si demandé
    if args.check_deps:
        print("Modules disponibles:")
        for module, available in MODULES.items():
            status = "Disponible" if available else "Non disponible"
            print(f"- {module}: {status}")
        return 0
    
    # Générer un timestamp unique pour cette exécution
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"Début de la prédiction des combinaisons Euromillions (Timestamp: {timestamp}).")
    logger.info(f"Arguments: {args}")
    
    # Créer la configuration
    config = {
        "csv_file": args.csv,
        "output_dir": args.output,
        "model_dir": args.model_dir,
        "video_dir": args.video_dir,
        "method": args.method,
        "combinations": args.combinations,
        "save_json": args.save_json,
        "timestamp": timestamp,
        # ⚠️ CRITIQUE : Activer le système quantique pour toutes les prédictions
        "use_quantum": True,
        "use_qnn": True,
        "use_qlstm": True,
        "use_quantum_annealing": True
    }
    
    try:
        # Créer le prédicteur
        predictor = EuromillionsPredictor(config)
        
        # Exécuter le processus de prédiction si demandé
        if args.run:
            success, results_file = predictor.run()
            
            if success:
                print(f"Prédiction terminée avec succès.")
                print(f"Résultats sauvegardés dans: {results_file}")
            else:
                print("La prédiction a échoué. Consultez les logs pour plus d'informations.")
                return 1
        else:
            print("Utilisation: ajoutez l'option --run pour exécuter le processus de prédiction.")
        
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {str(e)}")
        logger.debug(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
