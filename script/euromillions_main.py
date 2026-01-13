#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module principal pour l'analyse et la prédiction EuroMillions
Ce module coordonne l'ensemble du pipeline d'analyse et de prédiction.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import importlib.util

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("EuromillionsMain")

# Vérifier si les modules optionnels sont disponibles
try:
    import ephem
    EPHEM_AVAILABLE = True
except ImportError:
    EPHEM_AVAILABLE = False
    logger.warning("Module 'ephem' non trouvé. L'analyse des cycles lunaires sera désactivée.")

# Vérifier si le module combination_optimizer est disponible
COMBINATION_OPTIMIZER_AVAILABLE = False
try:
    spec = importlib.util.find_spec("combination_optimizer")
    if spec is not None:
        import combination_optimizer
        COMBINATION_OPTIMIZER_AVAILABLE = True
except ImportError:
    logger.warning("Module 'combination_optimizer.py' non trouvé. L'optimisation des combinaisons sera désactivée.")

# Importer les modules locaux
try:
    from euromillions_predictor_optimizer import EuromillionsCombinedPredictor
    from error_analyzer import ErrorAnalyzer
except ImportError as e:
    logger.error(f"Erreur lors de l'importation des modules locaux: {str(e)}")
    sys.exit(1)

class EuromillionsMainAnalyzer:
    """
    Classe principale pour l'analyse et la prédiction EuroMillions.
    Coordonne l'ensemble du pipeline d'analyse et de prédiction.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialise l'analyseur avec la configuration spécifiée.
        
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
            "output_dir": "resultats_euromillions",  # Répertoire de sortie
            "random_seed": 42,  # Graine aléatoire pour la reproductibilité
            "test_size": 10,  # Nombre de tirages à utiliser pour le test
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
        self.lunar_analyzer = None  # Analyseur de cycle lunaire
        self.combination_optimizer = None  # Optimiseur de combinaisons
        self.predictor = None  # Prédicteur combiné
        self.error_analyzer = None  # Analyseur d'erreurs
        
        # Fixer la graine aléatoire
        random.seed(self.config["random_seed"])
        np.random.seed(self.config["random_seed"])
        
        logger.info("EuromillionsAnalyzer initialisé avec succès.")
    
    def load_data(self, csv_file: str) -> pd.DataFrame:
        """
        Charge les données des tirages depuis un fichier CSV.
        
        Args:
            csv_file: Chemin vers le fichier CSV des tirages
            
        Returns:
            pd.DataFrame: DataFrame des tirages
        """
        logger.info(f"Chargement des données depuis {csv_file}...")
        
        try:
            # Charger les données
            df = pd.read_csv(csv_file, sep=";")
            
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
            
            # Trier par date (plus récents en premier)
            df = df.sort_values(by=self.config["date_col"], ascending=False)
            
            # Stocker le DataFrame
            self.df = df
            
            logger.info(f"Données chargées avec succès: {len(df)} tirages.")
            return df
        
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données: {str(e)}")
            return None
    
    def initialize_components(self) -> bool:
        """
        Initialise les composants d'analyse et de prédiction.
        
        Returns:
            bool: True si l'initialisation est réussie, False sinon
        """
        logger.info("Initialisation des composants...")
        
        try:
            # Initialiser l'analyseur de cycle lunaire si disponible
            if EPHEM_AVAILABLE:
                from lunar_cycle_analyzer import LunarCycleAnalyzer
                self.lunar_analyzer = LunarCycleAnalyzer()
                logger.info("Analyseur de cycle lunaire initialisé.")
            
            # Initialiser l'optimiseur de combinaisons si disponible
            if COMBINATION_OPTIMIZER_AVAILABLE:
                self.combination_optimizer = combination_optimizer.CombinationOptimizer()
                logger.info("Optimiseur de combinaisons initialisé.")
            
            # Initialiser le prédicteur combiné
            predictor_config = {
                "max_number": self.config["max_number"],
                "max_star_number": self.config["max_star_number"],
                "num_numbers": self.config["num_numbers"],
                "num_stars": self.config["num_stars"],
                "number_cols": self.config["number_cols"],
                "star_cols": self.config["star_cols"],
                "date_col": self.config["date_col"],
                "output_dir": self.config["output_dir"],
                "random_seed": self.config["random_seed"],
                "use_lunar_data": EPHEM_AVAILABLE,
            }
            self.predictor = EuromillionsCombinedPredictor(predictor_config)
            logger.info("Prédicteur combiné initialisé.")
            
            # Initialiser l'analyseur d'erreurs
            error_output_dir = self.output_dir / "error_analysis"
            if not error_output_dir.exists():
                error_output_dir.mkdir(parents=True)
            
            error_analyzer_config = {
                "output_dir": str(error_output_dir),
            }
            self.error_analyzer = ErrorAnalyzer(str(error_output_dir))
            logger.info(f"ErrorAnalyzer initialisé avec output_dir: {error_output_dir}")
            logger.info("Analyseur d'erreurs initialisé.")
            
            logger.info("Initialisation des composants terminée.")
            return True
        
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation des composants: {str(e)}")
            return False
    
    def train_models(self, train_all: bool = False) -> bool:
        """
        Entraîne les modèles de prédiction.
        
        Args:
            train_all: Si True, entraîne tous les modèles, sinon utilise les modèles existants si disponibles
            
        Returns:
            bool: True si l'entraînement est réussi, False sinon
        """
        logger.info("Entraînement des modèles...")
        
        try:
            start_time = time.time()
            
            # Charger les données dans le prédicteur
            self.predictor.load_data(self.df)
            
            # Entraîner les modèles
            models = self.predictor.train_models()
            
            # Sauvegarder les modèles
            self.predictor.save_models()
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            logger.info(f"Modèles entraînés et sauvegardés avec succès en {elapsed_time:.2f} secondes.")
            return True
        
        except Exception as e:
            logger.error(f"Erreur lors de l'entraînement des modèles: {str(e)}")
            return False
    
    def generate_combinations(self, num_combinations: int = 5) -> List[Dict]:
        """
        Génère des combinaisons EuroMillions.
        
        Args:
            num_combinations: Nombre de combinaisons à générer
            
        Returns:
            List[Dict]: Liste de combinaisons, chaque combinaison étant un dictionnaire avec 'numbers' et 'stars'
        """
        logger.info(f"Génération de {num_combinations} combinaisons...")
        
        try:
            start_time = time.time()
            
            # Générer les combinaisons
            combinations = self.predictor.generate_combinations(num_combinations)
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # Afficher les combinaisons
            logger.info(f"Combinaisons générées en {elapsed_time:.2f} secondes:")
            for i, combo in enumerate(combinations, 1):
                numbers_str = ", ".join(map(str, combo['numbers']))
                stars_str = ", ".join(map(str, combo['stars']))
                logger.info(f"Combinaison {i}: Numéros [{numbers_str}] - Étoiles [{stars_str}]")
            
            # Sauvegarder les combinaisons
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            combinations_file = self.output_dir / f"combinations_{timestamp}.txt"
            
            with open(combinations_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("COMBINAISONS EUROMILLIONS\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Date de génération: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                for i, combo in enumerate(combinations, 1):
                    numbers_str = ", ".join(map(str, combo['numbers']))
                    stars_str = ", ".join(map(str, combo['stars']))
                    f.write(f"Combinaison {i}: Numéros [{numbers_str}] - Étoiles [{stars_str}]\n")
            
            logger.info(f"Combinaisons sauvegardées dans {combinations_file}")
            return combinations
        
        except Exception as e:
            logger.error(f"Erreur lors de la génération des combinaisons: {str(e)}")
            return []
    
    def run_backtesting(self, test_size: int = None) -> Dict:
        """
        Exécute un backtesting sur les derniers tirages.
        
        Args:
            test_size: Nombre de tirages à utiliser pour le test (si None, utilise la valeur de configuration)
            
        Returns:
            Dict: Résultats du backtesting
        """
        if test_size is None:
            test_size = self.config["test_size"]
        
        logger.info(f"Exécution du backtesting...")
        
        try:
            # Exécuter le backtesting
            results = self.predictor.run_backtesting(test_size)
            
            # Sauvegarder les résultats
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = self.output_dir / f"backtesting_results_{timestamp}.txt"
            
            with open(results_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("RÉSULTATS DU BACKTESTING EUROMILLIONS\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Date d'exécution: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Nombre de tirages testés: {test_size}\n\n")
                
                f.write("STATISTIQUES GLOBALES:\n")
                f.write(f"Précision moyenne (numéros): {results['mean_accuracy_numbers']:.4f}\n")
                f.write(f"Précision moyenne (étoiles): {results['mean_accuracy_stars']:.4f}\n")
                f.write(f"Précision moyenne (combinaisons): {results['mean_accuracy_combinations']:.4f}\n\n")
                
                f.write("RÉSULTATS DÉTAILLÉS:\n")
                for i, result in enumerate(results.get('detailed_results', []), 1):
                    f.write(f"Tirage {i} ({result['date']}):\n")
                    f.write(f"  Numéros réels: {result['actual_numbers']}\n")
                    f.write(f"  Étoiles réelles: {result['actual_stars']}\n")
                    f.write(f"  Numéros prédits: {result['predicted_numbers']}\n")
                    f.write(f"  Étoiles prédites: {result['predicted_stars']}\n")
                    f.write(f"  Numéros corrects: {result['correct_numbers']}/{self.config['num_numbers']}\n")
                    f.write(f"  Étoiles correctes: {result['correct_stars']}/{self.config['num_stars']}\n")
                    f.write(f"  Rang: {result['rank']}\n\n")
            
            logger.info(f"Résultats du backtesting sauvegardés dans {results_file}")
            return results
        
        except Exception as e:
            logger.error(f"Erreur lors du backtesting: {str(e)}")
            return {}
    
    def analyze_errors(self) -> Dict:
        """
        Analyse les erreurs de prédiction.
        
        Returns:
            Dict: Résultats de l'analyse d'erreurs
        """
        logger.info("Comparaison des prédictions avec les tirages réels (simulation)...")
        
        try:
            # Récupérer les prédictions et les tirages réels
            past_predictions = self.predictor.get_past_predictions()
            actual_draws = self.predictor.get_actual_draws_for_predictions()
            prediction_dates = self.predictor.get_prediction_dates()
            
            if not past_predictions or not actual_draws:
                logger.warning("Pas de données de prédiction ou de tirages réels disponibles.")
                return {}
            
            # Créer les entrées d'erreur
            error_entries = []
            for i, (pred, actual, date) in enumerate(zip(past_predictions, actual_draws, prediction_dates)):
                error_entry = {
                    'date': date,
                    'predicted_numbers': pred['numbers'],
                    'predicted_stars': pred['stars'],
                    'actual_numbers': actual['numbers'],
                    'actual_stars': actual['stars'],
                    'error_numbers': [n for n in pred['numbers'] if n not in actual['numbers']],
                    'error_stars': [s for s in pred['stars'] if s not in actual['stars']],
                    'missed_numbers': [n for n in actual['numbers'] if n not in pred['numbers']],
                    'missed_stars': [s for s in actual['stars'] if s not in pred['stars']],
                }
                error_entries.append(error_entry)
            
            logger.info(f"Comparaison terminée. {len(error_entries)} entrées d'erreur générées.")
            
            # Analyser les erreurs
            logger.info("Analyse des erreurs...")
            error_analysis = self.error_analyzer.analyze_errors(error_entries)
            logger.info("Analyse des erreurs terminée.")
            
            # Sauvegarder les résultats
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            analysis_file = self.output_dir / "error_analysis" / f"error_analysis_results_{timestamp}.txt"
            
            logger.info(f"Sauvegarde des résultats de l'analyse d'erreurs dans {analysis_file}...")
            with open(analysis_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("RÉSULTATS DE L'ANALYSE D'ERREURS EUROMILLIONS\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Date d'exécution: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Nombre d'entrées analysées: {len(error_entries)}\n\n")
                
                f.write("STATISTIQUES GLOBALES:\n")
                f.write(f"Taux d'erreur moyen (numéros): {error_analysis.get('mean_error_rate_numbers', 0):.4f}\n")
                f.write(f"Taux d'erreur moyen (étoiles): {error_analysis.get('mean_error_rate_stars', 0):.4f}\n\n")
                
                f.write("NUMÉROS LES PLUS SOUVENT MANQUÉS:\n")
                for num, count in error_analysis.get('most_missed_numbers', []):
                    f.write(f"  Numéro {num}: {count} fois\n")
                f.write("\n")
                
                f.write("ÉTOILES LES PLUS SOUVENT MANQUÉES:\n")
                for star, count in error_analysis.get('most_missed_stars', []):
                    f.write(f"  Étoile {star}: {count} fois\n")
                f.write("\n")
                
                f.write("NUMÉROS LES PLUS SOUVENT PRÉDITS À TORT:\n")
                for num, count in error_analysis.get('most_error_numbers', []):
                    f.write(f"  Numéro {num}: {count} fois\n")
                f.write("\n")
                
                f.write("ÉTOILES LES PLUS SOUVENT PRÉDITES À TORT:\n")
                for star, count in error_analysis.get('most_error_stars', []):
                    f.write(f"  Étoile {star}: {count} fois\n")
            
            logger.info("Résultats de l'analyse d'erreurs sauvegardés.")
            
            # Exporter les erreurs détaillées en CSV
            detailed_errors_file = self.output_dir / "error_analysis" / f"detailed_errors_{timestamp}.csv"
            logger.info(f"Exportation des erreurs détaillées vers {detailed_errors_file}...")
            
            # Créer un DataFrame à partir des entrées d'erreur
            error_df = pd.DataFrame(error_entries)
            
            # Convertir les listes en chaînes pour le CSV
            for col in ['predicted_numbers', 'predicted_stars', 'actual_numbers', 'actual_stars', 
                        'error_numbers', 'error_stars', 'missed_numbers', 'missed_stars']:
                error_df[col] = error_df[col].apply(lambda x: ','.join(map(str, x)))
            
            # Exporter en CSV
            error_df.to_csv(detailed_errors_file, index=False)
            logger.info("Erreurs détaillées exportées.")
            
            return error_analysis
        
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse des erreurs: {str(e)}")
            return {}
    
    def run_analysis(self, csv_file: str, num_combinations: int = 5, train_all: bool = False) -> Dict:
        """
        Exécute l'analyse complète EuroMillions.
        
        Args:
            csv_file: Chemin vers le fichier CSV des tirages
            num_combinations: Nombre de combinaisons à générer
            train_all: Si True, entraîne tous les modèles, sinon utilise les modèles existants si disponibles
            
        Returns:
            Dict: Résultats de l'analyse
        """
        try:
            # Charger les données
            loaded_df = self.load_data(csv_file)
            if loaded_df is None or self.df is None or self.df.empty:
                logger.error("Échec du chargement des données. Arrêt de l'analyse.")
                return {}
            
            # Initialiser les composants
            if not self.initialize_components():
                logger.error("Échec de l'initialisation des composants. Arrêt de l'analyse.")
                return {}
            
            # Entraîner les modèles si nécessaire
            if train_all:
                if not self.train_models(train_all):
                    logger.error("Échec de l'entraînement des modèles. Arrêt de l'analyse.")
                    return {}
            
            # Générer les combinaisons
            combinations = self.generate_combinations(num_combinations)
            if not combinations:
                logger.error("Échec de la génération des combinaisons. Arrêt de l'analyse.")
                return {}
            
            # Exécuter le backtesting
            backtesting_results = self.run_backtesting()
            
            # Analyser les erreurs
            error_analysis = self.analyze_errors()
            
            # Résultats
            results = {
                'combinations': combinations,
                'backtesting_results': backtesting_results,
                'error_analysis': error_analysis
            }
            
            logger.info("Analyse EuroMillions terminée avec succès.")
            return results
        
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse EuroMillions: {str(e)}")
            return {}


def main():
    """
    Fonction principale pour l'exécution du script.
    """
    import argparse
    
    # Définir les arguments de ligne de commande
    parser = argparse.ArgumentParser(description="Analyse et prédiction EuroMillions")
    parser.add_argument("--csv", required=True, help="Chemin vers le fichier CSV des tirages")
    parser.add_argument("--combinations", type=int, default=5, help="Nombre de combinaisons à générer")
    parser.add_argument("--output-dir", default="resultats_euromillions", help="Répertoire de sortie")
    parser.add_argument("--all", action="store_true", help="Exécuter l'analyse complète (entraînement, prédiction, backtesting)")
    
    # Analyser les arguments
    args = parser.parse_args()
    
    # Configuration
    config = {
        "output_dir": args.output_dir
    }
    
    # Créer l'analyseur
    analyzer = EuromillionsMainAnalyzer(config)
    
    # Exécuter l'analyse
    analyzer.run_analysis(args.csv, args.combinations, args.all)


if __name__ == "__main__":
    logger.info("Démarrage de l'analyse EuroMillions...")
    main()
