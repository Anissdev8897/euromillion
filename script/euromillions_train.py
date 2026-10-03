#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module d'entraînement unifié pour l'analyseur Euromillions.
Ce script centralise toutes les fonctions d'entraînement des différents modèles
d'analyse et de prédiction Euromillions.

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
try:
    from safe_model_loader import ModelIntegrityError  # type: ignore
except ImportError:  # pragma: no cover
    from script.safe_model_loader import ModelIntegrityError  # type: ignore

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("EuromillionsTrainer")

# Ignorer les avertissements
warnings.filterwarnings("ignore")

# Assurer que le répertoire courant est dans sys.path pour l'importation des modules locaux
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importation conditionnelle des modules
try:
    from euromillions_analyzer import EuromillionsAdvancedAnalyzer
    # Créer un alias pour compatibilité
    EuromillionsAnalyzer = EuromillionsAdvancedAnalyzer
    ANALYZER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Module euromillions_analyzer non disponible: {str(e)}")
    ANALYZER_AVAILABLE = False

try:
    from euromillions_fibonacci_analyzer import EuromillionsFibonacciAnalyzer
    FIBONACCI_AVAILABLE = True
except ImportError:
    logger.warning("Module euromillions_fibonacci_analyzer non disponible.")
    FIBONACCI_AVAILABLE = False

try:
    from lunar_cycle_analyzer import LunarCycleAnalyzer
    LUNAR_AVAILABLE = True
except ImportError:
    logger.warning("Module lunar_cycle_analyzer non disponible.")
    LUNAR_AVAILABLE = False

try:
    from incremental_learning import EuromillionsIncrementalLearning
    INCREMENTAL_AVAILABLE = True
except ImportError:
    logger.warning("Module incremental_learning non disponible.")
    INCREMENTAL_AVAILABLE = False

try:
    from error_analyzer import ErrorAnalyzer
    ERROR_ANALYZER_AVAILABLE = True
except ImportError:
    logger.warning("Module error_analyzer non disponible.")
    ERROR_ANALYZER_AVAILABLE = False

try:
    from advanced_encoder import AdvancedEuromillionsEncoder
    ADVANCED_ENCODER_AVAILABLE = True
    logger.info("✅ Encodeur avancé disponible - Amélioration de la précision activée")
except ImportError:
    ADVANCED_ENCODER_AVAILABLE = False
    logger.warning("⚠️ Encodeur avancé non disponible - Utilisation des features de base")

class EuromillionsTrainer:
    """Classe pour l'entraînement unifié des modèles Euromillions."""
    
    def __init__(self, config):
        """
        Initialise l'entraîneur avec la configuration spécifiée.
        
        Args:
            config: Dictionnaire de configuration
        """
        self.config = config
        self.csv_file = config.get("csv_file", "tirage_euromillions.csv")
        self.output_dir = Path(config.get("output_dir", "resultats_euromillions"))
        self.model_dir = Path(config.get("model_dir", "models_euromillions"))
        self.video_embeddings = config.get("video_embeddings", None)  # 🎥 NOUVEAU: Embeddings vidéo
        
        # Créer les répertoires nécessaires
        for directory in [self.output_dir, self.model_dir]:
            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)
                logger.info(f"Répertoire créé: {directory}")
        
        # Charger les données
        self.df = None
        self.load_data()
        
        # Vérifier et logger l'utilisation de l'encodeur avancé
        if ADVANCED_ENCODER_AVAILABLE:
            logger.info("✅ Tous les analyseurs utiliseront l'encodeur avancé pour améliorer la précision")
        
        # Initialiser les analyseurs disponibles
        self.analyzers = {}
        self.initialize_analyzers()
        
        logger.info(f"EuromillionsTrainer initialisé avec output_dir: {self.output_dir}")
    
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
    
    def initialize_analyzers(self):
        """Initialise les analyseurs disponibles avec encodeur avancé."""
        # Analyseur principal (avec encodeur avancé intégré)
        if ANALYZER_AVAILABLE:
            try:
                # L'encodeur avancé est automatiquement initialisé dans EuromillionsAnalyzer
                # 🎥 NOUVEAU: Passer les embeddings vidéo à l'analyseur
                config_with_video = self.config.copy()
                config_with_video["video_embeddings"] = self.video_embeddings
                self.analyzers["main"] = EuromillionsAnalyzer(config_with_video)
                logger.info("Analyseur principal initialisé avec encodeur avancé.")
                if hasattr(self.analyzers["main"], 'advanced_encoder') and self.analyzers["main"].advanced_encoder:
                    logger.info("✅ Encodeur avancé activé dans l'analyseur principal")
                    if hasattr(self.analyzers["main"].advanced_encoder, 'ai_reflection') and self.analyzers["main"].advanced_encoder.ai_reflection:
                        logger.info("✅ Système de réflexion IA activé")
            except ModelIntegrityError:
                raise  # ne pas masquer un refus d'intégrité des modèles (mode strict)
            except Exception as e:
                logger.error(f"Erreur lors de l'initialisation de l'analyseur principal: {str(e)}")
        
        # Analyseur Fibonacci
        if FIBONACCI_AVAILABLE:
            try:
                self.analyzers["fibonacci"] = EuromillionsFibonacciAnalyzer(self.csv_file, self.output_dir / "fibonacci")
                logger.info("Analyseur Fibonacci initialisé.")
            except Exception as e:
                logger.error(f"Erreur lors de l'initialisation de l'analyseur Fibonacci: {str(e)}")
        
        # Analyseur de cycle lunaire
        if LUNAR_AVAILABLE:
            try:
                self.analyzers["lunar"] = LunarCycleAnalyzer(self.output_dir / "lunar")
                logger.info("Analyseur de cycle lunaire initialisé.")
            except Exception as e:
                logger.error(f"Erreur lors de l'initialisation de l'analyseur de cycle lunaire: {str(e)}")
        
        # Apprentissage incrémental
        if INCREMENTAL_AVAILABLE:
            try:
                # ⚠️ CRITIQUE : L'apprentissage incrémental nécessite l'analyseur principal
                if "main" in self.analyzers:
                    from incremental_learning import EuromillionsIncrementalLearning
                    self.analyzers["incremental"] = EuromillionsIncrementalLearning(self.analyzers["main"])
                    logger.info("Module d'apprentissage incrémental initialisé avec l'analyseur principal.")
                else:
                    logger.warning("⚠️ Analyseur principal non disponible. Apprentissage incrémental désactivé.")
            except ModelIntegrityError:
                raise  # ne pas masquer un refus d'intégrité des modèles (mode strict)
            except Exception as e:
                logger.error(f"Erreur lors de l'initialisation du module d'apprentissage incrémental: {str(e)}")
                logger.debug(traceback.format_exc())
        
        # Analyseur d'erreurs
        if ERROR_ANALYZER_AVAILABLE:
            try:
                self.analyzers["error"] = ErrorAnalyzer(self.output_dir / "errors")
                logger.info("Analyseur d'erreurs initialisé.")
            except Exception as e:
                logger.error(f"Erreur lors de l'initialisation de l'analyseur d'erreurs: {str(e)}")
    
    def train_main_analyzer(self):
        """Entraîne l'analyseur principal."""
        if "main" not in self.analyzers:
            logger.error("Analyseur principal non disponible.")
            return False
        
        try:
            logger.info("Entraînement de l'analyseur principal...")
            success = self.analyzers["main"].run_analysis()
            if success:
                logger.info("Entraînement de l'analyseur principal terminé avec succès.")
                # Les modèles sont déjà sauvegardés dans train_ml_models()
                logger.info(f"Modèles sauvegardés dans: {self.model_dir}")
            else:
                logger.error("Échec de l'entraînement de l'analyseur principal.")
            return success
        except Exception as e:
            logger.error(f"Erreur lors de l'entraînement de l'analyseur principal: {str(e)}")
            logger.debug(traceback.format_exc())
            return False
    
    def train_fibonacci_analyzer(self):
        """Entraîne l'analyseur Fibonacci."""
        if "fibonacci" not in self.analyzers:
            logger.error("Analyseur Fibonacci non disponible.")
            return False
        
        try:
            logger.info("Entraînement de l'analyseur Fibonacci...")
            self.analyzers["fibonacci"].analyze_frequencies()
            self.analyzers["fibonacci"].apply_fibonacci_weighting()
            
            # Sauvegarder les poids
            try:
                import pickle
                weights_path = self.model_dir / "fibonacci_weights.pkl"
                weights_data = {
                    'number_weights': self.analyzers["fibonacci"].number_weights,
                    'star_weights': self.analyzers["fibonacci"].star_weights,
                    'number_freq': self.analyzers["fibonacci"].number_freq,
                    'star_freq': self.analyzers["fibonacci"].star_freq
                }
                with open(weights_path, 'wb') as f:
                    pickle.dump(weights_data, f)
                logger.info(f"Poids Fibonacci sauvegardés: {weights_path}")
            except Exception as e:
                logger.warning(f"Impossible de sauvegarder les poids Fibonacci: {str(e)}")
            
            # Générer et sauvegarder les prédictions
            try:
                combinations = self.analyzers["fibonacci"].generate_combinations(num_combinations=10)
                if combinations:
                    self.analyzers["fibonacci"].predictions = combinations
                    result_file = self.analyzers["fibonacci"].save_predictions()
                    if result_file:
                        logger.info(f"Résultats Fibonacci sauvegardés: {result_file}")
            except Exception as e:
                logger.warning(f"Impossible de sauvegarder les prédictions Fibonacci: {str(e)}")
            
            # Générer une visualisation
            if self.config.get("visualize", False):
                self.analyzers["fibonacci"].visualize_weights()
            
            logger.info("Entraînement de l'analyseur Fibonacci terminé avec succès.")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de l'entraînement de l'analyseur Fibonacci: {str(e)}")
            logger.debug(traceback.format_exc())
            return False
    
    def train_lunar_analyzer(self):
        """Entraîne l'analyseur de cycle lunaire."""
        if "lunar" not in self.analyzers:
            logger.error("Analyseur de cycle lunaire non disponible.")
            return False
        
        try:
            logger.info("Entraînement de l'analyseur de cycle lunaire...")
            
            # Vérifier si la colonne de date existe
            date_column = "Date"
            if date_column not in self.df.columns:
                logger.error(f"Colonne de date '{date_column}' non trouvée dans le DataFrame.")
                return False
            
            # Enrichir les données avec les informations lunaires
            enriched_df = self.analyzers["lunar"].enrich_dataframe_with_lunar_data(self.df)
            
            # Analyser l'influence lunaire
            number_cols = [f"N{i}" for i in range(1, 6)]
            star_cols = [f"E{i}" for i in range(1, 3)]
            
            results = self.analyzers["lunar"].analyze_lunar_influence(enriched_df, number_cols, star_cols)
            
            # Sauvegarder les résultats
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.analyzers["lunar"].save_lunar_analysis(results, timestamp)
            
            logger.info("Entraînement de l'analyseur de cycle lunaire terminé avec succès.")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de l'entraînement de l'analyseur de cycle lunaire: {str(e)}")
            logger.debug(traceback.format_exc())
            return False
    
    def run_incremental_learning(self):
        """Exécute l'apprentissage incrémental."""
        if "incremental" not in self.analyzers:
            logger.error("Module d'apprentissage incrémental non disponible.")
            return False
        
        try:
            logger.info("Exécution de l'apprentissage incrémental...")
            
            # ⚠️ CRITIQUE : L'apprentissage incrémental utilise l'analyseur principal
            # qui a déjà chargé les données depuis le fichier de cycles
            if "main" not in self.analyzers:
                logger.error("Analyseur principal non disponible pour l'apprentissage incrémental.")
                return False
            
            # L'apprentissage incrémental est géré par EuromillionsIncrementalLearning
            # qui utilise directement l'analyseur principal et ses données
            logger.info("✅ Apprentissage incrémental configuré avec l'analyseur principal")
            logger.info("   Les données du fichier de cycles sont utilisées automatiquement")
            
            # Exécuter l'apprentissage incrémental
            incremental_analyzer = self.analyzers["incremental"]
            start_idx = self.config.get("incremental_start_idx", 50)
            step_size = self.config.get("incremental_step_size", 1)
            
            results = incremental_analyzer.run_incremental_learning(
                start_idx=start_idx,
                step_size=step_size
            )
            
            # Sauvegarder les modèles
            if hasattr(incremental_analyzer, 'save_models'):
                incremental_analyzer.save_models()
            
            logger.info("Apprentissage incrémental terminé avec succès.")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de l'apprentissage incrémental: {str(e)}")
            logger.debug(traceback.format_exc())
            return False
    
    def run_error_analysis(self):
        """Exécute l'analyse des erreurs."""
        if "error" not in self.analyzers:
            logger.error("Analyseur d'erreurs non disponible.")
            return False
        
        try:
            logger.info("Exécution de l'analyse des erreurs...")
            
            error_analyzer = self.analyzers["error"]
            
            # Vérifier si l'analyseur principal a des prédictions
            if "main" in self.analyzers and hasattr(self.analyzers["main"], 'predictions'):
                # Utiliser les prédictions de l'analyseur principal
                predictions = self.analyzers["main"].predictions
                actual_draws = []
                
                # Extraire les tirages réels depuis le DataFrame
                if self.df is not None and not self.df.empty:
                    for _, row in self.df.tail(len(predictions)).iterrows():
                        numbers = [int(row[f"N{i}"]) for i in range(1, 6)]
                        stars = [int(row[f"E{i}"]) for i in range(1, 3)]
                        actual_draws.append(numbers + stars)
                    
                    # Comparer les prédictions avec les tirages réels
                    if len(predictions) > 0 and len(actual_draws) > 0:
                        error_df = error_analyzer.compare_predictions_with_actual(
                            predictions[:len(actual_draws)],
                            actual_draws[:len(predictions)],
                            dates=None,
                            num_main_numbers=5,
                            num_stars=2
                        )
                        
                        # Analyser les erreurs
                        results = error_analyzer.analyze_errors(error_df)
                        
                        # Sauvegarder les résultats
                        error_analyzer.save_error_analysis(results)
                        
                        # Exporter vers CSV
                        error_analyzer.export_errors_to_csv(error_df)
                        
                        logger.info("Analyse des erreurs terminée avec succès.")
                        return True
                    else:
                        logger.warning("Pas assez de données pour l'analyse des erreurs.")
                        return False
                else:
                    logger.warning("DataFrame vide - Impossible d'analyser les erreurs.")
                    return False
            else:
                logger.warning("Aucune prédiction disponible pour l'analyse des erreurs.")
                # Créer un fichier vide pour indiquer que l'analyse a été tentée
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                empty_file = error_analyzer.output_dir / f"error_analysis_{timestamp}.txt"
                with open(empty_file, 'w', encoding='utf-8') as f:
                    f.write("Analyse des erreurs - Aucune prédiction disponible pour l'analyse.\n")
                logger.info("Fichier d'analyse des erreurs créé (vide).")
                return True
                
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse des erreurs: {str(e)}")
            logger.debug(traceback.format_exc())
            return False
    
    def run_backtesting(self, num_draws=10):
        """
        Exécute le backtesting sur les analyseurs disponibles.
        
        Args:
            num_draws: Nombre de tirages à utiliser pour le backtesting
        """
        logger.info(f"Exécution du backtesting sur {num_draws} tirages...")
        
        results = {}
        
        # Backtesting avec l'analyseur principal
        if "main" in self.analyzers and hasattr(self.analyzers["main"], "run_backtesting"):
            try:
                logger.info("Backtesting avec l'analyseur principal...")
                main_results = self.analyzers["main"].run_backtesting(num_draws)
                results["main"] = main_results
                logger.info("Backtesting avec l'analyseur principal terminé.")
            except Exception as e:
                logger.error(f"Erreur lors du backtesting avec l'analyseur principal: {str(e)}")
        
        # Backtesting avec l'analyseur Fibonacci
        if "fibonacci" in self.analyzers and hasattr(self.analyzers["fibonacci"], "run_backtesting"):
            try:
                logger.info("Backtesting avec l'analyseur Fibonacci...")
                fibonacci_results = self.analyzers["fibonacci"].run_backtesting(num_draws)
                results["fibonacci"] = fibonacci_results
                
                # Sauvegarder les résultats
                if fibonacci_results:
                    self.analyzers["fibonacci"].save_backtesting_results(fibonacci_results)
                
                logger.info("Backtesting avec l'analyseur Fibonacci terminé.")
            except Exception as e:
                logger.error(f"Erreur lors du backtesting avec l'analyseur Fibonacci: {str(e)}")
        
        return results
    
    def train_all(self):
        """Entraîne tous les analyseurs disponibles."""
        logger.info("Entraînement de tous les analyseurs...")
        
        results = {
            "main": False,
            "fibonacci": False,
            "lunar": False,
            "incremental": False,
            "error": False
        }
        
        # Entraîner l'analyseur principal
        if "main" in self.analyzers:
            results["main"] = self.train_main_analyzer()
        
        # Entraîner l'analyseur Fibonacci
        if "fibonacci" in self.analyzers:
            results["fibonacci"] = self.train_fibonacci_analyzer()
        
        # Entraîner l'analyseur de cycle lunaire
        if "lunar" in self.analyzers:
            results["lunar"] = self.train_lunar_analyzer()
        
        # Exécuter l'apprentissage incrémental
        if "incremental" in self.analyzers:
            results["incremental"] = self.run_incremental_learning()
        
        # Exécuter l'analyse des erreurs
        if "error" in self.analyzers:
            results["error"] = self.run_error_analysis()
        
        # Exécuter le backtesting si demandé
        if self.config.get("backtesting", False):
            backtesting_results = self.run_backtesting(self.config.get("backtesting_draws", 10))
            results["backtesting"] = backtesting_results
        
        # Afficher un résumé
        logger.info("Résumé de l'entraînement:")
        for analyzer, success in results.items():
            if analyzer != "backtesting":
                status = "Succès" if success else "Échec ou non exécuté"
                logger.info(f"- {analyzer}: {status}")
        
        return results

def parse_arguments():
    """
    Parse les arguments de ligne de commande pour l'entraînement Euromillions.
    """
    parser = argparse.ArgumentParser(description="Entraîneur unifié pour les modèles Euromillions")
    
    parser.add_argument("--csv", type=str, default="tirage_euromillions.csv",
                        help="Chemin vers le fichier CSV des tirages EuroMillions")
    parser.add_argument("--output", type=str, default="resultats_euromillions",
                        help="Répertoire de sortie pour les résultats")
    parser.add_argument("--model-dir", type=str, default="models_euromillions",
                        help="Répertoire pour les modèles entraînés")
    
    parser.add_argument("--method", type=str, choices=["all", "main", "fibonacci", "lunar", "incremental", "error"],
                        default="all", help="Méthode d'entraînement à utiliser (all = toutes les méthodes)")
    
    parser.add_argument("--enable-ai-reflection", action="store_true", default=True,
                        help="Activer la réflexion IA pour améliorer les features (défaut: True)")
    
    parser.add_argument("--llm-config", type=str, choices=["grok-4-fast", "claude-opus-4.1", "gpt-5-image"],
                        default="grok-4-fast", help="Configuration LLM pour la réflexion IA")
    
    parser.add_argument("--backtesting", action="store_true",
                        help="Activer le backtesting")
    parser.add_argument("--backtesting-draws", type=int, default=10,
                        help="Nombre de tirages à utiliser pour le backtesting")
    
    parser.add_argument("--visualize", action="store_true",
                        help="Générer les visualisations")
    parser.add_argument("--fibonacci-inverse", action="store_true",
                        help="Activer la pondération Fibonacci inversée")
    
    return parser.parse_args()

def main():
    """
    Fonction principale pour l'entraînement des modèles Euromillions.
    """
    args = parse_arguments()
    
    # Générer un timestamp unique pour cette exécution
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"Début de l'entraînement des modèles Euromillions (Timestamp: {timestamp}).")
    logger.info(f"Arguments: {args}")
    
    # Créer la configuration avec options d'encodeur avancé
    config = {
        "csv_file": args.csv,
        "output_dir": args.output,
        "model_dir": args.model_dir,
        "method": args.method,
        "backtesting": args.backtesting,
        "backtesting_draws": args.backtesting_draws,
        "visualize": args.visualize,
        "use_fibonacci_inverse": args.fibonacci_inverse,
        "enable_ai_reflection": args.enable_ai_reflection,
        "llm_config": args.llm_config,
        "timestamp": timestamp,
        # ⚠️ CRITIQUE : Activer le système quantique par défaut pour tous les entraînements
        "use_quantum": True,
        "use_qnn": True,
        "use_qlstm": True,
        "use_quantum_annealing": True,
    }
    
    # Logger la configuration
    logger.info("Configuration de l'entraînement:")
    logger.info(f"  - Méthode: {args.method}")
    logger.info(f"  - Encodeur avancé: Activé")
    logger.info(f"  - Réflexion IA: {'Activée' if args.enable_ai_reflection else 'Désactivée'}")
    if args.enable_ai_reflection:
        logger.info(f"  - Configuration LLM: {args.llm_config}")
    logger.info(f"  - Backtesting: {'Activé' if args.backtesting else 'Désactivé'}")
    
    try:
        # Créer l'entraîneur
        trainer = EuromillionsTrainer(config)
        
        # Exécuter l'entraînement selon la méthode spécifiée
        if args.method == "all":
            results = trainer.train_all()
        elif args.method == "main":
            results = {"main": trainer.train_main_analyzer()}
        elif args.method == "fibonacci":
            results = {"fibonacci": trainer.train_fibonacci_analyzer()}
        elif args.method == "lunar":
            results = {"lunar": trainer.train_lunar_analyzer()}
        elif args.method == "incremental":
            results = {"incremental": trainer.run_incremental_learning()}
        elif args.method == "error":
            results = {"error": trainer.run_error_analysis()}
        
        # Exécuter le backtesting si demandé
        if args.backtesting and args.method != "all":
            backtesting_results = trainer.run_backtesting(args.backtesting_draws)
            results["backtesting"] = backtesting_results
        
        # Afficher un résumé
        logger.info("Résumé de l'entraînement:")
        for analyzer, success in results.items():
            if analyzer != "backtesting":
                status = "Succès" if success else "Échec ou non exécuté"
                logger.info(f"- {analyzer}: {status}")
        
        logger.info(f"Entraînement terminé (Timestamp: {timestamp}).")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement: {str(e)}")
        logger.debug(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
