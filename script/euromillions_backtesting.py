#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module de backtesting pour l'analyseur Euromillions
Ce module implémente un système complet de backtesting pour évaluer la performance
des prédictions sur les tirages historiques d'Euromillions.

Améliorations:
- Ajout de méthodes `save_results` et `plot_results` pour correspondre aux appels dans `euromillions_main.py`.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple, Set, Optional, Union, Any
from collections import Counter
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import traceback

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("EuromillionsBacktesting")

class EuromillionsBacktesting:
    """Classe pour le backtesting des prédictions Euromillions."""
    
    def __init__(self, analyzer, output_dir=None):
        """
        Initialise le module de backtesting.
        
        Args:
            analyzer: Instance de EuromillionsAdvancedAnalyzer
            output_dir: Répertoire de sortie pour les résultats (optionnel)
        """
        self.analyzer = analyzer
        self.df = analyzer.df # Utiliser le DataFrame de l'analyseur
        self.number_cols = analyzer.number_cols
        self.star_cols = analyzer.star_cols
        
        # Répertoire de sortie
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            # Assurez-vous que l'output_dir de l'analyseur est bien défini
            self.output_dir = Path(analyzer.output_dir) / "backtesting"
        
        # Créer le répertoire de sortie s'il n'existe pas
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)
            logger.info(f"Répertoire de backtesting créé: {self.output_dir}")

    def run_backtesting(self, min_history: int = 50, step_size: int = 1) -> Dict:
        """
        Effectue un backtesting des prédictions sur les données historiques.
        
        Args:
            min_history: Nombre minimum de tirages requis pour l'entraînement initial du modèle.
            step_size: Nombre de tirages à avancer à chaque étape.
            
        Returns:
            Dict: Résultats du backtesting
        """
        logger.info("Démarrage du backtesting...")
        
        try:
            if self.df is None or len(self.df) < min_history:
                logger.error(f"Données insuffisantes pour le backtesting (minimum {min_history} tirages requis).")
                return {}
            
            # Initialiser les résultats
            results = {
                'correct_numbers': [],
                'correct_stars': [],
                'accuracy_numbers': [],
                'accuracy_stars': [],
                'predictions': []
            }
            
            # Boucle de backtesting
            # Le backtesting commence après le min_history initial
            for i in range(min_history, len(self.df), step_size):
                logger.info(f"Backtesting: tirage {i+1}/{len(self.df)}")
                
                # Créer un sous-ensemble des données jusqu'à l'indice i (exclus) pour l'entraînement
                train_df = self.df.iloc[:i].copy()
                
                # Créer un analyseur temporaire pour chaque étape du backtesting
                # Utilise la configuration de l'analyseur principal, mais avec le sous-ensemble de données
                temp_config = self.analyzer.config.copy()
                # Ajuster la fenêtre de tirages pour le temp_analyzer si elle est trop grande
                temp_config["window_draws"] = min(temp_config["window_draws"], len(train_df))
                
                temp_analyzer = self.analyzer.__class__(temp_config) # Utilise la même classe que l'analyseur principal
                temp_analyzer.df = train_df
                temp_analyzer.number_cols = self.number_cols
                temp_analyzer.star_cols = self.star_cols
                
                # Exécuter l'analyse sur le sous-ensemble de données
                temp_analyzer.compute_global_stats()
                temp_analyzer.compute_window_stats()
                temp_analyzer.identify_hot_cold()
                
                if temp_config["use_correlation"]:
                    temp_analyzer.analyze_correlations()
                
                if temp_config["use_temporal"]:
                    temp_analyzer.analyze_temporal_patterns()
                
                if temp_config["use_clustering"]:
                    temp_analyzer.analyze_clustering()
                
                temp_analyzer.compute_gap_scores() # Calculer les scores d'écart pour l'analyseur temporaire

                if temp_config["use_ml"]:
                    temp_analyzer.train_ml_models() # Entraîner les modèles ML sur le sous-ensemble
                
                # Prédire le prochain tirage (le tirage à l'indice i)
                # Cela met à jour temp_analyzer.number_predictions et temp_analyzer.star_predictions
                temp_analyzer.predict_numbers()

                # Calculer les scores finaux combinés pour la prédiction
                temp_number_scores, temp_star_scores = temp_analyzer.compute_scores()

                # Sélectionner les numéros et étoiles prédits basés sur les scores finaux
                predicted_numbers = sorted(temp_number_scores, key=temp_number_scores.get, reverse=True)[:temp_config["propose_size"]]
                predicted_stars = sorted(temp_star_scores, key=temp_star_scores.get, reverse=True)[:temp_config["star_size"]]
                
                # Comparer avec le tirage réel (le tirage à l'indice i)
                actual_row = self.df.iloc[i]
                actual_numbers = actual_row[self.number_cols].dropna().astype(int).tolist()
                actual_stars = actual_row[self.star_cols].dropna().astype(int).tolist()
                
                # Compter les numéros corrects
                correct_numbers = len(set(predicted_numbers) & set(actual_numbers))
                correct_stars = len(set(predicted_stars) & set(actual_stars))
                
                # Calculer les précisions
                accuracy_numbers = correct_numbers / temp_config["propose_size"] if temp_config["propose_size"] > 0 else 0
                accuracy_stars = correct_stars / temp_config["star_size"] if temp_config["star_size"] > 0 else 0
                
                # Enregistrer les résultats
                results['correct_numbers'].append(correct_numbers)
                results['correct_stars'].append(correct_stars)
                results['accuracy_numbers'].append(accuracy_numbers)
                results['accuracy_stars'].append(accuracy_stars)
                
                # Enregistrer la prédiction détaillée
                results['predictions'].append({
                    'index': i,
                    'date': actual_row['Date'] if 'Date' in actual_row else None,
                    'predicted_numbers': predicted_numbers,
                    'predicted_stars': predicted_stars,
                    'actual_numbers': actual_numbers,
                    'actual_stars': actual_stars,
                    'correct_numbers': correct_numbers,
                    'correct_stars': correct_stars
                })
            
            # Calculer les statistiques globales de backtesting
            results['avg_correct_numbers'] = sum(results['correct_numbers']) / len(results['correct_numbers']) if results['correct_numbers'] else 0
            results['avg_correct_stars'] = sum(results['correct_stars']) / len(results['correct_stars']) if results['correct_stars'] else 0
            results['avg_accuracy_numbers'] = sum(results['accuracy_numbers']) / len(results['accuracy_numbers']) if results['accuracy_numbers'] else 0
            results['avg_accuracy_stars'] = sum(results['accuracy_stars']) / len(results['accuracy_stars']) if results['accuracy_stars'] else 0
            
            # Distribution des numéros corrects
            results['distribution_correct_numbers'] = Counter(results['correct_numbers'])
            results['distribution_correct_stars'] = Counter(results['correct_stars'])
            
            logger.info(f"Backtesting terminé: {len(results['predictions'])} prédictions évaluées")
            logger.info(f"Nombre moyen de numéros corrects: {results['avg_correct_numbers']:.2f}")
            logger.info(f"Nombre moyen d'étoiles correctes: {results['avg_correct_stars']:.2f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Erreur lors du backtesting: {str(e)}")
            logger.debug(traceback.format_exc())
            return {}

    def save_results(self, backtest_results: Dict) -> str:
        """
        Sauvegarde les résultats détaillés du backtesting dans un fichier texte.
        
        Args:
            backtest_results: Dictionnaire contenant les résultats du backtesting.
            
        Returns:
            str: Chemin du fichier de résultats sauvegardé.
        """
        logger.info("Sauvegarde des résultats du backtesting...")
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = self.output_dir / f"backtesting_results_{timestamp}.txt"
            
            with open(results_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("RAPPORT DE BACKTESTING EUROMILLIONS\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Date du rapport: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Nombre de tirages évalués: {len(backtest_results['predictions'])}\n\n")
                
                f.write(f"Moyenne de numéros corrects par tirage: {backtest_results['avg_correct_numbers']:.2f}\n")
                f.write(f"Moyenne d'étoiles correctes par tirage: {backtest_results['avg_correct_stars']:.2f}\n")
                f.write(f"Précision moyenne des numéros: {backtest_results['avg_accuracy_numbers']*100:.2f}%\n")
                f.write(f"Précision moyenne des étoiles: {backtest_results['avg_accuracy_stars']*100:.2f}%\n\n")
                
                f.write("Distribution du nombre de numéros corrects:\n")
                for count, freq in sorted(backtest_results['distribution_correct_numbers'].items()):
                    f.write(f"  {count} numéros corrects: {freq} fois\n")
                f.write("\n")
                
                f.write("Distribution du nombre d'étoiles correctes:\n")
                for count, freq in sorted(backtest_results['distribution_correct_stars'].items()):
                    f.write(f"  {count} étoiles correctes: {freq} fois\n")
                f.write("\n")
                
                f.write("-" * 80 + "\n")
                f.write("Détail des prédictions:\n")
                f.write("-" * 80 + "\n")
                for p in backtest_results['predictions']:
                    f.write(f"Tirage Index: {p['index']}, Date: {p['date'] if p['date'] else 'N/A'}\n")
                    f.write(f"  Prédit: Numéros {p['predicted_numbers']}, Étoiles {p['predicted_stars']}\n")
                    f.write(f"  Réel:   Numéros {p['actual_numbers']}, Étoiles {p['actual_stars']}\n")
                    f.write(f"  Corrects: Numéros {p['correct_numbers']}, Étoiles {p['correct_stars']}\n\n")
            
            logger.info(f"Résultats du backtesting sauvegardés dans: {results_file}")
            return str(results_file)
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des résultats du backtesting: {str(e)}")
            logger.debug(traceback.format_exc())
            return ""

    def plot_results(self, backtest_results: Dict) -> List[str]:
        """
        Génère des visualisations des résultats du backtesting.
        
        Args:
            backtest_results: Dictionnaire contenant les résultats du backtesting.
            
        Returns:
            List[str]: Liste des chemins des fichiers de visualisation générés.
        """
        logger.info("Génération des graphiques de backtesting...")
        if not backtest_results or 'predictions' not in backtest_results or not backtest_results['predictions']:
            logger.error("Résultats de backtesting invalides ou vides pour la génération de graphiques.")
            return []
        
        plot_files = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plots_dir = self.output_dir # Utilise le répertoire de backtesting comme répertoire de plots

        try:
            indices = [p['index'] for p in backtest_results['predictions']]
            correct_numbers = [p['correct_numbers'] for p in backtest_results['predictions']]
            correct_stars = [p['correct_stars'] for p in backtest_results['predictions']]
            
            # Graphique 1: Évolution du nombre de numéros et étoiles corrects
            plt.figure(figsize=(12, 6))
            plt.plot(indices, correct_numbers, label='Numéros corrects', color='blue')
            plt.plot(indices, correct_stars, label='Étoiles correctes', color='red')
            plt.axhline(y=backtest_results['avg_correct_numbers'], color='lightblue', linestyle='--', label=f'Moyenne Numéros ({backtest_results["avg_correct_numbers"]:.2f})')
            plt.axhline(y=backtest_results['avg_correct_stars'], color='lightcoral', linestyle='--', label=f'Moyenne Étoiles ({backtest_results["avg_correct_stars"]:.2f})')
            plt.title('Évolution du nombre de numéros et d\'étoiles corrects par tirage')
            plt.xlabel('Index du tirage')
            plt.ylabel('Nombre de correspondances')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plot_path = plots_dir / f"backtesting_evolution_{timestamp}.png"
            plt.savefig(plot_path)
            plt.close()
            plot_files.append(str(plot_path))

            # Graphique 2: Distribution des numéros corrects
            plt.figure(figsize=(10, 6))
            sns.histplot(correct_numbers, bins=np.arange(self.analyzer.config["propose_size"] + 2) - 0.5, kde=False, color='skyblue', stat='count')
            plt.title('Distribution du nombre de numéros corrects par tirage')
            plt.xlabel('Nombre de numéros corrects')
            plt.ylabel('Fréquence')
            plt.xticks(np.arange(self.analyzer.config["propose_size"] + 1))
            plt.grid(axis='y', alpha=0.3)
            plot_path = plots_dir / f"backtesting_dist_numbers_{timestamp}.png"
            plt.savefig(plot_path)
            plt.close()
            plot_files.append(str(plot_path))

            # Graphique 3: Distribution des étoiles correctes
            plt.figure(figsize=(10, 6))
            sns.histplot(correct_stars, bins=np.arange(self.analyzer.config["star_size"] + 2) - 0.5, kde=False, color='lightcoral', stat='count')
            plt.title('Distribution du nombre d\'étoiles correctes par tirage')
            plt.xlabel('Nombre d\'étoiles correctes')
            plt.ylabel('Fréquence')
            plt.xticks(np.arange(self.analyzer.config["star_size"] + 1))
            plt.grid(axis='y', alpha=0.3)
            plot_path = plots_dir / f"backtesting_dist_stars_{timestamp}.png"
            plt.savefig(plot_path)
            plt.close()
            plot_files.append(str(plot_path))
            
            logger.info(f"Graphiques de backtesting sauvegardés dans: {plots_dir}")
            return plot_files
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération des graphiques de backtesting: {str(e)}")
            logger.debug(traceback.format_exc())
            return []

