#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module de comparaison des erreurs de prédiction pour l'analyseur Euromillions
Ce module analyse les prédictions passées, les compare aux tirages réels,
et génère un fichier CSV détaillé des erreurs de prédiction.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Set, Optional, Union, Any
from collections import Counter
from pathlib import Path
import traceback

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("EuromillionsErrorAnalyzer")

class ErrorAnalyzer:
    """Classe pour analyser les erreurs de prédiction Euromillions."""
    
    def __init__(self, output_dir=None):
        """
        Initialise l'analyseur d'erreurs.
        
        Args:
            output_dir: Répertoire de sortie pour les résultats (optionnel)
        """
        # Répertoire de sortie
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path("resultats_error_analysis")
        
        # Créer le répertoire de sortie s'il n'existe pas
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)
            logger.info(f"Répertoire d'analyse d'erreurs créé: {self.output_dir}")

    def compare_predictions_with_actual(self, 
                                       predictions: List[List[int]], 
                                       actual_draws: List[List[int]], 
                                       dates: List[datetime] = None,
                                       num_main_numbers: int = 5,
                                       num_stars: int = 2) -> pd.DataFrame:
        """
        Compare les prédictions avec les tirages réels et génère un DataFrame d'erreurs.
        
        Args:
            predictions: Liste des combinaisons prédites (chaque combinaison contient les numéros principaux suivis des étoiles)
            actual_draws: Liste des tirages réels (même format que les prédictions)
            dates: Liste des dates correspondant aux tirages (optionnel)
            num_main_numbers: Nombre de numéros principaux dans une combinaison
            num_stars: Nombre d'étoiles dans une combinaison
            
        Returns:
            pd.DataFrame: DataFrame contenant les erreurs de prédiction
        """
        logger.info("Comparaison des prédictions avec les tirages réels...")
        
        try:
            # Vérifier que les listes ont la même longueur
            if len(predictions) != len(actual_draws):
                logger.error(f"Les listes de prédictions et de tirages réels n'ont pas la même longueur: {len(predictions)} vs {len(actual_draws)}")
                return pd.DataFrame()
            
            # Préparer les données pour le DataFrame
            data = []
            
            for i, (pred, actual) in enumerate(zip(predictions, actual_draws)):
                # Vérifier que les combinaisons ont le bon format
                if len(pred) != num_main_numbers + num_stars or len(actual) != num_main_numbers + num_stars:
                    logger.warning(f"Combinaison {i} avec format incorrect: {len(pred)} ou {len(actual)} numéros au lieu de {num_main_numbers + num_stars}")
                    continue
                
                # Séparer les numéros principaux et les étoiles
                pred_numbers = pred[:num_main_numbers]
                pred_stars = pred[num_main_numbers:]
                actual_numbers = actual[:num_main_numbers]
                actual_stars = actual[num_main_numbers:]
                
                # Calculer les erreurs
                correct_numbers = len(set(pred_numbers) & set(actual_numbers))
                correct_stars = len(set(pred_stars) & set(actual_stars))
                
                # Calculer les erreurs par position
                position_errors_numbers = []
                for j in range(num_main_numbers):
                    if j < len(pred_numbers) and j < len(actual_numbers):
                        position_errors_numbers.append(abs(pred_numbers[j] - actual_numbers[j]))
                    else:
                        position_errors_numbers.append(np.nan)
                
                position_errors_stars = []
                for j in range(num_stars):
                    if j < len(pred_stars) and j < len(actual_stars):
                        position_errors_stars.append(abs(pred_stars[j] - actual_stars[j]))
                    else:
                        position_errors_stars.append(np.nan)
                
                # Calculer les erreurs moyennes
                avg_error_numbers = np.mean(position_errors_numbers) if position_errors_numbers else np.nan
                avg_error_stars = np.mean(position_errors_stars) if position_errors_stars else np.nan
                
                # Calculer les erreurs relatives (en pourcentage du maximum possible)
                max_number = 50  # Maximum pour Euromillions
                max_star = 12    # Maximum pour Euromillions
                rel_error_numbers = avg_error_numbers / max_number if not np.isnan(avg_error_numbers) else np.nan
                rel_error_stars = avg_error_stars / max_star if not np.isnan(avg_error_stars) else np.nan
                
                # Créer une entrée pour le DataFrame
                entry = {
                    'Index': i,
                    'Date': dates[i] if dates and i < len(dates) else None,
                    'PredictedNumbers': pred_numbers,
                    'PredictedStars': pred_stars,
                    'ActualNumbers': actual_numbers,
                    'ActualStars': actual_stars,
                    'CorrectNumbers': correct_numbers,
                    'CorrectStars': correct_stars,
                    'AccuracyNumbers': correct_numbers / num_main_numbers,
                    'AccuracyStars': correct_stars / num_stars,
                    'AvgErrorNumbers': avg_error_numbers,
                    'AvgErrorStars': avg_error_stars,
                    'RelErrorNumbers': rel_error_numbers,
                    'RelErrorStars': rel_error_stars
                }
                
                # Ajouter les erreurs par position
                for j in range(num_main_numbers):
                    entry[f'ErrorN{j+1}'] = position_errors_numbers[j] if j < len(position_errors_numbers) else np.nan
                
                for j in range(num_stars):
                    entry[f'ErrorE{j+1}'] = position_errors_stars[j] if j < len(position_errors_stars) else np.nan
                
                data.append(entry)
            
            # Créer le DataFrame
            error_df = pd.DataFrame(data)
            
            logger.info(f"Comparaison terminée: {len(error_df)} entrées générées.")
            return error_df
        
        except Exception as e:
            logger.error(f"Erreur lors de la comparaison des prédictions: {str(e)}")
            logger.debug(traceback.format_exc())
            return pd.DataFrame()

    def analyze_errors(self, error_df: pd.DataFrame) -> Dict:
        """
        Analyse les erreurs de prédiction et génère des statistiques.
        
        Args:
            error_df: DataFrame contenant les erreurs de prédiction
            
        Returns:
            Dict: Résultats de l'analyse
        """
        logger.info("Analyse des erreurs de prédiction...")
        
        try:
            if error_df.empty:
                logger.error("DataFrame d'erreurs vide. Impossible d'analyser.")
                return {}
            
            # Initialiser les résultats
            results = {
                'global_stats': {},
                'number_stats': {},
                'star_stats': {},
                'position_stats': {},
                'temporal_stats': {}
            }
            
            # Statistiques globales
            results['global_stats'] = {
                'num_predictions': len(error_df),
                'avg_correct_numbers': error_df['CorrectNumbers'].mean(),
                'avg_correct_stars': error_df['CorrectStars'].mean(),
                'avg_accuracy_numbers': error_df['AccuracyNumbers'].mean(),
                'avg_accuracy_stars': error_df['AccuracyStars'].mean(),
                'avg_error_numbers': error_df['AvgErrorNumbers'].mean(),
                'avg_error_stars': error_df['AvgErrorStars'].mean(),
                'avg_rel_error_numbers': error_df['RelErrorNumbers'].mean(),
                'avg_rel_error_stars': error_df['RelErrorStars'].mean()
            }
            
            # Distribution des numéros corrects
            results['number_stats']['correct_distribution'] = error_df['CorrectNumbers'].value_counts().sort_index().to_dict()
            
            # Distribution des étoiles correctes
            results['star_stats']['correct_distribution'] = error_df['CorrectStars'].value_counts().sort_index().to_dict()
            
            # Statistiques par position
            error_cols_numbers = [col for col in error_df.columns if col.startswith('ErrorN')]
            error_cols_stars = [col for col in error_df.columns if col.startswith('ErrorE')]
            
            for col in error_cols_numbers:
                pos = int(col[5:])  # Extraire le numéro de position (ErrorN1 -> 1)
                results['position_stats'][f'number_{pos}'] = {
                    'avg_error': error_df[col].mean(),
                    'median_error': error_df[col].median(),
                    'std_error': error_df[col].std(),
                    'min_error': error_df[col].min(),
                    'max_error': error_df[col].max()
                }
            
            for col in error_cols_stars:
                pos = int(col[5:])  # Extraire le numéro de position (ErrorE1 -> 1)
                results['position_stats'][f'star_{pos}'] = {
                    'avg_error': error_df[col].mean(),
                    'median_error': error_df[col].median(),
                    'std_error': error_df[col].std(),
                    'min_error': error_df[col].min(),
                    'max_error': error_df[col].max()
                }
            
            # Analyse temporelle (si les dates sont disponibles)
            if 'Date' in error_df.columns and not error_df['Date'].isna().all():
                # Convertir en datetime si nécessaire
                if not pd.api.types.is_datetime64_dtype(error_df['Date']):
                    error_df['Date'] = pd.to_datetime(error_df['Date'], errors='coerce')
                
                # Grouper par mois
                error_df['Month'] = error_df['Date'].dt.to_period('M')
                monthly_stats = error_df.groupby('Month').agg({
                    'CorrectNumbers': 'mean',
                    'CorrectStars': 'mean',
                    'AvgErrorNumbers': 'mean',
                    'AvgErrorStars': 'mean'
                }).reset_index()
                
                # Convertir en dictionnaire pour les résultats
                monthly_stats['Month'] = monthly_stats['Month'].astype(str)
                results['temporal_stats']['monthly'] = monthly_stats.set_index('Month').to_dict()
            
            logger.info("Analyse des erreurs terminée.")
            return results
        
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse des erreurs: {str(e)}")
            logger.debug(traceback.format_exc())
            return {}

    def export_errors_to_csv(self, error_df: pd.DataFrame, timestamp: str = None) -> str:
        """
        Exporte les erreurs de prédiction vers un fichier CSV.
        
        Args:
            error_df: DataFrame contenant les erreurs de prédiction
            timestamp: Horodatage pour le nom du fichier (optionnel)
            
        Returns:
            str: Chemin du fichier CSV sauvegardé
        """
        logger.info("Exportation des erreurs de prédiction vers CSV...")
        
        try:
            if error_df.empty:
                logger.error("DataFrame d'erreurs vide. Impossible d'exporter.")
                return ""
            
            if timestamp is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            csv_file = self.output_dir / f"euromillions_prediction_errors_{timestamp}.csv"
            
            # Convertir les listes en chaînes pour le CSV
            export_df = error_df.copy()
            list_columns = ['PredictedNumbers', 'PredictedStars', 'ActualNumbers', 'ActualStars']
            for col in list_columns:
                if col in export_df.columns:
                    export_df[col] = export_df[col].apply(lambda x: ','.join(map(str, x)) if isinstance(x, (list, tuple)) else x)
            
            # Exporter vers CSV
            export_df.to_csv(csv_file, index=False)
            
            logger.info(f"Erreurs de prédiction exportées vers: {csv_file}")
            return str(csv_file)
        
        except Exception as e:
            logger.error(f"Erreur lors de l'exportation des erreurs vers CSV: {str(e)}")
            logger.debug(traceback.format_exc())
            return ""

    def save_error_analysis(self, results: Dict, timestamp: str = None) -> str:
        """
        Sauvegarde les résultats de l'analyse d'erreurs dans un fichier texte.
        
        Args:
            results: Dictionnaire contenant les résultats de l'analyse
            timestamp: Horodatage pour le nom du fichier (optionnel)
            
        Returns:
            str: Chemin du fichier de résultats sauvegardé
        """
        logger.info("Sauvegarde des résultats de l'analyse d'erreurs...")
        
        try:
            if not results:
                logger.error("Aucun résultat à sauvegarder.")
                return ""
            
            if timestamp is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            results_file = self.output_dir / f"error_analysis_{timestamp}.txt"
            
            with open(results_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("ANALYSE DES ERREURS DE PRÉDICTION EUROMILLIONS\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Date de l'analyse: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Statistiques globales
                f.write("-" * 80 + "\n")
                f.write("STATISTIQUES GLOBALES\n")
                f.write("-" * 80 + "\n")
                for stat, value in results.get('global_stats', {}).items():
                    f.write(f"{stat.replace('_', ' ').capitalize()}: {value:.4f}\n")
                f.write("\n")
                
                # Distribution des numéros corrects
                f.write("-" * 80 + "\n")
                f.write("DISTRIBUTION DES NUMÉROS CORRECTS\n")
                f.write("-" * 80 + "\n")
                for count, freq in sorted(results.get('number_stats', {}).get('correct_distribution', {}).items()):
                    f.write(f"{count} numéros corrects: {freq} prédictions ({freq/results['global_stats']['num_predictions']*100:.2f}%)\n")
                f.write("\n")
                
                # Distribution des étoiles correctes
                f.write("-" * 80 + "\n")
                f.write("DISTRIBUTION DES ÉTOILES CORRECTES\n")
                f.write("-" * 80 + "\n")
                for count, freq in sorted(results.get('star_stats', {}).get('correct_distribution', {}).items()):
                    f.write(f"{count} étoiles correctes: {freq} prédictions ({freq/results['global_stats']['num_predictions']*100:.2f}%)\n")
                f.write("\n")
                
                # Statistiques par position
                f.write("-" * 80 + "\n")
                f.write("STATISTIQUES PAR POSITION\n")
                f.write("-" * 80 + "\n")
                
                # Numéros
                f.write("Numéros principaux:\n")
                for pos in sorted([k for k in results.get('position_stats', {}).keys() if k.startswith('number_')], 
                                 key=lambda x: int(x.split('_')[1])):
                    pos_stats = results['position_stats'][pos]
                    pos_num = pos.split('_')[1]
                    f.write(f"  Position {pos_num}:\n")
                    for stat, value in pos_stats.items():
                        f.write(f"    {stat.replace('_', ' ').capitalize()}: {value:.4f}\n")
                    f.write("\n")
                
                # Étoiles
                f.write("Étoiles:\n")
                for pos in sorted([k for k in results.get('position_stats', {}).keys() if k.startswith('star_')], 
                                 key=lambda x: int(x.split('_')[1])):
                    pos_stats = results['position_stats'][pos]
                    pos_num = pos.split('_')[1]
                    f.write(f"  Position {pos_num}:\n")
                    for stat, value in pos_stats.items():
                        f.write(f"    {stat.replace('_', ' ').capitalize()}: {value:.4f}\n")
                    f.write("\n")
                
                # Analyse temporelle
                if 'temporal_stats' in results and 'monthly' in results['temporal_stats']:
                    f.write("-" * 80 + "\n")
                    f.write("ÉVOLUTION TEMPORELLE DES ERREURS (MENSUELLE)\n")
                    f.write("-" * 80 + "\n")
                    
                    monthly_data = results['temporal_stats']['monthly']
                    months = sorted(list(monthly_data.get('CorrectNumbers', {}).keys()))
                    
                    f.write(f"{'Mois':<10} {'Numéros corrects':<20} {'Étoiles correctes':<20} {'Erreur moyenne numéros':<25} {'Erreur moyenne étoiles':<25}\n")
                    for month in months:
                        correct_numbers = monthly_data.get('CorrectNumbers', {}).get(month, 0)
                        correct_stars = monthly_data.get('CorrectStars', {}).get(month, 0)
                        error_numbers = monthly_data.get('AvgErrorNumbers', {}).get(month, 0)
                        error_stars = monthly_data.get('AvgErrorStars', {}).get(month, 0)
                        
                        f.write(f"{month:<10} {correct_numbers:<20.2f} {correct_stars:<20.2f} {error_numbers:<25.2f} {error_stars:<25.2f}\n")
                    f.write("\n")
            
            logger.info(f"Résultats de l'analyse d'erreurs sauvegardés dans: {results_file}")
            return str(results_file)
        
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des résultats de l'analyse d'erreurs: {str(e)}")
            logger.debug(traceback.format_exc())
            return ""

    def plot_error_analysis(self, error_df: pd.DataFrame, results: Dict = None, timestamp: str = None) -> List[str]:
        """
        Génère des visualisations de l'analyse d'erreurs.
        
        Args:
            error_df: DataFrame contenant les erreurs de prédiction
            results: Dictionnaire contenant les résultats de l'analyse (optionnel)
            timestamp: Horodatage pour les noms de fichiers (optionnel)
            
        Returns:
            List[str]: Liste des chemins des fichiers de visualisation générés
        """
        logger.info("Génération des graphiques d'analyse d'erreurs...")
        
        try:
            if error_df.empty:
                logger.error("DataFrame d'erreurs vide. Impossible de générer des graphiques.")
                return []
            
            if timestamp is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            plot_files = []
            
            # Graphique 1: Distribution des numéros corrects
            plt.figure(figsize=(10, 6))
            sns.countplot(x='CorrectNumbers', data=error_df, palette='viridis')
            plt.title('Distribution du nombre de numéros corrects')
            plt.xlabel('Nombre de numéros corrects')
            plt.ylabel('Nombre de prédictions')
            plt.grid(axis='y', alpha=0.3)
            plot_path = self.output_dir / f"error_analysis_numbers_dist_{timestamp}.png"
            plt.savefig(plot_path)
            plt.close()
            plot_files.append(str(plot_path))
            
            # Graphique 2: Distribution des étoiles correctes
            plt.figure(figsize=(10, 6))
            sns.countplot(x='CorrectStars', data=error_df, palette='viridis')
            plt.title('Distribution du nombre d\'étoiles correctes')
            plt.xlabel('Nombre d\'étoiles correctes')
            plt.ylabel('Nombre de prédictions')
            plt.grid(axis='y', alpha=0.3)
            plot_path = self.output_dir / f"error_analysis_stars_dist_{timestamp}.png"
            plt.savefig(plot_path)
            plt.close()
            plot_files.append(str(plot_path))
            
            # Graphique 3: Erreurs moyennes par position
            error_cols_numbers = [col for col in error_df.columns if col.startswith('ErrorN')]
            error_cols_stars = [col for col in error_df.columns if col.startswith('ErrorE')]
            
            # Erreurs des numéros par position
            plt.figure(figsize=(12, 6))
            error_means = [error_df[col].mean() for col in error_cols_numbers]
            error_stds = [error_df[col].std() for col in error_cols_numbers]
            positions = [int(col[5:]) for col in error_cols_numbers]
            
            plt.bar(positions, error_means, yerr=error_stds, alpha=0.7, capsize=10, color='skyblue')
            plt.title('Erreur moyenne par position (Numéros principaux)')
            plt.xlabel('Position')
            plt.ylabel('Erreur moyenne')
            plt.xticks(positions)
            plt.grid(axis='y', alpha=0.3)
            plot_path = self.output_dir / f"error_analysis_numbers_pos_{timestamp}.png"
            plt.savefig(plot_path)
            plt.close()
            plot_files.append(str(plot_path))
            
            # Erreurs des étoiles par position
            plt.figure(figsize=(10, 6))
            error_means = [error_df[col].mean() for col in error_cols_stars]
            error_stds = [error_df[col].std() for col in error_cols_stars]
            positions = [int(col[5:]) for col in error_cols_stars]
            
            plt.bar(positions, error_means, yerr=error_stds, alpha=0.7, capsize=10, color='salmon')
            plt.title('Erreur moyenne par position (Étoiles)')
            plt.xlabel('Position')
            plt.ylabel('Erreur moyenne')
            plt.xticks(positions)
            plt.grid(axis='y', alpha=0.3)
            plot_path = self.output_dir / f"error_analysis_stars_pos_{timestamp}.png"
            plt.savefig(plot_path)
            plt.close()
            plot_files.append(str(plot_path))
            
            # Graphique 4: Évolution temporelle des erreurs (si les dates sont disponibles)
            if 'Date' in error_df.columns and not error_df['Date'].isna().all():
                # Convertir en datetime si nécessaire
                if not pd.api.types.is_datetime64_dtype(error_df['Date']):
                    error_df['Date'] = pd.to_datetime(error_df['Date'], errors='coerce')
                
                # Trier par date
                temp_df = error_df.sort_values('Date')
                
                plt.figure(figsize=(14, 7))
                plt.plot(temp_df['Date'], temp_df['AvgErrorNumbers'], label='Erreur moyenne (Numéros)', color='blue')
                plt.plot(temp_df['Date'], temp_df['AvgErrorStars'], label='Erreur moyenne (Étoiles)', color='red')
                plt.title('Évolution temporelle des erreurs de prédiction')
                plt.xlabel('Date')
                plt.ylabel('Erreur moyenne')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plot_path = self.output_dir / f"error_analysis_temporal_{timestamp}.png"
                plt.savefig(plot_path)
                plt.close()
                plot_files.append(str(plot_path))
                
                # Graphique 5: Évolution temporelle du nombre de numéros et étoiles corrects
                plt.figure(figsize=(14, 7))
                plt.plot(temp_df['Date'], temp_df['CorrectNumbers'], label='Numéros corrects', color='green')
                plt.plot(temp_df['Date'], temp_df['CorrectStars'], label='Étoiles correctes', color='purple')
                plt.title('Évolution temporelle du nombre de numéros et étoiles corrects')
                plt.xlabel('Date')
                plt.ylabel('Nombre correct')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plot_path = self.output_dir / f"error_analysis_correct_temporal_{timestamp}.png"
                plt.savefig(plot_path)
                plt.close()
                plot_files.append(str(plot_path))
            
            logger.info(f"Graphiques d'analyse d'erreurs générés: {len(plot_files)} fichiers")
            return plot_files
        
        except Exception as e:
            logger.error(f"Erreur lors de la génération des graphiques d'analyse d'erreurs: {str(e)}")
            logger.debug(traceback.format_exc())
            return []

# Fonction principale pour tester le module
def main():
    """Fonction principale pour tester le module."""
    try:
        # Créer l'analyseur d'erreurs
        error_analyzer = ErrorAnalyzer(output_dir="resultats_error_analysis")
        
        # Générer des données de test
        num_predictions = 50
        num_main_numbers = 5
        num_stars = 2
        
        # Prédictions aléatoires
        predictions = []
        for _ in range(num_predictions):
            pred_numbers = sorted(np.random.choice(range(1, 51), size=num_main_numbers, replace=False))
            pred_stars = sorted(np.random.choice(range(1, 13), size=num_stars, replace=False))
            predictions.append(pred_numbers + pred_stars)
        
        # Tirages réels aléatoires
        actual_draws = []
        for _ in range(num_predictions):
            actual_numbers = sorted(np.random.choice(range(1, 51), size=num_main_numbers, replace=False))
            actual_stars = sorted(np.random.choice(range(1, 13), size=num_stars, replace=False))
            actual_draws.append(actual_numbers + actual_stars)
        
        # Dates de test
        dates = [datetime.now() - timedelta(days=i*7) for i in range(num_predictions)]
        
        # Comparer les prédictions avec les tirages réels
        error_df = error_analyzer.compare_predictions_with_actual(
            predictions, actual_draws, dates, num_main_numbers, num_stars)
        
        # Analyser les erreurs
        results = error_analyzer.analyze_errors(error_df)
        
        # Sauvegarder les résultats
        error_analyzer.save_error_analysis(results)
        
        # Exporter les erreurs vers CSV
        error_analyzer.export_errors_to_csv(error_df)
        
        # Générer des graphiques
        error_analyzer.plot_error_analysis(error_df, results)
        
        print("Test du module d'analyse d'erreurs terminé avec succès.")
    
    except Exception as e:
        print(f"Erreur lors du test du module d'analyse d'erreurs: {str(e)}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
