#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyseur Euromillions basé sur la série de Fibonacci inversée
Ce script implémente l'algorithme décrit dans l'article d'Atelier-de-France
qui utilise une série de Fibonacci inversée pour pondérer les numéros.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from pathlib import Path
import logging
import random
from datetime import datetime

# Importer notre module de pondération Fibonacci
sys.path.append('/home/ubuntu/upload')
from fibonacci_weighting import apply_inverse_fibonacci_weights

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("EuromillionsFibonacciAnalyzer")

class EuromillionsFibonacciAnalyzer:
    """Classe pour l'analyse et la prédiction Euromillions basée sur Fibonacci inversé."""
    
    def __init__(self, csv_file, output_dir="resultats_fibonacci"):
        """
        Initialise l'analyseur avec le fichier CSV des tirages.
        
        Args:
            csv_file: Chemin du fichier CSV contenant les tirages
            output_dir: Répertoire de sortie pour les résultats
        """
        self.csv_file = csv_file
        self.output_dir = Path(output_dir)
        
        # Créer le répertoire de sortie s'il n'existe pas
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)
            logger.info(f"Répertoire créé: {self.output_dir}")
        
        # Charger les données
        self.df = None
        self.load_data()
        
        # Colonnes pour les numéros et étoiles
        self.number_cols = [f"N{i}" for i in range(1, 6)]
        self.star_cols = [f"E{i}" for i in range(1, 3)]
        
        # Statistiques
        self.number_freq = None
        self.star_freq = None
        self.number_weights = None
        self.star_weights = None
        
        # Résultats
        self.predictions = []
    
    def load_data(self):
        """
        Charge les données du fichier CSV.
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
                    raise FileNotFoundError(f"Fichier CSV {self.csv_file} non trouvé.")
                
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
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données: {str(e)}")
            raise
    
    def analyze_frequencies(self):
        """Analyse les fréquences d'apparition des numéros et étoiles."""
        logger.info("Analyse des fréquences...")
        
        # Extraire tous les numéros et étoiles
        all_numbers = []
        all_stars = []
        
        for _, row in self.df.iterrows():
            all_numbers.extend([int(row[col]) for col in self.number_cols])
            all_stars.extend([int(row[col]) for col in self.star_cols])
        
        # Calculer les fréquences
        self.number_freq = Counter(all_numbers)
        self.star_freq = Counter(all_stars)
        
        logger.info(f"Fréquences calculées: {len(self.number_freq)} numéros, {len(self.star_freq)} étoiles")
    
    def apply_fibonacci_weighting(self):
        """
        Applique la pondération basée sur la série de Fibonacci inversée.
        Les numéros/étoiles moins fréquents reçoivent des poids plus élevés.
        """
        logger.info("Application de la pondération Fibonacci inversée...")
        
        if self.number_freq is None or self.star_freq is None:
            self.analyze_frequencies()
        
        # Appliquer la pondération Fibonacci inversée (moins fréquent = poids plus élevé)
        self.number_weights = apply_inverse_fibonacci_weights(self.number_freq, reverse_order=False)
        self.star_weights = apply_inverse_fibonacci_weights(self.star_freq, reverse_order=False)
        
        logger.info("Pondération Fibonacci inversée appliquée")
        
        # Afficher les 10 numéros avec les poids les plus élevés
        top_numbers = sorted(self.number_weights.items(), key=lambda x: x[1], reverse=True)[:10]
        logger.info("Top 10 numéros (pondération Fibonacci inversée):")
        for num, weight in top_numbers:
            logger.info(f"Numéro {num}: {weight:.4f}")
        
        # Afficher les étoiles avec les poids les plus élevés
        top_stars = sorted(self.star_weights.items(), key=lambda x: x[1], reverse=True)
        logger.info("Étoiles (pondération Fibonacci inversée):")
        for star, weight in top_stars:
            logger.info(f"Étoile {star}: {weight:.4f}")
    
    def generate_combinations(self, num_combinations=5):
        """
        Génère des combinaisons basées sur la pondération Fibonacci inversée.
        
        Args:
            num_combinations: Nombre de combinaisons à générer
            
        Returns:
            Liste de tuples (numéros, étoiles)
        """
        logger.info(f"Génération de {num_combinations} combinaisons...")
        
        if self.number_weights is None or self.star_weights is None:
            self.apply_fibonacci_weighting()
        
        combinations = []
        
        for _ in range(num_combinations):
            # Générer 5 numéros uniques en utilisant les poids Fibonacci
            numbers = []
            number_items = list(self.number_weights.items())
            number_values = [num for num, _ in number_items]
            number_weights = [weight for _, weight in number_items]
            
            # Normaliser les poids
            if sum(number_weights) > 0:
                number_weights = [w / sum(number_weights) for w in number_weights]
            
            while len(numbers) < 5:
                num = np.random.choice(number_values, p=number_weights)
                if num not in numbers:
                    numbers.append(num)
            
            # Trier les numéros
            numbers.sort()
            
            # Générer 2 étoiles uniques en utilisant les poids Fibonacci
            stars = []
            star_items = list(self.star_weights.items())
            star_values = [star for star, _ in star_items]
            star_weights = [weight for _, weight in star_items]
            
            # Normaliser les poids
            if sum(star_weights) > 0:
                star_weights = [w / sum(star_weights) for w in star_weights]
            
            while len(stars) < 2:
                star = np.random.choice(star_values, p=star_weights)
                if star not in stars:
                    stars.append(star)
            
            # Trier les étoiles
            stars.sort()
            
            combinations.append((numbers, stars))
        
        self.predictions = combinations
        return combinations
    
    def save_predictions(self, filename=None):
        """
        Sauvegarde les prédictions dans un fichier texte.
        
        Args:
            filename: Nom du fichier (optionnel)
            
        Returns:
            Chemin du fichier sauvegardé
        """
        if not self.predictions:
            logger.warning("Aucune prédiction à sauvegarder")
            return None
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"predictions_fibonacci_{timestamp}.txt"
        
        file_path = self.output_dir / filename
        
        try:
            with open(file_path, 'w') as f:
                f.write("Prédictions Euromillions basées sur la série de Fibonacci inversée\n")
                f.write("=" * 60 + "\n\n")
                
                for i, (numbers, stars) in enumerate(self.predictions, 1):
                    f.write(f"Combinaison {i}:\n")
                    f.write(f"  Numéros: {', '.join(map(str, numbers))}\n")
                    f.write(f"  Étoiles: {', '.join(map(str, stars))}\n\n")
            
            logger.info(f"Prédictions sauvegardées: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des prédictions: {str(e)}")
            return None
    
    def visualize_weights(self):
        """
        Visualise les poids Fibonacci inversés.
        
        Returns:
            Chemin du fichier de graphique généré
        """
        logger.info("Génération de la visualisation des poids Fibonacci...")
        
        if self.number_weights is None or self.star_weights is None:
            self.apply_fibonacci_weighting()
        
        try:
            # Créer une figure avec deux sous-graphiques
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), tight_layout=True)
            
            # Graphique 1: Poids des numéros
            number_df = pd.DataFrame({
                'Numéro': list(self.number_weights.keys()),
                'Poids': list(self.number_weights.values())
            })
            number_df = number_df.sort_values('Numéro')
            
            sns.barplot(x='Numéro', y='Poids', data=number_df, ax=ax1, palette='viridis')
            
            ax1.set_title("Poids Fibonacci inversés des numéros")
            ax1.set_xlabel("Numéro")
            ax1.set_ylabel("Poids")
            
            # Graphique 2: Poids des étoiles
            star_df = pd.DataFrame({
                'Étoile': list(self.star_weights.keys()),
                'Poids': list(self.star_weights.values())
            })
            star_df = star_df.sort_values('Étoile')
            
            sns.barplot(x='Étoile', y='Poids', data=star_df, ax=ax2, palette='rocket')
            
            ax2.set_title("Poids Fibonacci inversés des étoiles")
            ax2.set_xlabel("Étoile")
            ax2.set_ylabel("Poids")
            
            # Sauvegarder le graphique
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = self.output_dir / f"fibonacci_weights_{timestamp}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualisation des poids Fibonacci sauvegardée: {plot_path}")
            
            return str(plot_path)
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération de la visualisation: {str(e)}")
            return None
    
    def run_backtesting(self, num_draws=10):
        """
        Effectue un backtesting sur les derniers tirages.
        
        Args:
            num_draws: Nombre de tirages à utiliser pour le backtesting
            
        Returns:
            Dictionnaire des résultats de backtesting
        """
        logger.info(f"Backtesting sur les {num_draws} derniers tirages...")
        
        if len(self.df) <= num_draws:
            logger.warning(f"Pas assez de données pour le backtesting. {len(self.df)} tirages disponibles, {num_draws} requis.")
            return None
        
        results = []
        
        # Utiliser les derniers tirages pour le backtesting
        test_draws = self.df.iloc[-num_draws:].copy()
        
        for i in range(num_draws):
            # Utiliser les données jusqu'à l'indice i pour l'entraînement
            train_df = self.df.iloc[:-num_draws+i]
            
            # Créer un analyseur temporaire avec les données d'entraînement
            temp_analyzer = EuromillionsFibonacciAnalyzer(self.csv_file, self.output_dir)
            temp_analyzer.df = train_df
            
            # Générer des prédictions
            temp_analyzer.apply_fibonacci_weighting()
            combinations = temp_analyzer.generate_combinations(num_combinations=5)
            
            # Tirage réel à prédire
            actual_draw = test_draws.iloc[i]
            actual_numbers = [int(actual_draw[col]) for col in self.number_cols]
            actual_stars = [int(actual_draw[col]) for col in self.star_cols]
            
            # Évaluer les prédictions
            best_match = 0
            best_combination = None
            
            for numbers, stars in combinations:
                # Calculer les correspondances
                correct_numbers = len(set(numbers) & set(actual_numbers))
                correct_stars = len(set(stars) & set(actual_stars))
                
                # Calculer le score total
                match_score = correct_numbers + correct_stars
                
                if match_score > best_match:
                    best_match = match_score
                    best_combination = (numbers, stars, correct_numbers, correct_stars)
            
            # Enregistrer les résultats
            result = {
                'tirage': i + 1,
                'actual_numbers': actual_numbers,
                'actual_stars': actual_stars,
                'best_combination': best_combination[0] if best_combination else None,
                'best_stars': best_combination[1] if best_combination else None,
                'correct_numbers': best_combination[2] if best_combination else 0,
                'correct_stars': best_combination[3] if best_combination else 0,
                'total_match': best_match
            }
            
            results.append(result)
        
        # Calculer les statistiques globales
        total_correct_numbers = sum(r['correct_numbers'] for r in results)
        total_correct_stars = sum(r['correct_stars'] for r in results)
        
        avg_correct_numbers = total_correct_numbers / num_draws
        avg_correct_stars = total_correct_stars / num_draws
        
        stats = {
            'num_draws': num_draws,
            'total_correct_numbers': total_correct_numbers,
            'total_correct_stars': total_correct_stars,
            'avg_correct_numbers': avg_correct_numbers,
            'avg_correct_stars': avg_correct_stars
        }
        
        logger.info(f"Backtesting terminé. Moyenne: {avg_correct_numbers:.2f} numéros, {avg_correct_stars:.2f} étoiles")
        
        return {
            'results': results,
            'stats': stats
        }
    
    def save_backtesting_results(self, results, filename=None):
        """
        Sauvegarde les résultats de backtesting dans un fichier texte.
        
        Args:
            results: Résultats du backtesting
            filename: Nom du fichier (optionnel)
            
        Returns:
            Chemin du fichier sauvegardé
        """
        if not results or 'stats' not in results or 'results' not in results:
            logger.warning("Aucun résultat de backtesting valide à sauvegarder")
            return None
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtesting_fibonacci_{timestamp}.txt"
        
        file_path = self.output_dir / filename
        
        try:
            with open(file_path, 'w') as f:
                f.write("Résultats de backtesting - Algorithme Fibonacci inversé\n")
                f.write("=" * 60 + "\n\n")
                
                # Statistiques globales
                stats = results['stats']
                f.write("Statistiques globales:\n")
                f.write(f"  Nombre de tirages testés: {stats['num_draws']}\n")
                f.write(f"  Nombre total de numéros corrects: {stats['total_correct_numbers']}\n")
                f.write(f"  Nombre total d'étoiles correctes: {stats['total_correct_stars']}\n")
                f.write(f"  Moyenne de numéros corrects par tirage: {stats['avg_correct_numbers']:.2f}\n")
                f.write(f"  Moyenne d'étoiles correctes par tirage: {stats['avg_correct_stars']:.2f}\n\n")
                
                # Résultats détaillés
                f.write("Résultats détaillés:\n")
                for result in results['results']:
                    f.write(f"Tirage {result['tirage']}:\n")
                    f.write(f"  Numéros réels: {', '.join(map(str, result['actual_numbers']))}\n")
                    f.write(f"  Étoiles réelles: {', '.join(map(str, result['actual_stars']))}\n")
                    
                    if result['best_combination']:
                        f.write(f"  Meilleure combinaison: {', '.join(map(str, result['best_combination']))}\n")
                    else:
                        f.write("  Meilleure combinaison: Aucune\n")
                        
                    if result['best_stars']:
                        f.write(f"  Meilleures étoiles: {', '.join(map(str, result['best_stars']))}\n")
                    else:
                        f.write("  Meilleures étoiles: Aucune\n")
                        
                    f.write(f"  Numéros corrects: {result['correct_numbers']}\n")
                    f.write(f"  Étoiles correctes: {result['correct_stars']}\n")
                    f.write(f"  Score total: {result['total_match']}\n\n")
            
            logger.info(f"Résultats de backtesting sauvegardés: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des résultats de backtesting: {str(e)}")
            return None

# Fonction principale pour exécuter l'analyse
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyseur Euromillions basé sur la série de Fibonacci inversée")
    parser.add_argument("--csv", required=True, help="Chemin du fichier CSV contenant les tirages")
    parser.add_argument("--output", default="resultats_fibonacci", help="Répertoire de sortie pour les résultats")
    parser.add_argument("--combinations", type=int, default=5, help="Nombre de combinaisons à générer")
    parser.add_argument("--backtesting", action="store_true", help="Effectuer un backtesting")
    parser.add_argument("--backtesting-draws", type=int, default=10, help="Nombre de tirages pour le backtesting")
    parser.add_argument("--visualize", action="store_true", help="Générer des visualisations")
    
    args = parser.parse_args()
    
    # Créer l'analyseur
    analyzer = EuromillionsFibonacciAnalyzer(args.csv, args.output)
    
    # Analyser les fréquences
    analyzer.analyze_frequencies()
    
    # Appliquer la pondération Fibonacci inversée
    analyzer.apply_fibonacci_weighting()
    
    # Générer des combinaisons
    combinations = analyzer.generate_combinations(args.combinations)
    
    # Afficher les combinaisons
    print("\nCombinations générées:")
    for i, (numbers, stars) in enumerate(combinations, 1):
        print(f"Combinaison {i}: Numéros {numbers}, Étoiles {stars}")
    
    # Sauvegarder les prédictions
    predictions_file = analyzer.save_predictions()
    
    # Générer des visualisations
    if args.visualize:
        visualization_file = analyzer.visualize_weights()
    
    # Effectuer un backtesting
    if args.backtesting:
        backtesting_results = analyzer.run_backtesting(args.backtesting_draws)
        if backtesting_results:
            backtesting_file = analyzer.save_backtesting_results(backtesting_results)

if __name__ == "__main__":
    main()
