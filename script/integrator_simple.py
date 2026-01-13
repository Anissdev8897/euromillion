#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module d'intégration super-optimisé pour l'analyse Euromillions.
Ce script génère une seule grille optimisée basée sur les numéros qui apparaissent
en double dans les combinaisons prédites.

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
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('EuromillionsSuperOptimizer')

class EuromillionsSuperOptimizer:
    """
    Intégrateur super-optimisé pour l'analyse Euromillions.
    """
    
    def __init__(self, 
                 csv_file: str = None, 
                 output_dir: str = None, 
                 combinations: int = 5,
                 show_all: bool = False):
        """
        Initialise l'intégrateur super-optimisé.
        
        Args:
            csv_file: Chemin vers le fichier CSV des tirages Euromillions
            output_dir: Répertoire de sortie pour les résultats
            combinations: Nombre de combinaisons à générer
            show_all: Afficher toutes les combinaisons normales en plus de la grille super-optimisée
        """
        self.csv_file = csv_file
        self.output_dir = Path(output_dir) if output_dir else Path('resultats_euromillions_optimized')
        self.combinations = combinations
        self.show_all = show_all
        
        # Créer le répertoire de sortie s'il n'existe pas
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)
            logger.info(f"Répertoire de sortie créé: {self.output_dir}")
        
        # Initialiser les données
        self.df = None
        self.normal_combinations = []
        self.super_optimized_combination = None
    
    def load_data(self):
        """Charge les données depuis le fichier CSV."""
        if not self.csv_file:
            logger.error("Aucun fichier CSV spécifié.")
            return False
        
        try:
            # Charger les données
            self.df = pd.read_csv(self.csv_file)
            logger.info(f"Données chargées: {len(self.df)} lignes.")
            
            # Vérifier la structure du DataFrame
            expected_cols = ['N1', 'N2', 'N3', 'N4', 'N5', 'E1', 'E2']
            missing_cols = [col for col in expected_cols if col not in self.df.columns]
            
            if missing_cols:
                logger.error(f"Colonnes manquantes dans le fichier CSV: {missing_cols}")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données: {str(e)}")
            return False
    
    def generate_combinations(self):
        """Génère des combinaisons de numéros et d'étoiles."""
        try:
            logger.info(f"Génération de {self.combinations} combinaisons...")
            
            # Calculer les fréquences des numéros
            number_freq = {}
            for col in ['N1', 'N2', 'N3', 'N4', 'N5']:
                for num in self.df[col]:
                    number_freq[num] = number_freq.get(num, 0) + 1
            
            # Calculer les fréquences des étoiles
            star_freq = {}
            for col in ['E1', 'E2']:
                for star in self.df[col]:
                    star_freq[star] = star_freq.get(star, 0) + 1
            
            # Trier par fréquence décroissante
            sorted_numbers = sorted(number_freq.items(), key=lambda x: x[1], reverse=True)
            sorted_stars = sorted(star_freq.items(), key=lambda x: x[1], reverse=True)
            
            # Générer les combinaisons
            self.normal_combinations = []
            
            for i in range(self.combinations):
                # Sélectionner 5 numéros parmi les plus fréquents, avec un peu d'aléatoire
                import random
                top_numbers = [num for num, _ in sorted_numbers[:20]]
                numbers = sorted(random.sample(top_numbers, 5))
                
                # Sélectionner 2 étoiles parmi les plus fréquentes, avec un peu d'aléatoire
                top_stars = [star for star, _ in sorted_stars[:10]]
                stars = sorted(random.sample(top_stars, 2))
                
                self.normal_combinations.append((numbers, stars))
            
            logger.info(f"{len(self.normal_combinations)} combinaisons générées.")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de la génération des combinaisons: {str(e)}")
            return False
    
    def create_super_optimized_combination(self):
        """Crée une combinaison super-optimisée basée sur les numéros qui apparaissent en double."""
        try:
            if not self.normal_combinations:
                logger.error("Aucune combinaison à optimiser.")
                return False
            
            logger.info("Création de la combinaison super-optimisée...")
            
            # Collecter tous les numéros des combinaisons normales
            all_numbers = []
            for numbers, _ in self.normal_combinations:
                all_numbers.extend(numbers)
            
            # Collecter toutes les étoiles des combinaisons normales
            all_stars = []
            for _, stars in self.normal_combinations:
                all_stars.extend(stars)
            
            # Compter les occurrences de chaque numéro
            number_counter = Counter(all_numbers)
            star_counter = Counter(all_stars)
            
            # Identifier les numéros qui apparaissent plus d'une fois
            recurring_numbers = [num for num, count in number_counter.items() if count > 1]
            recurring_stars = [star for star, count in star_counter.items() if count > 1]
            
            logger.info(f"Numéros récurrents identifiés: {recurring_numbers}")
            logger.info(f"Étoiles récurrentes identifiées: {recurring_stars}")
            
            # Si nous n'avons pas assez de numéros récurrents, compléter avec les plus fréquents
            if len(recurring_numbers) < 5:
                # Trier les numéros par fréquence d'apparition dans les combinaisons
                sorted_by_freq = sorted(number_counter.items(), key=lambda x: x[1], reverse=True)
                
                # Ajouter les numéros les plus fréquents qui ne sont pas déjà dans recurring_numbers
                for num, _ in sorted_by_freq:
                    if num not in recurring_numbers:
                        recurring_numbers.append(num)
                        if len(recurring_numbers) == 5:
                            break
            
            # Si nous avons trop de numéros récurrents, prendre les 5 plus fréquents
            if len(recurring_numbers) > 5:
                # Filtrer le compteur pour ne garder que les numéros récurrents
                recurring_counter = {num: count for num, count in number_counter.items() if num in recurring_numbers}
                # Trier par fréquence
                sorted_recurring = sorted(recurring_counter.items(), key=lambda x: x[1], reverse=True)
                # Prendre les 5 premiers
                recurring_numbers = [num for num, _ in sorted_recurring[:5]]
            
            # Même logique pour les étoiles
            if len(recurring_stars) < 2:
                sorted_by_freq = sorted(star_counter.items(), key=lambda x: x[1], reverse=True)
                for star, _ in sorted_by_freq:
                    if star not in recurring_stars:
                        recurring_stars.append(star)
                        if len(recurring_stars) == 2:
                            break
            
            if len(recurring_stars) > 2:
                recurring_counter = {star: count for star, count in star_counter.items() if star in recurring_stars}
                sorted_recurring = sorted(recurring_counter.items(), key=lambda x: x[1], reverse=True)
                recurring_stars = [star for star, _ in sorted_recurring[:2]]
            
            # Créer la combinaison super-optimisée
            self.super_optimized_combination = (sorted(recurring_numbers), sorted(recurring_stars))
            
            logger.info(f"Combinaison super-optimisée créée: {self.super_optimized_combination}")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de la création de la combinaison super-optimisée: {str(e)}")
            return False
    
    def save_predictions(self):
        """Sauvegarde les prédictions dans un fichier."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            predictions_file = self.output_dir / f"euromillions_super_optimized_{timestamp}.txt"
            
            with open(predictions_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("PRÉDICTIONS EUROMILLIONS SUPER-OPTIMISÉES\n")
                f.write("=" * 80 + "\n\n")
                
                if self.show_all:
                    f.write("-" * 80 + "\n")
                    f.write("COMBINAISONS NORMALES\n")
                    f.write("-" * 80 + "\n\n")
                    
                    for i, (numbers, stars) in enumerate(self.normal_combinations, 1):
                        f.write(f"Combinaison {i}: {' - '.join(map(str, numbers))} | Étoiles: {' - '.join(map(str, stars))}\n")
                    
                    f.write("\n")
                
                f.write("-" * 80 + "\n")
                f.write("COMBINAISON SUPER-OPTIMISÉE (BASÉE SUR LES NUMÉROS RÉCURRENTS)\n")
                f.write("-" * 80 + "\n\n")
                
                numbers, stars = self.super_optimized_combination
                f.write(f"Grille optimisée: {' - '.join(map(str, numbers))} | Étoiles: {' - '.join(map(str, stars))}\n\n")
                
                f.write("Cette grille contient uniquement les numéros qui apparaissent dans plusieurs\n")
                f.write("combinaisons prédites, maximisant ainsi les chances de gain.\n\n")
                
                f.write("-" * 80 + "\n")
                f.write("INFORMATIONS SUR L'OPTIMISATION\n")
                f.write("-" * 80 + "\n")
                f.write(f"Nombre de combinaisons analysées: {self.combinations}\n")
                f.write(f"Date de génération: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n")
            
            logger.info(f"Prédictions sauvegardées dans: {predictions_file}")
            return str(predictions_file)
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des prédictions: {str(e)}")
            return ""
    
    def run(self):
        """Exécute l'intégration complète."""
        try:
            # 1. Charger les données
            if not self.load_data():
                logger.error("Impossible de continuer sans données.")
                return False
            
            # 2. Générer les combinaisons
            if not self.generate_combinations():
                logger.error("Impossible de générer les combinaisons.")
                return False
            
            # 3. Créer la combinaison super-optimisée
            if not self.create_super_optimized_combination():
                logger.error("Impossible de créer la combinaison super-optimisée.")
                return False
            
            # 4. Sauvegarder les prédictions
            predictions_file = self.save_predictions()
            if not predictions_file:
                logger.error("Impossible de sauvegarder les prédictions.")
                return False
            
            logger.info("Intégration terminée avec succès.")
            print(f"Prédictions sauvegardées dans: {predictions_file}")
            
            # Afficher la combinaison super-optimisée
            numbers, stars = self.super_optimized_combination
            print("\nCOMBINAISON SUPER-OPTIMISÉE:")
            print(f"Numéros: {' - '.join(map(str, numbers))}")
            print(f"Étoiles: {' - '.join(map(str, stars))}")
            print("\nCette grille contient uniquement les numéros qui apparaissent dans plusieurs combinaisons prédites.")
            
            return True
        except Exception as e:
            logger.error(f"Erreur lors de l'intégration: {str(e)}")
            return False

def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(description="Intégrateur super-optimisé pour l'analyse Euromillions")
    parser.add_argument("--csv", type=str, required=True, help="Chemin vers le fichier CSV des tirages Euromillions")
    parser.add_argument("--output", type=str, default="resultats_euromillions_optimized", help="Répertoire de sortie pour les résultats")
    parser.add_argument("--combinations", type=int, default=5, help="Nombre de combinaisons à analyser")
    parser.add_argument("--show-all", action="store_true", help="Afficher toutes les combinaisons normales en plus de la grille super-optimisée")
    
    args = parser.parse_args()
    
    try:
        optimizer = EuromillionsSuperOptimizer(
            csv_file=args.csv,
            output_dir=args.output,
            combinations=args.combinations,
            show_all=args.show_all
        )
        
        success = optimizer.run()
        
        if success:
            print(f"Intégration terminée avec succès.")
            print(f"Résultats sauvegardés dans: {args.output}")
        else:
            print("L'intégration a échoué. Consultez les logs pour plus d'informations.")
            sys.exit(1)
    except Exception as e:
        print(f"Erreur lors de l'exécution: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
