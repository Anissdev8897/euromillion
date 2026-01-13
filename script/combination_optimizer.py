#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module d'optimisation des combinaisons pour l'analyseur Euromillions - Version optimisée
Ce module implémente un système d'optimisation des combinaisons qui identifie
les numéros récurrents dans les prédictions et crée des grilles optimisées.
Optimisations: parallélisation, vectorisation et mise en cache
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Set, Optional, Union, Any
from collections import Counter
from pathlib import Path
import traceback
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("EuromillionsCombinationOptimizer")

# Détection du nombre de cœurs pour le parallélisme
num_cores = multiprocessing.cpu_count()
logger.info(f"Nombre de cœurs détectés: {num_cores}")

class CombinationOptimizer:
    """Classe pour optimiser les combinaisons Euromillions."""
    
    def __init__(self, num_main_numbers: int = 5, num_stars: int = 2, output_dir=None):
        """
        Initialise l'optimiseur de combinaisons.
        
        Args:
            num_main_numbers: Nombre de numéros principaux dans une combinaison Euromillions
            num_stars: Nombre d'étoiles dans une combinaison Euromillions
            output_dir: Répertoire de sortie pour les résultats (optionnel)
        """
        self.num_main_numbers = num_main_numbers
        self.num_stars = num_stars
        
        # Répertoire de sortie
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path("resultats_combination_optimizer")
        
        # Créer le répertoire de sortie s'il n'existe pas
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)
            logger.info(f"Répertoire d'optimisation de combinaisons créé: {self.output_dir}")
        
        # Cache pour stocker les résultats intermédiaires
        self.cache = {
            "number_frequency": {},
            "star_frequency": {},
            "optimized_grid": {},
            "optimized_stars": {}
        }

    def optimize_combinations(self, combinations: List[List[int]], number_scores: Dict[int, float] = None, star_scores: Dict[int, float] = None) -> List[List[int]]:
        """
        Optimise les combinaisons en identifiant les numéros récurrents et en créant des grilles optimisées.
        
        Args:
            combinations: Liste des combinaisons prédites (chaque combinaison contient les numéros principaux suivis des étoiles)
            number_scores: Dictionnaire des scores des numéros principaux (optionnel)
            star_scores: Dictionnaire des scores des étoiles (optionnel)
            
        Returns:
            List[List[int]]: Liste des combinaisons optimisées
        """
        logger.info("Optimisation des combinaisons Euromillions...")
        
        try:
            if not combinations:
                logger.error("Aucune combinaison à optimiser.")
                return []
            
            # Vérifier que les combinaisons ont le bon format
            for i, combo in enumerate(combinations):
                if len(combo) != self.num_main_numbers + self.num_stars:
                    logger.warning(f"Combinaison {i} avec format incorrect: {len(combo)} numéros au lieu de {self.num_main_numbers + self.num_stars}")
                    return combinations  # Retourner les combinaisons originales en cas d'erreur
            
            # Créer une copie des combinaisons originales
            optimized_combinations = combinations.copy()
            
            # Utiliser le parallélisme pour créer la grille et les étoiles optimisées simultanément
            with ThreadPoolExecutor(max_workers=2) as executor:
                # Soumettre les tâches en parallèle
                grid_future = executor.submit(self._create_optimized_grid, combinations, number_scores)
                stars_future = executor.submit(self._create_optimized_stars, combinations, star_scores)
                
                # Récupérer les résultats
                optimized_grid = grid_future.result()
                optimized_stars = stars_future.result()
            
            # Remplacer la dernière combinaison par la grille optimisée
            if optimized_grid and optimized_stars:
                optimized_combinations[-1] = optimized_grid + optimized_stars
                logger.info(f"Grille optimisée créée: Numéros {optimized_grid}, Étoiles {optimized_stars}")
            
            return optimized_combinations
        
        except Exception as e:
            logger.error(f"Erreur lors de l'optimisation des combinaisons: {str(e)}")
            logger.debug(traceback.format_exc())
            return combinations  # Retourner les combinaisons originales en cas d'erreur

    def _create_optimized_grid(self, combinations: List[List[int]], number_scores: Dict[int, float] = None) -> List[int]:
        """
        Crée une grille optimisée à partir des numéros récurrents dans les combinaisons.
        
        Args:
            combinations: Liste des combinaisons prédites
            number_scores: Dictionnaire des scores des numéros principaux (optionnel)
            
        Returns:
            List[int]: Liste des numéros principaux optimisés
        """
        try:
            # Vérifier si le résultat est déjà en cache
            cache_key = str(sorted([tuple(combo[:self.num_main_numbers]) for combo in combinations]))
            if cache_key in self.cache["optimized_grid"]:
                logger.info("Utilisation du cache pour la grille optimisée")
                return self.cache["optimized_grid"][cache_key]
            
            # Extraire les numéros principaux de chaque combinaison (vectorisé)
            all_numbers = np.array([combo[:self.num_main_numbers] for combo in combinations]).flatten()
            
            # Compter la fréquence de chaque numéro (vectorisé)
            unique_numbers, counts = np.unique(all_numbers, return_counts=True)
            number_counter = dict(zip(unique_numbers, counts))
            
            # Sélectionner les numéros récurrents (qui apparaissent plus d'une fois)
            recurring_numbers = [num for num, count in number_counter.items() if count > 1]
            
            # Si pas assez de numéros récurrents, compléter avec les meilleurs scores
            if len(recurring_numbers) < self.num_main_numbers:
                if number_scores:
                    # Trier les numéros par score (vectorisé)
                    top_numbers = sorted(number_scores.keys(), key=lambda x: number_scores[x], reverse=True)
                    
                    # Ajouter les meilleurs numéros qui ne sont pas déjà dans recurring_numbers
                    for num in top_numbers:
                        if num not in recurring_numbers:
                            recurring_numbers.append(num)
                        if len(recurring_numbers) >= self.num_main_numbers:
                            break
                else:
                    # Si pas de scores disponibles, utiliser les numéros les plus fréquents
                    most_common = sorted(number_counter.items(), key=lambda x: x[1], reverse=True)
                    for num, _ in most_common:
                        if num not in recurring_numbers:
                            recurring_numbers.append(num)
                        if len(recurring_numbers) >= self.num_main_numbers:
                            break
            
            # Prendre les num_main_numbers premiers numéros et les trier
            optimized_numbers = sorted(recurring_numbers[:self.num_main_numbers])
            
            # Mettre en cache le résultat
            self.cache["optimized_grid"][cache_key] = optimized_numbers
            
            logger.info(f"Grille optimisée créée avec les numéros récurrents: {optimized_numbers}")
            return optimized_numbers
        
        except Exception as e:
            logger.error(f"Erreur lors de la création de la grille optimisée: {str(e)}")
            logger.debug(traceback.format_exc())
            return []

    def _create_optimized_stars(self, combinations: List[List[int]], star_scores: Dict[int, float] = None) -> List[int]:
        """
        Crée des étoiles optimisées à partir des étoiles récurrentes dans les combinaisons.
        
        Args:
            combinations: Liste des combinaisons prédites
            star_scores: Dictionnaire des scores des étoiles (optionnel)
            
        Returns:
            List[int]: Liste des étoiles optimisées
        """
        try:
            # Vérifier si le résultat est déjà en cache
            cache_key = str(sorted([tuple(combo[self.num_main_numbers:self.num_main_numbers+self.num_stars]) for combo in combinations]))
            if cache_key in self.cache["optimized_stars"]:
                logger.info("Utilisation du cache pour les étoiles optimisées")
                return self.cache["optimized_stars"][cache_key]
            
            # Extraire les étoiles de chaque combinaison (vectorisé)
            all_stars = np.array([combo[self.num_main_numbers:self.num_main_numbers+self.num_stars] for combo in combinations]).flatten()
            
            # Compter la fréquence de chaque étoile (vectorisé)
            unique_stars, counts = np.unique(all_stars, return_counts=True)
            star_counter = dict(zip(unique_stars, counts))
            
            # Sélectionner les étoiles récurrentes (qui apparaissent plus d'une fois)
            recurring_stars = [star for star, count in star_counter.items() if count > 1]
            
            # Si pas assez d'étoiles récurrentes, compléter avec les meilleurs scores
            if len(recurring_stars) < self.num_stars:
                if star_scores:
                    # Trier les étoiles par score (vectorisé)
                    top_stars = sorted(star_scores.keys(), key=lambda x: star_scores[x], reverse=True)
                    
                    # Ajouter les meilleures étoiles qui ne sont pas déjà dans recurring_stars
                    for star in top_stars:
                        if star not in recurring_stars:
                            recurring_stars.append(star)
                        if len(recurring_stars) >= self.num_stars:
                            break
                else:
                    # Si pas de scores disponibles, utiliser les étoiles les plus fréquentes
                    most_common = sorted(star_counter.items(), key=lambda x: x[1], reverse=True)
                    for star, _ in most_common:
                        if star not in recurring_stars:
                            recurring_stars.append(star)
                        if len(recurring_stars) >= self.num_stars:
                            break
            
            # Prendre les num_stars premières étoiles et les trier
            optimized_stars = sorted(recurring_stars[:self.num_stars])
            
            # Mettre en cache le résultat
            self.cache["optimized_stars"][cache_key] = optimized_stars
            
            logger.info(f"Étoiles optimisées créées: {optimized_stars}")
            return optimized_stars
        
        except Exception as e:
            logger.error(f"Erreur lors de la création des étoiles optimisées: {str(e)}")
            logger.debug(traceback.format_exc())
            return []

    def batch_optimize_combinations(self, batch_combinations: List[List[List[int]]], number_scores: Dict[int, float] = None, star_scores: Dict[int, float] = None) -> List[List[List[int]]]:
        """
        Optimise plusieurs lots de combinaisons en parallèle.
        
        Args:
            batch_combinations: Liste de lots de combinaisons à optimiser
            number_scores: Dictionnaire des scores des numéros principaux (optionnel)
            star_scores: Dictionnaire des scores des étoiles (optionnel)
            
        Returns:
            List[List[List[int]]]: Liste des lots de combinaisons optimisées
        """
        logger.info(f"Optimisation parallèle de {len(batch_combinations)} lots de combinaisons...")
        
        try:
            if not batch_combinations:
                logger.error("Aucun lot de combinaisons à optimiser.")
                return []
            
            # Fonction pour optimiser un lot de combinaisons
            def optimize_batch(batch):
                return self.optimize_combinations(batch, number_scores, star_scores)
            
            # Optimiser les lots en parallèle
            with ProcessPoolExecutor(max_workers=min(num_cores, len(batch_combinations))) as executor:
                optimized_batches = list(executor.map(optimize_batch, batch_combinations))
            
            logger.info(f"Optimisation parallèle terminée: {len(optimized_batches)} lots optimisés")
            return optimized_batches
        
        except Exception as e:
            logger.error(f"Erreur lors de l'optimisation parallèle des combinaisons: {str(e)}")
            logger.debug(traceback.format_exc())
            return batch_combinations  # Retourner les lots originaux en cas d'erreur

    def save_optimized_combinations(self, original_combinations: List[List[int]], optimized_combinations: List[List[int]], timestamp: str = None) -> str:
        """
        Sauvegarde les combinaisons originales et optimisées dans un fichier texte.
        
        Args:
            original_combinations: Liste des combinaisons originales
            optimized_combinations: Liste des combinaisons optimisées
            timestamp: Horodatage pour le nom du fichier (optionnel)
            
        Returns:
            str: Chemin du fichier de résultats sauvegardé
        """
        logger.info("Sauvegarde des combinaisons optimisées...")
        
        try:
            if not original_combinations or not optimized_combinations:
                logger.error("Aucune combinaison à sauvegarder.")
                return ""
            
            if timestamp is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            results_file = self.output_dir / f"optimized_combinations_{timestamp}.txt"
            
            with open(results_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("COMBINAISONS EUROMILLIONS OPTIMISÉES\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Date de l'optimisation: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Combinaisons originales
                f.write("-" * 80 + "\n")
                f.write("COMBINAISONS ORIGINALES\n")
                f.write("-" * 80 + "\n")
                for i, combo in enumerate(original_combinations, 1):
                    if len(combo) >= self.num_main_numbers + self.num_stars:
                        main_nums = combo[:self.num_main_numbers]
                        stars = combo[self.num_main_numbers:self.num_main_numbers + self.num_stars]
                        f.write(f"Combinaison {i}: {main_nums} | Étoiles: {stars}\n")
                f.write("\n")
                
                # Combinaisons optimisées
                f.write("-" * 80 + "\n")
                f.write("COMBINAISONS OPTIMISÉES\n")
                f.write("-" * 80 + "\n")
                for i, combo in enumerate(optimized_combinations, 1):
                    if len(combo) >= self.num_main_numbers + self.num_stars:
                        main_nums = combo[:self.num_main_numbers]
                        stars = combo[self.num_main_numbers:self.num_main_numbers + self.num_stars]
                        f.write(f"Combinaison {i}: {main_nums} | Étoiles: {stars}\n")
                f.write("\n")
            logger.info(f"Combinaisons optimisées sauvegardées dans: {results_file}")
            return str(results_file)
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des combinaisons: {str(e)}")
            logger.debug(traceback.format_exc())
            return ""
