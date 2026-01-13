#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module d'Analyse de Gaps (Écarts entre Tirages)
Analyse les écarts entre apparitions de chaque numéro/étoile pour:
- Calculer les gaps moyens et distributions
- Prédire le prochain gap probable
- Détecter patterns de gaps récurrents
- Analyser la "dette" d'apparition
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from datetime import datetime
import traceback
from collections import defaultdict

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("GapAnalyzer")

# Imports statistiques
try:
    from scipy import stats
    from scipy.stats import norm, poisson
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("⚠️ scipy non disponible")


class GapAnalyzer:
    """
    Classe pour l'analyse des gaps (écarts) entre apparitions
    des numéros et étoiles dans les tirages EuroMillions.
    """
    
    def __init__(
        self,
        max_number: int = 50,
        max_star: int = 12,
        min_history: int = 100
    ):
        """
        Initialise l'analyseur de gaps.
        
        Args:
            max_number: Numéro maximum (50 pour EuroMillions)
            max_star: Étoile maximum (12 pour EuroMillions)
            min_history: Nombre minimum de tirages pour analyse fiable
        """
        self.max_number = max_number
        self.max_star = max_star
        self.min_history = min_history
        
        # Données de gaps
        self.number_gaps = defaultdict(list)  # {numéro: [gap1, gap2, ...]}
        self.star_gaps = defaultdict(list)    # {étoile: [gap1, gap2, ...]}
        
        # Gaps actuels (depuis dernière apparition)
        self.current_number_gaps = {}
        self.current_star_gaps = {}
        
        # Statistiques de gaps
        self.number_gap_stats = {}
        self.star_gap_stats = {}
        
        # Historique des tirages
        self.draw_history = []
        
        logger.info(f"✅ GapAnalyzer initialisé")
        logger.info(f"   Numéros: 1-{max_number}")
        logger.info(f"   Étoiles: 1-{max_star}")
    
    def load_data(self, df: pd.DataFrame, date_col: str = 'date_de_tirage',
                  number_cols: List[str] = None, star_cols: List[str] = None):
        """
        Charge les données historiques et calcule les gaps.
        
        Args:
            df: DataFrame avec l'historique des tirages
            date_col: Nom de la colonne de date
            number_cols: Colonnes des numéros (ex: ['N1', 'N2', 'N3', 'N4', 'N5'])
            star_cols: Colonnes des étoiles (ex: ['E1', 'E2'])
        """
        if number_cols is None:
            number_cols = ['N1', 'N2', 'N3', 'N4', 'N5']
        if star_cols is None:
            star_cols = ['E1', 'E2']
        
        logger.info(f"📊 Chargement de {len(df)} tirages...")
        
        # Trier par date (du plus ancien au plus récent)
        df = df.sort_values(date_col, ascending=True).reset_index(drop=True)
        
        # Initialiser les compteurs de gaps
        last_seen_number = {i: -1 for i in range(1, self.max_number + 1)}
        last_seen_star = {i: -1 for i in range(1, self.max_star + 1)}
        
        # Parcourir les tirages
        for idx, row in df.iterrows():
            # Extraire les numéros et étoiles
            numbers = [row[col] for col in number_cols if col in row]
            stars = [row[col] for col in star_cols if col in row]
            
            # Enregistrer le tirage
            self.draw_history.append({
                'index': idx,
                'date': row[date_col] if date_col in row else None,
                'numbers': numbers,
                'stars': stars
            })
            
            # Calculer les gaps pour les numéros
            for num in numbers:
                if last_seen_number[num] >= 0:
                    gap = idx - last_seen_number[num]
                    self.number_gaps[num].append(gap)
                last_seen_number[num] = idx
            
            # Calculer les gaps pour les étoiles
            for star in stars:
                if last_seen_star[star] >= 0:
                    gap = idx - last_seen_star[star]
                    self.star_gaps[star].append(gap)
                last_seen_star[star] = idx
        
        # Calculer les gaps actuels (depuis dernière apparition)
        total_draws = len(df)
        for num in range(1, self.max_number + 1):
            if last_seen_number[num] >= 0:
                self.current_number_gaps[num] = total_draws - 1 - last_seen_number[num]
            else:
                self.current_number_gaps[num] = total_draws  # Jamais apparu
        
        for star in range(1, self.max_star + 1):
            if last_seen_star[star] >= 0:
                self.current_star_gaps[star] = total_draws - 1 - last_seen_star[star]
            else:
                self.current_star_gaps[star] = total_draws
        
        # Calculer les statistiques
        self._calculate_gap_statistics()
        
        logger.info(f"✅ Données chargées: {total_draws} tirages")
        logger.info(f"   Gaps numéros calculés: {sum(len(g) for g in self.number_gaps.values())}")
        logger.info(f"   Gaps étoiles calculés: {sum(len(g) for g in self.star_gaps.values())}")
    
    def _calculate_gap_statistics(self):
        """Calcule les statistiques de gaps pour chaque numéro et étoile."""
        logger.info("📈 Calcul des statistiques de gaps...")
        
        # Statistiques pour les numéros
        for num in range(1, self.max_number + 1):
            gaps = self.number_gaps.get(num, [])
            
            if len(gaps) >= 2:
                self.number_gap_stats[num] = {
                    'count': len(gaps),
                    'mean': np.mean(gaps),
                    'median': np.median(gaps),
                    'std': np.std(gaps),
                    'min': np.min(gaps),
                    'max': np.max(gaps),
                    'current_gap': self.current_number_gaps.get(num, 0),
                    'gap_debt': self.current_number_gaps.get(num, 0) - np.mean(gaps)
                }
            else:
                self.number_gap_stats[num] = {
                    'count': len(gaps),
                    'mean': 0,
                    'median': 0,
                    'std': 0,
                    'min': 0,
                    'max': 0,
                    'current_gap': self.current_number_gaps.get(num, 0),
                    'gap_debt': 0
                }
        
        # Statistiques pour les étoiles
        for star in range(1, self.max_star + 1):
            gaps = self.star_gaps.get(star, [])
            
            if len(gaps) >= 2:
                self.star_gap_stats[star] = {
                    'count': len(gaps),
                    'mean': np.mean(gaps),
                    'median': np.median(gaps),
                    'std': np.std(gaps),
                    'min': np.min(gaps),
                    'max': np.max(gaps),
                    'current_gap': self.current_star_gaps.get(star, 0),
                    'gap_debt': self.current_star_gaps.get(star, 0) - np.mean(gaps)
                }
            else:
                self.star_gap_stats[star] = {
                    'count': len(gaps),
                    'mean': 0,
                    'median': 0,
                    'std': 0,
                    'min': 0,
                    'max': 0,
                    'current_gap': self.current_star_gaps.get(star, 0),
                    'gap_debt': 0
                }
        
        logger.info("✅ Statistiques calculées")
    
    def get_overdue_numbers(self, top_n: int = 10) -> List[Tuple[int, float]]:
        """
        Retourne les numéros les plus en retard (dette de gap élevée).
        
        Args:
            top_n: Nombre de numéros à retourner
            
        Returns:
            Liste de (numéro, dette_de_gap) triée par dette décroissante
        """
        overdue = []
        
        for num, stats in self.number_gap_stats.items():
            if stats['count'] >= 5:  # Minimum d'historique
                overdue.append((num, stats['gap_debt']))
        
        # Trier par dette décroissante
        overdue.sort(key=lambda x: x[1], reverse=True)
        
        return overdue[:top_n]
    
    def get_overdue_stars(self, top_n: int = 5) -> List[Tuple[int, float]]:
        """
        Retourne les étoiles les plus en retard.
        
        Args:
            top_n: Nombre d'étoiles à retourner
            
        Returns:
            Liste de (étoile, dette_de_gap) triée par dette décroissante
        """
        overdue = []
        
        for star, stats in self.star_gap_stats.items():
            if stats['count'] >= 3:
                overdue.append((star, stats['gap_debt']))
        
        overdue.sort(key=lambda x: x[1], reverse=True)
        
        return overdue[:top_n]
    
    def predict_next_appearance_probability(
        self,
        number: int,
        is_star: bool = False
    ) -> float:
        """
        Prédit la probabilité qu'un numéro/étoile sorte au prochain tirage
        basé sur son gap actuel et sa distribution historique.
        
        Args:
            number: Numéro ou étoile
            is_star: True si c'est une étoile
            
        Returns:
            Probabilité entre 0 et 1
        """
        if is_star:
            stats = self.star_gap_stats.get(number)
            gaps = self.star_gaps.get(number, [])
        else:
            stats = self.number_gap_stats.get(number)
            gaps = self.number_gaps.get(number, [])
        
        if not stats or not gaps or len(gaps) < 5:
            # Pas assez d'historique, probabilité uniforme
            return 1.0 / (self.max_star if is_star else self.max_number)
        
        current_gap = stats['current_gap']
        mean_gap = stats['mean']
        std_gap = stats['std']
        
        if std_gap == 0:
            std_gap = 1  # Éviter division par zéro
        
        # Utiliser une distribution normale pour modéliser les gaps
        if SCIPY_AVAILABLE:
            # Probabilité cumulative que le gap soit >= current_gap
            z_score = (current_gap - mean_gap) / std_gap
            prob = 1 - norm.cdf(z_score)
            
            # Normaliser entre 0 et 1
            prob = max(0.01, min(0.99, prob))
        else:
            # Approximation simple sans scipy
            if current_gap > mean_gap:
                # Plus le gap actuel est grand, plus la probabilité est élevée
                prob = min(0.99, 0.5 + (current_gap - mean_gap) / (2 * mean_gap))
            else:
                prob = max(0.01, 0.5 - (mean_gap - current_gap) / (2 * mean_gap))
        
        return prob
    
    def get_number_scores(self) -> Dict[int, float]:
        """
        Calcule les scores de probabilité pour tous les numéros.
        
        Returns:
            Dict {numéro: score} avec scores entre 0 et 1
        """
        scores = {}
        
        for num in range(1, self.max_number + 1):
            scores[num] = self.predict_next_appearance_probability(num, is_star=False)
        
        return scores
    
    def get_star_scores(self) -> Dict[int, float]:
        """
        Calcule les scores de probabilité pour toutes les étoiles.
        
        Returns:
            Dict {étoile: score} avec scores entre 0 et 1
        """
        scores = {}
        
        for star in range(1, self.max_star + 1):
            scores[star] = self.predict_next_appearance_probability(star, is_star=True)
        
        return scores
    
    def predict(
        self,
        num_numbers: int = 5,
        num_stars: int = 2
    ) -> Dict[str, List[int]]:
        """
        Prédit les numéros et étoiles basés sur l'analyse des gaps.
        
        Args:
            num_numbers: Nombre de numéros à prédire
            num_stars: Nombre d'étoiles à prédire
            
        Returns:
            Dict avec 'numbers' et 'stars'
        """
        # Obtenir les scores
        number_scores = self.get_number_scores()
        star_scores = self.get_star_scores()
        
        # Sélectionner les top numéros
        top_numbers = sorted(
            number_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:num_numbers]
        
        # Sélectionner les top étoiles
        top_stars = sorted(
            star_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:num_stars]
        
        prediction = {
            'numbers': sorted([num for num, _ in top_numbers]),
            'stars': sorted([star for star, _ in top_stars])
        }
        
        logger.info(f"🎯 Prédiction basée sur gaps:")
        logger.info(f"   Numéros: {prediction['numbers']}")
        logger.info(f"   Étoiles: {prediction['stars']}")
        
        return prediction
    
    def get_gap_report(self, output_file: Optional[str] = None) -> str:
        """
        Génère un rapport détaillé sur les gaps.
        
        Args:
            output_file: Fichier de sortie (optionnel)
            
        Returns:
            Chemin du rapport ou contenu si pas de fichier
        """
        report_lines = []
        report_lines.append("# Rapport d'Analyse des Gaps - EuroMillions\n")
        report_lines.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        report_lines.append("---\n\n")
        
        # Numéros en retard
        report_lines.append("## 🔴 Numéros les Plus en Retard\n\n")
        overdue_numbers = self.get_overdue_numbers(15)
        
        report_lines.append("| Rang | Numéro | Gap Actuel | Gap Moyen | Dette |\n")
        report_lines.append("|------|--------|------------|-----------|-------|\n")
        
        for rank, (num, debt) in enumerate(overdue_numbers, 1):
            stats = self.number_gap_stats[num]
            report_lines.append(
                f"| {rank} | **{num}** | {stats['current_gap']} | "
                f"{stats['mean']:.1f} | {debt:.1f} |\n"
            )
        
        report_lines.append("\n")
        
        # Étoiles en retard
        report_lines.append("## ⭐ Étoiles les Plus en Retard\n\n")
        overdue_stars = self.get_overdue_stars(5)
        
        report_lines.append("| Rang | Étoile | Gap Actuel | Gap Moyen | Dette |\n")
        report_lines.append("|------|--------|------------|-----------|-------|\n")
        
        for rank, (star, debt) in enumerate(overdue_stars, 1):
            stats = self.star_gap_stats[star]
            report_lines.append(
                f"| {rank} | **{star}** | {stats['current_gap']} | "
                f"{stats['mean']:.1f} | {debt:.1f} |\n"
            )
        
        report_lines.append("\n")
        
        # Statistiques globales
        report_lines.append("## 📊 Statistiques Globales\n\n")
        
        all_number_gaps = [gap for gaps in self.number_gaps.values() for gap in gaps]
        all_star_gaps = [gap for gaps in self.star_gaps.values() for gap in gaps]
        
        if all_number_gaps:
            report_lines.append(f"**Numéros**:\n")
            report_lines.append(f"- Gap moyen global: {np.mean(all_number_gaps):.2f}\n")
            report_lines.append(f"- Gap médian global: {np.median(all_number_gaps):.2f}\n")
            report_lines.append(f"- Écart-type: {np.std(all_number_gaps):.2f}\n\n")
        
        if all_star_gaps:
            report_lines.append(f"**Étoiles**:\n")
            report_lines.append(f"- Gap moyen global: {np.mean(all_star_gaps):.2f}\n")
            report_lines.append(f"- Gap médian global: {np.median(all_star_gaps):.2f}\n")
            report_lines.append(f"- Écart-type: {np.std(all_star_gaps):.2f}\n\n")
        
        report_lines.append("---\n\n")
        report_lines.append("*Rapport généré par GapAnalyzer*\n")
        
        report_content = "".join(report_lines)
        
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            logger.info(f"✅ Rapport sauvegardé: {output_path}")
            return str(output_path)
        
        return report_content


def main():
    """Fonction de test."""
    logger.info("=== Test GapAnalyzer ===")
    
    # Créer des données de test
    np.random.seed(42)
    n_draws = 200
    
    data = {
        'date_de_tirage': pd.date_range('2020-01-01', periods=n_draws, freq='3D'),
        'N1': np.random.randint(1, 51, n_draws),
        'N2': np.random.randint(1, 51, n_draws),
        'N3': np.random.randint(1, 51, n_draws),
        'N4': np.random.randint(1, 51, n_draws),
        'N5': np.random.randint(1, 51, n_draws),
        'E1': np.random.randint(1, 13, n_draws),
        'E2': np.random.randint(1, 13, n_draws)
    }
    
    df = pd.DataFrame(data)
    
    # Créer l'analyseur
    analyzer = GapAnalyzer(max_number=50, max_star=12)
    
    # Charger les données
    analyzer.load_data(df)
    
    # Numéros en retard
    overdue = analyzer.get_overdue_numbers(10)
    logger.info(f"Numéros en retard: {overdue[:5]}")
    
    # Prédiction
    prediction = analyzer.predict()
    logger.info(f"Prédiction: {prediction}")
    
    # Rapport
    report = analyzer.get_gap_report("test_gap_report.md")
    logger.info(f"Rapport: {report[:200]}...")
    
    logger.info("✅ Test terminé")


if __name__ == "__main__":
    main()
