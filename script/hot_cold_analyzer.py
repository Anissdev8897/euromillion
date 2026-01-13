#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module d'Analyse Hot/Cold/Warm
Classifie les numéros et étoiles selon leur "température":
- Hot: Sortis fréquemment récemment
- Cold: Absents depuis longtemps
- Warm: Fréquence moyenne
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
from collections import Counter, defaultdict

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("HotColdAnalyzer")


class HotColdAnalyzer:
    """
    Classe pour l'analyse Hot/Cold/Warm des numéros et étoiles.
    """
    
    def __init__(
        self,
        max_number: int = 50,
        max_star: int = 12,
        recent_window: int = 20,
        hot_threshold_sigma: float = 1.0,
        cold_threshold_sigma: float = 1.0
    ):
        """
        Initialise l'analyseur Hot/Cold.
        
        Args:
            max_number: Numéro maximum
            max_star: Étoile maximum
            recent_window: Nombre de tirages récents à considérer
            hot_threshold_sigma: Seuil en écarts-types pour "hot"
            cold_threshold_sigma: Seuil en écarts-types pour "cold"
        """
        self.max_number = max_number
        self.max_star = max_star
        self.recent_window = recent_window
        self.hot_threshold_sigma = hot_threshold_sigma
        self.cold_threshold_sigma = cold_threshold_sigma
        
        # Classifications
        self.number_classification = {}
        self.star_classification = {}
        
        # Fréquences
        self.number_frequencies = {}
        self.star_frequencies = {}
        
        # Historique
        self.draw_history = []
        
        logger.info(f"✅ HotColdAnalyzer initialisé")
        logger.info(f"   Fenêtre récente: {recent_window} tirages")
        logger.info(f"   Seuils: ±{hot_threshold_sigma}σ")
    
    def load_data(
        self,
        df: pd.DataFrame,
        date_col: str = 'date_de_tirage',
        number_cols: List[str] = None,
        star_cols: List[str] = None
    ):
        """
        Charge les données et effectue la classification.
        
        Args:
            df: DataFrame avec l'historique
            date_col: Colonne de date
            number_cols: Colonnes des numéros
            star_cols: Colonnes des étoiles
        """
        if number_cols is None:
            number_cols = ['N1', 'N2', 'N3', 'N4', 'N5']
        if star_cols is None:
            star_cols = ['E1', 'E2']
        
        logger.info(f"📊 Chargement de {len(df)} tirages...")
        
        # Trier par date (plus récent en premier pour analyse)
        df = df.sort_values(date_col, ascending=False).reset_index(drop=True)
        
        # Extraire les tirages récents
        recent_df = df.head(self.recent_window)
        
        # Compter les fréquences dans la fenêtre récente
        recent_numbers = []
        recent_stars = []
        
        for idx, row in recent_df.iterrows():
            numbers = [row[col] for col in number_cols if col in row]
            stars = [row[col] for col in star_cols if col in row]
            
            recent_numbers.extend(numbers)
            recent_stars.extend(stars)
            
            self.draw_history.append({
                'date': row[date_col] if date_col in row else None,
                'numbers': numbers,
                'stars': stars
            })
        
        # Calculer les fréquences
        number_counter = Counter(recent_numbers)
        star_counter = Counter(recent_stars)
        
        # Normaliser les fréquences
        for num in range(1, self.max_number + 1):
            self.number_frequencies[num] = number_counter.get(num, 0)
        
        for star in range(1, self.max_star + 1):
            self.star_frequencies[star] = star_counter.get(star, 0)
        
        # Classifier
        self._classify_numbers()
        self._classify_stars()
        
        logger.info(f"✅ Classification terminée")
        self._log_classification_summary()
    
    def _classify_numbers(self):
        """Classifie les numéros en hot/cold/warm."""
        frequencies = list(self.number_frequencies.values())
        
        if not frequencies:
            return
        
        mean_freq = np.mean(frequencies)
        std_freq = np.std(frequencies)
        
        if std_freq == 0:
            std_freq = 1  # Éviter division par zéro
        
        hot_threshold = mean_freq + self.hot_threshold_sigma * std_freq
        cold_threshold = mean_freq - self.cold_threshold_sigma * std_freq
        
        for num, freq in self.number_frequencies.items():
            if freq >= hot_threshold:
                category = "hot"
                score = 1.0 + (freq - hot_threshold) / (std_freq + 1)
            elif freq <= cold_threshold:
                category = "cold"
                score = 0.3 - (cold_threshold - freq) / (std_freq + 1)
                score = max(0.1, score)
            else:
                category = "warm"
                # Score proportionnel à la position entre cold et hot
                score = 0.5 + 0.3 * (freq - mean_freq) / (std_freq + 1)
            
            self.number_classification[num] = {
                'category': category,
                'frequency': freq,
                'score': score,
                'z_score': (freq - mean_freq) / std_freq
            }
    
    def _classify_stars(self):
        """Classifie les étoiles en hot/cold/warm."""
        frequencies = list(self.star_frequencies.values())
        
        if not frequencies:
            return
        
        mean_freq = np.mean(frequencies)
        std_freq = np.std(frequencies)
        
        if std_freq == 0:
            std_freq = 1
        
        hot_threshold = mean_freq + self.hot_threshold_sigma * std_freq
        cold_threshold = mean_freq - self.cold_threshold_sigma * std_freq
        
        for star, freq in self.star_frequencies.items():
            if freq >= hot_threshold:
                category = "hot"
                score = 1.0 + (freq - hot_threshold) / (std_freq + 1)
            elif freq <= cold_threshold:
                category = "cold"
                score = 0.3 - (cold_threshold - freq) / (std_freq + 1)
                score = max(0.1, score)
            else:
                category = "warm"
                score = 0.5 + 0.3 * (freq - mean_freq) / (std_freq + 1)
            
            self.star_classification[star] = {
                'category': category,
                'frequency': freq,
                'score': score,
                'z_score': (freq - mean_freq) / std_freq
            }
    
    def _log_classification_summary(self):
        """Affiche un résumé de la classification."""
        # Compter par catégorie
        num_hot = sum(1 for c in self.number_classification.values() if c['category'] == 'hot')
        num_warm = sum(1 for c in self.number_classification.values() if c['category'] == 'warm')
        num_cold = sum(1 for c in self.number_classification.values() if c['category'] == 'cold')
        
        star_hot = sum(1 for c in self.star_classification.values() if c['category'] == 'hot')
        star_warm = sum(1 for c in self.star_classification.values() if c['category'] == 'warm')
        star_cold = sum(1 for c in self.star_classification.values() if c['category'] == 'cold')
        
        logger.info(f"📊 Classification des numéros:")
        logger.info(f"   🔥 Hot: {num_hot}")
        logger.info(f"   🌡️ Warm: {num_warm}")
        logger.info(f"   ❄️ Cold: {num_cold}")
        
        logger.info(f"📊 Classification des étoiles:")
        logger.info(f"   🔥 Hot: {star_hot}")
        logger.info(f"   🌡️ Warm: {star_warm}")
        logger.info(f"   ❄️ Cold: {star_cold}")
    
    def get_hot_numbers(self, limit: int = 10) -> List[Tuple[int, Dict]]:
        """Retourne les numéros hot triés par score."""
        hot = [(num, data) for num, data in self.number_classification.items()
               if data['category'] == 'hot']
        hot.sort(key=lambda x: x[1]['score'], reverse=True)
        return hot[:limit]
    
    def get_cold_numbers(self, limit: int = 10) -> List[Tuple[int, Dict]]:
        """Retourne les numéros cold triés par score (les plus froids)."""
        cold = [(num, data) for num, data in self.number_classification.items()
                if data['category'] == 'cold']
        cold.sort(key=lambda x: x[1]['score'])
        return cold[:limit]
    
    def get_warm_numbers(self, limit: int = 10) -> List[Tuple[int, Dict]]:
        """Retourne les numéros warm."""
        warm = [(num, data) for num, data in self.number_classification.items()
                if data['category'] == 'warm']
        warm.sort(key=lambda x: x[1]['score'], reverse=True)
        return warm[:limit]
    
    def get_hot_stars(self, limit: int = 5) -> List[Tuple[int, Dict]]:
        """Retourne les étoiles hot."""
        hot = [(star, data) for star, data in self.star_classification.items()
               if data['category'] == 'hot']
        hot.sort(key=lambda x: x[1]['score'], reverse=True)
        return hot[:limit]
    
    def get_cold_stars(self, limit: int = 5) -> List[Tuple[int, Dict]]:
        """Retourne les étoiles cold."""
        cold = [(star, data) for star, data in self.star_classification.items()
                if data['category'] == 'cold']
        cold.sort(key=lambda x: x[1]['score'])
        return cold[:limit]
    
    def get_number_scores(self) -> Dict[int, float]:
        """Retourne les scores de tous les numéros."""
        return {num: data['score'] for num, data in self.number_classification.items()}
    
    def get_star_scores(self) -> Dict[int, float]:
        """Retourne les scores de toutes les étoiles."""
        return {star: data['score'] for star, data in self.star_classification.items()}
    
    def predict(
        self,
        strategy: str = "balanced",
        num_numbers: int = 5,
        num_stars: int = 2
    ) -> Dict[str, List[int]]:
        """
        Prédit les numéros et étoiles selon une stratégie.
        
        Args:
            strategy: "hot" (favoriser hot), "cold" (favoriser cold), 
                     "balanced" (mix), "contrarian" (inverser tendance)
            num_numbers: Nombre de numéros
            num_stars: Nombre d'étoiles
            
        Returns:
            Dict avec 'numbers' et 'stars'
        """
        if strategy == "hot":
            # Favoriser les numéros hot
            numbers = [num for num, _ in self.get_hot_numbers(num_numbers)]
            stars = [star for star, _ in self.get_hot_stars(num_stars)]
        
        elif strategy == "cold":
            # Favoriser les numéros cold (théorie du retour à la moyenne)
            numbers = [num for num, _ in self.get_cold_numbers(num_numbers)]
            stars = [star for star, _ in self.get_cold_stars(num_stars)]
        
        elif strategy == "balanced":
            # Mix de hot et warm
            hot_nums = self.get_hot_numbers(3)
            warm_nums = self.get_warm_numbers(2)
            numbers = [num for num, _ in (hot_nums + warm_nums)][:num_numbers]
            
            hot_st = self.get_hot_stars(1)
            warm_st = [(s, d) for s, d in self.star_classification.items() 
                       if d['category'] == 'warm'][:1]
            stars = [star for star, _ in (hot_st + warm_st)][:num_stars]
        
        elif strategy == "contrarian":
            # Stratégie contrarian: cold + quelques warm
            cold_nums = self.get_cold_numbers(3)
            warm_nums = self.get_warm_numbers(2)
            numbers = [num for num, _ in (cold_nums + warm_nums)][:num_numbers]
            
            cold_st = self.get_cold_stars(1)
            warm_st = [(s, d) for s, d in self.star_classification.items()
                       if d['category'] == 'warm'][:1]
            stars = [star for star, _ in (cold_st + warm_st)][:num_stars]
        
        else:
            logger.warning(f"Stratégie inconnue: {strategy}, utilisation de 'balanced'")
            return self.predict("balanced", num_numbers, num_stars)
        
        # S'assurer qu'on a le bon nombre
        if len(numbers) < num_numbers:
            # Compléter avec des warm
            all_nums = sorted(self.number_classification.keys(),
                            key=lambda x: self.number_classification[x]['score'],
                            reverse=True)
            for num in all_nums:
                if num not in numbers:
                    numbers.append(num)
                    if len(numbers) >= num_numbers:
                        break
        
        if len(stars) < num_stars:
            all_stars = sorted(self.star_classification.keys(),
                             key=lambda x: self.star_classification[x]['score'],
                             reverse=True)
            for star in all_stars:
                if star not in stars:
                    stars.append(star)
                    if len(stars) >= num_stars:
                        break
        
        prediction = {
            'numbers': sorted(numbers[:num_numbers]),
            'stars': sorted(stars[:num_stars])
        }
        
        logger.info(f"🎯 Prédiction Hot/Cold (stratégie: {strategy}):")
        logger.info(f"   Numéros: {prediction['numbers']}")
        logger.info(f"   Étoiles: {prediction['stars']}")
        
        return prediction
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """Génère un rapport Hot/Cold."""
        lines = []
        lines.append("# Rapport d'Analyse Hot/Cold/Warm - EuroMillions\n\n")
        lines.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        lines.append(f"**Fenêtre d'analyse**: {self.recent_window} derniers tirages\n\n")
        lines.append("---\n\n")
        
        # Numéros Hot
        lines.append("## 🔥 Numéros HOT\n\n")
        hot_nums = self.get_hot_numbers(15)
        lines.append("| Rang | Numéro | Fréquence | Score | Z-Score |\n")
        lines.append("|------|--------|-----------|-------|----------|\n")
        for rank, (num, data) in enumerate(hot_nums, 1):
            lines.append(f"| {rank} | **{num}** | {data['frequency']} | "
                        f"{data['score']:.2f} | {data['z_score']:.2f} |\n")
        lines.append("\n")
        
        # Numéros Cold
        lines.append("## ❄️ Numéros COLD\n\n")
        cold_nums = self.get_cold_numbers(15)
        lines.append("| Rang | Numéro | Fréquence | Score | Z-Score |\n")
        lines.append("|------|--------|-----------|-------|----------|\n")
        for rank, (num, data) in enumerate(cold_nums, 1):
            lines.append(f"| {rank} | **{num}** | {data['frequency']} | "
                        f"{data['score']:.2f} | {data['z_score']:.2f} |\n")
        lines.append("\n")
        
        # Étoiles Hot
        lines.append("## 🔥 Étoiles HOT\n\n")
        hot_stars = self.get_hot_stars(5)
        lines.append("| Rang | Étoile | Fréquence | Score | Z-Score |\n")
        lines.append("|------|--------|-----------|-------|----------|\n")
        for rank, (star, data) in enumerate(hot_stars, 1):
            lines.append(f"| {rank} | **{star}** | {data['frequency']} | "
                        f"{data['score']:.2f} | {data['z_score']:.2f} |\n")
        lines.append("\n")
        
        # Étoiles Cold
        lines.append("## ❄️ Étoiles COLD\n\n")
        cold_stars = self.get_cold_stars(5)
        lines.append("| Rang | Étoile | Fréquence | Score | Z-Score |\n")
        lines.append("|------|--------|-----------|-------|----------|\n")
        for rank, (star, data) in enumerate(cold_stars, 1):
            lines.append(f"| {rank} | **{star}** | {data['frequency']} | "
                        f"{data['score']:.2f} | {data['z_score']:.2f} |\n")
        lines.append("\n")
        
        lines.append("---\n\n")
        lines.append("*Rapport généré par HotColdAnalyzer*\n")
        
        content = "".join(lines)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"✅ Rapport sauvegardé: {output_file}")
            return str(output_file)
        
        return content


def main():
    """Test du module."""
    logger.info("=== Test HotColdAnalyzer ===")
    
    # Données de test
    np.random.seed(42)
    n_draws = 100
    
    # Simuler des numéros avec certains plus fréquents (hot)
    hot_numbers = [7, 14, 23, 31, 42]
    cold_numbers = [3, 11, 19, 27, 48]
    
    data = {
        'date_de_tirage': pd.date_range('2024-01-01', periods=n_draws, freq='3D'),
        'N1': [], 'N2': [], 'N3': [], 'N4': [], 'N5': [],
        'E1': [], 'E2': []
    }
    
    for i in range(n_draws):
        # Biaiser vers hot numbers dans les tirages récents
        if i < 20:
            nums = np.random.choice(hot_numbers + list(range(1, 51)), 5, replace=False)
        else:
            nums = np.random.randint(1, 51, 5)
        
        for j, col in enumerate(['N1', 'N2', 'N3', 'N4', 'N5']):
            data[col].append(nums[j] if j < len(nums) else np.random.randint(1, 51))
        
        data['E1'].append(np.random.randint(1, 13))
        data['E2'].append(np.random.randint(1, 13))
    
    df = pd.DataFrame(data)
    
    # Analyser
    analyzer = HotColdAnalyzer(recent_window=20)
    analyzer.load_data(df)
    
    # Prédictions
    for strategy in ['hot', 'cold', 'balanced', 'contrarian']:
        pred = analyzer.predict(strategy=strategy)
        logger.info(f"Stratégie {strategy}: {pred}")
    
    # Rapport
    report = analyzer.generate_report("test_hot_cold_report.md")
    
    logger.info("✅ Test terminé")


if __name__ == "__main__":
    main()
