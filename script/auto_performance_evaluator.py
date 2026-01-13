#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module de Performance Auto-Évaluée
Ce module évalue et ajuste automatiquement le système après chaque tirage:
- Scoring automatique de chaque module
- Calcul de métriques détaillées
- Ajustement dynamique des poids
- Détection des modèles défaillants
- Recommandations d'amélioration
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from datetime import datetime
import json
import traceback
from collections import defaultdict

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("AutoPerformanceEvaluator")

# Imports ML
try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report
    )
    import matplotlib.pyplot as plt
    import seaborn as sns
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("⚠️ scikit-learn ou matplotlib non disponible")


class AutoPerformanceEvaluator:
    """
    Classe pour l'évaluation automatique des performances du système
    et l'ajustement dynamique des modèles.
    """
    
    def __init__(
        self,
        output_dir: str = "performance_reports",
        auto_retrain_threshold: float = 0.6,
        disable_threshold: float = 0.3,
        history_window: int = 50
    ):
        """
        Initialise l'évaluateur de performances.
        
        Args:
            output_dir: Répertoire pour les rapports
            auto_retrain_threshold: Seuil pour réentraînement automatique
            disable_threshold: Seuil pour désactivation d'un modèle
            history_window: Nombre de tirages à garder en historique
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.auto_retrain_threshold = auto_retrain_threshold
        self.disable_threshold = disable_threshold
        self.history_window = history_window
        
        # Historique des performances
        self.performance_history = defaultdict(list)
        self.global_history = []
        
        # Statistiques par modèle
        self.model_stats = defaultdict(lambda: {
            'total_predictions': 0,
            'correct_predictions': 0,
            'accuracy_history': [],
            'f1_history': [],
            'status': 'active'  # active, warning, disabled
        })
        
        # Gains simulés
        self.roi_history = []
        
        logger.info(f"✅ AutoPerformanceEvaluator initialisé")
        logger.info(f"   Répertoire: {self.output_dir}")
        logger.info(f"   Seuil réentraînement: {auto_retrain_threshold}")
        logger.info(f"   Seuil désactivation: {disable_threshold}")
    
    def evaluate_prediction(
        self,
        model_name: str,
        predicted_numbers: List[int],
        predicted_stars: List[int],
        actual_numbers: List[int],
        actual_stars: List[int],
        draw_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Évalue une prédiction d'un modèle par rapport au tirage réel.
        
        Args:
            model_name: Nom du modèle évalué
            predicted_numbers: Numéros prédits
            predicted_stars: Étoiles prédites
            actual_numbers: Numéros réels
            actual_stars: Étoiles réelles
            draw_date: Date du tirage (optionnel)
            
        Returns:
            Dict avec les métriques de performance
        """
        # Calculer les correspondances
        correct_numbers = len(set(predicted_numbers) & set(actual_numbers))
        correct_stars = len(set(predicted_stars) & set(actual_stars))
        
        # Calculer le rang EuroMillions
        rank = self._calculate_rank(correct_numbers, correct_stars)
        
        # Calculer les métriques
        total_correct = correct_numbers + correct_stars
        total_predicted = len(predicted_numbers) + len(predicted_stars)
        
        accuracy = total_correct / total_predicted if total_predicted > 0 else 0
        
        # Précision par type
        number_precision = correct_numbers / len(predicted_numbers) if predicted_numbers else 0
        star_precision = correct_stars / len(predicted_stars) if predicted_stars else 0
        
        # Score global (pondéré)
        global_score = (
            0.4 * accuracy +
            0.3 * number_precision +
            0.2 * star_precision +
            0.1 * (1.0 if rank != "Aucun gain" else 0)
        )
        
        # Créer le résultat
        result = {
            'model_name': model_name,
            'draw_date': draw_date or datetime.now().strftime('%Y-%m-%d'),
            'predicted_numbers': predicted_numbers,
            'predicted_stars': predicted_stars,
            'actual_numbers': actual_numbers,
            'actual_stars': actual_stars,
            'correct_numbers': correct_numbers,
            'correct_stars': correct_stars,
            'rank': rank,
            'accuracy': accuracy,
            'number_precision': number_precision,
            'star_precision': star_precision,
            'global_score': global_score,
            'timestamp': datetime.now().isoformat()
        }
        
        # Mettre à jour les statistiques
        self._update_model_stats(model_name, result)
        
        logger.info(f"📊 Évaluation {model_name}:")
        logger.info(f"   Numéros corrects: {correct_numbers}/5")
        logger.info(f"   Étoiles correctes: {correct_stars}/2")
        logger.info(f"   Rang: {rank}")
        logger.info(f"   Score global: {global_score:.3f}")
        
        return result
    
    def _calculate_rank(self, correct_numbers: int, correct_stars: int) -> str:
        """Calcule le rang EuroMillions selon les numéros et étoiles corrects."""
        if correct_numbers == 5 and correct_stars == 2:
            return "Jackpot (1er rang)"
        elif correct_numbers == 5 and correct_stars == 1:
            return "2e rang"
        elif correct_numbers == 5 and correct_stars == 0:
            return "3e rang"
        elif correct_numbers == 4 and correct_stars == 2:
            return "4e rang"
        elif (correct_numbers == 4 and correct_stars == 1) or \
             (correct_numbers == 3 and correct_stars == 2):
            return "5e rang"
        elif (correct_numbers == 4 and correct_stars == 0) or \
             (correct_numbers == 3 and correct_stars == 1) or \
             (correct_numbers == 2 and correct_stars == 2):
            return "6e rang"
        elif (correct_numbers == 3 and correct_stars == 0) or \
             (correct_numbers == 1 and correct_stars == 2) or \
             (correct_numbers == 2 and correct_stars == 1):
            return "7e rang"
        elif (correct_numbers == 2 and correct_stars == 0) or \
             (correct_numbers == 1 and correct_stars == 1) or \
             (correct_numbers == 0 and correct_stars == 2):
            return "8e rang"
        else:
            return "Aucun gain"
    
    def _update_model_stats(self, model_name: str, result: Dict[str, Any]):
        """Met à jour les statistiques d'un modèle."""
        stats = self.model_stats[model_name]
        
        stats['total_predictions'] += 1
        if result['rank'] != "Aucun gain":
            stats['correct_predictions'] += 1
        
        stats['accuracy_history'].append(result['accuracy'])
        
        # Calculer F1-score approximatif
        precision = result['number_precision']
        recall = result['correct_numbers'] / 5  # Sur 5 numéros possibles
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        stats['f1_history'].append(f1)
        
        # Garder seulement les N dernières prédictions
        if len(stats['accuracy_history']) > self.history_window:
            stats['accuracy_history'] = stats['accuracy_history'][-self.history_window:]
            stats['f1_history'] = stats['f1_history'][-self.history_window:]
        
        # Enregistrer dans l'historique global
        self.performance_history[model_name].append(result)
    
    def evaluate_all_models(
        self,
        predictions: Dict[str, Dict[str, List[int]]],
        actual_numbers: List[int],
        actual_stars: List[int],
        draw_date: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Évalue tous les modèles en une fois.
        
        Args:
            predictions: Dict {model_name: {'numbers': [...], 'stars': [...]}}
            actual_numbers: Numéros réels
            actual_stars: Étoiles réelles
            draw_date: Date du tirage
            
        Returns:
            Dict {model_name: résultat_évaluation}
        """
        results = {}
        
        for model_name, pred in predictions.items():
            result = self.evaluate_prediction(
                model_name=model_name,
                predicted_numbers=pred.get('numbers', []),
                predicted_stars=pred.get('stars', []),
                actual_numbers=actual_numbers,
                actual_stars=actual_stars,
                draw_date=draw_date
            )
            results[model_name] = result
        
        # Enregistrer dans l'historique global
        self.global_history.append({
            'draw_date': draw_date or datetime.now().strftime('%Y-%m-%d'),
            'actual_numbers': actual_numbers,
            'actual_stars': actual_stars,
            'results': results
        })
        
        return results
    
    def get_model_scores(self) -> Dict[str, float]:
        """
        Calcule les scores actuels de tous les modèles.
        
        Returns:
            Dict {model_name: score} avec scores entre 0 et 1
        """
        scores = {}
        
        for model_name, stats in self.model_stats.items():
            if not stats['accuracy_history']:
                scores[model_name] = 0.5  # Score neutre si pas d'historique
                continue
            
            # Calculer le score basé sur les performances récentes
            recent_accuracy = np.mean(stats['accuracy_history'][-10:])
            recent_f1 = np.mean(stats['f1_history'][-10:]) if stats['f1_history'] else 0
            
            # Score pondéré
            score = 0.6 * recent_accuracy + 0.4 * recent_f1
            scores[model_name] = score
        
        return scores
    
    def get_recommendations(self) -> List[Dict[str, str]]:
        """
        Génère des recommandations d'amélioration basées sur les performances.
        
        Returns:
            Liste de recommandations
        """
        recommendations = []
        scores = self.get_model_scores()
        
        for model_name, score in scores.items():
            stats = self.model_stats[model_name]
            
            if score < self.disable_threshold:
                recommendations.append({
                    'model': model_name,
                    'priority': 'CRITIQUE',
                    'action': 'DÉSACTIVER',
                    'reason': f'Score très faible ({score:.2f})',
                    'suggestion': 'Désactiver temporairement et analyser les causes'
                })
                stats['status'] = 'disabled'
            
            elif score < self.auto_retrain_threshold:
                recommendations.append({
                    'model': model_name,
                    'priority': 'HAUTE',
                    'action': 'RÉENTRAÎNER',
                    'reason': f'Score sous le seuil ({score:.2f})',
                    'suggestion': 'Réentraîner avec données récentes ou ajuster hyperparamètres'
                })
                stats['status'] = 'warning'
            
            elif score > 0.8:
                recommendations.append({
                    'model': model_name,
                    'priority': 'INFO',
                    'action': 'AUGMENTER_POIDS',
                    'reason': f'Excellentes performances ({score:.2f})',
                    'suggestion': 'Augmenter le poids dans la fusion'
                })
                stats['status'] = 'active'
        
        return recommendations
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """
        Génère un rapport détaillé des performances.
        
        Args:
            output_file: Fichier de sortie (optionnel)
            
        Returns:
            Chemin du rapport généré
        """
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = self.output_dir / f"performance_report_{timestamp}.md"
        else:
            output_file = Path(output_file)
        
        logger.info(f"📝 Génération du rapport: {output_file}")
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("# Rapport de Performance - Système EuroMillions\n\n")
                f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("---\n\n")
                
                # Résumé global
                f.write("## 📊 Résumé Global\n\n")
                scores = self.get_model_scores()
                f.write(f"- **Nombre de modèles**: {len(self.model_stats)}\n")
                f.write(f"- **Score moyen**: {np.mean(list(scores.values())):.3f}\n")
                f.write(f"- **Meilleur modèle**: {max(scores.items(), key=lambda x: x[1])[0]}\n")
                f.write(f"- **Tirages évalués**: {len(self.global_history)}\n\n")
                
                # Performances par modèle
                f.write("## 🎯 Performances par Modèle\n\n")
                for model_name, stats in sorted(self.model_stats.items()):
                    score = scores.get(model_name, 0)
                    status_emoji = "✅" if stats['status'] == 'active' else \
                                   "⚠️" if stats['status'] == 'warning' else "❌"
                    
                    f.write(f"### {status_emoji} {model_name}\n\n")
                    f.write(f"- **Score actuel**: {score:.3f}\n")
                    f.write(f"- **Statut**: {stats['status'].upper()}\n")
                    f.write(f"- **Prédictions totales**: {stats['total_predictions']}\n")
                    f.write(f"- **Prédictions correctes**: {stats['correct_predictions']}\n")
                    
                    if stats['accuracy_history']:
                        f.write(f"- **Précision moyenne**: {np.mean(stats['accuracy_history']):.3f}\n")
                        f.write(f"- **F1-score moyen**: {np.mean(stats['f1_history']):.3f}\n")
                    
                    f.write("\n")
                
                # Recommandations
                f.write("## 💡 Recommandations\n\n")
                recommendations = self.get_recommendations()
                
                if not recommendations:
                    f.write("✅ Aucune action requise. Tous les modèles fonctionnent correctement.\n\n")
                else:
                    for rec in sorted(recommendations, key=lambda x: x['priority']):
                        priority_emoji = "🔴" if rec['priority'] == 'CRITIQUE' else \
                                       "🟠" if rec['priority'] == 'HAUTE' else "🔵"
                        
                        f.write(f"### {priority_emoji} {rec['model']}\n\n")
                        f.write(f"- **Priorité**: {rec['priority']}\n")
                        f.write(f"- **Action**: {rec['action']}\n")
                        f.write(f"- **Raison**: {rec['reason']}\n")
                        f.write(f"- **Suggestion**: {rec['suggestion']}\n\n")
                
                # Historique récent
                f.write("## 📈 Historique Récent (10 derniers tirages)\n\n")
                recent_history = self.global_history[-10:]
                
                for entry in recent_history:
                    f.write(f"### Tirage du {entry['draw_date']}\n\n")
                    f.write(f"**Numéros**: {entry['actual_numbers']}\n")
                    f.write(f"**Étoiles**: {entry['actual_stars']}\n\n")
                    
                    f.write("| Modèle | Numéros corrects | Étoiles correctes | Rang | Score |\n")
                    f.write("|--------|------------------|-------------------|------|-------|\n")
                    
                    for model_name, result in entry['results'].items():
                        f.write(f"| {model_name} | {result['correct_numbers']}/5 | "
                               f"{result['correct_stars']}/2 | {result['rank']} | "
                               f"{result['global_score']:.3f} |\n")
                    
                    f.write("\n")
                
                f.write("---\n\n")
                f.write("*Rapport généré automatiquement par AutoPerformanceEvaluator*\n")
            
            logger.info(f"✅ Rapport généré: {output_file}")
            return str(output_file)
        
        except Exception as e:
            logger.error(f"❌ Erreur génération rapport: {e}")
            logger.debug(traceback.format_exc())
            return ""
    
    def save_history(self, output_file: Optional[str] = None):
        """Sauvegarde l'historique des performances."""
        if output_file is None:
            output_file = self.output_dir / "performance_history.json"
        else:
            output_file = Path(output_file)
        
        try:
            data = {
                'model_stats': dict(self.model_stats),
                'global_history': self.global_history,
                'performance_history': {k: list(v) for k, v in self.performance_history.items()}
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Historique sauvegardé: {output_file}")
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde historique: {e}")
    
    def load_history(self, input_file: str):
        """Charge l'historique des performances."""
        input_path = Path(input_file)
        
        if not input_path.exists():
            logger.warning(f"Fichier historique non trouvé: {input_file}")
            return
        
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.model_stats = defaultdict(
                lambda: {'total_predictions': 0, 'correct_predictions': 0,
                        'accuracy_history': [], 'f1_history': [], 'status': 'active'},
                data.get('model_stats', {})
            )
            self.global_history = data.get('global_history', [])
            self.performance_history = defaultdict(list, data.get('performance_history', {}))
            
            logger.info(f"✅ Historique chargé: {input_file}")
        except Exception as e:
            logger.error(f"❌ Erreur chargement historique: {e}")


def main():
    """Fonction de test."""
    logger.info("=== Test AutoPerformanceEvaluator ===")
    
    evaluator = AutoPerformanceEvaluator(
        output_dir="test_performance_reports",
        auto_retrain_threshold=0.6,
        disable_threshold=0.3
    )
    
    # Simuler des prédictions
    predictions = {
        'model_a': {'numbers': [5, 12, 23, 34, 45], 'stars': [3, 8]},
        'model_b': {'numbers': [7, 14, 21, 28, 35], 'stars': [2, 9]},
        'model_c': {'numbers': [3, 15, 27, 39, 48], 'stars': [1, 10]}
    }
    
    # Tirage réel
    actual_numbers = [5, 12, 19, 34, 42]
    actual_stars = [3, 7]
    
    # Évaluer
    results = evaluator.evaluate_all_models(
        predictions=predictions,
        actual_numbers=actual_numbers,
        actual_stars=actual_stars,
        draw_date="2025-11-18"
    )
    
    # Scores
    scores = evaluator.get_model_scores()
    logger.info(f"Scores: {scores}")
    
    # Recommandations
    recommendations = evaluator.get_recommendations()
    logger.info(f"Recommandations: {len(recommendations)}")
    
    # Générer rapport
    report_path = evaluator.generate_report()
    logger.info(f"Rapport: {report_path}")
    
    logger.info("✅ Test terminé")


if __name__ == "__main__":
    main()
