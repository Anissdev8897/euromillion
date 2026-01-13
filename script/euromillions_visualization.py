#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module de visualisation pour l'analyseur Euromillions.
Ce module implémente une structure minimale pour permettre l'exécution du script principal.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple, Set, Optional, Union, Any
from collections import Counter
from pathlib import Path
import traceback
from scipy import stats # Ajouter l'import pour la régression linéaire

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger("EuromillionsVisualization")

class EuromillionsVisualization:
    """Classe pour la visualisation des données et résultats d'analyse Euromillions."""
    
    def __init__(self, analyzer, output_dir=None):
        """
        Initialise le module de visualisation avec l'analyseur fourni.
        
        Args:
            analyzer: Instance de EuromillionsAdvancedAnalyzer
            output_dir: Répertoire de sortie pour les graphiques (optionnel)
        """
        self.analyzer = analyzer
        
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(analyzer.output_dir) / "visualizations"
        
        # Créer le répertoire de sortie s'il n'existe pas
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)
            logger.info(f"Répertoire créé: {self.output_dir}")
        
        # Définir une palette de couleurs cohérente
        self.colors = {
            'primary': '#1f77b4',  # Bleu
            'secondary': '#ff7f0e',  # Orange
            'tertiary': '#2ca02c',  # Vert
            'quaternary': '#d62728',  # Rouge
            'positive': '#2ca02c',  # Vert pour les valeurs positives
            'negative': '#d62728',  # Rouge pour les valeurs négatives
            'neutral': '#7f7f7f',   # Gris pour les valeurs neutres
            'highlight': '#ff7f0e'  # Orange pour les éléments à mettre en évidence
        }
    
    def plot_frequency_distribution(self):
        """
        Génère un graphique de la distribution des fréquences des numéros et étoiles.
        
        Returns:
            str: Chemin du fichier de graphique généré
        """
        logger.info("Génération du graphique de distribution des fréquences...")
        
        try:
            # Vérifier si les données de fréquence sont disponibles
            if not hasattr(self.analyzer, 'number_frequencies') or not self.analyzer.number_frequencies:
                logger.warning("Données de fréquence non disponibles")
                return ""
            
            # Créer une figure avec deux sous-graphiques
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), tight_layout=True)
            
            # Graphique 1: Fréquences des numéros principaux
            numbers = sorted(self.analyzer.number_frequencies.keys())
            freqs = [self.analyzer.number_frequencies.get(n, 0) for n in numbers]
            
            # Créer un DataFrame pour faciliter le tracé
            freq_df = pd.DataFrame({
                'Numéro': numbers,
                'Fréquence': freqs
            })
            
            # Calculer la fréquence moyenne pour référence
            avg_freq = np.mean(freqs)
            
            # Colorer selon la fréquence (au-dessus de la moyenne = bleu, en-dessous = gris)
            colors = [self.colors['primary'] if f > avg_freq else self.colors['neutral'] for f in freqs]
            
            # Tracer le graphique à barres
            sns.barplot(x='Numéro', y='Fréquence', data=freq_df, ax=ax1, palette=colors)
            
            ax1.set_title("Distribution des fréquences des numéros principaux")
            ax1.set_xlabel("Numéro")
            ax1.set_ylabel("Fréquence")
            
            # Ajouter une ligne horizontale à la fréquence moyenne
            ax1.axhline(y=avg_freq, color='red', linestyle='--', alpha=0.7, label=f'Moyenne ({avg_freq:.2f})')
            ax1.legend()
            
            # Réduire le nombre d'étiquettes sur l'axe x pour éviter l'encombrement
            if len(freq_df) > 20:
                ax1.set_xticks(ax1.get_xticks()[::5])
                ax1.set_xticklabels([freq_df['Numéro'].iloc[i] if i < len(freq_df) else '' for i in range(0, len(freq_df), 5)])
            
            # Graphique 2: Fréquences des étoiles
            if hasattr(self.analyzer, 'star_frequencies') and self.analyzer.star_frequencies:
                stars = sorted(self.analyzer.star_frequencies.keys())
                star_freqs = [self.analyzer.star_frequencies.get(s, 0) for s in stars]
                
                star_freq_df = pd.DataFrame({
                    'Étoile': stars,
                    'Fréquence': star_freqs
                })
                
                avg_star_freq = np.mean(star_freqs)
                colors = [self.colors['secondary'] if f > avg_star_freq else self.colors['neutral'] for f in star_freqs]
                
                sns.barplot(x='Étoile', y='Fréquence', data=star_freq_df, ax=ax2, palette=colors)
                
                ax2.set_title("Distribution des fréquences des étoiles")
                ax2.set_xlabel("Étoile")
                ax2.set_ylabel("Fréquence")
                
                ax2.axhline(y=avg_star_freq, color='red', linestyle='--', alpha=0.7, label=f'Moyenne ({avg_star_freq:.2f})')
                ax2.legend()
            else:
                ax2.text(0.5, 0.5, "Données de fréquence des étoiles non disponibles", 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax2.transAxes)
                ax2.set_title("Distribution des fréquences des étoiles (non disponibles)")
            
            # Sauvegarder le graphique
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = self.output_dir / f"frequency_distribution_{timestamp}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Graphique de distribution des fréquences sauvegardé: {plot_path}")
            
            return str(plot_path)
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération du graphique de distribution des fréquences: {str(e)}")
            logger.debug(traceback.format_exc())
            return ""
    
    def plot_temporal_trends(self):
        """
        Génère un graphique des tendances temporelles des numéros et étoiles.
        
        Returns:
            str: Chemin du fichier de graphique généré
        """
        logger.info("Génération du graphique des tendances temporelles...")
        
        try:
            # Vérifier si les données temporelles sont disponibles
            if not hasattr(self.analyzer, 'temporal_patterns') or not self.analyzer.temporal_patterns:
                logger.warning("Données de tendances temporelles non disponibles")
                return ""
            
            # Créer une figure avec deux sous-graphiques
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), tight_layout=True)
             # Graphique 1: Tendances des numéros principaux
            calculated_trends = {}
            for num in sorted(self.analyzer.temporal_patterns.keys()):
                data_list = self.analyzer.temporal_patterns.get(num, [])
                if len(data_list) > 1:
                    # Extraire les fréquences et créer des indices numériques pour la régression
                    freqs = [freq for period, freq in data_list]
                    indices = np.arange(len(freqs))
                    try:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(indices, freqs)
                        calculated_trends[num] = slope
                    except ValueError:
                        logger.warning(f"Impossible de calculer la tendance pour le numéro {num} (données insuffisantes ou constantes)")
                        calculated_trends[num] = 0 # Tendance neutre en cas d'erreurs
                else:
                    calculated_trends[num] = 0 # Pas assez de données pour une tendance
            
            trend_nums = sorted(calculated_trends.keys())
            trend_vals = [calculated_trends.get(n, 0) for n in trend_nums]
            
            # Créer un DataFrame pour faciliter le tracé
            trends_df = pd.DataFrame({
                "Numéro": trend_nums,
                "Tendance": trend_vals
            })
            
            # Colorer selon la tendance (positif = vert, négatif = rouge)
            colors = [self.colors["positive"] if val > 0 else (self.colors["negative"] if val < 0 else self.colors["neutral"]) for val in trend_vals]
            
            # Tracer le graphique à barres
            sns.barplot(x="Numéro", y="Tendance", data=trends_df, ax=ax1, palette=colors)
            ax1.set_title("Tendances temporelles des numéros principaux")
            ax1.set_xlabel("Numéro")
            ax1.set_ylabel("Tendance (pente)")
            
            # Ajouter une ligne horizontale à y=0
            ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Réduire le nombre d'étiquettes sur l'axe x pour éviter l'encombrement
            if len(trends_df) > 20:
                ax1.set_xticks(ax1.get_xticks()[::5])
                ax1.set_xticklabels([trends_df['Numéro'].iloc[i] if i < len(trends_df) else '' for i in range(0, len(trends_df), 5)])
              # Graphique 2: Tendances des étoiles
            if hasattr(self.analyzer, "star_temporal_patterns") and self.analyzer.star_temporal_patterns:
                calculated_star_trends = {}
                for star in sorted(self.analyzer.star_temporal_patterns.keys()):
                    data_list = self.analyzer.star_temporal_patterns.get(star, [])
                    if len(data_list) > 1:
                        freqs = [freq for period, freq in data_list]
                        indices = np.arange(len(freqs))
                        try:
                            slope, intercept, r_value, p_value, std_err = stats.linregress(indices, freqs)
                            calculated_star_trends[star] = slope
                        except ValueError:
                            logger.warning(f"Impossible de calculer la tendance pour l'étoile {star} (données insuffisantes ou constantes)")
                            calculated_star_trends[star] = 0
                    else:
                        calculated_star_trends[star] = 0
                
                star_trend_nums = sorted(calculated_star_trends.keys())
                star_trend_vals = [calculated_star_trends.get(n, 0) for n in star_trend_nums]
                
                star_trends_df = pd.DataFrame({
                    "Étoile": star_trend_nums,
                    "Tendance": star_trend_vals
                })
                
                colors = [self.colors["positive"] if val > 0 else (self.colors["negative"] if val < 0 else self.colors["neutral"]) for val in star_trend_vals]
                
                sns.barplot(x="Étoile", y="Tendance", data=star_trends_df, ax=ax2, palette=colors) 
                ax2.set_title("Tendances temporelles des étoiles")
                ax2.set_xlabel("Étoile")
                ax2.set_ylabel("Tendance (pente)")
                
                ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            else:
                ax2.text(0.5, 0.5, "Données de tendances des étoiles non disponibles", 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax2.transAxes)
                ax2.set_title("Tendances temporelles des étoiles (non disponibles)")
            
            # Sauvegarder le graphique
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = self.output_dir / f"temporal_trends_{timestamp}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Graphique des tendances temporelles sauvegardé: {plot_path}")
            
            return str(plot_path)
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération du graphique des tendances temporelles: {str(e)}")
            logger.debug(traceback.format_exc())
            return ""
    
    def plot_correlation_heatmap(self):
        """
        Génère une heatmap des corrélations entre numéros et entre étoiles.
        
        Returns:
            str: Chemin du fichier de graphique généré
        """
        logger.info("Génération de la heatmap des corrélations...")
        
        try:
            # Vérifier si les données de corrélation sont disponibles
            if not hasattr(self.analyzer, 'number_correlations') or not self.analyzer.number_correlations:
                logger.warning("Données de corrélation non disponibles")
                return ""
            
            # Créer une figure avec deux sous-graphiques
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), tight_layout=True)
            
            # Graphique 1: Corrélations des numéros principaux
            # Convertir le dictionnaire de corrélations en DataFrame
            numbers = sorted(set([n for pair in self.analyzer.number_correlations.keys() for n in pair]))
            corr_matrix = pd.DataFrame(0, index=numbers, columns=numbers)
            
            for (n1, n2), corr in self.analyzer.number_correlations.items():
                corr_matrix.loc[n1, n2] = corr
                corr_matrix.loc[n2, n1] = corr  # La matrice est symétrique
            
            # Diagonale à 1 (auto-corrélation)
            for n in numbers:
                corr_matrix.loc[n, n] = 1.0
            
            # Tracer la heatmap
            sns.heatmap(corr_matrix, ax=ax1, cmap='coolwarm', vmin=-1, vmax=1, 
                       annot=False, square=True, linewidths=.5)
            
            ax1.set_title("Corrélations entre numéros principaux")
            
            # Graphique 2: Corrélations des étoiles
            if hasattr(self.analyzer, 'star_correlations') and self.analyzer.star_correlations:
                stars = sorted(set([s for pair in self.analyzer.star_correlations.keys() for s in pair]))
                star_corr_matrix = pd.DataFrame(0, index=stars, columns=stars)
                
                for (s1, s2), corr in self.analyzer.star_correlations.items():
                    star_corr_matrix.loc[s1, s2] = corr
                    star_corr_matrix.loc[s2, s1] = corr  # La matrice est symétrique
                
                # Diagonale à 1 (auto-corrélation)
                for s in stars:
                    star_corr_matrix.loc[s, s] = 1.0
                
                sns.heatmap(star_corr_matrix, ax=ax2, cmap='coolwarm', vmin=-1, vmax=1, 
                           annot=True, square=True, linewidths=.5)
                
                ax2.set_title("Corrélations entre étoiles")
            else:
                ax2.text(0.5, 0.5, "Données de corrélation des étoiles non disponibles", 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax2.transAxes)
                ax2.set_title("Corrélations entre étoiles (non disponibles)")
            
            # Sauvegarder le graphique
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = self.output_dir / f"correlation_heatmap_{timestamp}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Heatmap des corrélations sauvegardée: {plot_path}")
            
            return str(plot_path)
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération de la heatmap des corrélations: {str(e)}")
            logger.debug(traceback.format_exc())
            return ""
    
    def plot_historical_draws(self):
        """
        Génère un graphique des tirages historiques.
        
        Returns:
            str: Chemin du fichier de graphique généré
        """
        logger.info("Génération du graphique des tirages historiques...")
        
        try:
            # Vérifier si les données de tirage sont disponibles
            if not hasattr(self.analyzer, 'df') or self.analyzer.df.empty:
                logger.warning("Données de tirage non disponibles")
                return ""
            
            # Créer une figure
            fig, ax = plt.subplots(figsize=(15, 10), tight_layout=True)
            
            # Extraire les données
            df = self.analyzer.df.copy()
            
            # S'assurer que la colonne Date est au format datetime
            if 'Date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['Date']):
                df['Date'] = pd.to_datetime(df['Date'])
            
            # Préparer les données pour le graphique
            dates = df['Date'].tolist() if 'Date' in df.columns else list(range(len(df)))
            
            # Créer un tableau pour stocker les numéros tirés à chaque date
            all_numbers = set()
            for _, row in df.iterrows():
                numbers = row[self.analyzer.number_cols].dropna().astype(int).tolist()
                all_numbers.update(numbers)
            
            all_numbers = sorted(all_numbers)
            
            # Créer une matrice de présence (1 si le numéro est présent, 0 sinon)
            presence_matrix = np.zeros((len(dates), len(all_numbers)))
            
            for i, (_, row) in enumerate(df.iterrows()):
                numbers = row[self.analyzer.number_cols].dropna().astype(int).tolist()
                for num in numbers:
                    j = all_numbers.index(num)
                    presence_matrix[i, j] = 1
            
            # Tracer le graphique de chaleur
            im = ax.imshow(presence_matrix.T, aspect='auto', cmap='Blues', interpolation='nearest')
            
            # Configurer les axes
            ax.set_yticks(range(len(all_numbers)))
            ax.set_yticklabels(all_numbers)
            
            # Formater l'axe des x pour afficher les dates
            if 'Date' in df.columns:
                # Déterminer un intervalle raisonnable pour les ticks de date
                date_range = (dates[-1] - dates[0]).days
                if date_range > 365 * 5:  # Plus de 5 ans
                    interval = 365 * 2  # Tous les 2 ans
                elif date_range > 365 * 2:  # Plus de 2 ans
                    interval = 365  # Tous les ans
                elif date_range > 180:  # Plus de 6 mois
                    interval = 90  # Tous les 3 mois
                else:
                    interval = 30  # Tous les mois
                
                # Créer des ticks à intervalles réguliers
                date_ticks = pd.date_range(start=dates[0], end=dates[-1], freq=f'{interval}D')
                tick_positions = [dates.index(date) if date in dates else -1 for date in date_ticks]
                tick_positions = [pos for pos in tick_positions if pos >= 0]
                
                ax.set_xticks(tick_positions)
                ax.set_xticklabels([date_ticks[i].strftime('%Y-%m-%d') for i in range(len(tick_positions))], 
                                  rotation=45, ha='right')
            else:
                # Si pas de dates, utiliser des indices
                ax.set_xticks(range(0, len(df), max(1, len(df) // 10)))
            
            # Ajouter une barre de couleur
            plt.colorbar(im, ax=ax, label='Présence du numéro')
            
            # Ajouter des titres et des étiquettes
            ax.set_title("Historique des numéros tirés")
            ax.set_xlabel("Date du tirage")
            ax.set_ylabel("Numéro")
            
            # Sauvegarder le graphique
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = self.output_dir / f"historical_draws_{timestamp}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Graphique des tirages historiques sauvegardé: {plot_path}")
            
            return str(plot_path)
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération du graphique des tirages historiques: {str(e)}")
            logger.debug(traceback.format_exc())
            return ""
    
    def plot_gap_analysis(self):
        """
        Génère un graphique d'analyse des écarts entre apparitions des numéros.
        
        Returns:
            str: Chemin du fichier de graphique généré
        """
        logger.info("Génération du graphique d'analyse des écarts...")
        
        try:
            # Vérifier si les données d'écart sont disponibles
            if not hasattr(self.analyzer, 'number_gaps') or not self.analyzer.number_gaps:
                logger.warning("Données d'écart non disponibles")
                return ""
            
            # Créer une figure avec deux sous-graphiques
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), tight_layout=True)
            
            # Graphique 1: Écarts moyens des numéros principaux
            numbers = sorted(self.analyzer.number_gaps.keys())
            avg_gaps = [np.mean(self.analyzer.number_gaps.get(n, [0])) for n in numbers]
            
            # Créer un DataFrame pour faciliter le tracé
            gap_df = pd.DataFrame({
                'Numéro': numbers,
                'Écart moyen': avg_gaps
            })
            
            # Calculer l'écart moyen global pour référence
            global_avg_gap = np.mean(avg_gaps)
            
            # Colorer selon l'écart (au-dessus de la moyenne = rouge, en-dessous = vert)
            colors = [self.colors['negative'] if g > global_avg_gap else self.colors['positive'] for g in avg_gaps]
            
            # Tracer le graphique à barres
            sns.barplot(x='Numéro', y='Écart moyen', data=gap_df, ax=ax1, palette=colors)
            
            ax1.set_title("Écarts moyens entre apparitions des numéros principaux")
            ax1.set_xlabel("Numéro")
            ax1.set_ylabel("Écart moyen (tirages)")
            
            # Ajouter une ligne horizontale à l'écart moyen global
            ax1.axhline(y=global_avg_gap, color='black', linestyle='--', alpha=0.7, label=f'Moyenne globale ({global_avg_gap:.2f})')
            ax1.legend()
            
            # Réduire le nombre d'étiquettes sur l'axe x pour éviter l'encombrement
            if len(gap_df) > 20:
                ax1.set_xticks(ax1.get_xticks()[::5])
                ax1.set_xticklabels([gap_df['Numéro'].iloc[i] if i < len(gap_df) else '' for i in range(0, len(gap_df), 5)])
            
            # Graphique 2: Écarts moyens des étoiles
            if hasattr(self.analyzer, 'star_gaps') and self.analyzer.star_gaps:
                stars = sorted(self.analyzer.star_gaps.keys())
                star_avg_gaps = [np.mean(self.analyzer.star_gaps.get(s, [0])) for s in stars]
                
                star_gap_df = pd.DataFrame({
                    'Étoile': stars,
                    'Écart moyen': star_avg_gaps
                })
                
                star_global_avg_gap = np.mean(star_avg_gaps)
                colors = [self.colors['negative'] if g > star_global_avg_gap else self.colors['positive'] for g in star_avg_gaps]
                
                sns.barplot(x='Étoile', y='Écart moyen', data=star_gap_df, ax=ax2, palette=colors)
                
                ax2.set_title("Écarts moyens entre apparitions des étoiles")
                ax2.set_xlabel("Étoile")
                ax2.set_ylabel("Écart moyen (tirages)")
                
                ax2.axhline(y=star_global_avg_gap, color='black', linestyle='--', alpha=0.7, label=f'Moyenne globale ({star_global_avg_gap:.2f})')
                ax2.legend()
            else:
                ax2.text(0.5, 0.5, "Données d'écart des étoiles non disponibles", 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax2.transAxes)
                ax2.set_title("Écarts moyens entre apparitions des étoiles (non disponibles)")
            
            # Sauvegarder le graphique
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = self.output_dir / f"gap_analysis_{timestamp}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Graphique d'analyse des écarts sauvegardé: {plot_path}")
            
            return str(plot_path)
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération du graphique d'analyse des écarts: {str(e)}")
            logger.debug(traceback.format_exc())
            return ""
    
    def plot_parity_distribution(self):
        """
        Génère un graphique de la distribution de parité des numéros.
        
        Returns:
            str: Chemin du fichier de graphique généré
        """
        logger.info("Génération du graphique de distribution de parité...")
        
        try:
            # Vérifier si les données de parité sont disponibles
            if not hasattr(self.analyzer, 'parity_stats') or not self.analyzer.parity_stats:
                logger.warning("Données de parité non disponibles")
                return ""
            
            # Créer une figure
            fig, ax = plt.subplots(figsize=(12, 8), tight_layout=True)
            
            # Préparer les données
            parity_types = list(self.analyzer.parity_stats.keys())
            percentages = [self.analyzer.parity_stats[p]['percentage'] for p in parity_types]
            
            # Trier par pourcentage décroissant
            sorted_indices = np.argsort(percentages)[::-1]
            parity_types = [parity_types[i] for i in sorted_indices]
            percentages = [percentages[i] for i in sorted_indices]
            
            # Créer un DataFrame pour faciliter le tracé
            parity_df = pd.DataFrame({
                'Configuration': parity_types,
                'Pourcentage': percentages
            })
            
            # Tracer le graphique à barres
            bars = sns.barplot(x='Configuration', y='Pourcentage', data=parity_df, ax=ax, palette='viridis')
            
            # Ajouter les valeurs sur les barres
            for i, p in enumerate(percentages):
                ax.text(i, p + 0.5, f'{p:.1f}%', ha='center', va='bottom')
            
            ax.set_title("Distribution des configurations de parité")
            ax.set_xlabel("Configuration (Pairs-Impairs)")
            ax.set_ylabel("Pourcentage des tirages (%)")
            
            # Ajuster la rotation des étiquettes si nécessaire
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            # Sauvegarder le graphique
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = self.output_dir / f"parity_distribution_{timestamp}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Graphique de distribution de parité sauvegardé: {plot_path}")
            
            return str(plot_path)
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération du graphique de distribution de parité: {str(e)}")
            logger.debug(traceback.format_exc())
            return ""
    
    def plot_sum_distribution(self):
        """
        Génère un graphique de la distribution des sommes des numéros.
        
        Returns:
            str: Chemin du fichier de graphique généré
        """
        logger.info("Génération du graphique de distribution des sommes...")
        
        try:
            # Vérifier si les données de somme sont disponibles
            if not hasattr(self.analyzer, 'sum_ranges') or not self.analyzer.sum_ranges:
                logger.warning("Données de somme non disponibles")
                return ""
            
            # Créer une figure avec deux sous-graphiques
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), tight_layout=True)
            
            # Graphique 1: Distribution des plages de sommes
            sum_ranges = list(self.analyzer.sum_ranges.keys())
            percentages = [self.analyzer.sum_ranges[r]['percentage'] for r in sum_ranges]
            
            # Créer un DataFrame pour faciliter le tracé
            sum_df = pd.DataFrame({
                'Plage': sum_ranges,
                'Pourcentage': percentages
            })
            
            # Tracer le graphique à barres
            bars = sns.barplot(x='Plage', y='Pourcentage', data=sum_df, ax=ax1, palette='viridis')
            
            # Ajouter les valeurs sur les barres
            for i, p in enumerate(percentages):
                ax1.text(i, p + 0.5, f'{p:.1f}%', ha='center', va='bottom')
            
            ax1.set_title("Distribution des plages de sommes des numéros")
            ax1.set_xlabel("Plage de sommes")
            ax1.set_ylabel("Pourcentage des tirages (%)")
            
            # Graphique 2: Sommes les plus fréquentes
            if hasattr(self.analyzer, 'most_common_sums') and self.analyzer.most_common_sums:
                sums = [s for s, _ in self.analyzer.most_common_sums]
                counts = [c for _, c in self.analyzer.most_common_sums]
                
                # Calculer les pourcentages
                total_draws = sum(counts)
                percentages = [(c / total_draws) * 100 for c in counts]
                
                common_sum_df = pd.DataFrame({
                    'Somme': sums,
                    'Pourcentage': percentages
                })
                
                sns.barplot(x='Somme', y='Pourcentage', data=common_sum_df, ax=ax2, palette='viridis')
                
                # Ajouter les valeurs sur les barres
                for i, p in enumerate(percentages):
                    ax2.text(i, p + 0.1, f'{p:.1f}%', ha='center', va='bottom')
                
                ax2.set_title("Sommes les plus fréquentes")
                ax2.set_xlabel("Somme des numéros")
                ax2.set_ylabel("Pourcentage des tirages (%)")
                
                # Ajouter des statistiques dans un encadré
                if hasattr(self.analyzer, 'sum_stats'):
                    stats_text = (
                        f"Min: {self.analyzer.sum_stats['min']}\n"
                        f"Max: {self.analyzer.sum_stats['max']}\n"
                        f"Moyenne: {self.analyzer.sum_stats['avg']:.2f}\n"
                        f"Médiane: {self.analyzer.sum_stats['median']:.2f}\n"
                        f"Écart-type: {self.analyzer.sum_stats['std']:.2f}"
                    )
                    ax2.text(0.02, 0.95, stats_text, transform=ax2.transAxes, 
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                ax2.text(0.5, 0.5, "Données des sommes les plus fréquentes non disponibles", 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax2.transAxes)
                ax2.set_title("Sommes les plus fréquentes (non disponibles)")
            
            # Sauvegarder le graphique
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = self.output_dir / f"sum_distribution_{timestamp}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Graphique de distribution des sommes sauvegardé: {plot_path}")
            
            return str(plot_path)
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération du graphique de distribution des sommes: {str(e)}")
            logger.debug(traceback.format_exc())
            return ""
    
    def plot_sequence_distribution(self):
        """
        Génère un graphique de la distribution des séquences de numéros consécutifs.
        
        Returns:
            str: Chemin du fichier de graphique généré
        """
        logger.info("Génération du graphique de distribution des séquences...")
        
        try:
            # Vérifier si les données de séquence sont disponibles
            if not hasattr(self.analyzer, 'sequence_stats') or not self.analyzer.sequence_stats:
                logger.warning("Données de séquence non disponibles")
                return ""
            
            # Créer une figure
            fig, ax = plt.subplots(figsize=(12, 8), tight_layout=True)
            
            # Préparer les données
            seq_counts = sorted(self.analyzer.sequence_stats.keys())
            percentages = [self.analyzer.sequence_stats[c]['percentage'] for c in seq_counts]
            
            # Créer un DataFrame pour faciliter le tracé
            seq_df = pd.DataFrame({
                'Nombre de séquences': seq_counts,
                'Pourcentage': percentages
            })
            
            # Tracer le graphique à barres
            bars = sns.barplot(x='Nombre de séquences', y='Pourcentage', data=seq_df, ax=ax, palette='viridis')
            
            # Ajouter les valeurs sur les barres
            for i, p in enumerate(percentages):
                ax.text(i, p + 0.5, f'{p:.1f}%', ha='center', va='bottom')
            
            ax.set_title("Distribution du nombre de séquences de numéros consécutifs")
            ax.set_xlabel("Nombre de séquences")
            ax.set_ylabel("Pourcentage des tirages (%)")
            
            # Sauvegarder le graphique
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = self.output_dir / f"sequence_distribution_{timestamp}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Graphique de distribution des séquences sauvegardé: {plot_path}")
            
            return str(plot_path)
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération du graphique de distribution des séquences: {str(e)}")
            logger.debug(traceback.format_exc())
            return ""
    
    def plot_cluster_analysis(self):
        """
        Génère un graphique d'analyse des clusters de numéros.
        
        Returns:
            str: Chemin du fichier de graphique généré
        """
        logger.info("Génération du graphique d'analyse des clusters...")
        
        try:
            # Vérifier si les données de cluster sont disponibles
            if not hasattr(self.analyzer, 'number_clusters') or not self.analyzer.number_clusters:
                logger.warning("Données de cluster non disponibles")
                return ""
            
            # Déterminer le nombre de clusters
            n_clusters = len(self.analyzer.number_clusters)
            
            # Créer une figure avec un sous-graphique par cluster
            fig, axes = plt.subplots(n_clusters, 1, figsize=(14, 5 * n_clusters), tight_layout=True)
            
            # S'assurer que axes est toujours une liste, même avec un seul cluster
            if n_clusters == 1:
                axes = [axes]
            
            # Pour chaque cluster
            for i, (cluster_id, stats) in enumerate(sorted(self.analyzer.number_clusters.items())):
                ax = axes[i]
                
                # Extraire les numéros les plus fréquents dans ce cluster
                top_numbers = stats['top_numbers']
                numbers = [n for n, _ in top_numbers]
                freqs = [f for _, f in top_numbers]
                
                # Créer un DataFrame pour faciliter le tracé
                cluster_df = pd.DataFrame({
                    'Numéro': numbers,
                    'Fréquence': freqs
                })
                
                # Tracer le graphique à barres
                sns.barplot(x='Numéro', y='Fréquence', data=cluster_df, ax=ax, palette='viridis')
                
                ax.set_title(f"Cluster {cluster_id+1}: {stats['size']} tirages, Fréquence moyenne: {stats['avg_frequency']:.3f}")
                ax.set_xlabel("Numéro")
                ax.set_ylabel("Fréquence dans le cluster")
                
                # Ajouter les valeurs sur les barres
                for j, f in enumerate(freqs):
                    ax.text(j, f + 0.01, f'{f:.2f}', ha='center', va='bottom')
            
            # Sauvegarder le graphique
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = self.output_dir / f"cluster_analysis_{timestamp}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Graphique d'analyse des clusters sauvegardé: {plot_path}")
            
            return str(plot_path)
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération du graphique d'analyse des clusters: {str(e)}")
            logger.debug(traceback.format_exc())
            return ""
    
    def plot_all(self):
        """
        Génère tous les graphiques disponibles.
        
        Returns:
            List[str]: Liste des chemins des fichiers de graphiques générés
        """
        logger.info("Génération de tous les graphiques...")
        
        plots = []
        
        # Fréquences
        freq_plot = self.plot_frequency_distribution()
        if freq_plot:
            plots.append(freq_plot)
        
        # Tendances temporelles
        trend_plot = self.plot_temporal_trends()
        if trend_plot:
            plots.append(trend_plot)
        
        # Corrélations
        corr_plot = self.plot_correlation_heatmap()
        if corr_plot:
            plots.append(corr_plot)
        
        # Historique des tirages
        hist_plot = self.plot_historical_draws()
        if hist_plot:
            plots.append(hist_plot)
        
        # Analyse des écarts
        gap_plot = self.plot_gap_analysis()
        if gap_plot:
            plots.append(gap_plot)
        
        # Distribution de parité
        parity_plot = self.plot_parity_distribution()
        if parity_plot:
            plots.append(parity_plot)
        
        # Distribution des sommes
        sum_plot = self.plot_sum_distribution()
        if sum_plot:
            plots.append(sum_plot)
        
        # Distribution des séquences
        seq_plot = self.plot_sequence_distribution()
        if seq_plot:
            plots.append(seq_plot)
        
        # Analyse des clusters
        cluster_plot = self.plot_cluster_analysis()
        if cluster_plot:
            plots.append(cluster_plot)
        
        logger.info(f"Génération terminée: {len(plots)} graphiques générés")
        
        return plots
