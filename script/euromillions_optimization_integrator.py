#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module d'intégration du cycle lunaire pour l'analyseur Euromillions
Ce module calcule les phases lunaires pour chaque date de tirage et les intègre
comme features supplémentaires dans les modèles de prédiction.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
import ephem  # Pour calculer les phases lunaires
from typing import Dict, List, Tuple
from pathlib import Path

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("EuromillionsLunarCycle")

class LunarCycleAnalyzer:
    """Classe pour analyser l'influence du cycle lunaire sur les tirages Euromillions."""
    
    def __init__(self, output_dir=None):
        """
        Initialise l'analyseur de cycle lunaire.
        
        Args:
            output_dir: Répertoire de sortie pour les résultats (optionnel)
        """
        # Répertoire de sortie
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path("resultats_lunar_cycle")
        
        # Créer le répertoire de sortie s'il n'existe pas
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)
            logger.info(f"Répertoire d'analyse lunaire créé: {self.output_dir}")
        
        # Phases lunaires nommées
        self.lunar_phases = {
            0: "Nouvelle Lune",
            1: "Premier Croissant",
            2: "Premier Quartier",
            3: "Gibbeuse Croissante",
            4: "Pleine Lune",
            5: "Gibbeuse Décroissante",
            6: "Dernier Quartier",
            7: "Dernier Croissant"
        }
        
        # Date du premier tirage EuroMillions (13 février 2004)
        self.first_draw_date = datetime(2004, 2, 13)

    def calculate_moon_phase(self, date: datetime) -> Tuple[float, int, str]:
        """
        Calcule la phase lunaire pour une date donnée.
        
        Args:
            date: Date pour laquelle calculer la phase lunaire
            
        Returns:
            Tuple[float, int, str]: (illumination en pourcentage, phase numérique (0-7), nom de la phase)
        """
        try:
            # Créer un objet Moon d'ephem
            moon = ephem.Moon()
            # Convertir la date au format ephem
            ephem_date = ephem.Date(date)
            # Calculer la position de la lune à cette date
            moon.compute(ephem_date)
            
            # Obtenir l'illumination (0-1)
            # La propriété 'phase' de ephem.Moon donne le pourcentage d'illumination (0-100)
            illumination = moon.phase / 100.0
            
            # Calculer la phase lunaire (0-7) en utilisant l'âge de la lune
            # Un cycle lunaire complet dure environ 29.53 jours
            # Nous pouvons calculer l'âge de la lune à partir de la dernière nouvelle lune
            
            # Trouver la date de la dernière nouvelle lune
            previous_new_moon = ephem.previous_new_moon(ephem_date)
            # Calculer l'âge de la lune en jours
            moon_age = (ephem_date - previous_new_moon) * 24 * 60 * 60 / 86400.0  # Convertir en jours
            
            # Convertir l'âge en phase (0-7)
            # 0: Nouvelle Lune (0-3.69 jours)
            # 1: Premier Croissant (3.69-7.38 jours)
            # 2: Premier Quartier (7.38-11.07 jours)
            # 3: Gibbeuse Croissante (11.07-14.76 jours)
            # 4: Pleine Lune (14.76-18.45 jours)
            # 5: Gibbeuse Décroissante (18.45-22.14 jours)
            # 6: Dernier Quartier (22.14-25.83 jours)
            # 7: Dernier Croissant (25.83-29.53 jours)
            phase_numeric = int((moon_age / 29.53) * 8) % 8
            
            # Obtenir le nom de la phase
            phase_name = self.lunar_phases[phase_numeric]
            
            return (illumination, phase_numeric, phase_name)
        except Exception as e:
            logger.error(f"Erreur lors du calcul de la phase lunaire: {str(e)}")
            return (0.0, 0, "Inconnue")

    def enrich_dataframe_with_lunar_data(self, df: pd.DataFrame, date_column: str = 'Date') -> pd.DataFrame:
        """
        Enrichit un DataFrame avec des données lunaires pour chaque date.
        
        Args:
            df: DataFrame contenant les tirages
            date_column: Nom de la colonne contenant les dates
            
        Returns:
            pd.DataFrame: DataFrame enrichi avec les données lunaires
        """
        logger.info("Enrichissement du DataFrame avec les données lunaires...")
        
        try:
            # Vérifier que la colonne de date existe
            if date_column not in df.columns:
                logger.error(f"Colonne de date '{date_column}' non trouvée dans le DataFrame.")
                return df
            
            # Créer une copie du DataFrame pour ne pas modifier l'original
            enriched_df = df.copy()
            
            # Sauvegarder l'ordre original des lignes
            # Important: les tirages les plus anciens sont en bas, les plus récents en haut
            enriched_df['_original_index'] = np.arange(len(enriched_df))
            
            # S'assurer que la colonne de date est au format datetime
            if not pd.api.types.is_datetime64_dtype(enriched_df[date_column]):
                try:
                    enriched_df[date_column] = pd.to_datetime(enriched_df[date_column], errors='coerce')
                    logger.info(f"Colonne {date_column} convertie en datetime.")
                except Exception as e:
                    logger.error(f"Impossible de convertir la colonne {date_column} en datetime: {str(e)}")
                    return df
            
            # Vérifier les dates et corriger si nécessaire
            # Aucune date ne devrait être antérieure au premier tirage (13 février 2004)
            mask_invalid_dates = enriched_df[date_column] < self.first_draw_date
            if mask_invalid_dates.any():
                logger.warning(f"Détection de {mask_invalid_dates.sum()} dates antérieures au premier tirage EuroMillions (13 février 2004). Ces dates seront corrigées.")
                enriched_df.loc[mask_invalid_dates, date_column] = self.first_draw_date
            
            # Ajouter les colonnes de données lunaires
            illuminations = []
            phases_numeric = []
            phases_name = []
            
            for date in enriched_df[date_column]:
                if pd.isna(date):
                    illuminations.append(np.nan)
                    phases_numeric.append(np.nan)
                    phases_name.append("Inconnue")
                else:
                    illumination, phase_numeric, phase_name = self.calculate_moon_phase(date)
                    illuminations.append(illumination)
                    phases_numeric.append(phase_numeric)
                    phases_name.append(phase_name)
            
            enriched_df['LunarIllumination'] = illuminations
            enriched_df['LunarPhaseNumeric'] = phases_numeric
            enriched_df['LunarPhaseName'] = phases_name
            
            # Restaurer l'ordre original des lignes (plus récents en haut, plus anciens en bas)
            enriched_df = enriched_df.sort_values('_original_index')
            enriched_df = enriched_df.drop('_original_index', axis=1)
            
            logger.info(f"DataFrame enrichi avec les données lunaires: {len(enriched_df)} lignes traitées.")
            return enriched_df
        
        except Exception as e:
            logger.error(f"Erreur lors de l'enrichissement du DataFrame avec les données lunaires: {str(e)}")
            return df

    def analyze_lunar_influence(self, df: pd.DataFrame, number_cols: List[str], star_cols: List[str]) -> Dict:
        """
        Analyse l'influence du cycle lunaire sur les tirages.
        
        Args:
            df: DataFrame enrichi avec les données lunaires
            number_cols: Liste des colonnes contenant les numéros principaux
            star_cols: Liste des colonnes contenant les étoiles
            
        Returns:
            Dict: Résultats de l'analyse
        """
        logger.info("Analyse de l'influence du cycle lunaire sur les tirages...")
        
        try:
            # Vérifier que les colonnes nécessaires existent
            required_cols = ['LunarPhaseNumeric', 'LunarIllumination'] + number_cols + star_cols
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Colonnes manquantes pour l'analyse lunaire: {missing_cols}")
                return {}
            
            # Initialiser les résultats
            results = {
                'number_frequency_by_phase': {},
                'star_frequency_by_phase': {},
                'number_frequency_by_illumination': {},
                'star_frequency_by_illumination': {},
                'phase_distribution': {},
                'illumination_stats': {}
            }
            
            # Analyser la distribution des phases lunaires
            phase_counts = df['LunarPhaseNumeric'].value_counts().to_dict()
            for phase in range(8):
                results['phase_distribution'][self.lunar_phases.get(phase, f"Phase {phase}")] = phase_counts.get(phase, 0)
            
            # Statistiques d'illumination
            results['illumination_stats'] = {
                'mean': df['LunarIllumination'].mean(),
                'median': df['LunarIllumination'].median(),
                'std': df['LunarIllumination'].std(),
                'min': df['LunarIllumination'].min(),
                'max': df['LunarIllumination'].max()
            }
            
            # Analyser la fréquence des numéros par phase lunaire
            for phase in range(8):
                phase_df = df[df['LunarPhaseNumeric'] == phase]
                
                # Fréquence des numéros principaux
                number_freq = {}
                for col in number_cols:
                    for num in phase_df[col].dropna():
                        number_freq[int(num)] = number_freq.get(int(num), 0) + 1
                results['number_frequency_by_phase'][self.lunar_phases.get(phase, f"Phase {phase}")] = number_freq
                
                # Fréquence des étoiles
                star_freq = {}
                for col in star_cols:
                    for star in phase_df[col].dropna():
                        star_freq[int(star)] = star_freq.get(int(star), 0) + 1
                results['star_frequency_by_phase'][self.lunar_phases.get(phase, f"Phase {phase}")] = star_freq
            
            # Analyser la fréquence des numéros par niveau d'illumination
            # Diviser l'illumination en 4 quartiles
            df['IlluminationQuartile'] = pd.qcut(df['LunarIllumination'], 4, labels=False)
            
            for quartile in range(4):
                quartile_df = df[df['IlluminationQuartile'] == quartile]
                quartile_name = f"Quartile {quartile+1}"
                
                # Fréquence des numéros principaux
                number_freq = {}
                for col in number_cols:
                    for num in quartile_df[col].dropna():
                        number_freq[int(num)] = number_freq.get(int(num), 0) + 1
                results['number_frequency_by_illumination'][quartile_name] = number_freq
                
                # Fréquence des étoiles
                star_freq = {}
                for col in star_cols:
                    for star in quartile_df[col].dropna():
                        star_freq[int(star)] = star_freq.get(int(star), 0) + 1
                results['star_frequency_by_illumination'][quartile_name] = star_freq
            
            logger.info("Analyse de l'influence lunaire terminée.")
            return results
        
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse de l'influence lunaire: {str(e)}")
            return {}

    def save_lunar_analysis(self, results: Dict, timestamp: str = None) -> str:
        """
        Sauvegarde les résultats de l'analyse lunaire dans un fichier texte.
        
        Args:
            results: Dictionnaire contenant les résultats de l'analyse
            timestamp: Horodatage pour le nom du fichier (optionnel)
            
        Returns:
            str: Chemin du fichier de résultats sauvegardé
        """
        logger.info("Sauvegarde des résultats de l'analyse lunaire...")
        
        try:
            if not results:
                logger.error("Aucun résultat à sauvegarder.")
                return ""
            
            if timestamp is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            results_file = self.output_dir / f"lunar_analysis_{timestamp}.txt"
            
            with open(results_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("ANALYSE DE L'INFLUENCE DU CYCLE LUNAIRE SUR LES TIRAGES EUROMILLIONS\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Date de l'analyse: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"Note: Les tirages sont analysés depuis le premier tirage EuroMillions (13 février 2004)\n")
                f.write(f"      jusqu'au tirage le plus récent, avec les plus récents en haut du fichier.\n\n")
                
                # Distribution des phases lunaires
                f.write("-" * 80 + "\n")
                f.write("DISTRIBUTION DES PHASES LUNAIRES\n")
                f.write("-" * 80 + "\n")
                for phase, count in results.get('phase_distribution', {}).items():
                    f.write(f"{phase}: {count} tirages\n")
                f.write("\n")
                
                # Statistiques d'illumination
                f.write("-" * 80 + "\n")
                f.write("STATISTIQUES D'ILLUMINATION LUNAIRE\n")
                f.write("-" * 80 + "\n")
                for stat, value in results.get('illumination_stats', {}).items():
                    f.write(f"{stat.capitalize()}: {value:.4f}\n")
                f.write("\n")
                
                # Fréquence des numéros par phase lunaire
                f.write("-" * 80 + "\n")
                f.write("FRÉQUENCE DES NUMÉROS PAR PHASE LUNAIRE\n")
                f.write("-" * 80 + "\n")
                for phase, freq in results.get('number_frequency_by_phase', {}).items():
                    f.write(f"{phase}:\n")
                    # Trier par fréquence décroissante
                    sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
                    for num, count in sorted_freq[:10]:  # Top 10
                        f.write(f"  Numéro {num}: {count} occurrences\n")
                    f.write("\n")
                
                # Fréquence des étoiles par phase lunaire
                f.write("-" * 80 + "\n")
                f.write("FRÉQUENCE DES ÉTOILES PAR PHASE LUNAIRE\n")
                f.write("-" * 80 + "\n")
                for phase, freq in results.get('star_frequency_by_phase', {}).items():
                    f.write(f"{phase}:\n")
                    # Trier par fréquence décroissante
                    sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
                    for star, count in sorted_freq:
                        f.write(f"  Étoile {star}: {count} occurrences\n")
                    f.write("\n")
                
                # Fréquence des numéros par niveau d'illumination
                f.write("-" * 80 + "\n")
                        # Fréquence des numéros par niveau d'illumination
                f.write("-" * 80 + "\n")
                f.write("FRÉQUENCE DES NUMÉROS PAR NIVEAU D'ILLUMINATION\n")
                f.write("-" * 80 + "\n")
                for quartile, freq in results.get('number_frequency_by_illumination', {}).items():
                    f.write(f"{quartile}:\n")
                    sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
                    for num, count in sorted_freq[:10]:  # Top 10
                        f.write(f"  Numéro {num}: {count} occurrences\n")
                    f.write("\n")

                # Fréquence des étoiles par niveau d'illumination
                f.write("-" * 80 + "\n")
                f.write("FRÉQUENCE DES ÉTOILES PAR NIVEAU D'ILLUMINATION\n")
                f.write("-" * 80 + "\n")
                for quartile, freq in results.get('star_frequency_by_illumination', {}).items():
                    f.write(f"{quartile}:\n")
                    sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
                    for star, count in sorted_freq:
                        f.write(f"  Étoile {star}: {count} occurrences\n")
                    f.write("\n")

            logger.info(f"Résultats sauvegardés dans {results_file}")
            return str(results_file)
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des résultats de l'analyse lunaire: {str(e)}")
            return ""