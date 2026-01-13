#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Générateur de fichier de cycles avec dates automatiques
Crée un fichier avec les cycles depuis le premier tirage jusqu'au dernier
pour l'utilisation des modèles de cycles (lunaire, temporel, etc.)
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import traceback

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CycleDataGenerator")

try:
    import ephem
    EPHEM_AVAILABLE = True
except ImportError:
    EPHEM_AVAILABLE = False
    logger.warning("Module ephem non disponible. Les cycles lunaires ne seront pas calculés.")


class CycleDataGenerator:
    """Générateur de données de cycles pour les modèles de prédiction"""
    
    def __init__(self, csv_file: str = "tirage_euromillions_complet.csv"):
        """
        Initialise le générateur de cycles.
        
        Args:
            csv_file: Chemin vers le fichier CSV des tirages
        """
        self.csv_file = Path(csv_file)
        self.cycle_file = self.csv_file.parent / f"{self.csv_file.stem}_cycles.csv"
        
        # Date du premier tirage EuroMillions (13 février 2004)
        self.first_draw_date = datetime(2004, 2, 13)
        
        # Phases lunaires
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
    
    def load_draws(self) -> pd.DataFrame:
        """
        Charge les tirages depuis le CSV.
        
        Returns:
            DataFrame avec les tirages
        """
        if not self.csv_file.exists():
            logger.error(f"Fichier CSV non trouvé: {self.csv_file}")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(self.csv_file)
            logger.info(f"Chargé {len(df)} tirages depuis {self.csv_file}")
            
            # Convertir la colonne Date en datetime si elle existe
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            
            return df
        except Exception as e:
            logger.error(f"Erreur lors du chargement du CSV: {str(e)}")
            logger.debug(traceback.format_exc())
            return pd.DataFrame()
    
    def calculate_moon_phase(self, date: datetime) -> Tuple[float, int, str]:
        """
        Calcule la phase lunaire pour une date donnée.
        
        Args:
            date: Date pour laquelle calculer la phase lunaire
            
        Returns:
            Tuple[float, int, str]: (illumination en pourcentage, phase numérique (0-7), nom de la phase)
        """
        # Vérifier si la date est valide (pas NaT)
        if pd.isna(date):
            return (0.0, 0, "Date invalide")
        
        if not EPHEM_AVAILABLE:
            return (0.0, 0, "Non disponible")
        
        try:
            moon = ephem.Moon()
            ephem_date = ephem.Date(date)
            moon.compute(ephem_date)
            
            illumination = moon.phase / 100.0
            
            previous_new_moon = ephem.previous_new_moon(ephem_date)
            moon_age = (ephem_date - previous_new_moon) * 24 * 60 * 60 / 86400.0
            
            phase_numeric = int((moon_age / 29.53) * 8) % 8
            phase_name = self.lunar_phases[phase_numeric]
            
            return (illumination, phase_numeric, phase_name)
        except Exception as e:
            logger.warning(f"Erreur lors du calcul de la phase lunaire: {str(e)}")
            return (0.0, 0, "Erreur")
    
    def calculate_draw_cycle(self, date: datetime) -> int:
        """
        Calcule le numéro de cycle depuis le premier tirage.
        Un cycle = 1 semaine (2 tirages par semaine : mardi et vendredi)
        
        Args:
            date: Date du tirage
            
        Returns:
            Numéro de cycle
        """
        # Vérifier si la date est valide (pas NaT)
        if pd.isna(date):
            return 0
        
        try:
            if date < self.first_draw_date:
                return 0
            
            # Calculer le nombre de jours depuis le premier tirage
            days_since_first = (date - self.first_draw_date).days
            
            # Un cycle = 1 semaine = 7 jours
            # Mais les tirages sont le mardi et vendredi, donc environ 2 tirages par semaine
            cycle = (days_since_first // 7) + 1
            
            return cycle
        except (TypeError, ValueError):
            return 0
    
    def calculate_week_of_year(self, date: datetime) -> int:
        """
        Calcule le numéro de semaine dans l'année.
        
        Args:
            date: Date du tirage
            
        Returns:
            Numéro de semaine (1-53)
        """
        # Vérifier si la date est valide (pas NaT)
        if pd.isna(date):
            return 0
        try:
            return date.isocalendar()[1]
        except (AttributeError, ValueError):
            return 0
    
    def calculate_day_of_week(self, date: datetime) -> int:
        """
        Calcule le jour de la semaine (0 = lundi, 6 = dimanche).
        
        Args:
            date: Date du tirage
            
        Returns:
            Jour de la semaine (0-6)
        """
        # Vérifier si la date est valide (pas NaT)
        if pd.isna(date):
            return 0
        try:
            return date.weekday()
        except (AttributeError, ValueError):
            return 0
    
    def calculate_month_cycle(self, date: datetime) -> int:
        """
        Calcule le cycle mensuel (1-12).
        
        Args:
            date: Date du tirage
            
        Returns:
            Mois (1-12)
        """
        # Vérifier si la date est valide (pas NaT)
        if pd.isna(date):
            return 0
        try:
            return date.month
        except (AttributeError, ValueError):
            return 0
    
    def generate_cycle_data(self) -> pd.DataFrame:
        """
        Génère le fichier de cycles avec toutes les informations depuis le premier tirage.
        
        Returns:
            DataFrame avec les données de cycles
        """
        logger.info("Génération du fichier de cycles...")
        
        # Charger les tirages
        df = self.load_draws()
        if df.empty:
            logger.error("Aucun tirage à traiter")
            return pd.DataFrame()
        
        # Créer une liste pour stocker les données de cycles
        cycle_data = []
        
        # ⚠️ CRITIQUE : Vérifier si la colonne Date existe
        if 'Date' not in df.columns:
            logger.warning("Colonne 'Date' non trouvée. Création de dates automatiques...")
            # Créer des dates automatiques depuis le premier tirage
            start_date = self.first_draw_date
            dates_created = []
            for i in range(len(df)):
                # Les tirages sont le mardi et vendredi
                # Calculer la date en fonction de l'index
                weeks = i // 2
                day_in_week = (i % 2) * 3  # 0 pour mardi, 3 pour vendredi
                date = start_date + timedelta(weeks=weeks, days=day_in_week)
                dates_created.append(date)
                df.loc[i, 'Date'] = date
            logger.info(f"✅ {len(dates_created)} dates automatiques créées (du {dates_created[0].strftime('%Y-%m-%d')} au {dates_created[-1].strftime('%Y-%m-%d')})")
        else:
            # Vérifier que les dates ne sont pas toutes vides
            date_col = pd.to_datetime(df['Date'], errors='coerce')
            if date_col.isna().all():
                logger.warning("Colonne 'Date' présente mais toutes les dates sont invalides. Création de dates automatiques...")
                start_date = self.first_draw_date
                dates_created = []
                for i in range(len(df)):
                    weeks = i // 2
                    day_in_week = (i % 2) * 3
                    date = start_date + timedelta(weeks=weeks, days=day_in_week)
                    dates_created.append(date)
                    df.loc[i, 'Date'] = date
                logger.info(f"✅ {len(dates_created)} dates automatiques créées")
            else:
                # Convertir les dates en datetime
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                valid_dates = df['Date'].notna().sum()
                logger.info(f"✅ {valid_dates}/{len(df)} dates valides trouvées dans le fichier")
        
        # ⚠️ CRITIQUE : Trier par date pour s'assurer de l'ordre chronologique (du premier au dernier)
        df = df.sort_values('Date', ascending=True).reset_index(drop=True)
        logger.info(f"✅ Données triées par date (ordre chronologique: {df['Date'].min()} → {df['Date'].max()})")
        
        logger.info(f"Traitement de {len(df)} tirages...")
        
        for idx, row in df.iterrows():
            try:
                date = pd.to_datetime(row['Date'])
                
                # ⚠️ CRITIQUE : Vérifier si la date est valide (pas NaT)
                if pd.isna(date):
                    # Générer une date automatique basée sur l'index
                    # Les tirages sont le mardi et vendredi
                    weeks = idx // 2
                    day_in_week = (idx % 2) * 3  # 0 pour mardi, 3 pour vendredi
                    date = self.first_draw_date + timedelta(weeks=weeks, days=day_in_week)
                    logger.debug(f"Date automatique générée pour le tirage {idx + 1}: {date.strftime('%Y-%m-%d')}")
                
                # Calculer tous les cycles
                draw_cycle = self.calculate_draw_cycle(date)
                week_of_year = self.calculate_week_of_year(date)
                day_of_week = self.calculate_day_of_week(date)
                month_cycle = self.calculate_month_cycle(date)
                
                # Calculer la phase lunaire
                illumination, phase_numeric, phase_name = self.calculate_moon_phase(date)
                
                # Créer l'entrée de cycle
                cycle_entry = {
                    'Index': idx + 1,
                    'Date': date.strftime('%Y-%m-%d'),
                    'Cycle_Draw': draw_cycle,
                    'Week_Of_Year': week_of_year,
                    'Day_Of_Week': day_of_week,
                    'Day_Name': ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche'][day_of_week],
                    'Month': month_cycle,
                    'Month_Name': date.strftime('%B'),
                    'Year': date.year,
                    'Lunar_Illumination': round(illumination, 4),
                    'Lunar_Phase': phase_numeric,
                    'Lunar_Phase_Name': phase_name,
                    'Days_Since_First': (date - self.first_draw_date).days,
                }
                
                # Ajouter les numéros et étoiles si disponibles
                number_cols = ['N1', 'N2', 'N3', 'N4', 'N5']
                star_cols = ['E1', 'E2']
                
                for col in number_cols + star_cols:
                    if col in row:
                        cycle_entry[col] = row[col]
                
                cycle_data.append(cycle_entry)
                
            except Exception as e:
                logger.warning(f"Erreur lors du traitement du tirage {idx + 1}: {str(e)}")
                continue
        
        # Créer le DataFrame
        cycle_df = pd.DataFrame(cycle_data)
        
        logger.info(f"Génération terminée: {len(cycle_df)} cycles créés")
        
        return cycle_df
    
    def save_cycle_file(self, cycle_df: pd.DataFrame) -> bool:
        """
        Sauvegarde le fichier de cycles.
        
        Args:
            cycle_df: DataFrame avec les données de cycles
            
        Returns:
            True si sauvegarde réussie, False sinon
        """
        if cycle_df.empty:
            logger.error("Aucune donnée de cycle à sauvegarder")
            return False
        
        try:
            cycle_df.to_csv(self.cycle_file, index=False, encoding='utf-8')
            logger.info(f"✅ Fichier de cycles sauvegardé: {self.cycle_file}")
            logger.info(f"   - {len(cycle_df)} cycles")
            logger.info(f"   - Du {cycle_df['Date'].min()} au {cycle_df['Date'].max()}")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du fichier de cycles: {str(e)}")
            logger.debug(traceback.format_exc())
            return False
    
    def generate_and_save(self) -> bool:
        """
        Génère et sauvegarde le fichier de cycles.
        
        Returns:
            True si succès, False sinon
        """
        cycle_df = self.generate_cycle_data()
        if cycle_df.empty:
            return False
        
        return self.save_cycle_file(cycle_df)
    
    def update_cycles(self) -> bool:
        """
        Met à jour le fichier de cycles (alias pour generate_and_save).
        
        Returns:
            True si succès, False sinon
        """
        return self.generate_and_save()


def main():
    """Fonction principale pour tester le générateur"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Générateur de fichier de cycles EuroMillions")
    parser.add_argument("--csv", default="tirage_euromillions_complet.csv", help="Fichier CSV des tirages")
    parser.add_argument("--generate", action="store_true", help="Générer le fichier de cycles")
    
    args = parser.parse_args()
    
    generator = CycleDataGenerator(args.csv)
    
    if args.generate:
        success = generator.generate_and_save()
        if success:
            print(f"✅ Fichier de cycles généré: {generator.cycle_file}")
        else:
            print("❌ Erreur lors de la génération du fichier de cycles")
    else:
        print("Utilisez --generate pour générer le fichier de cycles")


if __name__ == "__main__":
    main()

