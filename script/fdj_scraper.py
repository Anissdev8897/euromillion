#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scraper pour récupérer automatiquement les tirages EuroMillions depuis le site FDJ
Récupère les données depuis https://www.fdj.fr/jeux-de-tirage/euromillions-my-million/resultats
"""

import os
import sys
import logging
import requests
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import time
import re
import traceback

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FDJScraper")

class FDJEuromillionsScraper:
    """Scraper pour récupérer les tirages EuroMillions depuis FDJ"""
    
    def __init__(self, csv_file: str = "tirage_euromillions_complet.csv"):
        """
        Initialise le scraper.
        
        Args:
            csv_file: Chemin vers le fichier CSV des tirages
        """
        self.base_url = "https://www.fdj.fr/jeux-de-tirage/euromillions-my-million/resultats"
        self.csv_file = Path(csv_file)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
    def fetch_page(self, date: Optional[str] = None) -> Optional[BeautifulSoup]:
        """
        Récupère la page des résultats.
        
        Args:
            date: Date au format YYYY-MM-DD (optionnel)
            
        Returns:
            BeautifulSoup object ou None en cas d'erreur
        """
        try:
            url = self.base_url
            if date:
                url += f"?date={date}"
            
            logger.info(f"Récupération de la page: {url}")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # ⚠️ CRITIQUE : Utiliser 'lxml' parser si disponible pour une meilleure compatibilité
            try:
                soup = BeautifulSoup(response.content, 'lxml')
            except:
                # Fallback vers html.parser
                soup = BeautifulSoup(response.content, 'html.parser')
            
            # ⚠️ DEBUG : Logger un extrait du HTML pour diagnostiquer
            html_preview = str(soup)[:500] if len(str(soup)) > 500 else str(soup)
            logger.debug(f"Extrait HTML (500 premiers caractères): {html_preview}")
            
            return soup
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de la page: {str(e)}")
            return None
    
    def extract_draw_from_page(self, soup: BeautifulSoup) -> Optional[Dict]:
        """
        Extrait les données d'un tirage depuis la page HTML.
        Utilise le sélecteur CSS recommandé : #result-grid-1 li span span.heading4
        
        Args:
            soup: BeautifulSoup object de la page
            
        Returns:
            Dictionnaire avec les numéros et étoiles ou None
        """
        try:
            # Méthode 1: Utiliser le sélecteur CSS recommandé (#result-grid-1 li span span.heading4)
            numbers = []
            stars = []
            
            # ⚠️ CRITIQUE : Méthode 1 - Utiliser le sélecteur CSS recommandé (#result-grid-1 li span span.heading4)
            # Chercher la liste ul avec id="result-grid-1"
            result_grid = soup.find('ul', {'id': 'result-grid-1'})
            if result_grid:
                logger.debug(f"✅ Élément ul#result-grid-1 trouvé")
                # Extraire tous les éléments span.heading4 dans cette liste
                heading4_elements = result_grid.select('li span span.heading4')
                logger.debug(f"Nombre d'éléments span.heading4 trouvés: {len(heading4_elements)}")
                
                for elem in heading4_elements:
                    text = elem.get_text(strip=True)
                    if text.isdigit():
                        num = int(text)
                        # Les 5 premiers numéros valides (1-50) sont les numéros principaux
                        if 1 <= num <= 50 and num not in numbers and len(numbers) < 5:
                            numbers.append(num)
                            logger.debug(f"Numéro trouvé: {num}")
                        # Les 2 numéros suivants valides (1-12) sont les étoiles
                        elif 1 <= num <= 12 and num not in stars and len(stars) < 2:
                            stars.append(num)
                            logger.debug(f"Étoile trouvée: {num}")
                
                logger.info(f"Extraction via sélecteur CSS: {len(numbers)} numéros, {len(stars)} étoiles")
            else:
                logger.warning("⚠️ Élément ul#result-grid-1 non trouvé, tentative avec d'autres sélecteurs...")
            
            # Méthode 2: Fallback - Chercher dans result-wrapper-grid-1
            if len(numbers) < 5 or len(stars) < 2:
                result_wrapper = soup.find('div', {'id': 'result-wrapper-grid-1'})
                if result_wrapper:
                    logger.debug(f"✅ Élément div#result-wrapper-grid-1 trouvé")
                    # Chercher tous les span.heading4 dans le wrapper
                    heading4_elements = result_wrapper.select('span.heading4')
                    logger.debug(f"Nombre d'éléments span.heading4 dans wrapper: {len(heading4_elements)}")
                    
                    for elem in heading4_elements:
                        text = elem.get_text(strip=True)
                        if text.isdigit():
                            num = int(text)
                            if 1 <= num <= 50 and num not in numbers and len(numbers) < 5:
                                numbers.append(num)
                                logger.debug(f"Numéro trouvé (fallback 1): {num}")
                            elif 1 <= num <= 12 and num not in stars and len(stars) < 2:
                                stars.append(num)
                                logger.debug(f"Étoile trouvée (fallback 1): {num}")
                    
                    logger.info(f"Extraction via fallback: {len(numbers)} numéros, {len(stars)} étoiles")
                else:
                    logger.warning("⚠️ Élément div#result-wrapper-grid-1 non trouvé")
            
            # Méthode 3: Fallback - Chercher dans les éléments avec des classes spécifiques
            if len(numbers) < 5 or len(stars) < 2:
                number_elements = soup.find_all(['span', 'div', 'li'], 
                                               class_=re.compile(r'number|num|ball|heading4', re.I))
                
                for elem in number_elements:
                    text = elem.get_text(strip=True)
                    if text.isdigit():
                        num = int(text)
                        if 1 <= num <= 50 and num not in numbers and len(numbers) < 5:
                            numbers.append(num)
                        elif 1 <= num <= 12 and num not in stars and len(stars) < 2:
                            stars.append(num)
                
                logger.info(f"Extraction via fallback 2: {len(numbers)} numéros, {len(stars)} étoiles")
            
            # Méthode 4: Fallback - Chercher dans le texte de la page
            if len(numbers) < 5 or len(stars) < 2:
                page_text = soup.get_text()
                # Chercher des patterns comme "26-27-45-48-9"
                number_pattern = r'\b([1-4]?[0-9]|50)\b'
                matches = re.findall(number_pattern, page_text)
                
                for match in matches:
                    num = int(match)
                    if 1 <= num <= 50 and num not in numbers and len(numbers) < 5:
                        numbers.append(num)
                    elif 1 <= num <= 12 and num not in stars and len(stars) < 2:
                        stars.append(num)
                    
                    if len(numbers) >= 5 and len(stars) >= 2:
                        break
                
                logger.info(f"Extraction via regex: {len(numbers)} numéros, {len(stars)} étoiles")
            
            # Vérifier que nous avons tous les numéros et étoiles
            if len(numbers) == 5 and len(stars) == 2:
                # Trier les numéros par ordre croissant
                numbers.sort()
                stars.sort()
                return {
                    'N1': numbers[0],
                    'N2': numbers[1],
                    'N3': numbers[2],
                    'N4': numbers[3],
                    'N5': numbers[4],
                    'E1': stars[0],
                    'E2': stars[1]
                }
            else:
                logger.warning(f"Nombre de numéros/étoiles insuffisant: {len(numbers)} numéros, {len(stars)} étoiles")
                logger.debug(f"Numéros trouvés: {numbers}, Étoiles trouvées: {stars}")
                return None
                
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction des données: {str(e)}")
            logger.debug(traceback.format_exc())
            return None
    
    def extract_date_from_page(self, soup: BeautifulSoup) -> Optional[str]:
        """
        Extrait la date du tirage depuis la page.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            Date au format YYYY-MM-DD ou None
        """
        try:
            # Chercher la date dans différents formats
            date_patterns = [
                r'(\d{2})/(\d{2})/(\d{4})',
                r'(\d{4})-(\d{2})-(\d{2})',
                r'(\d{1,2})\s+(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+(\d{4})'
            ]
            
            page_text = soup.get_text()
            for pattern in date_patterns:
                match = re.search(pattern, page_text, re.IGNORECASE)
                if match:
                    if 'janvier' in pattern.lower():
                        # Format français
                        months = {
                            'janvier': '01', 'février': '02', 'mars': '03', 'avril': '04',
                            'mai': '05', 'juin': '06', 'juillet': '07', 'août': '08',
                            'septembre': '09', 'octobre': '10', 'novembre': '11', 'décembre': '12'
                        }
                        day, month_name, year = match.groups()
                        month = months.get(month_name.lower(), '01')
                        return f"{year}-{month}-{day.zfill(2)}"
                    else:
                        # Format numérique
                        parts = match.groups()
                        if len(parts) == 3:
                            if len(parts[0]) == 4:  # YYYY-MM-DD
                                return f"{parts[0]}-{parts[1]}-{parts[2]}"
                            else:  # DD/MM/YYYY
                                return f"{parts[2]}-{parts[1]}-{parts[0]}"
            
            # Si aucune date trouvée, utiliser la date actuelle
            return datetime.now().strftime('%Y-%m-%d')
            
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction de la date: {str(e)}")
            return datetime.now().strftime('%Y-%m-%d')
    
    def get_latest_draw(self) -> Optional[Dict]:
        """
        Récupère le dernier tirage disponible.
        
        Returns:
            Dictionnaire avec les données du tirage ou None
        """
        soup = self.fetch_page()
        if not soup:
            return None
        
        draw_data = self.extract_draw_from_page(soup)
        if draw_data:
            date = self.extract_date_from_page(soup)
            draw_data['Date'] = date
            logger.info(f"Tirage récupéré pour le {date}: {draw_data}")
        
        return draw_data
    
    def load_existing_draws(self) -> pd.DataFrame:
        """
        Charge les tirages existants depuis le CSV.
        
        Returns:
            DataFrame avec les tirages existants
        """
        if self.csv_file.exists():
            try:
                df = pd.read_csv(self.csv_file)
                logger.info(f"Chargé {len(df)} tirages existants")
                return df
            except Exception as e:
                logger.error(f"Erreur lors du chargement du CSV: {str(e)}")
                return pd.DataFrame()
        else:
            logger.info("Aucun fichier CSV existant, création d'un nouveau")
            return pd.DataFrame()
    
    def is_draw_new(self, draw_data: Dict, existing_df: pd.DataFrame) -> bool:
        """
        Vérifie si un tirage est nouveau.
        
        Args:
            draw_data: Données du tirage
            existing_df: DataFrame des tirages existants
            
        Returns:
            True si le tirage est nouveau, False sinon
        """
        if existing_df.empty:
            return True
        
        # Vérifier par date
        if 'Date' in draw_data and 'Date' in existing_df.columns:
            if draw_data['Date'] in existing_df['Date'].values:
                return False
        
        # Vérifier par combinaison
        cols = ['N1', 'N2', 'N3', 'N4', 'N5', 'E1', 'E2']
        if all(col in draw_data for col in cols) and all(col in existing_df.columns for col in cols):
            numbers = sorted([draw_data[col] for col in cols[:5]])
            stars = sorted([draw_data[col] for col in cols[5:]])
            
            for _, row in existing_df.iterrows():
                existing_numbers = sorted([row[col] for col in cols[:5]])
                existing_stars = sorted([row[col] for col in cols[5:]])
                
                if numbers == existing_numbers and stars == existing_stars:
                    return False
        
        return True
    
    def add_draw_to_csv(self, draw_data: Dict) -> bool:
        """
        Ajoute un nouveau tirage au fichier CSV.
        
        Args:
            draw_data: Données du tirage
            
        Returns:
            True si ajouté avec succès, False sinon
        """
        try:
            existing_df = self.load_existing_draws()
            
            if not self.is_draw_new(draw_data, existing_df):
                logger.info("Tirage déjà présent dans le fichier CSV")
                return False
            
            # Préparer les données
            new_row = {
                'N1': draw_data.get('N1'),
                'N2': draw_data.get('N2'),
                'N3': draw_data.get('N3'),
                'N4': draw_data.get('N4'),
                'N5': draw_data.get('N5'),
                'E1': draw_data.get('E1'),
                'E2': draw_data.get('E2')
            }
            
            # Ajouter la date si disponible
            if 'Date' in draw_data:
                new_row['Date'] = draw_data['Date']
            
            new_df = pd.DataFrame([new_row])
            
            # Concaténer avec les données existantes
            if existing_df.empty:
                final_df = new_df
            else:
                final_df = pd.concat([new_df, existing_df], ignore_index=True)
            
            # Sauvegarder
            final_df.to_csv(self.csv_file, index=False)
            logger.info(f"Nouveau tirage ajouté au fichier CSV: {new_row}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de l'ajout du tirage au CSV: {str(e)}")
            return False
    
    def update_draws(self) -> bool:
        """
        Met à jour les tirages en récupérant le dernier tirage.
        Génère automatiquement le fichier de cycles après la mise à jour.
        
        Returns:
            True si mise à jour réussie, False sinon
        """
        logger.info("Mise à jour des tirages EuroMillions...")
        
        draw_data = self.get_latest_draw()
        if not draw_data:
            logger.warning("Aucun tirage récupéré")
            return False
        
        success = self.add_draw_to_csv(draw_data)
        if success:
            logger.info("Mise à jour réussie")
            
            # ⚠️ CRITIQUE : Générer automatiquement le fichier de cycles après mise à jour
            try:
                from cycle_data_generator import CycleDataGenerator
                generator = CycleDataGenerator(str(self.csv_file))
                cycle_success = generator.generate_and_save()
                if cycle_success:
                    logger.info("✅ Fichier de cycles mis à jour automatiquement")
                else:
                    logger.warning("⚠️ Échec de la génération du fichier de cycles")
            except ImportError:
                logger.warning("Module cycle_data_generator non disponible. Fichier de cycles non généré.")
            except Exception as e:
                logger.warning(f"Erreur lors de la génération du fichier de cycles: {str(e)}")
        else:
            logger.info("Aucun nouveau tirage à ajouter")
        
        return success


def main():
    """Fonction principale pour tester le scraper"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Scraper FDJ EuroMillions")
    parser.add_argument("--csv", default="tirage_euromillions_complet.csv", help="Fichier CSV de sortie")
    parser.add_argument("--update", action="store_true", help="Mettre à jour les tirages")
    
    args = parser.parse_args()
    
    scraper = FDJEuromillionsScraper(args.csv)
    
    if args.update:
        scraper.update_draws()
    else:
        draw = scraper.get_latest_draw()
        if draw:
            print(f"Tirage récupéré: {draw}")
        else:
            print("Aucun tirage récupéré")


if __name__ == "__main__":
    main()

