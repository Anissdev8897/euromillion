#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module d'intégration pour l'analyseur vidéo Euromillions.
Ce module permet d'intégrer l'analyseur vidéo au système d'analyse Euromillions existant.
"""

import os
import sys
import logging
from pathlib import Path
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any

# Importer le module d'analyse vidéo
from euromillions_video_analyzer import EuromillionsVideoAnalyzer

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger("EuromillionsVideoIntegration")

class EuromillionsVideoIntegration:
    """Classe pour l'intégration de l'analyseur vidéo au système Euromillions."""
    
    def __init__(self, analyzer=None, video_dir=None, output_dir=None):
        """
        Initialise l'intégrateur vidéo.
        
        Args:
            analyzer: Instance de EuromillionsAnalyzer (optionnel)
            video_dir: Répertoire contenant les vidéos de tirage (optionnel)
            output_dir: Répertoire de sortie pour les résultats (optionnel)
        """
        self.analyzer = analyzer
        self.video_dir = Path(video_dir) if video_dir else Path("tirage_video")
        self.output_dir = Path(output_dir) if output_dir else Path("resultats_euromillions")
        
        # Créer l'analyseur vidéo
        self.video_analyzer = EuromillionsVideoAnalyzer(
            video_dir=self.video_dir,
            output_dir=self.output_dir / "video_analysis"
        )
        
        logger.info(f"EuromillionsVideoIntegration initialisé avec video_dir: {self.video_dir}")
    
    def analyze_latest_video(self) -> Dict[str, Any]:
        """
        Analyse la vidéo la plus récente et intègre les résultats.
        
        Returns:
            Dict[str, Any]: Résultats de l'analyse
        """
        # Lister les vidéos disponibles
        videos = self.video_analyzer.list_videos()
        if not videos:
            logger.warning("Aucune vidéo disponible pour analyse.")
            return {"success": False, "error": "Aucune vidéo disponible"}
        
        # Trier par date de modification (la plus récente en premier)
        videos.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        latest_video = videos[0]
        
        logger.info(f"Analyse de la vidéo la plus récente: {latest_video}")
        
        # Analyser la vidéo
        results = self.video_analyzer.analyze_video(latest_video)
        
        # Intégrer les résultats si l'analyse a réussi
        if results["success"] and results["main_numbers"] and results["stars"]:
            self.integrate_results(results)
        
        return results
    
    def integrate_results(self, video_results: Dict[str, Any]) -> bool:
        """
        Intègre les résultats de l'analyse vidéo au système Euromillions.
        
        Args:
            video_results: Résultats de l'analyse vidéo
            
        Returns:
            bool: True si l'intégration a réussi, False sinon
        """
        if not video_results["success"]:
            logger.error("Impossible d'intégrer des résultats d'analyse échoués.")
            return False
        
        try:
            # Extraire les numéros et étoiles détectés
            main_numbers = video_results["main_numbers"]
            stars = video_results["stars"]
            
            if not main_numbers or not stars:
                logger.error("Numéros ou étoiles manquants dans les résultats.")
                return False
            
            logger.info(f"Intégration des numéros: {main_numbers} et étoiles: {stars}")
            
            # Si un analyseur Euromillions est fourni, on peut l'utiliser pour intégrer les résultats
            if self.analyzer:
                # Exemple d'intégration (à adapter selon l'API de l'analyseur)
                if hasattr(self.analyzer, 'add_tirage'):
                    # Format de date à adapter selon le format attendu par l'analyseur
                    date_str = datetime.now().strftime("%d/%m/%Y")
                    self.analyzer.add_tirage(date_str, main_numbers, stars)
                    logger.info(f"Tirage ajouté à l'analyseur pour la date {date_str}")
                else:
                    logger.warning("L'analyseur ne dispose pas de méthode add_tirage.")
            
            # Sauvegarder les résultats dans un fichier CSV pour intégration ultérieure
            # Format: date,num1,num2,num3,num4,num5,star1,star2
            csv_path = self.output_dir / "video_detected_tirages.csv"
            
            # Créer l'en-tête si le fichier n'existe pas
            if not csv_path.exists():
                with open(csv_path, 'w', encoding='utf-8') as f:
                    f.write("Date,Numéro1,Numéro2,Numéro3,Numéro4,Numéro5,Etoile1,Etoile2\n")
            
            # Ajouter le tirage détecté
            with open(csv_path, 'a', encoding='utf-8') as f:
                date_str = datetime.now().strftime("%d/%m/%Y")
                nums_str = ",".join(map(str, main_numbers))
                stars_str = ",".join(map(str, stars))
                f.write(f"{date_str},{nums_str},{stars_str}\n")
            
            logger.info(f"Résultats sauvegardés dans {csv_path}")
            
            # Créer un fichier de rapport détaillé
            report_path = self.output_dir / f"video_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=== RAPPORT D'ANALYSE VIDÉO EUROMILLIONS ===\n\n")
                f.write(f"Date d'analyse: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
                f.write(f"Vidéo analysée: {video_results['video_path']}\n")
                f.write(f"Durée: {video_results['duration']:.2f} secondes\n\n")
                f.write("=== RÉSULTATS DÉTECTÉS ===\n")
                f.write(f"Numéros principaux: {main_numbers}\n")
                f.write(f"Étoiles: {stars}\n\n")
                f.write("=== FRAMES CLÉS ===\n")
                for key, path in video_results["frames"].items():
                    if path:
                        f.write(f"{key}: {path}\n")
            
            logger.info(f"Rapport détaillé sauvegardé dans {report_path}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de l'intégration des résultats: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
            return False
    
    def launch_video_gui(self):
        """
        Lance l'interface graphique de l'analyseur vidéo.
        """
        logger.info("Lancement de l'interface graphique de l'analyseur vidéo...")
        self.video_analyzer.start_gui()

def main():
    """
    Fonction principale pour l'exécution autonome du module.
    """
    parser = argparse.ArgumentParser(description="Intégrateur vidéo pour Euromillions")
    parser.add_argument("--video-dir", type=str, default="tirage_video",
                        help="Répertoire contenant les vidéos de tirage")
    parser.add_argument("--output-dir", type=str, default="resultats_euromillions",
                        help="Répertoire de sortie pour les résultats")
    parser.add_argument("--analyze-latest", action="store_true",
                        help="Analyser la vidéo la plus récente")
    parser.add_argument("--gui", action="store_true",
                        help="Lancer l'interface graphique")
    
    args = parser.parse_args()
    
    integrator = EuromillionsVideoIntegration(
        video_dir=args.video_dir,
        output_dir=args.output_dir
    )
    
    if args.analyze_latest:
        results = integrator.analyze_latest_video()
        if results["success"]:
            print(f"Analyse terminée avec succès.")
            print(f"Numéros détectés: {results['main_numbers']}")
            print(f"Étoiles détectées: {results['stars']}")
        else:
            print(f"Échec de l'analyse: {results.get('error', 'Erreur inconnue')}")
    
    if args.gui:
        integrator.launch_video_gui()
    
    if not args.analyze_latest and not args.gui:
        print("Aucune action spécifiée. Utilisez --analyze-latest ou --gui.")
        parser.print_help()

if __name__ == "__main__":
    main()
