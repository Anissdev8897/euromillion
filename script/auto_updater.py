#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Système de mise à jour automatique des tirages et réentraînement des modèles
Récupère les tirages tous les mardis et vendredis à 22h
"""

import os
import sys
import logging
import schedule
import time
from pathlib import Path
from datetime import datetime
from threading import Thread
import traceback

# Ajouter le répertoire script au path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from fdj_scraper import FDJEuromillionsScraper
from euromillions_train import EuromillionsTrainer

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('auto_updater.log')
    ]
)
logger = logging.getLogger("AutoUpdater")

class AutoUpdater:
    """Système de mise à jour automatique"""
    
    def __init__(self, csv_file: str = "tirage_euromillions_complet.csv",
                 output_dir: str = "resultats_euromillions",
                 model_dir: str = "models_euromillions",
                 enable_training: bool = False):
        """
        Initialise le système de mise à jour.
        
        Args:
            csv_file: Chemin vers le fichier CSV
            output_dir: Répertoire de sortie
            model_dir: Répertoire des modèles
            enable_training: Si True, réentraîne les modèles après mise à jour
        """
        self.csv_file = csv_file
        self.output_dir = output_dir
        self.model_dir = model_dir
        self.enable_training = enable_training
        self.scraper = FDJEuromillionsScraper(csv_file)
        self.running = False

    def update_and_retrain(self):
        """Met à jour les tirages et réentraîne les modèles (si activé)"""
        try:
            logger.info("=" * 80)
            logger.info("Démarrage de la mise à jour automatique")
            logger.info("=" * 80)
            
            # Étape 1: Mettre à jour les tirages
            logger.info("Étape 1: Mise à jour des tirages...")
            updated = self.scraper.update_draws()
            
            if updated:
                logger.info("Nouveau tirage ajouté au fichier CSV")
                
                # Étape 2: Réentraîner les modèles (seulement si activé)
                if self.enable_training:
                    logger.info("Étape 2: Réentraînement des modèles...")
                    config = {
                        "csv_file": self.csv_file,
                        "output_dir": self.output_dir,
                        "model_dir": self.model_dir
                    }
                    
                    trainer = EuromillionsTrainer(config)
                    results = trainer.train_all()
                    
                    logger.info("Réentraînement terminé")
                    logger.info(f"Résultats: {results}")
                else:
                    logger.info("Étape 2: Réentraînement désactivé (à faire sur PC local)")
                    logger.info("Pour entraîner: python script/euromillions_train.py")
            else:
                logger.info("Aucun nouveau tirage à ajouter")
            
            logger.info("=" * 80)
            logger.info("Mise à jour terminée")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour: {str(e)}")
            logger.error(traceback.format_exc())
    
    def schedule_updates(self):
        """Planifie les mises à jour automatiques"""
        # Mardi à 22h
        schedule.every().tuesday.at("22:00").do(self.update_and_retrain)
        
        # Vendredi à 22h
        schedule.every().friday.at("22:00").do(self.update_and_retrain)
        
        logger.info("Mises à jour planifiées:")
        logger.info("  - Tous les mardis à 22h00")
        logger.info("  - Tous les vendredis à 22h00")
    
    def run_scheduler(self):
        """Lance le scheduler en arrière-plan"""
        self.running = True
        logger.info("Démarrage du scheduler...")
        
        while self.running:
            schedule.run_pending()
            time.sleep(60)  # Vérifier toutes les minutes
    
    def start(self, run_immediately: bool = False):
        """
        Démarre le système de mise à jour.
        
        Args:
            run_immediately: Si True, exécute une mise à jour immédiate
        """
        if run_immediately:
            logger.info("Exécution immédiate de la mise à jour...")
            self.update_and_retrain()
        
        # Planifier les mises à jour
        self.schedule_updates()
        
        # Lancer le scheduler dans un thread séparé
        scheduler_thread = Thread(target=self.run_scheduler, daemon=True)
        scheduler_thread.start()
        
        logger.info("Système de mise à jour automatique démarré")
    
    def stop(self):
        """Arrête le système de mise à jour"""
        self.running = False
        logger.info("Arrêt du système de mise à jour")


def main():
    """Fonction principale"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Système de mise à jour automatique")
    parser.add_argument("--csv", default="tirage_euromillions_complet.csv", help="Fichier CSV")
    parser.add_argument("--output", default="resultats_euromillions", help="Répertoire de sortie")
    parser.add_argument("--model-dir", default="models_euromillions", help="Répertoire des modèles")
    parser.add_argument("--now", action="store_true", help="Exécuter une mise à jour immédiate")
    parser.add_argument("--daemon", action="store_true", help="Lancer en mode daemon")
    
    args = parser.parse_args()
    
    # Par défaut, désactiver l'entraînement (à faire sur PC local)
    updater = AutoUpdater(args.csv, args.output, args.model_dir, enable_training=False)
    
    if args.daemon:
        updater.start(run_immediately=args.now)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Arrêt demandé...")
            updater.stop()
    else:
        # Exécution unique
        updater.update_and_retrain()


if __name__ == "__main__":
    main()

