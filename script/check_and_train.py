#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script pour vérifier les nouveaux tirages et lancer l'entraînement
Vérifie automatiquement s'il y a de nouveaux tirages avant d'entraîner
VERSION MODIFIÉE: Intégration des features vidéo
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# Ajouter le répertoire script au path
script_dir = Path(__file__).parent / "script"
sys.path.insert(0, str(script_dir))

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CheckAndTrain")

def check_and_update_draws(csv_file: str = "tirage_euromillions_complet.csv") -> bool:
    """
    Vérifie et met à jour les tirages depuis FDJ.
    
    Args:
        csv_file: Chemin vers le fichier CSV
        
    Returns:
        True si de nouveaux tirages ont été ajoutés, False sinon
    """
    try:
        from fdj_scraper import FDJEuromillionsScraper
        
        logger.info("=" * 80)
        logger.info("Vérification des nouveaux tirages...")
        logger.info("=" * 80)
        
        scraper = FDJEuromillionsScraper(csv_file)
        updated = scraper.update_draws()
        
        if updated:
            logger.info("✅ Nouveaux tirages détectés et ajoutés au fichier CSV")
            return True
        else:
            logger.info("ℹ️ Aucun nouveau tirage disponible")
            return False
            
    except Exception as e:
        logger.error(f"Erreur lors de la vérification des tirages: {str(e)}")
        logger.warning("L'entraînement continuera avec les données existantes")
        return False

def get_csv_count(csv_file: str) -> int:
    """
    Compte le nombre de tirages dans le fichier CSV.
    
    Args:
        csv_file: Chemin vers le fichier CSV
        
    Returns:
        Nombre de tirages
    """
    try:
        import pandas as pd
        csv_path = Path(csv_file)
        
        if not csv_path.exists():
            return 0
        
        df = pd.read_csv(csv_path)
        return len(df)
    except Exception as e:
        logger.error(f"Erreur lors du comptage des tirages: {str(e)}")
        return 0

def main():
    """Fonction principale"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Vérifier les tirages et entraîner les modèles")
    parser.add_argument("--csv", default="tirage_euromillions_complet.csv", help="Fichier CSV")
    parser.add_argument("--output", default="resultats_euromillions", help="Répertoire de sortie")
    parser.add_argument("--model-dir", default="models_euromillions", help="Répertoire des modèles")
    parser.add_argument("--method", default="all", help="Méthode d'entraînement")
    parser.add_argument("--skip-check", action="store_true", help="Ignorer la vérification des tirages")
    parser.add_argument("--force", action="store_true", help="Forcer l'entraînement même sans nouveaux tirages")
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("Système de Vérification et Entraînement EuroMillions")
    logger.info("=" * 80)
    logger.info("")
    
    # Compter les tirages avant
    count_before = get_csv_count(args.csv)
    logger.info(f"Nombre de tirages dans le fichier CSV: {count_before}")
    logger.info("")
    
    # Vérifier les nouveaux tirages
    new_draws_found = False
    if not args.skip_check:
        new_draws_found = check_and_update_draws(args.csv)
        logger.info("")
    else:
        logger.info("Vérification des tirages ignorée (--skip-check)")
        logger.info("")
    
    # Compter les tirages après
    count_after = get_csv_count(args.csv)
    if count_after > count_before:
        logger.info(f"✅ {count_after - count_before} nouveau(x) tirage(s) ajouté(s)")
        logger.info(f"Total: {count_before} → {count_after} tirages")
        new_draws_found = True
    else:
        logger.info(f"ℹ️ Aucun nouveau tirage (total: {count_after} tirages)")
    
    logger.info("")
    
    # ⚠️ CRITIQUE : Générer le fichier de cycles après mise à jour
    try:
        script_dir = Path(__file__).parent / "script"
        sys.path.insert(0, str(script_dir))
        from cycle_data_generator import CycleDataGenerator
        
        logger.info("Génération du fichier de cycles avec dates automatiques...")
        generator = CycleDataGenerator(args.csv)
        cycle_success = generator.generate_and_save()
        if cycle_success:
            logger.info("✅ Fichier de cycles généré avec succès")
            logger.info(f"   Fichier: {generator.cycle_file}")
        else:
            logger.warning("⚠️ Échec de la génération du fichier de cycles")
    except ImportError:
        logger.warning("Module cycle_data_generator non disponible. Fichier de cycles non généré.")
    except Exception as e:
        logger.warning(f"Erreur lors de la génération du fichier de cycles: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
    
    logger.info("")
    
    # Décider si on entraîne
    should_train = False
    
    if args.force:
        logger.info("Mode FORCE activé - Entraînement forcé")
        should_train = True
    elif new_draws_found:
        logger.info("Nouveaux tirages détectés - Entraînement recommandé")
        should_train = True
    else:
        logger.info("Aucun nouveau tirage - Entraînement non nécessaire")
        logger.info("Utilisez --force pour forcer l'entraînement")
        should_train = False
    
    if should_train:
        logger.info("")
        logger.info("=" * 80)
        logger.info("Démarrage de l'entraînement...")
        logger.info("=" * 80)
        logger.info("")
        
        try:
            # Vérifier que l'encodeur avancé est disponible
            script_dir = Path(__file__).parent / "script"
            sys.path.insert(0, str(script_dir))
            
            try:
                from advanced_encoder import AdvancedEuromillionsEncoder
                logger.info("✅ Encodeur avancé disponible - Amélioration de la précision activée")
            except ImportError:
                logger.warning("⚠️ Encodeur avancé non disponible - Utilisation des features de base")
            
            # 🎥 NOUVEAU: Charger ou extraire les embeddings vidéo automatiquement
            logger.info("DEBUG: Avant section video - ligne 180")
            logger.info("")
            logger.info("=" * 80)
            logger.info("GESTION DES EMBEDDINGS VIDEO")
            logger.info("=" * 80)
            logger.info("Verification de la disponibilite des modules video...")
            logger.info(f"DEBUG: Chemin actuel: {Path.cwd()}")
            logger.info(f"DEBUG: Repertoire script: {script_dir}")
            logger.info("DEBUG: Code video execute - ligne 188")
            
            video_embeddings = None
            try:
                # Path est déjà importé en haut du fichier, pas besoin de réimporter
                from video_embeddings_loader import load_video_embeddings
                logger.info("Module video_embeddings_loader importe avec succes")
                
                encoded_videos_dir = Path("encoded_videos")
                
                # Vérifier si le répertoire existe et contient des embeddings
                embeddings_exist = encoded_videos_dir.exists() and len(list(encoded_videos_dir.glob("*_embedding.pkl"))) > 0
                
                logger.info(f"Verification du repertoire encoded_videos: {encoded_videos_dir.absolute()}")
                logger.info(f"Repertoire existe: {encoded_videos_dir.exists()}")
                
                if embeddings_exist:
                    # Charger les embeddings existants
                    logger.info("Embeddings video detectes - Chargement...")
                    video_embeddings = load_video_embeddings("encoded_videos")
                    
                    if video_embeddings and len(video_embeddings) > 0:
                        logger.info(f"{len(video_embeddings)} embeddings video charges avec succes")
                        logger.info(f"Exemples: {list(video_embeddings.keys())[:3]}")
                    else:
                        logger.warning("Aucun embedding valide trouve")
                        video_embeddings = None
                else:
                    # Extraire automatiquement les embeddings depuis les vidéos
                    logger.info("Aucun embedding video trouve")
                    logger.info("Lancement de l'extraction automatique des embeddings video...")
                    
                    try:
                        from video_deep_analyzer import VideoDeepAnalyzer
                        
                        # Vérifier si des vidéos existent
                        video_dir = Path("tirage_videos")
                        if video_dir.exists():
                            video_files = list(video_dir.glob("*.webm")) + list(video_dir.glob("*.mkv")) + list(video_dir.glob("*.mp4"))
                            
                            if len(video_files) > 0:
                                logger.info(f"{len(video_files)} videos trouvees dans {video_dir}")
                                logger.info("Extraction des embeddings (cela peut prendre du temps)...")
                                
                                # Initialiser l'analyseur vidéo
                                video_analyzer = VideoDeepAnalyzer(
                                    video_dir=str(video_dir),
                                    output_dir="encoded_videos",
                                    frame_interval=30,  # Extraire 1 frame toutes les 30 frames
                                    use_gpu=False,  # Utiliser CPU par défaut
                                    cnn_model="resnet50"
                                )
                                
                                # Traiter toutes les vidéos
                                embeddings_dict = video_analyzer.process_all_videos(force_reprocess=False)
                                
                                if embeddings_dict and len(embeddings_dict) > 0:
                                    logger.info(f"{len(embeddings_dict)} embeddings video extraits avec succes")
                                    
                                    # Convertir le format pour compatibilité avec load_video_embeddings
                                    video_embeddings = embeddings_dict
                                    logger.info(f"Exemples: {list(video_embeddings.keys())[:3]}")
                                else:
                                    logger.warning("Aucun embedding extrait - Verifiez les videos")
                                    video_embeddings = None
                            else:
                                logger.warning(f"Aucune video trouvee dans {video_dir}")
                                logger.info("   Le systeme fonctionnera sans features video")
                                video_embeddings = None
                        else:
                            logger.warning(f"Repertoire video {video_dir} n'existe pas")
                            logger.info("   Le systeme fonctionnera sans features video")
                            video_embeddings = None
                            
                    except ImportError as e:
                        logger.warning(f"Module video_deep_analyzer non disponible: {str(e)}")
                        logger.info("   Le systeme fonctionnera sans features video")
                        import traceback
                        logger.error(traceback.format_exc())
                        video_embeddings = None
                    except Exception as e:
                        logger.error(f"ERREUR lors de l'extraction des embeddings video: {str(e)}")
                        import traceback
                        logger.error(traceback.format_exc())
                        logger.info("   Le systeme continuera sans features video")
                        video_embeddings = None
                        
            except ImportError as e:
                logger.error(f"ERREUR Import: Module video_embeddings_loader non disponible: {str(e)}")
                logger.warning("   Features video desactivees")
                import traceback
                logger.error(traceback.format_exc())
                video_embeddings = None
            except Exception as e:
                logger.error(f"ERREUR lors de la gestion des embeddings video: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                logger.warning("   Le systeme continuera sans features video")
                video_embeddings = None
            
            # Log final pour vérifier que les embeddings sont bien passés
            if video_embeddings:
                logger.info(f"RESUME: {len(video_embeddings)} embeddings video passes a l'entraineur")
            else:
                logger.warning("RESUME: Aucun embedding video - L'entrainement continuera sans features video")
            
            logger.info("=" * 80)
            logger.info("")
            
            from euromillions_train import EuromillionsTrainer
            
            config = {
                "csv_file": args.csv,
                "output_dir": args.output,
                "model_dir": args.model_dir,
                "video_embeddings": video_embeddings  # 🎥 NOUVEAU: Passer les embeddings
            }
            
            logger.info("Initialisation de l'entraîneur avec toutes les logiques...")
            trainer = EuromillionsTrainer(config)
            
            if args.method == "all":
                results = trainer.train_all()
            elif args.method == "main":
                results = {"main": trainer.train_main_analyzer()}
            elif args.method == "fibonacci":
                results = {"fibonacci": trainer.train_fibonacci_analyzer()}
            else:
                results = trainer.train_all()
            
            logger.info("")
            logger.info("=" * 80)
            logger.info("✅ Entraînement terminé avec succès")
            logger.info("=" * 80)
            logger.info(f"Résultats: {results}")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'entraînement: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return 1
    else:
        logger.info("")
        logger.info("Entraînement non effectué")
        logger.info("Pour forcer l'entraînement: python check_and_train.py --force")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
