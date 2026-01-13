#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Serveur API Flask pour les prédictions EuroMillions
Expose une API REST pour lancer les prédictions depuis l'interface web
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, Response
import traceback

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("EuromillionsAPI")

# Gestion de flask-cors optionnel
try:
    from flask_cors import CORS
    CORS_AVAILABLE = True
except ImportError:
    CORS_AVAILABLE = False
    logger.warning("flask-cors non disponible. Les requêtes CORS peuvent échouer.")

# Ajouter le répertoire script au path
script_dir = Path(__file__).parent / "script"
sys.path.insert(0, str(script_dir))

# Ajouter aussi le répertoire parent pour les imports
parent_dir = Path(__file__).parent
sys.path.insert(0, str(parent_dir))

app = Flask(__name__)
if CORS_AVAILABLE:
    CORS(app)  # Autoriser les requêtes cross-origin depuis l'interface HTML
else:
    # Solution de secours pour CORS
    @app.after_request
    def after_request(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response

# Configuration
# ⚠️ CRITIQUE : Type de jeu (euromillions ou loto) - peut être défini via variable d'environnement
GAME_TYPE = os.environ.get("GAME_TYPE", "euromillions").lower()  # Par défaut: euromillions

# Configuration des fichiers selon le type de jeu
CSV_FILE = Path(__file__).parent / f"tirage_{GAME_TYPE}_complet.csv"
OUTPUT_DIR = Path(__file__).parent / f"resultats_{GAME_TYPE}"
MODEL_DIR = Path(__file__).parent / f"models_{GAME_TYPE}"
SERVER_IP = os.environ.get("SERVER_IP", "107.189.17.46")  # IP du serveur (configurable)

# Configuration de l'entraînement
# Mettre à False pour désactiver l'entraînement automatique (entraînement sur PC local uniquement)
ENABLE_AUTO_TRAINING = os.environ.get("ENABLE_AUTO_TRAINING", "False").lower() == "true"  # Désactivé par défaut
ENABLE_AUTO_UPDATE = os.environ.get("ENABLE_AUTO_UPDATE", "True").lower() == "true"  # Activé par défaut

# Créer les répertoires nécessaires
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)


@app.route('/')
def index():
    """Route racine - sert l'interface HTML"""
    html_file = Path(__file__).parent / "euromillion.html"
    logger.info(f"Tentative de chargement du fichier HTML: {html_file}")
    logger.info(f"Fichier existe: {html_file.exists()}")
    
    if html_file.exists():
        try:
            # Lire le fichier et le renvoyer avec le bon Content-Type
            with open(html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            return Response(
                html_content,
                mimetype='text/html; charset=utf-8',
                headers={'Content-Type': 'text/html; charset=utf-8'}
            )
        except Exception as e:
            logger.error(f"Erreur lors du chargement du fichier HTML: {str(e)}")
            logger.error(traceback.format_exc())
            return f"<html><body><h1>Erreur</h1><p>Erreur lors du chargement: {str(e)}</p></body></html>", 500
    else:
        logger.error(f"Fichier HTML non trouvé: {html_file}")
        return jsonify({
            "status": "error",
            "message": f"Fichier HTML non trouvé: {html_file}",
            "endpoints": {
                "/api/predict": "POST - Générer des prédictions",
                "/api/methods": "GET - Liste des méthodes disponibles",
                "/api/status": "GET - Statut de l'API"
            }
        }), 404


@app.route('/euromillion.html')
def euromillion_html():
    """Route pour servir directement le fichier HTML"""
    html_file = Path(__file__).parent / "euromillion.html"
    logger.info(f"Route /euromillion.html - Fichier: {html_file}")
    
    if html_file.exists():
        try:
            # Lire le fichier et le renvoyer avec le bon Content-Type
            with open(html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            return Response(
                html_content,
                mimetype='text/html; charset=utf-8',
                headers={'Content-Type': 'text/html; charset=utf-8'}
            )
        except Exception as e:
            logger.error(f"Erreur lors du chargement du fichier HTML: {str(e)}")
            logger.error(traceback.format_exc())
            return f"<html><body><h1>Erreur</h1><p>Erreur lors du chargement: {str(e)}</p></body></html>", 500
    else:
        logger.error(f"Fichier HTML non trouvé: {html_file}")
        return jsonify({
            "status": "error",
            "message": f"Fichier euromillion.html non trouvé: {html_file}"
        }), 404


@app.route('/api/test', methods=['GET'])
@app.route('/euromillions/api/test', methods=['GET'])
def test():
    """Route de test pour vérifier que l'API fonctionne"""
    return jsonify({
        "status": "success",
        "message": "API EuroMillions fonctionnelle",
        "server_ip": SERVER_IP,
        "path": request.path,
        "timestamp": datetime.now().isoformat()
    })


@app.route('/api/status', methods=['GET'])
@app.route('/euromillions/api/status', methods=['GET'])
def status():
    """Vérifier le statut de l'API et des fichiers nécessaires"""
    try:
        csv_exists = CSV_FILE.exists()
        csv_size = CSV_FILE.stat().st_size if csv_exists else 0
        
        return jsonify({
            "status": "success",
            "api": "running",
            "server_ip": SERVER_IP,
            "csv_file": {
                "path": str(CSV_FILE),
                "exists": csv_exists,
                "size": csv_size
            },
            "output_dir": str(OUTPUT_DIR),
            "model_dir": str(MODEL_DIR)
        })
    except Exception as e:
        logger.error(f"Erreur lors de la vérification du statut: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route('/api/methods', methods=['GET'])
@app.route('/euromillions/api/methods', methods=['GET'])
def get_methods():
    """Obtenir la liste des méthodes de prédiction disponibles"""
    methods = [
        {
            "id": "all",
            "name": "Toutes les méthodes",
            "description": "Combine toutes les méthodes disponibles pour une prédiction optimale"
        },
        {
            "id": "main",
            "name": "Analyseur principal",
            "description": "Utilise l'analyseur principal avec ML et statistiques avancées"
        },
        {
            "id": "fibonacci",
            "name": "Analyse Fibonacci",
            "description": "Prédictions basées sur la série de Fibonacci inversée"
        },
        {
            "id": "super",
            "name": "Super-optimiseur",
            "description": "Combinaison optimisée de plusieurs méthodes"
        },
        {
            "id": "fallback",
            "name": "Aléatoire",
            "description": "Génération aléatoire de combinaisons"
        }
    ]
    
    return jsonify({
        "status": "success",
        "methods": methods
    })


@app.route('/api/predict', methods=['POST'])
@app.route('/euromillions/api/predict', methods=['POST'])
def predict():
    """Générer des prédictions EuroMillions"""
    try:
        # Récupérer les paramètres de la requête
        data = request.get_json() or {}
        method = data.get('method', 'all')
        num_combinations = data.get('combinations', 5)
        
        # Validation
        if not isinstance(num_combinations, int) or num_combinations < 1 or num_combinations > 20:
            return jsonify({
                "status": "error",
                "message": "Le nombre de combinaisons doit être entre 1 et 20"
            }), 400
        
        valid_methods = ['all', 'main', 'fibonacci', 'super', 'fallback']
        if method not in valid_methods:
            return jsonify({
                "status": "error",
                "message": f"Méthode invalide. Méthodes disponibles: {', '.join(valid_methods)}"
            }), 400
        
        # Vérifier que le fichier CSV existe
        if not CSV_FILE.exists():
            return jsonify({
                "status": "error",
                "message": f"Fichier CSV non trouvé: {CSV_FILE}"
            }), 404
        
        logger.info(f"Génération de {num_combinations} combinaisons avec la méthode '{method}'")
        
        # Importer et initialiser le prédicteur
        try:
            try:
                from script.euromillions_predict import EuromillionsPredictor
            except ImportError:
                from euromillions_predict import EuromillionsPredictor  # type: ignore
        except ImportError:
            logger.error("Impossible d'importer euromillions_predict")
            return jsonify({
                "status": "error",
                "message": "Module de prédiction non disponible"
            }), 500
        
        # Configuration avec système quantique activé par défaut
        config = {
            "use_quantum": True,  # ⚠️ CRITIQUE : Activer le système quantique pour les utilisateurs
            "use_qnn": True,
            "use_qlstm": True,
            "use_quantum_annealing": True,
            "csv_file": str(CSV_FILE),
            "output_dir": str(OUTPUT_DIR),
            "model_dir": str(MODEL_DIR),
            "method": method,
            "combinations": num_combinations,
            "save_json": True
        }
        
        # Créer le prédicteur
        predictor = EuromillionsPredictor(config)
        
        # Générer les prédictions
        combinations = predictor.generate_predictions()
        
        if not combinations:
            return jsonify({
                "status": "error",
                "message": "Aucune combinaison générée"
            }), 500
        
        # Formater les résultats
        formatted_combinations = []
        for i, combo in enumerate(combinations, 1):
            if len(combo) >= 7:
                formatted_combinations.append({
                    "id": i,
                    "numbers": combo[:5],
                    "stars": combo[5:7],
                    "display": f"{' - '.join(map(str, combo[:5]))} | ⭐ {' - '.join(map(str, combo[5:7]))}"
                })
            elif len(combo) >= 5:
                # Format alternatif si seulement les numéros sont présents
                formatted_combinations.append({
                    "id": i,
                    "numbers": combo[:5],
                    "stars": [],
                    "display": f"{' - '.join(map(str, combo[:5]))} | ⭐ Non disponibles"
                })
        
        # Sauvegarder les prédictions
        predictor.save_predictions()
        predictor.save_predictions_json()
        
        return jsonify({
            "status": "success",
            "method": method,
            "combinations": formatted_combinations,
            "count": len(formatted_combinations),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error",
            "message": f"Erreur lors de la génération des prédictions: {str(e)}"
        }), 500


@app.route('/api/predict/simple', methods=['POST'])
@app.route('/euromillions/api/predict/simple', methods=['POST'])
def predict_simple():
    """Générer des prédictions simples (méthode rapide)"""
    try:
        data = request.get_json() or {}
        num_combinations = data.get('combinations', 5)
        
        if not isinstance(num_combinations, int) or num_combinations < 1 or num_combinations > 20:
            return jsonify({
                "status": "error",
                "message": "Le nombre de combinaisons doit être entre 1 et 20"
            }), 400
        
        logger.info(f"Génération rapide de {num_combinations} combinaisons")
        
        # Utiliser la méthode fallback pour une génération rapide
        try:
            try:
                from script.euromillions_predict import EuromillionsPredictor
            except ImportError:
                from euromillions_predict import EuromillionsPredictor  # type: ignore
        except ImportError:
            return jsonify({
                "status": "error",
                "message": "Module de prédiction non disponible"
            }), 500
        
        config = {
            "csv_file": str(CSV_FILE),
            "output_dir": str(OUTPUT_DIR),
            "model_dir": str(MODEL_DIR),
            "method": "fallback",
            "combinations": num_combinations,
            "save_json": False,
            "use_quantum": True,  # ⚠️ CRITIQUE : Activer le système quantique pour les utilisateurs
            "use_qnn": True,
            "use_qlstm": True,
            "use_quantum_annealing": True,
        }
        
        predictor = EuromillionsPredictor(config)
        combinations = predictor.generate_predictions()
        
        formatted_combinations = []
        for i, combo in enumerate(combinations, 1):
            if len(combo) >= 7:
                formatted_combinations.append({
                    "id": i,
                    "numbers": combo[:5],
                    "stars": combo[5:7],
                    "display": f"{' - '.join(map(str, combo[:5]))} | ⭐ {' - '.join(map(str, combo[5:7]))}"
                })
        
        return jsonify({
            "status": "success",
            "method": "simple",
            "combinations": formatted_combinations,
            "count": len(formatted_combinations),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction simple: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.errorhandler(404)
def not_found(error):
    """Gestion des erreurs 404"""
    # Si la requête demande du JSON, renvoyer du JSON
    if request.is_json or request.path.startswith('/api/'):
        return jsonify({
            "status": "error",
            "message": f"Endpoint non trouvé: {request.path}",
            "available_endpoints": [
                "/api/predict",
                "/api/predict/simple",
                "/api/status",
                "/api/methods"
            ]
        }), 404
    # Sinon, renvoyer la page HTML par défaut
    html_file = Path(__file__).parent / "euromillion.html"
    if html_file.exists():
        try:
            with open(html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            return Response(
                html_content,
                mimetype='text/html; charset=utf-8',
                headers={'Content-Type': 'text/html; charset=utf-8'}
            )
        except:
            pass
    return jsonify({
        "status": "error",
        "message": "Endpoint non trouvé"
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Gestion des erreurs 500"""
    return jsonify({
        "status": "error",
        "message": "Erreur interne du serveur"
    }), 500


def initialize_system():
    """Initialise le système avant le démarrage du serveur"""
    logger.info("=" * 80)
    logger.info("Initialisation du système EuroMillions")
    logger.info("=" * 80)
    
    try:
        # Importer les modules nécessaires depuis le répertoire script
        script_dir = Path(__file__).parent / "script"
        sys.path.insert(0, str(script_dir))
        
        try:
            from script.auto_updater import AutoUpdater
            from script.fdj_scraper import FDJEuromillionsScraper
        except ImportError:
            from auto_updater import AutoUpdater  # type: ignore
            from fdj_scraper import FDJEuromillionsScraper  # type: ignore
        
        # Étape 1: Vérifier et mettre à jour les tirages
        logger.info("Étape 1: Vérification des tirages...")
        scraper = FDJEuromillionsScraper(str(CSV_FILE))
        
        # Récupérer le dernier tirage disponible
        updated = scraper.update_draws()
        if updated:
            logger.info("Nouveau tirage ajouté au fichier CSV")
        else:
            logger.info("Aucun nouveau tirage à ajouter")
        
        # Étape 2: Réentraîner les modèles avec tous les tirages (optionnel)
        if ENABLE_AUTO_TRAINING:
            logger.info("Étape 2: Réentraînement des modèles avec tous les tirages...")
            
            try:
                # Import depuis le répertoire script
                # Essayer d'abord avec le chemin absolu (pour l'IDE)
                try:
                    from script.euromillions_train import EuromillionsTrainer
                except ImportError:
                    # Fallback: import direct (si script_dir est dans sys.path)
                    from euromillions_train import EuromillionsTrainer  # type: ignore
                
                config = {
                    "csv_file": str(CSV_FILE),
                    "output_dir": str(OUTPUT_DIR),
                    "model_dir": str(MODEL_DIR)
                }
                
                trainer = EuromillionsTrainer(config)
                results = trainer.train_all()
                
                logger.info("Réentraînement terminé avec succès")
                logger.info(f"Résultats: {results}")
                
            except Exception as e:
                logger.warning(f"Erreur lors du réentraînement (non bloquant): {str(e)}")
                logger.warning("Le serveur continuera de fonctionner avec les modèles existants")
        else:
            logger.info("Étape 2: Réentraînement désactivé (entraînement sur PC local uniquement)")
            logger.info("Pour entraîner les modèles, utilisez: python script/euromillions_train.py")
        
        # Étape 3: Démarrer le système de mise à jour automatique (optionnel)
        if ENABLE_AUTO_UPDATE:
            logger.info("Étape 3: Démarrage du système de mise à jour automatique...")
            
            # Créer l'updater avec réentraînement désactivé (entraînement sur PC local uniquement)
            updater = AutoUpdater(
                csv_file=str(CSV_FILE),
                output_dir=str(OUTPUT_DIR),
                model_dir=str(MODEL_DIR),
                enable_training=False  # Désactivé - entraînement sur PC local uniquement
            )
            
            # Démarrer le scheduler en arrière-plan (sans réentraînement automatique)
            updater.start(run_immediately=False)
            
            logger.info("Système de mise à jour automatique démarré")
            logger.info("  - Mises à jour planifiées: Mardis et Vendredis à 22h00")
            logger.info("  - Réentraînement: Désactivé (à faire sur PC local)")
        else:
            logger.info("Étape 3: Mise à jour automatique désactivée")
        
        # Vérifier que l'encodeur avancé est disponible
        try:
            try:
                from script.advanced_encoder import AdvancedEuromillionsEncoder
            except ImportError:
                from advanced_encoder import AdvancedEuromillionsEncoder  # type: ignore
            logger.info("✅ Encodeur avancé disponible - Toutes les logiques intégrées")
            logger.info("   - Features temporelles, numériques et de séquence")
            logger.info("   - Système de réflexion IA activé")
            logger.info("   - Amélioration continue des prédictions")
        except ImportError:
            logger.warning("⚠️  Encodeur avancé non disponible - Utilisation des features de base")
        
        logger.info("=" * 80)
        logger.info("Initialisation terminée")
        logger.info("=" * 80)
        logger.info("✅ Toutes les logiques sont intégrées pour les futures prédictions:")
        logger.info("   - Encodeur avancé avec réflexion IA")
        logger.info("   - Mise à jour automatique des tirages")
        logger.info("   - Toutes les méthodes de prédiction disponibles")
        logger.info("   - Système de récompense pour amélioration continue")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation (non bloquant): {str(e)}")
        logger.error(traceback.format_exc())
        logger.warning("Le serveur continuera de fonctionner sans les fonctionnalités avancées")


if __name__ == '__main__':
    import argparse
    
    # Parser les arguments de ligne de commande
    parser = argparse.ArgumentParser(description='Serveur API EuroMillions/Loto')
    parser.add_argument('--host', default=os.environ.get('HOST', '0.0.0.0'), help='Adresse IP du serveur')
    parser.add_argument('--port', type=int, default=int(os.environ.get('PORT', 5002)), help='Port du serveur')
    parser.add_argument('--debug', action='store_true', default=os.environ.get('FLASK_ENV', 'production') == 'development', help='Mode debug')
    args = parser.parse_args()
    
    logger.info(f"Démarrage du serveur API {GAME_TYPE.upper()}...")
    logger.info(f"Type de jeu: {GAME_TYPE}")
    logger.info(f"Fichier CSV: {CSV_FILE}")
    logger.info(f"Répertoire de sortie: {OUTPUT_DIR}")
    
    # Vérifier que le fichier CSV existe
    if not CSV_FILE.exists():
        logger.warning(f"ATTENTION: Le fichier CSV {CSV_FILE} n'existe pas!")
        logger.warning("Certaines fonctionnalités peuvent ne pas fonctionner.")
    
    # Initialiser le système (mise à jour et réentraînement)
    initialize_system()
    
    # Démarrer le serveur
    logger.info("Démarrage du serveur Flask...")
    logger.info(f"IP du serveur: {SERVER_IP}")
    logger.info(f"Le serveur sera accessible sur: http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)

