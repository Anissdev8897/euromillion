#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Système de logs centralisé pour EuroMillions
Ce module implémente un système de logs centralisé pour tous les scripts
de prédiction EuroMillions.
"""

import os
import sys
import logging
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import json
import atexit
import threading
import queue
import time

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("EuromillionsLogSystem")

class EuromillionsLogSystem:
    """Système de logs centralisé pour les scripts EuroMillions."""
    
    def __init__(self, log_dir: str = "logs", max_log_files: int = 10, 
                 rotation_size_mb: float = 10.0, async_logging: bool = True):
        """
        Initialise le système de logs.
        
        Args:
            log_dir: Répertoire pour les fichiers de log
            max_log_files: Nombre maximum de fichiers de log à conserver
            rotation_size_mb: Taille maximale d'un fichier de log en Mo avant rotation
            async_logging: Utiliser la journalisation asynchrone pour de meilleures performances
        """
        self.log_dir = Path(log_dir)
        self.max_log_files = max_log_files
        self.rotation_size_mb = rotation_size_mb
        self.async_logging = async_logging
        
        # Créer le répertoire de logs s'il n'existe pas
        if not self.log_dir.exists():
            self.log_dir.mkdir(parents=True)
            logger.info(f"Répertoire de logs créé: {self.log_dir}")
        
        # Configurer le fichier de log principal
        self.main_log_file = self.log_dir / f"euromillions_main_{datetime.now().strftime('%Y%m%d')}.log"
        
        # Configurer le gestionnaire de fichiers
        self.file_handler = logging.FileHandler(self.main_log_file)
        self.file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        
        # Configurer le logger racine
        self.root_logger = logging.getLogger()
        self.root_logger.addHandler(self.file_handler)
        
        # Configurer les loggers spécifiques
        self.loggers = {}
        
        # Nettoyer les anciens fichiers de log
        self._cleanup_old_logs()
        
        # Configuration pour la journalisation asynchrone
        if self.async_logging:
            self.log_queue = queue.Queue()
            self.stop_event = threading.Event()
            self.log_thread = threading.Thread(target=self._async_log_worker)
            self.log_thread.daemon = True
            self.log_thread.start()
            
            # S'assurer que le thread de journalisation est arrêté proprement
            atexit.register(self.shutdown)
        
        logger.info("Système de logs EuroMillions initialisé")
    
    def _cleanup_old_logs(self):
        """Nettoie les anciens fichiers de log si leur nombre dépasse max_log_files."""
        log_files = sorted(self.log_dir.glob("euromillions_*.log"), key=os.path.getmtime)
        if len(log_files) > self.max_log_files:
            for old_file in log_files[:-self.max_log_files]:
                try:
                    old_file.unlink()
                    logger.info(f"Ancien fichier de log supprimé: {old_file}")
                except Exception as e:
                    logger.warning(f"Impossible de supprimer l'ancien fichier de log {old_file}: {str(e)}")
    
    def _check_rotation(self):
        """Vérifie si une rotation des logs est nécessaire."""
        if self.main_log_file.exists():
            size_mb = self.main_log_file.stat().st_size / (1024 * 1024)
            if size_mb >= self.rotation_size_mb:
                # Créer un nouveau fichier de log
                old_file = self.main_log_file
                self.main_log_file = self.log_dir / f"euromillions_main_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
                
                # Fermer et remplacer le gestionnaire de fichiers
                self.root_logger.removeHandler(self.file_handler)
                self.file_handler.close()
                
                self.file_handler = logging.FileHandler(self.main_log_file)
                self.file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
                self.root_logger.addHandler(self.file_handler)
                
                logger.info(f"Rotation des logs effectuée: {old_file} -> {self.main_log_file}")
                
                # Nettoyer les anciens fichiers de log
                self._cleanup_old_logs()
    
    def _async_log_worker(self):
        """Thread de travail pour la journalisation asynchrone."""
        while not self.stop_event.is_set():
            try:
                # Récupérer un message de log de la file d'attente (avec timeout pour vérifier stop_event)
                try:
                    log_record = self.log_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                # Vérifier si une rotation est nécessaire
                self._check_rotation()
                
                # Écrire le message de log dans le fichier
                with open(self.main_log_file, 'a', encoding='utf-8') as f:
                    f.write(log_record + '\n')
                
                # Marquer la tâche comme terminée
                self.log_queue.task_done()
            except Exception as e:
                print(f"Erreur dans le thread de journalisation: {str(e)}")
    
    def shutdown(self):
        """Arrête proprement le système de logs."""
        if self.async_logging:
            # Signaler au thread de s'arrêter
            self.stop_event.set()
            
            # Attendre que le thread se termine
            if self.log_thread.is_alive():
                self.log_thread.join(timeout=2.0)
            
            # Vider la file d'attente
            while not self.log_queue.empty():
                try:
                    log_record = self.log_queue.get_nowait()
                    with open(self.main_log_file, 'a', encoding='utf-8') as f:
                        f.write(log_record + '\n')
                    self.log_queue.task_done()
                except queue.Empty:
                    break
        
        # Fermer le gestionnaire de fichiers
        self.file_handler.close()
        logger.info("Système de logs EuroMillions arrêté")
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        Obtient un logger configuré pour un module spécifique.
        
        Args:
            name: Nom du module
            
        Returns:
            logging.Logger: Logger configuré
        """
        if name in self.loggers:
            return self.loggers[name]
        
        # Créer un nouveau logger
        module_logger = logging.getLogger(name)
        
        # Configurer le logger
        module_logger.setLevel(logging.INFO)
        
        # Ajouter un gestionnaire de fichiers spécifique au module
        module_log_file = self.log_dir / f"euromillions_{name}_{datetime.now().strftime('%Y%m%d')}.log"
        module_handler = logging.FileHandler(module_log_file)
        module_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        module_logger.addHandler(module_handler)
        
        # Stocker le logger
        self.loggers[name] = module_logger
        
        return module_logger
    
    def log(self, level: int, message: str, module: str = "main"):
        """
        Journalise un message.
        
        Args:
            level: Niveau de logging
            message: Message à journaliser
            module: Nom du module
        """
        if self.async_logging:
            # Formater le message de log
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
            level_name = logging.getLevelName(level)
            log_record = f"{timestamp} - {module} - {level_name} - {message}"
            
            # Ajouter le message à la file d'attente
            self.log_queue.put(log_record)
        else:
            # Journaliser directement
            logger = self.get_logger(module)
            logger.log(level, message)
            
            # Vérifier si une rotation est nécessaire
            self._check_rotation()
    
    def log_exception(self, e: Exception, context: str = "", module: str = "main"):
        """
        Journalise une exception.
        
        Args:
            e: L'exception à journaliser
            context: Contexte dans lequel l'exception s'est produite
            module: Nom du module
        """
        error_info = {
            "timestamp": datetime.now().isoformat(),
            "type": type(e).__name__,
            "message": str(e),
            "context": context,
            "traceback": traceback.format_exc()
        }
        
        # Journaliser l'erreur
        error_message = f"Exception dans {context}: {type(e).__name__}: {str(e)}"
        self.log(logging.ERROR, error_message, module)
        self.log(logging.DEBUG, traceback.format_exc(), module)
        
        # Sauvegarder les détails de l'erreur dans un fichier JSON
        error_file = self.log_dir / f"error_details_{module}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(error_file, 'w', encoding='utf-8') as f:
                json.dump(error_info, f, indent=2)
            self.log(logging.INFO, f"Détails de l'erreur sauvegardés dans {error_file}", module)
        except Exception as save_error:
            self.log(logging.WARNING, f"Impossible de sauvegarder les détails de l'erreur: {str(save_error)}", module)
    
    def log_performance(self, operation: str, execution_time: float, details: Dict = None, module: str = "main"):
        """
        Journalise des informations de performance.
        
        Args:
            operation: Nom de l'opération
            execution_time: Temps d'exécution en secondes
            details: Détails supplémentaires
            module: Nom du module
        """
        performance_info = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "execution_time": execution_time,
            "details": details or {}
        }
        
        # Journaliser les informations de performance
        self.log(logging.INFO, f"Performance de {operation}: {execution_time:.4f}s", module)
        
        # Sauvegarder les détails de performance dans un fichier JSON
        performance_file = self.log_dir / f"performance_{module}.json"
        try:
            # Charger les données existantes
            if performance_file.exists():
                with open(performance_file, 'r', encoding='utf-8') as f:
                    try:
                        performance_data = json.load(f)
                    except json.JSONDecodeError:
                        performance_data = {"entries": []}
            else:
                performance_data = {"entries": []}
            
            # Ajouter la nouvelle entrée
            performance_data["entries"].append(performance_info)
            
            # Limiter le nombre d'entrées
            if len(performance_data["entries"]) > 100:
                performance_data["entries"] = performance_data["entries"][-100:]
            
            # Sauvegarder les données
            with open(performance_file, 'w', encoding='utf-8') as f:
                json.dump(performance_data, f, indent=2)
        except Exception as save_error:
            self.log(logging.WARNING, f"Impossible de sauvegarder les détails de performance: {str(save_error)}", module)

# Fonction pour créer un système de logs global
_log_system = None

def get_log_system(log_dir: str = "logs", max_log_files: int = 10, 
                  rotation_size_mb: float = 10.0, async_logging: bool = True) -> EuromillionsLogSystem:
    """
    Obtient le système de logs global.
    
    Args:
        log_dir: Répertoire pour les fichiers de log
        max_log_files: Nombre maximum de fichiers de log à conserver
        rotation_size_mb: Taille maximale d'un fichier de log en Mo avant rotation
        async_logging: Utiliser la journalisation asynchrone pour de meilleures performances
        
    Returns:
        EuromillionsLogSystem: Système de logs global
    """
    global _log_system
    if _log_system is None:
        _log_system = EuromillionsLogSystem(log_dir, max_log_files, rotation_size_mb, async_logging)
    return _log_system

# Exemple d'utilisation
def main():
    """Fonction principale pour tester le module."""
    try:
        # Créer le système de logs
        log_system = get_log_system()
        
        # Obtenir un logger pour un module spécifique
        predictor_logger = log_system.get_logger("predictor")
        predictor_logger.info("Test du logger pour le module predictor")
        
        # Journaliser des messages
        log_system.log(logging.INFO, "Test du système de logs", "main")
        log_system.log(logging.WARNING, "Attention, ceci est un avertissement", "main")
        log_system.log(logging.ERROR, "Erreur de test", "main")
        
        # Journaliser une exception
        try:
            result = 1 / 0
        except Exception as e:
            log_system.log_exception(e, "Division par zéro", "main")
        
        # Journaliser des informations de performance
        import time
        start_time = time.time()
        time.sleep(0.5)
        execution_time = time.time() - start_time
        log_system.log_performance("opération de test", execution_time, {"iterations": 1}, "main")
        
        # Tester la journalisation asynchrone
        for i in range(100):
            log_system.log(logging.INFO, f"Message de test {i}", "async_test")
        
        # Attendre que tous les messages soient traités
        if log_system.async_logging:
            log_system.log_queue.join()
        
        print("Test du système de logs terminé avec succès.")
    
    except Exception as e:
        print(f"Erreur lors du test du système de logs: {str(e)}")
        traceback.print_exc()
    finally:
        # Arrêter proprement le système de logs
        if _log_system is not None:
            _log_system.shutdown()

if __name__ == "__main__":
    main()
