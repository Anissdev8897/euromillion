#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module de gestion des erreurs imprévues pour EuroMillions
Ce module implémente un système robuste de gestion des erreurs pour
les scripts de prédiction EuroMillions.
"""

import os
import sys
import logging
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Callable
import functools
import inspect
import json

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("EuromillionsErrorHandler")

class EuromillionsErrorHandler:
    """Gestionnaire d'erreurs pour les scripts EuroMillions."""
    
    def __init__(self, log_dir: str = "logs", max_log_files: int = 10):
        """
        Initialise le gestionnaire d'erreurs.
        
        Args:
            log_dir: Répertoire pour les fichiers de log
            max_log_files: Nombre maximum de fichiers de log à conserver
        """
        self.log_dir = Path(log_dir)
        self.max_log_files = max_log_files
        
        # Créer le répertoire de logs s'il n'existe pas
        if not self.log_dir.exists():
            self.log_dir.mkdir(parents=True)
            logger.info(f"Répertoire de logs créé: {self.log_dir}")
        
        # Configurer le fichier de log
        self.log_file = self.log_dir / f"euromillions_errors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.file_handler = logging.FileHandler(self.log_file)
        self.file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(self.file_handler)
        
        # Nettoyer les anciens fichiers de log
        self._cleanup_old_logs()
        
        logger.info("Gestionnaire d'erreurs EuroMillions initialisé")
    
    def _cleanup_old_logs(self):
        """Nettoie les anciens fichiers de log si leur nombre dépasse max_log_files."""
        log_files = sorted(self.log_dir.glob("euromillions_errors_*.log"), key=os.path.getmtime)
        if len(log_files) > self.max_log_files:
            for old_file in log_files[:-self.max_log_files]:
                try:
                    old_file.unlink()
                    logger.info(f"Ancien fichier de log supprimé: {old_file}")
                except Exception as e:
                    logger.warning(f"Impossible de supprimer l'ancien fichier de log {old_file}: {str(e)}")
    
    def handle_exception(self, e: Exception, context: str = ""):
        """
        Gère une exception en la journalisant et en fournissant des informations de débogage.
        
        Args:
            e: L'exception à gérer
            context: Contexte dans lequel l'exception s'est produite
            
        Returns:
            Dict: Informations sur l'erreur
        """
        error_info = {
            "timestamp": datetime.now().isoformat(),
            "type": type(e).__name__,
            "message": str(e),
            "context": context,
            "traceback": traceback.format_exc()
        }
        
        # Journaliser l'erreur
        logger.error(f"Exception dans {context}: {type(e).__name__}: {str(e)}")
        logger.debug(traceback.format_exc())
        
        # Sauvegarder les détails de l'erreur dans un fichier JSON
        error_file = self.log_dir / f"error_details_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(error_file, 'w', encoding='utf-8') as f:
                json.dump(error_info, f, indent=2)
            logger.info(f"Détails de l'erreur sauvegardés dans {error_file}")
        except Exception as save_error:
            logger.warning(f"Impossible de sauvegarder les détails de l'erreur: {str(save_error)}")
        
        return error_info
    
    def safe_execute(self, func: Callable, *args, **kwargs) -> Tuple[bool, Any, Optional[Dict]]:
        """
        Exécute une fonction de manière sécurisée en capturant les exceptions.
        
        Args:
            func: Fonction à exécuter
            *args: Arguments positionnels pour la fonction
            **kwargs: Arguments nommés pour la fonction
            
        Returns:
            Tuple[bool, Any, Optional[Dict]]: (succès, résultat, informations d'erreur)
        """
        try:
            result = func(*args, **kwargs)
            return True, result, None
        except Exception as e:
            context = f"{func.__module__}.{func.__name__}"
            error_info = self.handle_exception(e, context)
            return False, None, error_info
    
    def retry(self, max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0, exceptions: Tuple = (Exception,)):
        """
        Décorateur pour réessayer une fonction en cas d'échec.
        
        Args:
            max_attempts: Nombre maximum de tentatives
            delay: Délai initial entre les tentatives (en secondes)
            backoff: Facteur multiplicatif pour le délai entre les tentatives
            exceptions: Tuple des exceptions à capturer
            
        Returns:
            Callable: Décorateur
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                import time
                
                attempt = 1
                current_delay = delay
                
                while attempt <= max_attempts:
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        context = f"{func.__module__}.{func.__name__} (tentative {attempt}/{max_attempts})"
                        self.handle_exception(e, context)
                        
                        if attempt == max_attempts:
                            logger.error(f"Échec après {max_attempts} tentatives: {str(e)}")
                            raise
                        
                        logger.warning(f"Tentative {attempt}/{max_attempts} échouée, nouvelle tentative dans {current_delay:.2f}s")
                        time.sleep(current_delay)
                        current_delay *= backoff
                        attempt += 1
            
            return wrapper
        
        return decorator
    
    def validate_input(self, schema: Dict):
        """
        Décorateur pour valider les entrées d'une fonction selon un schéma.
        
        Args:
            schema: Schéma de validation pour les arguments
            
        Returns:
            Callable: Décorateur
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Obtenir les noms des arguments de la fonction
                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                
                # Valider les arguments selon le schéma
                for arg_name, arg_value in bound_args.arguments.items():
                    if arg_name in schema:
                        arg_schema = schema[arg_name]
                        
                        # Vérifier le type
                        if "type" in arg_schema and not isinstance(arg_value, arg_schema["type"]):
                            raise TypeError(f"Argument '{arg_name}' doit être de type {arg_schema['type'].__name__}, reçu {type(arg_value).__name__}")
                        
                        # Vérifier les contraintes
                        if "constraints" in arg_schema:
                            for constraint_func, error_msg in arg_schema["constraints"]:
                                if not constraint_func(arg_value):
                                    raise ValueError(f"Argument '{arg_name}': {error_msg}")
                
                return func(*args, **kwargs)
            
            return wrapper
        
        return decorator
    
    def log_execution(self, level: int = logging.INFO):
        """
        Décorateur pour journaliser l'exécution d'une fonction.
        
        Args:
            level: Niveau de logging
            
        Returns:
            Callable: Décorateur
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                logger.log(level, f"Début de l'exécution de {func.__name__}")
                start_time = datetime.now()
                
                try:
                    result = func(*args, **kwargs)
                    execution_time = (datetime.now() - start_time).total_seconds()
                    logger.log(level, f"Fin de l'exécution de {func.__name__} en {execution_time:.4f}s")
                    return result
                except Exception as e:
                    execution_time = (datetime.now() - start_time).total_seconds()
                    logger.log(level, f"Échec de l'exécution de {func.__name__} après {execution_time:.4f}s")
                    raise
            
            return wrapper
        
        return decorator

# Fonction pour créer un gestionnaire d'erreurs global
_error_handler = None

def get_error_handler(log_dir: str = "logs", max_log_files: int = 10) -> EuromillionsErrorHandler:
    """
    Obtient le gestionnaire d'erreurs global.
    
    Args:
        log_dir: Répertoire pour les fichiers de log
        max_log_files: Nombre maximum de fichiers de log à conserver
        
    Returns:
        EuromillionsErrorHandler: Gestionnaire d'erreurs global
    """
    global _error_handler
    if _error_handler is None:
        _error_handler = EuromillionsErrorHandler(log_dir, max_log_files)
    return _error_handler

# Exemple d'utilisation
def main():
    """Fonction principale pour tester le module."""
    try:
        # Créer le gestionnaire d'erreurs
        error_handler = get_error_handler()
        
        # Exemple d'utilisation du décorateur retry
        @error_handler.retry(max_attempts=3, delay=0.5)
        def function_that_might_fail(x):
            import random
            if random.random() < 0.7:
                raise ValueError(f"Échec aléatoire avec x={x}")
            return x * 2
        
        # Exemple d'utilisation du décorateur validate_input
        @error_handler.validate_input({
            "x": {
                "type": int,
                "constraints": [
                    (lambda x: x > 0, "doit être positif"),
                    (lambda x: x < 100, "doit être inférieur à 100")
                ]
            }
        })
        def process_positive_number(x):
            return x * 2
        
        # Exemple d'utilisation du décorateur log_execution
        @error_handler.log_execution()
        def long_running_function():
            import time
            time.sleep(1)
            return "Terminé"
        
        # Tester les fonctions
        print("Test de safe_execute:")
        success, result, error_info = error_handler.safe_execute(lambda x: x / 0, 10)
        print(f"Succès: {success}, Résultat: {result}, Erreur: {error_info is not None}")
        
        print("\nTest de retry:")
        try:
            result = function_that_might_fail(5)
            print(f"Résultat: {result}")
        except Exception as e:
            print(f"Échec final: {str(e)}")
        
        print("\nTest de validate_input:")
        try:
            result = process_positive_number(50)
            print(f"Résultat: {result}")
            
            # Ceci devrait échouer
            result = process_positive_number(-5)
            print(f"Résultat: {result}")
        except Exception as e:
            print(f"Validation échouée comme prévu: {str(e)}")
        
        print("\nTest de log_execution:")
        result = long_running_function()
        print(f"Résultat: {result}")
        
        print("\nTest du module de gestion des erreurs terminé avec succès.")
    
    except Exception as e:
        print(f"Erreur lors du test du module de gestion des erreurs: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
