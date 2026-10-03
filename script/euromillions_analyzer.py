#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyseur avancé de tirages d'Euromillions
Ce script analyse les statistiques des tirages d'Euromillions et propose des combinaisons
en utilisant des techniques avancées de prédiction.

Améliorations:
- Correction de l'affichage des valeurs np.int64
- Meilleure gestion des erreurs
- Optimisation du code
- Amélioration de la documentation
- Standardisation du format de présentation des combinaisons
- Intégration améliorée de la pondération Fibonacci inversée avec poids configurable.
- Optimisation des hyperparamètres des modèles de Machine Learning (Gradient Boosting, RandomForest).
- Évaluation enrichie des modèles ML avec F1-score, précision et rappel.
- Correction des avertissements Pylance concernant les variables non définies.
- Correction de l'initialisation du logger pour éviter les TypeError.
- Correction de l'erreur `ValueError: y should be a 1d array` lors de l'optimisation ML
  en adaptant `RandomizedSearchCV` pour travailler directement avec `OneVsRestClassifier`.
"""

import os
import sys
import argparse
import logging # Importation de logging en premier
import warnings
import random # Ajouter pour Monte Carlo
import platform  # Pour détecter le système d'exploitation
from itertools import combinations as iter_combinations # Pour les systèmes réducteurs
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Set, Counter as CounterType, Optional, Union, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from scipy import stats
# ⚠️ CRITIQUE : Configurer joblib AVANT l'import pour éviter l'erreur _winapi.CreateProcess
# Cette erreur se produit avec le backend 'loky' (par défaut) qui essaie de compter les cœurs CPU
if platform.system() == 'Windows':
    # Définir les variables d'environnement AVANT d'importer joblib
    os.environ['JOBLIB_START_METHOD'] = 'threading'
    os.environ['JOBLIB_TEMP_FOLDER'] = os.path.join(os.path.expanduser('~'), '.joblib')
    # Désactiver complètement le multiprocessing pour éviter l'erreur _winapi.CreateProcess
    os.environ['JOBLIB_MULTIPROCESSING'] = '0'
    # Forcer le nombre de cœurs à 1 pour éviter le comptage
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'

import joblib
from joblib import parallel_backend  # Pour forcer le backend threading sur Windows
from sklearn.cluster import KMeans
import traceback
try:
    from safe_model_loader import safe_joblib_load, ModelIntegrityError  # type: ignore
except ImportError:  # pragma: no cover
    from script.safe_model_loader import safe_joblib_load, ModelIntegrityError  # type: ignore

# ⚠️ CRITIQUE : Patcher joblib pour éviter l'erreur _count_physical_cores sur Windows
if platform.system() == 'Windows':
    try:
        # Patcher la fonction _count_physical_cores pour qu'elle retourne 1 sans essayer de compter
        import joblib.externals.loky.backend.context as loky_context
        original_count_physical_cores = loky_context._count_physical_cores
        
        def patched_count_physical_cores():
            """Version patchée qui retourne 1 sans essayer de compter les cœurs."""
            return 1
        
        loky_context._count_physical_cores = patched_count_physical_cores
    except Exception:
        # Si le patch échoue, continuer (l'erreur peut toujours se produire)
        pass

# Fonction pour obtenir le nombre de jobs de manière sûre (évite l'erreur joblib sur Windows)
def get_n_jobs():
    """
    Retourne le nombre de jobs à utiliser pour le parallélisme.
    ⚠️ CRITIQUE : Sur Windows, forcer n_jobs=1 pour éviter l'erreur _winapi.CreateProcess
    Cette erreur se produit avec joblib lors de la création de processus en parallèle.
    """
    # Sur Windows, forcer n_jobs=1 pour éviter l'erreur joblib
    if platform.system() == 'Windows':
        return 1
    try:
        # Essayer d'obtenir le nombre de cœurs CPU
        n_cores = os.cpu_count()
        if n_cores is None:
            return 1
        # Utiliser au maximum 2 cœurs pour éviter la surcharge
        return min(2, n_cores)
    except Exception:
        # En cas d'erreur, utiliser 1 seul thread
        return 1

# Configuration du logging (déplacée en haut)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("EuromillionsAdvancedAnalyzer")


# Importer le module de pondération Fibonacci
# Assurez-vous que fibonacci_weighting.py est accessible dans le PYTHONPATH
try:
    from fibonacci_weighting import apply_inverse_fibonacci_weights
except ImportError:
    # Fallback pour l'environnement de test si le fichier n'est pas directement dans le path
    # Ceci est une solution temporaire si le module n'est pas installé ou accessible
    # Dans un environnement de production, fibonacci_weighting.py devrait être correctement importable
    logger.warning("Le module 'fibonacci_weighting' n'a pas pu être importé directement. Tentative d'import via sys.path.")
    sys.path.append(os.path.dirname(__file__)) # Ajoute le répertoire courant au path
    try:
        from fibonacci_weighting import apply_inverse_fibonacci_weights
    except ImportError:
        logger.error("Impossible d'importer le module 'fibonacci_weighting'. La fonctionnalité Fibonacci ne sera pas disponible.")
        # Définir une fonction de remplacement pour éviter les erreurs
        def apply_inverse_fibonacci_weights(counts: CounterType[int], reverse_order: bool = True) -> Dict[int, float]:
            return {item: 0.0 for item in counts.keys()}

# Importer l'encodeur avancé
try:
    from advanced_encoder import AdvancedEuromillionsEncoder
    ADVANCED_ENCODER_AVAILABLE = True
except ImportError:
    ADVANCED_ENCODER_AVAILABLE = False
    logger.warning("Encodeur avancé non disponible. Utilisation des features de base.")

# Import du système quantique inspiré
try:
    from quantum_inspired_predictor import QuantumInspiredPredictor, PENNYLANE_AVAILABLE, TORCH_AVAILABLE
    QUANTUM_AVAILABLE = True
    logger.info("✅ Module quantum_inspired_predictor disponible")
except ImportError:
    QUANTUM_AVAILABLE = False
    logger.warning("⚠️ Module quantum_inspired_predictor non disponible - Installation: pip install pennylane torch")

# Ignorer les avertissements
warnings.filterwarnings("ignore")


# Constantes par défaut
DEFAULT_CONFIG = {
    "csv_file": "tirage_euromillions.csv",
    "window_draws": 1844, # Nombre de tirages à considérer pour les statistiques récentes
    "num_hot": 8, # Nombre de numéros "chauds" à identifier
    "num_cold": 8, # Nombre de numéros "froids" à identifier
    "propose_size": 5, # Nombre de numéros principaux dans une combinaison
    "star_hot": 3, # Nombre d'étoiles "chaudes" à identifier
    "star_cold": 3, # Nombre d'étoiles "froides" à identifier
    "star_size": 2, # Nombre d'étoiles dans une combinaison
    "output_dir": "resultats_euromillions_advanced", # Répertoire de sortie pour les rapports et graphiques
    "max_number": 50, # Nombre maximum de numéros principaux possibles
    "max_star": 12, # Nombre maximum d'étoiles possibles
    "combinations_to_generate": 10, # Nombre de combinaisons finales à proposer
    "use_ml": True, # Utiliser les modèles de Machine Learning
    "use_temporal": True, # Utiliser l'analyse des tendances temporelles
    "use_correlation": True, # Utiliser l'analyse des corrélations
    "use_clustering": True, # Utiliser l'analyse de clustering
    "prediction_weight": 0.7, # Poids des prédictions ML dans le score final
    "test_size": 0.2, # Taille de l'ensemble de test pour l'entraînement ML
    "cv_folds": 5, # Nombre de folds pour la validation croisée
    "model_dir": "models", # Répertoire pour sauvegarder les modèles ML
    "gap_weight": 0.1, # Poids du score d'écart dans le score final
    "export_excel": False, # Exporter les résultats vers Excel
    "analyze_parity": True, # Analyser la parité des numéros
    "analyze_sum": True, # Analyser la somme des numéros
    "use_fibonacci_inverse": True,  # Utiliser la pondération Fibonacci inversée
    "fibonacci_inverse_weight_blend": 0.3, # Poids pour le mélange Fibonacci (0.0 à 1.0)
    "monte_carlo_simulations": 5000, # Nombre de simulations Monte Carlo
    "max_wheeling_combinations": 50, # Limite le nombre de combinaisons générées par le système réducteur
    "wheeling_num_count": 10, # Nombre de numéros à inclure dans le système réducteur
    "wheeling_star_count": 5, # Nombre d'étoiles à inclure dans le système réducteur
    "max_combination_history": 100, # Taille maximale de l'historique des combinaisons générées
    "max_performance_history": 100, # Taille maximale de l'historique de performance
    "score_weight_prediction": 0.5, # Poids des prédictions ML/fréquence dans le score final combiné
    "score_weight_gap": 0.3, # Poids du score d'écart dans le score final combiné
    "score_weight_frequency": 0.2, # Poids de la fréquence récente dans le score final combiné
}

def _optimize_model_hyperparameters(
    model_name: str,
    estimator: Any, # L'estimateur à optimiser (maintenant peut être OneVsRestClassifier)
    X_train: np.ndarray,
    y_train_multi: np.ndarray,
    param_distributions: Dict[str, List[Any]],
    n_iter_search: int = 20, # Nombre d'itérations pour RandomizedSearchCV
    cv_folds: int = 3, # Nombre de folds pour la validation croisée
    random_state: int = 42
) -> Any:
    """
    Optimise les hyperparamètres d'un modèle en utilisant RandomizedSearchCV.

    Args:
        model_name (str): Nom du modèle (pour le logging).
        estimator (Any): L'instance du classifieur à optimiser (peut être OneVsRestClassifier).
        X_train (np.ndarray): Les données d'entraînement (features).
        y_train_multi (np.ndarray): Les cibles d'entraînement (format multi-label).
        param_distributions (Dict[str, List[Any]]): Dictionnaire des distributions de paramètres à échantillonner.
            Les clés doivent être au format 'estimator__paramètre' si l'estimateur est encapsulé.
        n_iter_search (int): Nombre d'itérations pour RandomizedSearchCV.
        cv_folds (int): Nombre de folds pour la validation croisée.
        random_state (int): Graine pour la reproductibilité.

    Returns:
        Any: Le meilleur estimateur trouvé après optimisation.
    """
    logger.info(f"Démarrage de l'optimisation des hyperparamètres pour {model_name}...")
    
    # ⚠️ CRITIQUE : Sur Windows, forcer n_jobs=1 pour éviter l'erreur _winapi.CreateProcess
    # Même avec get_n_jobs(), RandomizedSearchCV peut causer des problèmes sur Windows
    if platform.system() == 'Windows':
        n_jobs = 1  # Forcer n_jobs=1 sur Windows
        logger.debug("Windows détecté - Utilisation de n_jobs=1 pour RandomizedSearchCV")
    else:
        n_jobs = get_n_jobs()
    
    random_search = RandomizedSearchCV(
        estimator=estimator, # L'estimateur peut être OneVsRestClassifier ici
        param_distributions=param_distributions,
        n_iter=n_iter_search,
        cv=cv_folds,
        scoring='f1_micro', # Utilisation du F1-score micro pour les problèmes multi-label
        random_state=random_state,
        n_jobs=n_jobs, # ⚠️ CRITIQUE : n_jobs=1 sur Windows pour éviter l'erreur joblib
        verbose=1 # Affiche la progression
    )
    
    random_search.fit(X_train, y_train_multi)
    
    logger.info(f"Optimisation terminée pour {model_name}.")
    logger.info(f"Meilleurs paramètres pour {model_name}: {random_search.best_params_}")
    logger.info(f"Meilleur score F1-micro pour {model_name}: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_


class EuromillionsAdvancedAnalyzer:
    """Classe principale pour l'analyse avancée des tirages d'Euromillions."""
    
    def __init__(self, config: Dict = None):
        """
        Initialise l'analyseur avec la configuration spécifiée.
        
        Args:
            config: Dictionnaire de configuration ou None pour utiliser les valeurs par défaut
        """
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)
        
        # 🎥 NOUVEAU: Récupérer les embeddings vidéo depuis la config
        self.video_embeddings = self.config.get("video_embeddings", None)
        if self.video_embeddings:
            logger.info(f"🎥 Embeddings vidéo reçus: {len(self.video_embeddings)} vidéos")
        
        self.df = None
        self.number_cols = []
        self.star_cols = []
        self.output_dir = Path(self.config["output_dir"])
        self.model_dir = Path(self.config["model_dir"])
        
        # Création des répertoires nécessaires
        for directory in [self.output_dir, self.model_dir]:
            if not directory.exists():
                directory.mkdir(parents=True)
                logger.info(f"Répertoire créé: {directory}")
        
        # Initialisation des modèles (MLP par défaut, d'autres peuvent être ajoutés)
        self.number_model = None # Sera initialisé comme OneVsRestClassifier(GradientBoostingClassifier)
        self.star_model = None   # Sera initialisé comme OneVsRestClassifier(GradientBoostingClassifier)
        self.rf_number_model = None # Modèle RandomForest pour numéros
        self.rf_star_model = None   # Modèle RandomForest pour étoiles
        self.scaler_numbers = StandardScaler()
        self.scaler_stars = StandardScaler()  # Scaler séparé pour les étoiles
        
        # Stockage des résultats d'analyse
        self.freq = None # Fréquence globale des numéros
        self.var_monthly = None # Variance mensuelle des numéros
        self.star_freq = None # Fréquence globale des étoiles
        self.star_var_monthly = None # Variance mensuelle des étoiles
        
        # ⚠️ CRITIQUE : Charger les scalers sauvegardés si disponibles (après initialisation de toutes les variables)
        self._load_saved_scalers()
    
    def _load_saved_scalers(self) -> None:
        """
        Charge les scalers sauvegardés depuis les fichiers joblib si disponibles.
        ⚠️ CRITIQUE : Cette méthode doit être appelée pour éviter l'erreur "not fitted yet".
        """
        try:
            scaler_numbers_path = self.model_dir / "scaler_numbers.joblib"
            if scaler_numbers_path.exists():
                try:
                    self.scaler_numbers = safe_joblib_load(scaler_numbers_path)
                    logger.info(f"✅ Scaler numéros chargé depuis {scaler_numbers_path}")
                except ModelIntegrityError:
                    raise  # mode strict : ne pas substituer silencieusement un scaler vide
                except Exception as e:
                    logger.warning(f"Impossible de charger le scaler numéros: {str(e)}. Création d'un nouveau scaler.")
                    self.scaler_numbers = StandardScaler()
            
            scaler_stars_path = self.model_dir / "scaler_stars.joblib"
            if scaler_stars_path.exists():
                try:
                    self.scaler_stars = safe_joblib_load(scaler_stars_path)
                    logger.info(f"✅ Scaler étoiles chargé depuis {scaler_stars_path}")
                except ModelIntegrityError:
                    raise  # mode strict : ne pas substituer silencieusement un scaler vide
                except Exception as e:
                    logger.warning(f"Impossible de charger le scaler étoiles: {str(e)}. Création d'un nouveau scaler.")
                    self.scaler_stars = StandardScaler()
        except ModelIntegrityError:
            raise  # propager le refus d'intégrité au lieu de l'avaler ici
        except Exception as e:
            logger.warning(f"Erreur lors du chargement des scalers: {str(e)}")
            # Continuer avec des scalers vides
        self.window_counts = None # Fréquence des numéros sur la fenêtre récente
        self.star_window_counts = None # Fréquence des étoiles sur la fenêtre récente
        self.hot = None # Numéros "chauds"
        self.cold = None # Numéros "froids"
        self.star_hot = None # Étoiles "chaudes"
        self.star_cold = None # Étoiles "froides"
        self.number_correlations = None # Corrélations entre numéros
        self.star_correlations = None # Corrélations entre étoiles
        self.number_clusters = None # Statistiques des clusters de numéros
        self.star_clusters = None # Statistiques des clusters d'étoiles
        self.temporal_patterns = None # Tendances temporelles des numéros
        self.star_temporal_patterns = None # Tendances temporelles des étoiles
        self.number_predictions = None # Probabilités/scores prédits pour les numéros
        self.star_predictions = None # Probabilités/scores prédits pour les étoiles
        self.number_gap_scores = None # Scores d'écart pour les numéros
        self.star_gap_scores = None # Scores d'écart pour les étoiles
        
        # Nouvelles variables pour les analyses supplémentaires
        self.parity_stats = None # Statistiques de parité
        self.sum_stats = None # Statistiques de somme
        self.sum_ranges = None # Distribution des sommes par plage
        self.most_common_sums = None # Sommes les plus fréquentes
        self.sequence_stats = None # Statistiques des séquences de numéros
        
        # Variables pour le suivi des performances
        self.generated_combinations_history = [] # Liste de tuples (date, [combinaisons])
        self.performance_history = [] # Liste de dictionnaires {date: ..., metrics: ...}
        
        # Initialiser l'encodeur avancé si disponible
        self.advanced_encoder = None
        if ADVANCED_ENCODER_AVAILABLE:
            try:
                enable_ai_reflection = self.config.get('enable_ai_reflection', True)
                llm_config = self.config.get('llm_config', 'openai')
                self.advanced_encoder = AdvancedEuromillionsEncoder(
                    enable_ai_reflection=enable_ai_reflection,
                    llm_config=llm_config
                )
                logger.info("Encodeur avancé initialisé - Amélioration de la précision activée")
            except Exception as e:
                logger.warning(f"Erreur lors de l'initialisation de l'encodeur avancé: {str(e)}")
                self.advanced_encoder = None
        
        # Initialiser le prédicteur quantique si disponible
        # ⚠️ CRITIQUE : Activer le système quantique par défaut pour tous les entraînements et prédictions
        self.quantum_predictor = None
        use_quantum = self.config.get('use_quantum', True)  # Activé par défaut
        if QUANTUM_AVAILABLE and use_quantum:
            try:
                quantum_config = {
                    'max_number': self.config['max_number'],
                    'max_star': self.config['max_star'],
                    'n_numbers': self.config['propose_size'],
                    'n_stars': self.config['star_size'],
                    'use_qnn': self.config.get('use_qnn', True),
                    'use_qlstm': self.config.get('use_qlstm', True),
                    'use_quantum_annealing': self.config.get('use_quantum_annealing', True),
                }
                self.quantum_predictor = QuantumInspiredPredictor(quantum_config)
                self.config['use_quantum'] = True  # Activer dans la config
                logger.info("✅ Prédicteur quantique initialisé - Système Quantum-Inspired activé")
            except Exception as e:
                logger.warning(f"Erreur lors de l'initialisation du prédicteur quantique: {str(e)}")
                logger.debug(traceback.format_exc())
                self.quantum_predictor = None
                self.config['use_quantum'] = False
    
    def _convert_to_int_list(self, number_list):
        """
        Convertit une liste de nombres (potentiellement des np.int64) en liste d'entiers Python standard.
        
        Args:
            number_list: Liste de nombres à convertir
            
        Returns:
            Liste d'entiers Python standard
        """
        if number_list is None:
            return []
        return [int(num) for num in number_list]
    
    def load_data(self) -> bool:
        """
        Charge les données du fichier CSV.
        Utilise le fichier de cycles s'il existe (tirage_euromillions_complet_cycles.csv).
        Respecte l'ordre chronologique du premier au dernier tirage.
        
        Returns:
            bool: True si le chargement a réussi, False sinon
        """
        csv_path = Path(self.config["csv_file"])
        
        # ⚠️ CRITIQUE : Vérifier si le fichier de cycles existe
        cycle_file = csv_path.parent / f"{csv_path.stem}_cycles.csv"
        use_cycle_file = False
        
        if cycle_file.exists():
            logger.info(f"Fichier de cycles trouvé: {cycle_file}")
            logger.info("Vérification du contenu du fichier de cycles...")
            
            try:
                # Vérifier que le fichier cycle a les colonnes nécessaires
                cycle_df_test = pd.read_csv(cycle_file, nrows=1)
                required_cols = ['Date', 'N1', 'N2', 'N3', 'N4', 'N5', 'E1', 'E2']
                missing_cols = [col for col in required_cols if col not in cycle_df_test.columns]
                
                if not missing_cols:
                    # Vérifier que la colonne Date existe et n'est pas vide
                    cycle_df_full = pd.read_csv(cycle_file)
                    if 'Date' in cycle_df_full.columns and not cycle_df_full['Date'].isna().all():
                        use_cycle_file = True
                        logger.info("✅ Fichier de cycles valide avec dates - Utilisation du fichier de cycles")
                    else:
                        logger.warning("⚠️ Fichier de cycles sans dates valides - Utilisation du fichier principal")
                else:
                    logger.warning(f"⚠️ Fichier de cycles incomplet (colonnes manquantes: {missing_cols}) - Utilisation du fichier principal")
            except Exception as e:
                logger.warning(f"⚠️ Erreur lors de la vérification du fichier de cycles: {str(e)} - Utilisation du fichier principal")
        
        try:
            # Utiliser le fichier de cycles si disponible et valide
            if use_cycle_file:
                logger.info(f"Chargement des données depuis le fichier de cycles: {cycle_file}")
                self.df = pd.read_csv(cycle_file)
                
                # ⚠️ CRITIQUE : Vérifier et convertir la colonne Date
                if 'Date' in self.df.columns:
                    self.df['Date'] = pd.to_datetime(self.df['Date'], errors='coerce')
                    # Vérifier qu'il n'y a pas de dates manquantes
                    if self.df['Date'].isna().any():
                        logger.warning(f"⚠️ {self.df['Date'].isna().sum()} dates manquantes dans le fichier de cycles")
                
                # ⚠️ CRITIQUE : Trier par date du premier au dernier tirage (ordre chronologique)
                if 'Date' in self.df.columns:
                    self.df = self.df.sort_values('Date', ascending=True).reset_index(drop=True)
                    logger.info(f"✅ Données triées par date (ordre chronologique: {self.df['Date'].min()} → {self.df['Date'].max()})")
                else:
                    # Trier par Index si pas de Date
                    if 'Index' in self.df.columns:
                        self.df = self.df.sort_values('Index', ascending=True).reset_index(drop=True)
                        logger.info("✅ Données triées par Index (ordre chronologique)")
            else:
                # Utiliser le fichier principal
                if not csv_path.exists():
                    logger.error(f"Fichier CSV introuvable: {csv_path}")
                    return False
                    
                logger.info(f"Chargement des données depuis {csv_path}")
                self.df = pd.read_csv(csv_path)
                
                # ⚠️ CRITIQUE : Vérifier et créer la colonne Date si manquante
                if 'Date' not in self.df.columns:
                    logger.warning("Colonne 'Date' non trouvée. Création de dates automatiques...")
                    from datetime import datetime, timedelta
                    first_draw_date = datetime(2004, 2, 13)
                    for i in range(len(self.df)):
                        weeks = i // 2
                        day_in_week = (i % 2) * 3  # 0 pour mardi, 3 pour vendredi
                        date = first_draw_date + timedelta(weeks=weeks, days=day_in_week)
                        self.df.loc[i, 'Date'] = date
                    logger.info("✅ Dates automatiques créées")
                
                # Convertir la colonne Date en datetime
                if 'Date' in self.df.columns:
                    self.df['Date'] = pd.to_datetime(self.df['Date'], errors='coerce')
                
                # ⚠️ CRITIQUE : Trier par date du premier au dernier tirage (ordre chronologique)
                if 'Date' in self.df.columns and not self.df['Date'].isna().all():
                    self.df = self.df.sort_values('Date', ascending=True).reset_index(drop=True)
                    logger.info(f"✅ Données triées par date (ordre chronologique: {self.df['Date'].min()} → {self.df['Date'].max()})")
            
            # Appliquer l'encodeur avancé si disponible pour améliorer les features
            if self.advanced_encoder is not None:
                logger.info("Application de l'encodeur avancé pour améliorer les features...")
                try:
                    # Encoder toutes les features avancées (incluant les vidéos si disponibles)
                    self.df = self.advanced_encoder.encode_features(self.df, video_embeddings=self.video_embeddings)
                    logger.info("Features avancées encodées avec succès")
                except Exception as e:
                    logger.warning(f"Erreur lors de l'encodage avancé (utilisation des features de base): {str(e)}")
            
            # Identification des colonnes de numéros et d'étoiles
            # Pour Euromillions, on s'attend à des colonnes N1-N5 et E1-E2
            self.number_cols = [col for col in self.df.columns if col.startswith('N')]
            self.star_cols = [col for col in self.df.columns if col.startswith('E')]
            
            if not self.number_cols or len(self.number_cols) != 5:
                logger.error(f"Format de colonnes incorrect pour les numéros principaux. Attendu: 5 colonnes commençant par 'N', trouvé: {len(self.number_cols)}")
                return False
                
            if not self.star_cols or len(self.star_cols) != 2:
                logger.error(f"Format de colonnes incorrect pour les étoiles. Attendu: 2 colonnes commençant par 'E', trouvé: {len(self.star_cols)}")
                return False
            
            # Vérification des types de données
            for col in self.number_cols + self.star_cols:
                if not pd.api.types.is_numeric_dtype(self.df[col]):
                    try:
                        self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                        logger.warning(f"Colonne {col} convertie en type numérique")
                    except:
                        logger.error(f"Impossible de convertir la colonne {col} en type numérique")
                        return False
            
            # Suppression des lignes avec des valeurs manquantes
            initial_rows = len(self.df)
            self.df = self.df.dropna(subset=self.number_cols + self.star_cols)
            if len(self.df) < initial_rows:
                logger.warning(f"{initial_rows - len(self.df)} lignes supprimées car contenant des valeurs manquantes")
            
            # Ajout d'une colonne de date si elle n'existe pas
            if 'Date' not in self.df.columns:
                logger.warning("Colonne 'Date' non trouvée, création d'une colonne de date fictive")
                # Créer une série de dates en partant de la plus récente (aujourd'hui)
                end_date = datetime.now()
                # Supposer un tirage tous les 3-4 jours (mardi et vendredi pour Euromillions)
                dates = [(end_date - timedelta(days=i*3.5)).strftime('%Y-%m-%d') for i in range(len(self.df))]
                dates.reverse()  # Pour avoir les dates dans l'ordre chronologique
                self.df['Date'] = dates
            
            # Conversion de la date
            try:
                self.df['Date'] = pd.to_datetime(self.df['Date'], errors='coerce')
                # Tri par date
                self.df = self.df.sort_values('Date')
            except Exception as e:
                logger.warning(f"Erreur lors de la conversion des dates: {str(e)}")
            
            logger.info(f"Données chargées avec succès: {len(self.df)} tirages, {len(self.number_cols)} numéros principaux et {len(self.star_cols)} étoiles par tirage")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données: {str(e)}")
            logger.debug(traceback.format_exc())
            return False
    
    def compute_global_stats(self) -> None:
        """
        Calcule les statistiques globales sur tous les tirages.
        """
        logger.info("Calcul des statistiques globales...")
        
        try:
            # Statistiques des numéros principaux
            all_numbers = self.df[self.number_cols].values.flatten()
            all_numbers = all_numbers[~pd.isna(all_numbers)].astype(int)
            self.freq = Counter(all_numbers)
            
            # Statistiques des étoiles
            all_stars = self.df[self.star_cols].values.flatten()
            all_stars = all_stars[~pd.isna(all_stars)].astype(int)
            self.star_freq = Counter(all_stars)
            
            # Calcul des variances (si date disponible)
            self.var_monthly = {}
            self.star_var_monthly = {}
            
            if 'Date' in self.df.columns:
                try:
                    # Variance des numéros principaux
                    df_long = self.df.melt(id_vars=["Date"], value_vars=self.number_cols, value_name="number").dropna()
                    df_long["number"] = df_long["number"].astype(int)
                    df_long["month"] = df_long["Date"].dt.to_period("M")
                    monthly = df_long.groupby(["month", "number"]).size().unstack(fill_value=0)
                    self.var_monthly = monthly.var(axis=0).to_dict()
                    
                    # Variance des étoiles
                    df_long_stars = self.df.melt(id_vars=["Date"], value_vars=self.star_cols, value_name="star").dropna()
                    df_long_stars["star"] = df_long_stars["star"].astype(int)
                    df_long_stars["month"] = df_long_stars["Date"].dt.to_period("M")
                    monthly_stars = df_long_stars.groupby(["month", "star"]).size().unstack(fill_value=0)
                    self.star_var_monthly = monthly_stars.var(axis=0).to_dict()
                except Exception as e:
                    logger.warning(f"Impossible de calculer les variances mensuelles: {str(e)}")
                    logger.debug(traceback.format_exc())
        except Exception as e:
            logger.error(f"Erreur lors du calcul des statistiques globales: {str(e)}")
            logger.debug(traceback.format_exc())
    
    def compute_window_stats(self) -> None:
        """
        Calcule les statistiques sur les derniers tirages (fenêtre).
        """
        window = self.config["window_draws"]
        logger.info(f"Calcul des statistiques sur les {window} derniers tirages...")
        
        try:
            if len(self.df) < window:
                window = len(self.df)
                logger.warning(f"Nombre de tirages disponibles ({window}) inférieur à la fenêtre demandée")
            
            recent = self.df.tail(window)
            
            # Statistiques des numéros principaux
            nums = recent[self.number_cols].values.flatten()
            nums = nums[~pd.isna(nums)].astype(int)
            self.window_counts = Counter(nums)
            
            # Statistiques des étoiles
            stars = recent[self.star_cols].values.flatten()
            stars = stars[~pd.isna(stars)].astype(int)
            self.star_window_counts = Counter(stars)
        except Exception as e:
            logger.error(f"Erreur lors du calcul des statistiques de fenêtre: {str(e)}")
            logger.debug(traceback.format_exc())
    
    def identify_hot_cold(self) -> None:
        """
        Identifie les nombres chauds et froids pour les numéros principaux et les étoiles.
        """
        logger.info("Identification des numéros chauds et froids...")
        
        try:
            # Numéros principaux
            num_hot = self.config["num_hot"]
            num_cold = self.config["num_cold"]
            unique_nums = len(self.window_counts)
            
            if unique_nums < num_hot + num_cold:
                logger.warning(f"Seulement {unique_nums} numéros principaux uniques trouvés, ajustement des paramètres hot/cold")
                if unique_nums <= num_hot:
                    num_hot = max(1, unique_nums - 1)
                    num_cold = 1
                else:
                    num_cold = unique_nums - num_hot
            
            self.hot = [n for n, _ in self.window_counts.most_common(num_hot)]
            self.cold = [n for n, _ in self.window_counts.most_common()][-num_cold:]
            
            # Étoiles
            star_hot = self.config["star_hot"]
            star_cold = self.config["star_cold"]
            unique_stars = len(self.star_window_counts)
            
            if unique_stars < star_hot + star_cold:
                logger.warning(f"Seulement {unique_stars} étoiles uniques trouvées, ajustement des paramètres hot/cold")
                if unique_stars <= star_hot:
                    star_hot = max(1, unique_stars - 1)
                    star_cold = 1
                else:
                    star_cold = unique_stars - star_hot
            
            self.star_hot = [n for n, _ in self.star_window_counts.most_common(star_hot)]
            self.star_cold = [n for n, _ in self.star_window_counts.most_common()][-star_cold:]
        except Exception as e:
            logger.error(f"Erreur lors de l'identification des numéros chauds/froids: {str(e)}")
            logger.debug(traceback.format_exc())
    
    def analyze_correlations(self) -> None:
        """
        Analyse les corrélations entre les numéros et entre les étoiles.
        """
        if not self.config["use_correlation"]:
            return
            
        logger.info("Analyse des corrélations entre numéros...")
        
        try:
            # Préparation des données pour les numéros principaux
            # Créer un DataFrame où chaque colonne représente un numéro et chaque ligne un tirage
            # Les valeurs sont 1 si le numéro est présent, 0 sinon
            number_presence_df = pd.DataFrame(0, index=self.df.index, columns=range(1, self.config["max_number"] + 1))
            for col in self.number_cols:
                for num_val in self.df[col].dropna().unique():
                    number_presence_df[int(num_val)] = self.df[col].apply(lambda x: 1 if x == num_val else 0)
            
            # Calcul de la matrice de corrélation
            self.number_correlations = number_presence_df.corr()
            
            # Même chose pour les étoiles
            star_presence_df = pd.DataFrame(0, index=self.df.index, columns=range(1, self.config["max_star"] + 1))
            for col in self.star_cols:
                for star_val in self.df[col].dropna().unique():
                    star_presence_df[int(star_val)] = self.df[col].apply(lambda x: 1 if x == star_val else 0)
            
            self.star_correlations = star_presence_df.corr()
            
            logger.info("Analyse des corrélations terminée")
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse des corrélations: {str(e)}")
            logger.debug(traceback.format_exc())
    
    def analyze_temporal_patterns(self) -> None:
        """
        Analyse les tendances temporelles dans les tirages.
        """
        if not self.config["use_temporal"] or 'Date' not in self.df.columns:
            return
            
        logger.info("Analyse des tendances temporelles...")
        
        try:
            # Préparation des données
            temporal_data = {}
            star_temporal_data = {}
            
            # Grouper par mois ou semaine selon la quantité de données
            # Créer une copie pour éviter SettingWithCopyWarning
            df_temp = self.df.copy()
            if len(df_temp) > 100:
                # Assez de données pour une analyse mensuelle
                df_temp['period'] = df_temp['Date'].dt.to_period('M')
                period_type = "mensuelle"
            else:
                # Moins de données, analyse hebdomadaire
                df_temp['period'] = df_temp['Date'].dt.to_period('W')
                period_type = "hebdomadaire"
            
            # Analyse des numéros principaux
            for num in range(1, self.config["max_number"] + 1):
                temporal_data[num] = []
                
                for period, group in df_temp.groupby('period'):
                    # Compter combien de fois le numéro apparaît dans cette période
                    count = 0
                    for _, row in group.iterrows():
                        numbers = row[self.number_cols].dropna().astype(int).tolist()
                        if num in numbers:
                            count += 1
                    
                    # Normaliser par le nombre de tirages dans la période
                    freq = count / len(group) if len(group) > 0 else 0
                    temporal_data[num].append((period, freq))
            
            # Analyse des étoiles
            for star in range(1, self.config["max_star"] + 1):
                star_temporal_data[star] = []
                
                for period, group in df_temp.groupby('period'):
                    count = 0
                    for _, row in group.iterrows():
                        stars = row[self.star_cols].dropna().astype(int).tolist()
                        if star in stars:
                            count += 1
                    
                    freq = count / len(group) if len(group) > 0 else 0
                    star_temporal_data[star].append((period, freq))
            
            # Stocker les résultats
            self.temporal_patterns = temporal_data
            self.star_temporal_patterns = star_temporal_data
            
            logger.info(f"Analyse temporelle {period_type} terminée")
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse des tendances temporelles: {str(e)}")
            logger.debug(traceback.format_exc())
    
    def analyze_clustering(self) -> None:
        """
        Analyse les clusters de numéros et d'étoiles.
        """
        if not self.config["use_clustering"]:
            return
            
        logger.info("Analyse des clusters de numéros...")
        
        # ⚠️ CRITIQUE : Forcer le backend threading sur Windows pour éviter l'erreur _winapi.CreateProcess
        # Cette erreur se produit avec le backend 'loky' (par défaut) qui essaie de compter les cœurs CPU
        if platform.system() == 'Windows':
            try:
                with parallel_backend('threading', n_jobs=1):
                    self._analyze_clustering_internal()
            except Exception as e:
                logger.warning(f"Erreur avec le backend threading, tentative sans contexte: {str(e)}")
                self._analyze_clustering_internal()
        else:
            self._analyze_clustering_internal()
    
    def _analyze_clustering_internal(self) -> None:
        """
        Implémentation interne de l'analyse de clustering.
        """
        try:
            # Préparation des données pour les numéros principaux
            draws_matrix = np.zeros((len(self.df), self.config["max_number"]))
            
            for i, (_, row) in enumerate(self.df.iterrows()):
                numbers = row[self.number_cols].dropna().astype(int).tolist()
                for num in numbers:
                    if 1 <= num <= self.config["max_number"]:
                        draws_matrix[i, num-1] = 1
            
            # Clustering des tirages
            n_clusters = min(8, len(self.df) // 10)  # Nombre de clusters adaptatif
            if n_clusters < 2:
                n_clusters = 2
            
            # ⚠️ CRITIQUE : Sur Windows, utiliser n_init=1 pour éviter l'erreur joblib
            # n_init=1 désactive l'optimisation multi-initialisation qui peut déclencher le parallélisme
            n_init_value = 1 if platform.system() == 'Windows' else 10
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=n_init_value, algorithm='lloyd')
            clusters = kmeans.fit_predict(draws_matrix)
            
            # Analyser les clusters
            cluster_stats = {}
            for cluster_id in range(n_clusters):
                cluster_draws = draws_matrix[clusters == cluster_id]
                cluster_sum = cluster_draws.sum(axis=0)
                cluster_freq = cluster_sum / len(cluster_draws) if len(cluster_draws) > 0 else np.zeros(self.config["max_number"])
                
                # Identifier les numéros les plus fréquents dans ce cluster
                top_indices = np.argsort(cluster_freq)[-10:]  # Top 10 numéros
                top_numbers = [(idx + 1, cluster_freq[idx]) for idx in top_indices]
                
                cluster_stats[cluster_id] = {
                    'size': int(np.sum(clusters == cluster_id)),
                    'top_numbers': top_numbers,
                    'avg_frequency': float(np.mean(cluster_freq))
                }
            
            self.number_clusters = cluster_stats
            
            # Même chose pour les étoiles
            star_matrix = np.zeros((len(self.df), self.config["max_star"]))
            
            for i, (_, row) in enumerate(self.df.iterrows()):
                stars = row[self.star_cols].dropna().astype(int).tolist()
                for star in stars:
                    if 1 <= star <= self.config["max_star"]:
                        star_matrix[i, star-1] = 1
            
            # Moins de clusters pour les étoiles car moins de combinaisons possibles
            n_star_clusters = min(4, len(self.df) // 20)
            if n_star_clusters < 2:
                n_star_clusters = 2
            
            # ⚠️ CRITIQUE : Sur Windows, utiliser n_init=1 pour éviter l'erreur joblib
            # n_init=1 désactive l'optimisation multi-initialisation qui peut déclencher le parallélisme
            n_init_value = 1 if platform.system() == 'Windows' else 10
            kmeans_stars = KMeans(n_clusters=n_star_clusters, random_state=42, n_init=n_init_value, algorithm='lloyd')
            star_clusters = kmeans_stars.fit_predict(star_matrix)
            
            # Analyser les clusters d'étoiles
            star_cluster_stats = {}
            for cluster_id in range(n_star_clusters):
                cluster_draws = star_matrix[star_clusters == cluster_id]
                cluster_sum = cluster_draws.sum(axis=0)
                cluster_freq = cluster_sum / len(cluster_draws) if len(cluster_draws) > 0 else np.zeros(self.config["max_star"])
                
                top_indices = np.argsort(cluster_freq)[-5:]  # Top 5 étoiles
                top_stars = [(idx + 1, cluster_freq[idx]) for idx in top_indices]
                
                star_cluster_stats[cluster_id] = {
                    'size': int(np.sum(star_clusters == cluster_id)),
                    'top_stars': top_stars,
                    'avg_frequency': float(np.mean(cluster_freq))
                }
            
            self.star_clusters = star_cluster_stats
            
            logger.info(f"Analyse de clustering terminée: {n_clusters} clusters de numéros, {n_star_clusters} clusters d'étoiles")
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse des clusters: {str(e)}")
            logger.debug(traceback.format_exc())

    def perform_clustering(self) -> None:
        """
        Exécute l'analyse de clustering. Alias pour analyze_clustering.
        """
        self.analyze_clustering()
    
    def analyze_parity(self) -> None:
        """
        Analyse la parité des numéros dans les tirages.
        """
        if not self.config["analyze_parity"]:
            return
            
        logger.info("Analyse de la parité des numéros...")
        
        try:
            parity_counts = []
            
            for _, row in self.df.iterrows():
                numbers = row[self.number_cols].dropna().astype(int).tolist()
                even_count = sum(1 for num in numbers if num % 2 == 0)
                odd_count = len(numbers) - even_count
                parity_counts.append((even_count, odd_count))
            
            # Calculer les statistiques de parité
            parity_distribution = Counter(parity_counts)
            most_common = parity_distribution.most_common()
            
            # Calculer les pourcentages
            total_draws = len(self.df)
            parity_stats = {}
            
            for (even, odd), count in most_common:
                parity_stats[f"{even}E-{odd}O"] = {
                    'count': count,
                    'percentage': (count / total_draws) * 100 if total_draws > 0 else 0
                }
            
            self.parity_stats = parity_stats
            
            logger.info("Analyse de parité terminée")
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse de parité: {str(e)}")
            logger.debug(traceback.format_exc())
    
    def analyze_sum(self) -> None:
        """
        Analyse la somme des numéros dans les tirages.
        """
        if not self.config["analyze_sum"]:
            return
            
        logger.info("Analyse de la somme des numéros...")
        
        try:
            sums = []
            
            for _, row in self.df.iterrows():
                numbers = row[self.number_cols].dropna().astype(int).tolist()
                total_sum = sum(numbers)
                sums.append(total_sum)
            
            # Statistiques de base
            min_sum = min(sums) if sums else 0
            max_sum = max(sums) if sums else 0
            avg_sum = sum(sums) / len(sums) if sums else 0
            
            # Distribution des sommes
            sum_counts = Counter(sums)
            most_common_sums = sum_counts.most_common(10)  # Top 10 sommes les plus fréquentes
            
            # Définir des plages de sommes
            # Calculer les limites des plages de manière plus robuste
            if sums:
                min_val = min(sums)
                max_val = max(sums)
                # Assurer au moins 5 plages, même si la plage est petite
                num_bins = 5
                bin_edges = np.linspace(min_val, max_val + 1, num_bins + 1) # +1 pour inclure max_val
                
                ranges_list = []
                for i in range(num_bins):
                    start = int(bin_edges[i])
                    end = int(bin_edges[i+1])
                    if i == num_bins - 1: # Pour la dernière plage, inclure la fin
                        ranges_list.append((start, end))
                    else:
                        ranges_list.append((start, end -1)) # Exclure la fin pour les autres plages
            else:
                ranges_list = []

            # Compter les tirages dans chaque plage
            range_counts = {f"{int(start)}-{int(end)}": 0 for start, end in ranges_list}
            
            for sum_val in sums:
                for i, (start, end) in enumerate(ranges_list):
                    # Ajuster la condition pour la dernière plage si nécessaire
                    if i == len(ranges_list) -1:
                        if start <= sum_val <= end:
                            range_key = f"{int(start)}-{int(end)}"
                            range_counts[range_key] += 1
                            break
                    else:
                        if start <= sum_val < end:
                            range_key = f"{int(start)}-{int(end)}"
                            range_counts[range_key] += 1
                            break
            
            # Calculer les pourcentages
            total_draws = len(sums)
            for range_key in range_counts:
                count = range_counts[range_key]
                range_counts[range_key] = {
                    'count': count,
                    'percentage': (count / total_draws) * 100 if total_draws > 0 else 0
                }
            
            # Stocker les résultats
            self.sum_stats = {
                'min': int(min_sum),
                'max': int(max_sum),
                'avg': float(avg_sum),
                'median': float(np.median(sums)) if sums else 0,
                'std': float(np.std(sums)) if sums else 0
            }
            
            self.sum_ranges = range_counts
            self.most_common_sums = [(int(sum_val), count) for sum_val, count in most_common_sums]
            
            logger.info("Analyse de somme terminée")
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse de somme: {str(e)}")
            logger.debug(traceback.format_exc())
    
    def analyze_sequences(self) -> None:
        """
        Analyse les séquences de numéros consécutifs dans les tirages.
        """
        logger.info("Analyse des séquences de numéros consécutifs...")
        
        try:
            sequence_counts = []
            
            for _, row in self.df.iterrows():
                numbers = sorted(row[self.number_cols].dropna().astype(int).tolist())
                
                # Compter les séquences de numéros consécutifs
                sequences = 0
                seq_length = 1
                
                for i in range(1, len(numbers)):
                    if numbers[i] == numbers[i-1] + 1:
                        seq_length += 1
                    else:
                        if seq_length > 1:
                            sequences += 1
                        seq_length = 1
                
                # Vérifier la dernière séquence
                if seq_length > 1:
                    sequences += 1
                
                sequence_counts.append(sequences)
            
            # Calculer les statistiques
            sequence_distribution = Counter(sequence_counts)
            
            # Calculer les pourcentages
            total_draws = len(self.df)
            sequence_stats = {}
            
            for seq_count, draw_count in sequence_distribution.items():
                sequence_stats[seq_count] = {
                    'count': draw_count,
                    'percentage': (draw_count / total_draws) * 100 if total_draws > 0 else 0
                }
            
            self.sequence_stats = sequence_stats
            
            logger.info("Analyse des séquences terminée")
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse des séquences: {str(e)}")
            logger.debug(traceback.format_exc())
    
    def train_ml_models(self) -> bool:
        """
        Entraîne des modèles de machine learning pour prédire les numéros et les étoiles.
        
        Returns:
            bool: True si l'entraînement a réussi, False sinon
        """
        if not self.config["use_ml"]:
            return False
            
        logger.info("Entraînement des modèles de prédiction...")
        
        try:
            # Vérifier qu'il y a suffisamment de données
            if len(self.df) < 50:
                logger.warning("Données insuffisantes pour l'entraînement des modèles ML (minimum 50 tirages)")
                return False
            
            # Préparation des données pour les numéros principaux
            # Utiliser l'encodeur avancé si disponible
            window_size = 5  # Définir window_size au début pour toutes les branches
            y_stars = None  # Initialiser y_stars à None pour pouvoir vérifier s'il a été créé
            
            if self.advanced_encoder is not None:
                try:
                    logger.info("Utilisation de l'encodeur avancé pour préparer les features ML...")
                    # ⚠️ CRITIQUE : Préparer les features SANS scaler de l'encodeur
                    # On utilisera le scaler de l'analyseur pour garantir la cohérence entre entraînement et prédiction
                    # 🎥 NOUVEAU: Passer les embeddings vidéo à l'encodeur
                    X_unscaled, y = self.advanced_encoder.prepare_ml_features(
                        self.df, 
                        use_scaler=False,
                        video_embeddings=self.video_embeddings
                    )
                    
                    # Séparer les targets pour numéros et étoiles
                    y_numbers = y[:, :5].tolist()  # N1-N5
                    y_stars = y[:, 5:7].tolist()   # E1-E2
                    
                    # ⚠️ CRITIQUE : Utiliser le scaler de l'analyseur (pas celui de l'encodeur)
                    # Cela garantit que les features à l'entraînement et à la prédiction sont identiques
                    X = self.scaler_numbers.fit_transform(X_unscaled)
                    
                    logger.info(f"Features avancées préparées: {X.shape[0]} échantillons, {X.shape[1]} features")
                    logger.info(f"Targets préparés: {len(y_numbers)} numéros, {len(y_stars)} étoiles")
                except Exception as e:
                    logger.warning(f"Erreur avec l'encodeur avancé, utilisation des features de base: {str(e)}")
                    # Continuer avec la méthode de base
                    X = []
                    y_numbers = []
                    y_stars = []
                    for i in range(len(self.df) - window_size):
                        features = []
                        for j in range(window_size):
                            row = self.df.iloc[i + j]
                            numbers = row[self.number_cols].dropna().astype(int).tolist()
                            stars = row[self.star_cols].dropna().astype(int).tolist()
                            features.extend(numbers)
                            features.extend(stars)
                        X.append(features)
                        next_row = self.df.iloc[i + window_size]
                        next_numbers = next_row[self.number_cols].dropna().astype(int).tolist()
                        next_stars = next_row[self.star_cols].dropna().astype(int).tolist()
                        y_numbers.append(next_numbers)
                        y_stars.append(next_stars)
                    X = np.array(X)
                    X = self.scaler_numbers.fit_transform(X)
            else:
                # Méthode de base sans encodeur avancé
                X = []
                y_numbers = []
                y_stars = []
                
                # Utiliser une fenêtre glissante pour créer les features
                for i in range(len(self.df) - window_size):
                    # Features: les window_size derniers tirages
                    features = []
                    
                    for j in range(window_size):
                        row = self.df.iloc[i + j]
                        numbers = row[self.number_cols].dropna().astype(int).tolist()
                        stars = row[self.star_cols].dropna().astype(int).tolist()
                        
                        # Ajouter les numéros et étoiles comme features
                        features.extend(numbers)
                        features.extend(stars)
                    
                    X.append(features)
                    
                    # Target: le tirage suivant
                    next_row = self.df.iloc[i + window_size]
                    next_numbers = next_row[self.number_cols].dropna().astype(int).tolist()
                    next_stars = next_row[self.star_cols].dropna().astype(int).tolist()
                    y_numbers.append(next_numbers)
                    y_stars.append(next_stars)
                
                # Conversion en arrays numpy
                X = np.array(X)
                
                # Normalisation des features
                X = self.scaler_numbers.fit_transform(X)
            
            # Séparation en ensembles d'entraînement et de test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_numbers, test_size=self.config["test_size"], random_state=42
            )
            
            # Transformation du problème en classification multi-label pour les numéros
            y_train_multi = np.zeros((len(y_train), self.config["max_number"]))
            for i, nums in enumerate(y_train):
                for num in nums:
                    if 1 <= num <= self.config["max_number"]:
                        y_train_multi[i, num-1] = 1

            # --- Optimisation et entraînement du modèle Gradient Boosting pour les numéros ---
            logger.info("Optimisation et entraînement du modèle Gradient Boosting pour les numéros...")
            gb_params_numbers = {
                'estimator__n_estimators': [50, 100, 150, 200], # Utiliser estimator__ pour cibler l'estimateur de base
                'estimator__learning_rate': [0.01, 0.05, 0.1, 0.2],
                'estimator__max_depth': [3, 4, 5, 6],
                'estimator__subsample': [0.7, 0.8, 0.9, 1.0] 
            }
            # Créer l'estimateur OneVsRestClassifier à optimiser
            # ⚠️ CRITIQUE : Sur Windows, utiliser n_jobs=1 pour éviter l'erreur joblib
            if platform.system() == 'Windows':
                n_jobs_ovr = 1
            else:
                n_jobs_ovr = get_n_jobs()
            gb_ovr_estimator_numbers = OneVsRestClassifier(GradientBoostingClassifier(random_state=42), n_jobs=n_jobs_ovr)
            best_gb_numbers_estimator = _optimize_model_hyperparameters(
                "GradientBoostingClassifier (Numéros)",
                gb_ovr_estimator_numbers, # Passer le OneVsRestClassifier ici
                X_train, y_train_multi,
                gb_params_numbers,
                n_iter_search=20 
            )
            self.number_model = best_gb_numbers_estimator # best_estimator_ est déjà un OneVsRestClassifier
            # Le .fit est déjà fait par RandomizedSearchCV
            logger.info("Entraînement du modèle Gradient Boosting de numéros terminé.")

            # --- Optimisation et entraînement du modèle RandomForest pour les numéros ---
            logger.info("Optimisation et entraînement du modèle RandomForest pour les numéros...")
            rf_params_numbers = {
                'n_estimators': [50, 100, 150, 200], 
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4] 
            }
            # RandomForestClassifier gère nativement le multi-label, pas besoin de OneVsRestClassifier pour l'optimisation
            # ⚠️ CRITIQUE : Sur Windows, utiliser n_jobs=1 pour éviter l'erreur joblib
            if platform.system() == 'Windows':
                n_jobs_rf = 1
            else:
                n_jobs_rf = get_n_jobs()
            best_rf_numbers_estimator = _optimize_model_hyperparameters(
                "RandomForestClassifier (Numéros)",
                RandomForestClassifier(random_state=42, n_jobs=n_jobs_rf), # Passer le RandomForestClassifier directement
                X_train, y_train_multi,
                rf_params_numbers,
                n_iter_search=20 
            )
            self.rf_number_model = best_rf_numbers_estimator 
            # Le .fit est déjà fait par RandomizedSearchCV
            logger.info("Entraînement du modèle RandomForest de numéros terminé.")
            
            # Évaluation des numéros
            logger.info("Évaluation du modèle de numéros...")
            # Initialisation explicite pour Pylance
            y_pred_numbers = np.array([]) 
            y_test_multi_eval = np.zeros((len(y_test), self.config["max_number"])) # Renommé pour clarté
            for i, nums in enumerate(y_test):
                for num in nums:
                    if 1 <= num <= self.config["max_number"]:
                        y_test_multi_eval[i, num-1] = 1

            # Utiliser le modèle optimisé pour la prédiction
            y_pred_numbers = self.number_model.predict(X_test)
            
            accuracy_numbers = accuracy_score(y_test_multi_eval, y_pred_numbers)
            f1_micro_numbers = f1_score(y_test_multi_eval, y_pred_numbers, average='micro')
            f1_macro_numbers = f1_score(y_test_multi_eval, y_pred_numbers, average='macro')
            precision_micro_numbers = precision_score(y_test_multi_eval, y_pred_numbers, average='micro')
            recall_micro_numbers = recall_score(y_test_multi_eval, y_pred_numbers, average='micro')

            logger.info(f"Précision (Accuracy) du modèle de numéros: {accuracy_numbers*100:.2f}%")
            logger.info(f"F1-score (Micro) du modèle de numéros: {f1_micro_numbers:.4f}")
            logger.info(f"F1-score (Macro) du modèle de numéros: {f1_macro_numbers:.4f}")
            logger.info(f"Précision (Micro) du modèle de numéros: {precision_micro_numbers:.4f}")
            logger.info(f"Rappel (Micro) du modèle de numéros: {recall_micro_numbers:.4f}")
            logger.info("Évaluation du modèle de numéros terminée.")
            
            # Récompenser la réflexion IA si disponible
            if self.advanced_encoder is not None and self.advanced_encoder.ai_reflection is not None:
                performance_metrics = {
                    'accuracy': accuracy_numbers,
                    'f1_score': f1_micro_numbers,
                    'precision': precision_micro_numbers,
                    'recall': recall_micro_numbers
                }
                self.advanced_encoder.reward_reflection(performance_metrics)
            
            # Préparer y_stars si pas déjà fait (cas méthode de base)
            # Si y_stars est None ou vide, le créer depuis les données
            if y_stars is None or len(y_stars) == 0:
                logger.info("Création de y_stars depuis les données...")
                y_stars = []
                for i in range(len(self.df) - window_size):
                    next_row = self.df.iloc[i + window_size]
                    next_stars = next_row[self.star_cols].dropna().astype(int).tolist()
                    y_stars.append(next_stars)
                logger.info(f"y_stars créé: {len(y_stars)} échantillons")
            
            # Séparation en ensembles d'entraînement et de test
            # X_train et X_test sont déjà définis et normalisés
            # Créer des indices pour la séparation train/test qui correspondent à ceux utilisés pour X
            indices = np.arange(len(y_stars))
            indices_train, indices_test = train_test_split(
                indices, 
                test_size=self.config["test_size"], 
                random_state=42
            )
            
            # Utiliser ces indices pour séparer y_stars
            y_train_stars = [y_stars[i] for i in indices_train]
            y_test_stars = [y_stars[i] for i in indices_test]
            
            # S'assurer que les scalers sont fitted (ils devraient l'être déjà via fit_transform)
            # Mais vérifier pour éviter l'erreur "not fitted yet"
            if not hasattr(self.scaler_numbers, 'mean_') or self.scaler_numbers.mean_ is None:
                logger.warning("Le scaler_numbers n'est pas fitted, le fit maintenant...")
                self.scaler_numbers.fit(X_train)
            
            # Initialiser et fit le scaler_stars si nécessaire
            if not hasattr(self, 'scaler_stars'):
                self.scaler_stars = StandardScaler()
            if not hasattr(self.scaler_stars, 'mean_') or self.scaler_stars.mean_ is None:
                logger.info("Fitting du scaler_stars...")
                self.scaler_stars.fit(X_train)
            
            # Transformation en format multi-label pour les étoiles
            y_train_stars_multi = np.zeros((len(y_train_stars), self.config["max_star"]))
            for i, stars in enumerate(y_train_stars):
                for star in stars:
                    if 1 <= star <= self.config["max_star"]:
                        y_train_stars_multi[i, star-1] = 1

            # --- Optimisation et entraînement du modèle Gradient Boosting pour les étoiles ---
            logger.info("Optimisation et entraînement du modèle Gradient Boosting pour les étoiles...")
            gb_params_stars = {
                'estimator__n_estimators': [50, 100, 150, 200],
                'estimator__learning_rate': [0.01, 0.05, 0.1, 0.2],
                'estimator__max_depth': [2, 3, 4, 5],
                'estimator__subsample': [0.7, 0.8, 0.9, 1.0]
            }
            # ⚠️ CRITIQUE : Sur Windows, utiliser n_jobs=1 pour éviter l'erreur joblib
            if platform.system() == 'Windows':
                n_jobs_ovr_stars = 1
            else:
                n_jobs_ovr_stars = get_n_jobs()
            gb_ovr_estimator_stars = OneVsRestClassifier(GradientBoostingClassifier(random_state=42), n_jobs=n_jobs_ovr_stars)
            best_gb_stars_estimator = _optimize_model_hyperparameters(
                "GradientBoostingClassifier (Étoiles)",
                gb_ovr_estimator_stars,
                X_train, y_train_stars_multi,
                gb_params_stars,
                n_iter_search=10
            )
            self.star_model = best_gb_stars_estimator
            logger.info("Entraînement du modèle Gradient Boosting d'étoiles terminé.")

            # --- Optimisation et entraînement du modèle RandomForest pour les étoiles ---
            logger.info("Optimisation et entraînement du modèle RandomForest pour les étoiles...")
            rf_params_stars = {
                'n_estimators': [50, 100, 150, 200],
                'max_depth': [3, 7, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            # ⚠️ CRITIQUE : Sur Windows, utiliser n_jobs=1 pour éviter l'erreur joblib
            if platform.system() == 'Windows':
                n_jobs_rf_stars = 1
            else:
                n_jobs_rf_stars = get_n_jobs()
            best_rf_stars_estimator = _optimize_model_hyperparameters(
                "RandomForestClassifier (Étoiles)",
                RandomForestClassifier(random_state=42, n_jobs=n_jobs_rf_stars),
                X_train, y_train_stars_multi,
                rf_params_stars,
                n_iter_search=10
            )
            self.rf_star_model = best_rf_stars_estimator
            logger.info("Entraînement du modèle RandomForest d'étoiles terminé.")
            
            # Évaluation des étoiles
            logger.info("Évaluation du modèle d'étoiles...")
            # Initialisation explicite pour Pylance
            y_pred_stars = np.array([]) 
            y_test_stars_multi_eval = np.zeros((len(y_test_stars), self.config["max_star"])) # Renommé pour clarté
            for i, stars in enumerate(y_test_stars):
                for star in stars:
                    if 1 <= star <= self.config["max_star"]:
                        y_test_stars_multi_eval[i, star-1] = 1

            y_pred_stars = self.star_model.predict(X_test)
            
            accuracy_stars = accuracy_score(y_test_stars_multi_eval, y_pred_stars)
            f1_micro_stars = f1_score(y_test_stars_multi_eval, y_pred_stars, average='micro')
            f1_macro_stars = f1_score(y_test_stars_multi_eval, y_pred_stars, average='macro')
            precision_micro_stars = precision_score(y_test_stars_multi_eval, y_pred_stars, average='micro')
            recall_micro_stars = recall_score(y_test_stars_multi_eval, y_pred_stars, average='micro')

            logger.info(f"Précision (Accuracy) du modèle d'étoiles: {accuracy_stars*100:.2f}%")
            logger.info(f"F1-score (Micro) du modèle d'étoiles: {f1_micro_stars:.4f}")
            logger.info(f"F1-score (Macro) du modèle d'étoiles: {f1_macro_stars:.4f}")
            logger.info(f"Précision (Micro) du modèle d'étoiles: {precision_micro_stars:.4f}")
            logger.info(f"Rappel (Micro) du modèle d'étoiles: {recall_micro_stars:.4f}")
            logger.info("Évaluation du modèle d'étoiles terminée.")
            
            # Récompenser la réflexion IA pour les étoiles aussi
            if self.advanced_encoder is not None and self.advanced_encoder.ai_reflection is not None:
                performance_metrics_stars = {
                    'accuracy': accuracy_stars,
                    'f1_score': f1_micro_stars,
                    'precision': precision_micro_stars,
                    'recall': recall_micro_stars
                }
                # Utiliser la moyenne des métriques numéros et étoiles pour la récompense globale
                combined_metrics = {
                    'accuracy': (accuracy_numbers + accuracy_stars) / 2,
                    'f1_score': (f1_micro_numbers + f1_micro_stars) / 2,
                    'precision': (precision_micro_numbers + precision_micro_stars) / 2,
                    'recall': (recall_micro_numbers + recall_micro_stars) / 2
                }
                # Récompenser la réflexion IA si la méthode existe
                if hasattr(self.advanced_encoder, 'reward_reflection'):
                    try:
                        self.advanced_encoder.reward_reflection(combined_metrics)
                    except Exception as e:
                        logger.debug(f"Impossible de récompenser la réflexion IA: {str(e)}")
            
            # Sauvegarder les modèles entraînés
            try:
                logger.info("Sauvegarde des modèles entraînés...")
                
                # Sauvegarder le modèle Gradient Boosting pour numéros
                if self.number_model is not None:
                    number_model_path = self.model_dir / "number_model_gb.joblib"
                    joblib.dump(self.number_model, number_model_path)
                    logger.info(f"Modèle Gradient Boosting numéros sauvegardé: {number_model_path}")
                
                # Sauvegarder le modèle RandomForest pour numéros
                if self.rf_number_model is not None:
                    rf_number_model_path = self.model_dir / "number_model_rf.joblib"
                    joblib.dump(self.rf_number_model, rf_number_model_path)
                    logger.info(f"Modèle RandomForest numéros sauvegardé: {rf_number_model_path}")
                
                # Sauvegarder le modèle Gradient Boosting pour étoiles
                if self.star_model is not None:
                    star_model_path = self.model_dir / "star_model_gb.joblib"
                    joblib.dump(self.star_model, star_model_path)
                    logger.info(f"Modèle Gradient Boosting étoiles sauvegardé: {star_model_path}")
                
                # Sauvegarder le modèle RandomForest pour étoiles
                if self.rf_star_model is not None:
                    rf_star_model_path = self.model_dir / "star_model_rf.joblib"
                    joblib.dump(self.rf_star_model, rf_star_model_path)
                    logger.info(f"Modèle RandomForest étoiles sauvegardé: {rf_star_model_path}")
                
                # Sauvegarder les scalers
                if self.scaler_numbers is not None:
                    scaler_path = self.model_dir / "scaler_numbers.joblib"
                    joblib.dump(self.scaler_numbers, scaler_path)
                    logger.info(f"Scaler numéros sauvegardé: {scaler_path}")
                
                if hasattr(self, 'scaler_stars') and self.scaler_stars is not None:
                    scaler_stars_path = self.model_dir / "scaler_stars.joblib"
                    joblib.dump(self.scaler_stars, scaler_stars_path)
                    logger.info(f"Scaler étoiles sauvegardé: {scaler_stars_path}")
                
                # Sauvegarder l'encodeur avancé si disponible
                if self.advanced_encoder is not None:
                    encoder_path = self.model_dir / "advanced_encoder.joblib"
                    joblib.dump(self.advanced_encoder, encoder_path)
                    logger.info(f"Encodeur avancé sauvegardé: {encoder_path}")
                
                logger.info("Tous les modèles ont été sauvegardés avec succès.")
                
            except Exception as e:
                logger.error(f"Erreur lors de la sauvegarde des modèles: {str(e)}")
                logger.debug(traceback.format_exc())
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de l'entraînement des modèles ML: {str(e)}")
            logger.debug(traceback.format_exc())
            return False
    
    def predict_numbers(self) -> Tuple[List[int], List[int]]:
        """
        Prédit les numéros et les étoiles pour le prochain tirage.
        
        Returns:
            Tuple contenant:
            - Liste des numéros prédits
            - Liste des étoiles prédites
        """
        logger.info("Prédiction des numéros pour le prochain tirage...")
        
        try:
            # Initialiser les prédictions
            number_probs = {}
            star_probs = {}
            
            # 1. Utiliser les statistiques de fréquence
            for num in range(1, self.config["max_number"] + 1):
                # Utiliser la fréquence sur la fenêtre récente, normalisée par le nombre total de numéros tirés dans la fenêtre
                freq = self.window_counts.get(num, 0) / (self.config["propose_size"] * min(self.config["window_draws"], len(self.df)))
                number_probs[num] = freq
            
            for star in range(1, self.config["max_star"] + 1):
                # Utiliser la fréquence sur la fenêtre récente, normalisée par le nombre total d'étoiles tirées dans la fenêtre
                freq = self.star_window_counts.get(star, 0) / (self.config["star_size"] * min(self.config["window_draws"], len(self.df)))
                star_probs[star] = freq
            
            # 2. Ajuster avec les tendances temporelles si disponibles
            if self.temporal_patterns and self.config["use_temporal"]:
                for num in range(1, self.config["max_number"] + 1):
                    if num in self.temporal_patterns and self.temporal_patterns[num]:
                        # Utiliser la tendance la plus récente
                        _, recent_freq = self.temporal_patterns[num][-1]
                        # Blend avec un poids de 0.3 pour la tendance temporelle
                        number_probs[num] = number_probs.get(num, 0) * 0.7 + recent_freq * 0.3
            
            if self.star_temporal_patterns and self.config["use_temporal"]:
                for star in range(1, self.config["max_star"] + 1):
                    if star in self.star_temporal_patterns and self.star_temporal_patterns[star]:
                        _, recent_freq = self.star_temporal_patterns[star][-1]
                        star_probs[star] = star_probs.get(star, 0) * 0.7 + recent_freq * 0.3 # Correction: utiliser recent_freq ici
            
            # 3. Utiliser les prédictions ML si disponibles
            if self.number_model and self.star_model and self.rf_number_model and self.rf_star_model and self.config["use_ml"]:
                # ⚠️ CRITIQUE : Utiliser la même méthode de préparation des features que lors de l'entraînement
                # Si l'encodeur avancé a été utilisé à l'entraînement, l'utiliser aussi à la prédiction
                if self.advanced_encoder is not None:
                    try:
                        # Utiliser prepare_ml_features pour obtenir les mêmes features qu'à l'entraînement
                        # On prend les derniers tirages pour créer les features de prédiction
                        window_size = 5
                        if len(self.df) < window_size:
                            logger.warning("Pas assez de données pour préparer les features ML pour la prédiction.")
                        else:
                            # Créer un DataFrame temporaire avec les derniers tirages
                            last_draws_df = self.df.iloc[-window_size:].copy()
                            
                            # ⚠️ CRITIQUE : Utiliser prepare_ml_features mais avec le scaler de l'analyseur (pas celui de l'encodeur)
                            # L'encodeur prépare les features, mais on utilise le scaler de l'analyseur qui a été entraîné
                            # 🎥 NOUVEAU: Passer les embeddings vidéo pour la prédiction
                            X_unscaled, _ = self.advanced_encoder.prepare_ml_features(
                                last_draws_df, 
                                use_scaler=False,
                                video_embeddings=self.video_embeddings
                            )
                            
                            # Prendre la dernière ligne (la plus récente) pour la prédiction
                            if len(X_unscaled) > 0:
                                features_unscaled = X_unscaled[-1:].reshape(1, -1)  # Reshape pour avoir (1, n_features)
                                
                                # Charger les scalers sauvegardés si pas encore chargés
                                if not hasattr(self.scaler_numbers, 'mean_') or self.scaler_numbers.mean_ is None:
                                    logger.warning("Le scaler_numbers n'est pas fitted. Tentative de chargement depuis le fichier sauvegardé...")
                                    self._load_saved_scalers()
                                
                                # Utiliser le scaler de l'analyseur (celui qui a été entraîné)
                                if hasattr(self.scaler_numbers, 'mean_') and self.scaler_numbers.mean_ is not None:
                                    features_scaled_numbers = self.scaler_numbers.transform(features_unscaled)
                                    features_scaled_stars = features_scaled_numbers  # Utiliser les mêmes features pour les étoiles
                                    
                                    logger.info(f"✅ Features préparées avec l'encodeur avancé: {features_scaled_numbers.shape[1]} features (identique à l'entraînement)")
                                else:
                                    raise ValueError("Scaler de l'analyseur non disponible")
                            else:
                                raise ValueError("Aucune feature générée par l'encodeur avancé")
                    except Exception as e:
                        logger.warning(f"Erreur lors de la préparation des features avec l'encodeur avancé: {str(e)}")
                        logger.warning("Tentative avec la méthode de base...")
                        # Fallback vers la méthode de base
                        window_size = 5
                        if len(self.df) < window_size:
                            logger.warning("Pas assez de données pour préparer les features ML pour la prédiction.")
                        else:
                            features_list = []
                            for i in range(1, window_size + 1):
                                row = self.df.iloc[-i] 
                                numbers = row[self.number_cols].dropna().astype(int).tolist()
                                stars = row[self.star_cols].dropna().astype(int).tolist()
                                features_list.extend(numbers)
                                features_list.extend(stars)
                            
                            features_list.reverse()
                            expected_feature_length = window_size * (len(self.number_cols) + len(self.star_cols))
                            while len(features_list) < expected_feature_length:
                                features_list.append(0)
                            
                            features = np.array([features_list[:expected_feature_length]])
                            
                            # Charger les scalers sauvegardés si pas encore chargés
                            if not hasattr(self.scaler_numbers, 'mean_') or self.scaler_numbers.mean_ is None:
                                logger.warning("Le scaler_numbers n'est pas fitted. Tentative de chargement depuis le fichier sauvegardé...")
                                self._load_saved_scalers()
                            
                            if not hasattr(self.scaler_numbers, 'mean_') or self.scaler_numbers.mean_ is None:
                                logger.error("Le scaler_numbers n'est pas fitted et n'a pas pu être chargé. Impossible de faire des prédictions ML.")
                                logger.warning("Utilisation uniquement des statistiques de fréquence pour la prédiction.")
                                features_scaled_numbers = None
                                features_scaled_stars = None
                            else:
                                features_scaled_numbers = self.scaler_numbers.transform(features)
                                if hasattr(self, 'scaler_stars') and hasattr(self.scaler_stars, 'mean_') and self.scaler_stars.mean_ is not None:
                                    features_scaled_stars = self.scaler_stars.transform(features)
                                else:
                                    features_scaled_stars = features_scaled_numbers
                else:
                    # Méthode de base sans encodeur avancé
                    window_size = 5
                    if len(self.df) < window_size:
                        logger.warning("Pas assez de données pour préparer les features ML pour la prédiction.")
                        features_scaled_numbers = None
                        features_scaled_stars = None
                    else:
                        features_list = []
                        for i in range(1, window_size + 1):
                            row = self.df.iloc[-i] 
                            numbers = row[self.number_cols].dropna().astype(int).tolist()
                            stars = row[self.star_cols].dropna().astype(int).tolist()
                            features_list.extend(numbers)
                            features_list.extend(stars)
                        
                        features_list.reverse()
                        expected_feature_length = window_size * (len(self.number_cols) + len(self.star_cols))
                        while len(features_list) < expected_feature_length:
                            features_list.append(0)
                        
                        features = np.array([features_list[:expected_feature_length]])
                        
                        # Charger les scalers sauvegardés si pas encore chargés
                        if not hasattr(self.scaler_numbers, 'mean_') or self.scaler_numbers.mean_ is None:
                            logger.warning("Le scaler_numbers n'est pas fitted. Tentative de chargement depuis le fichier sauvegardé...")
                            self._load_saved_scalers()
                        
                        if not hasattr(self.scaler_numbers, 'mean_') or self.scaler_numbers.mean_ is None:
                            logger.error("Le scaler_numbers n'est pas fitted et n'a pas pu être chargé. Impossible de faire des prédictions ML.")
                            logger.warning("Utilisation uniquement des statistiques de fréquence pour la prédiction.")
                            features_scaled_numbers = None
                            features_scaled_stars = None
                        else:
                            features_scaled_numbers = self.scaler_numbers.transform(features)
                            if hasattr(self, 'scaler_stars') and hasattr(self.scaler_stars, 'mean_') and self.scaler_stars.mean_ is not None:
                                features_scaled_stars = self.scaler_stars.transform(features)
                            else:
                                features_scaled_stars = features_scaled_numbers
                
                # Utiliser les features préparées pour la prédiction
                if features_scaled_numbers is not None and features_scaled_stars is not None:
                    try:
                        # Vérifier que le nombre de features correspond
                        if hasattr(self.scaler_numbers, 'n_features_in_') and features_scaled_numbers.shape[1] != self.scaler_numbers.n_features_in_:
                            logger.error(f"❌ Nombre de features incompatible: {features_scaled_numbers.shape[1]} features en prédiction vs {self.scaler_numbers.n_features_in_} features attendues par le scaler")
                            logger.warning("Utilisation uniquement des statistiques de fréquence pour la prédiction.")
                        else:
                            # Prédire les probabilités avec les deux modèles pour les numéros
                            # ⚠️ CRITIQUE : predict_proba pour OneVsRestClassifier retourne un array de shape (n_samples, n_classes)
                            # Chaque colonne correspond à la probabilité de la classe correspondante (numéro 1-50)
                            try:
                                gb_number_pred_proba = self.number_model.predict_proba(features_scaled_numbers)
                                rf_number_pred_proba = self.rf_number_model.predict_proba(features_scaled_numbers)
                                
                                # ⚠️ CRITIQUE : Extraire les probabilités correctement selon la forme
                                # OneVsRestClassifier.predict_proba() retourne un array 2D (n_samples, n_classes)
                                if isinstance(gb_number_pred_proba, list):
                                    # Si c'est une liste (format OneVsRestClassifier avec liste), convertir
                                    # Chaque élément est un array (n_samples, 2) pour la classe binaire
                                    gb_number_pred_probs = np.array([proba[0][1] if proba.shape[1] > 1 else proba[0][0] for proba in gb_number_pred_proba])
                                elif gb_number_pred_proba.ndim == 2:
                                    # Array 2D : (n_samples, n_classes) - prendre la première ligne
                                    gb_number_pred_probs = gb_number_pred_proba[0]
                                else:
                                    # Array 1D : utiliser directement
                                    gb_number_pred_probs = gb_number_pred_proba.flatten()
                                
                                if isinstance(rf_number_pred_proba, list):
                                    rf_number_pred_probs = np.array([proba[0][1] if proba.shape[1] > 1 else proba[0][0] for proba in rf_number_pred_proba])
                                elif rf_number_pred_proba.ndim == 2:
                                    rf_number_pred_probs = rf_number_pred_proba[0]
                                else:
                                    rf_number_pred_probs = rf_number_pred_proba.flatten()
                                
                                # ⚠️ CRITIQUE : S'assurer que les arrays ont la même forme et la bonne longueur
                                # Normaliser à la longueur attendue (max_number = 50)
                                expected_len = self.config["max_number"]
                                if len(gb_number_pred_probs) != expected_len:
                                    logger.warning(f"Longueur GB inattendue: {len(gb_number_pred_probs)} (attendu: {expected_len})")
                                    if len(gb_number_pred_probs) < expected_len:
                                        # Compléter avec des zéros
                                        gb_number_pred_probs = np.pad(gb_number_pred_probs, (0, expected_len - len(gb_number_pred_probs)), 'constant')
                                    else:
                                        # Tronquer
                                        gb_number_pred_probs = gb_number_pred_probs[:expected_len]
                                
                                if len(rf_number_pred_probs) != expected_len:
                                    logger.warning(f"Longueur RF inattendue: {len(rf_number_pred_probs)} (attendu: {expected_len})")
                                    if len(rf_number_pred_probs) < expected_len:
                                        rf_number_pred_probs = np.pad(rf_number_pred_probs, (0, expected_len - len(rf_number_pred_probs)), 'constant')
                                    else:
                                        rf_number_pred_probs = rf_number_pred_probs[:expected_len]
                                
                                # Calculer la moyenne des probabilités ML pour les numéros
                                avg_ml_prob_numbers = (gb_number_pred_probs + rf_number_pred_probs) / 2.0
                                
                                # Prédire les probabilités avec les deux modèles pour les étoiles
                                gb_star_pred_proba = self.star_model.predict_proba(features_scaled_stars)
                                rf_star_pred_proba = self.rf_star_model.predict_proba(features_scaled_stars)
                                
                                # Même traitement pour les étoiles
                                if isinstance(gb_star_pred_proba, list):
                                    gb_star_pred_probs = np.array([proba[0][1] if proba.shape[1] > 1 else proba[0][0] for proba in gb_star_pred_proba])
                                elif gb_star_pred_proba.ndim == 2:
                                    gb_star_pred_probs = gb_star_pred_proba[0]
                                else:
                                    gb_star_pred_probs = gb_star_pred_proba.flatten()
                                
                                if isinstance(rf_star_pred_proba, list):
                                    rf_star_pred_probs = np.array([proba[0][1] if proba.shape[1] > 1 else proba[0][0] for proba in rf_star_pred_proba])
                                elif rf_star_pred_proba.ndim == 2:
                                    rf_star_pred_probs = rf_star_pred_proba[0]
                                else:
                                    rf_star_pred_probs = rf_star_pred_proba.flatten()
                                
                                # Normaliser à la longueur attendue pour les étoiles (max_star = 12)
                                expected_star_len = self.config["max_star"]
                                if len(gb_star_pred_probs) != expected_star_len:
                                    if len(gb_star_pred_probs) < expected_star_len:
                                        gb_star_pred_probs = np.pad(gb_star_pred_probs, (0, expected_star_len - len(gb_star_pred_probs)), 'constant')
                                    else:
                                        gb_star_pred_probs = gb_star_pred_probs[:expected_star_len]
                                
                                if len(rf_star_pred_probs) != expected_star_len:
                                    if len(rf_star_pred_probs) < expected_star_len:
                                        rf_star_pred_probs = np.pad(rf_star_pred_probs, (0, expected_star_len - len(rf_star_pred_probs)), 'constant')
                                    else:
                                        rf_star_pred_probs = rf_star_pred_probs[:expected_star_len]
                                
                                # Calculer la moyenne des probabilités ML pour les étoiles
                                avg_ml_prob_stars = (gb_star_pred_probs + rf_star_pred_probs) / 2.0
                                
                                # Combiner avec les probabilités existantes
                                weight = self.config["prediction_weight"]
                                for num in range(1, self.config["max_number"] + 1):
                                    idx = num - 1
                                    if idx < len(avg_ml_prob_numbers):
                                        ml_prob = float(avg_ml_prob_numbers[idx])
                                    else:
                                        ml_prob = 0.0
                                    number_probs[num] = number_probs.get(num, 0) * (1 - weight) + ml_prob * weight
                                
                                for star in range(1, self.config["max_star"] + 1):
                                    idx = star - 1
                                    if idx < len(avg_ml_prob_stars):
                                        ml_prob = float(avg_ml_prob_stars[idx])
                                    else:
                                        ml_prob = 0.0
                                    star_probs[star] = star_probs.get(star, 0) * (1 - weight) + ml_prob * weight
                                
                                logger.debug(f"✅ Prédictions ML combinées: {len(avg_ml_prob_numbers)} numéros, {len(avg_ml_prob_stars)} étoiles")
                                
                                # 4. Utiliser le prédicteur quantique si disponible
                                if self.quantum_predictor is not None and self.config.get('use_quantum', False):
                                    try:
                                        logger.info("🌌 Utilisation du prédicteur quantique pour optimiser la sélection...")
                                        # Normaliser les probabilités pour le prédicteur quantique
                                        number_probs_normalized = {k: max(0.0, v) for k, v in number_probs.items()}
                                        star_probs_normalized = {k: max(0.0, v) for k, v in star_probs.items()}
                                        
                                        # Normaliser pour que la somme soit 1
                                        number_sum = sum(number_probs_normalized.values())
                                        star_sum = sum(star_probs_normalized.values())
                                        if number_sum > 0:
                                            number_probs_normalized = {k: v / number_sum for k, v in number_probs_normalized.items()}
                                        if star_sum > 0:
                                            star_probs_normalized = {k: v / star_sum for k, v in star_probs_normalized.items()}
                                        
                                        # Utiliser le prédicteur quantique pour optimiser la sélection
                                        quantum_numbers, quantum_stars = self.quantum_predictor.predict(
                                            features=features_scaled_numbers[0] if 'features_scaled_numbers' in locals() else None,
                                            historical_data=self.df,
                                            number_probs=number_probs_normalized,
                                            star_probs=star_probs_normalized
                                        )
                                        
                                        # Mélanger les prédictions quantiques avec les probabilités classiques (poids 0.3 pour le quantique)
                                        quantum_weight = 0.3
                                        for num in quantum_numbers:
                                            number_probs[num] = number_probs.get(num, 0) * (1 - quantum_weight) + quantum_weight
                                        for star in quantum_stars:
                                            star_probs[star] = star_probs.get(star, 0) * (1 - quantum_weight) + quantum_weight
                                        
                                        logger.info(f"✅ Prédictions quantiques appliquées: {quantum_numbers}, {quantum_stars}")
                                    except Exception as e:
                                        logger.warning(f"Erreur lors de l'utilisation du prédicteur quantique: {str(e)}")
                                        logger.debug(traceback.format_exc())
                            except Exception as pred_error:
                                logger.error(f"Erreur lors du calcul des probabilités ML: {str(pred_error)}")
                                logger.debug(traceback.format_exc())
                                # Ne pas lever l'exception, continuer avec les statistiques de fréquence
                                logger.warning("Utilisation uniquement des statistiques de fréquence pour la prédiction.")
                    except Exception as e:
                        logger.error(f"Erreur lors de la préparation des features ML: {str(e)}")
                        logger.debug(traceback.format_exc())
                        logger.warning("Utilisation uniquement des statistiques de fréquence pour la prédiction.")
                else:
                    logger.warning("Features ML non disponibles. Utilisation uniquement des statistiques de fréquence.")
            
            # 4. Ajuster avec les corrélations si disponibles
            if self.number_correlations is not None and self.config["use_correlation"]:
                # Obtenir les derniers numéros tirés
                last_numbers = self.df.iloc[-1][self.number_cols].dropna().astype(int).tolist()
                
                for num in range(1, self.config["max_number"] + 1):
                    # Calculer la corrélation moyenne avec les derniers numéros tirés
                    corr_sum = 0
                    count = 0
                    
                    for last_num in last_numbers:
                        if 1 <= last_num <= self.config["max_number"] and num in self.number_correlations.index and last_num in self.number_correlations.columns:
                            corr_val = self.number_correlations.loc[num, last_num]
                            # Ajouter une pondération pour la corrélation (ex: 0.1 pour la corrélation)
                            corr_sum += corr_val
                            count += 1
                    
                    if count > 0:
                        avg_corr = corr_sum / count
                        # Ajuster la probabilité en fonction de la corrélation (normaliser la corrélation de -1 à 1 vers 0 à 1)
                        # Puis mélanger avec la probabilité existante.
                        # Un poids de 0.2 est utilisé pour la corrélation.
                        number_probs[num] = number_probs.get(num, 0) * (1 - 0.2) + ((avg_corr + 1) / 2) * 0.2
            
            if self.star_correlations is not None and self.config["use_correlation"]:
                last_stars = self.df.iloc[-1][self.star_cols].dropna().astype(int).tolist()
                
                for star in range(1, self.config["max_star"] + 1):
                    corr_sum = 0
                    count = 0
                    
                    for last_star in last_stars:
                        if 1 <= last_star <= self.config["max_star"] and star in self.star_correlations.index and last_star in self.star_correlations.columns:
                            corr_val = self.star_correlations.loc[star, last_star]
                            corr_sum += corr_val
                            count += 1
                    
                    if count > 0:
                        avg_corr = corr_sum / count
                        star_probs[star] = star_probs.get(star, 0) * (1 - 0.2) + ((avg_corr + 1) / 2) * 0.2
            
            # 5. Appliquer la pondération Fibonacci inversée si activée
            if self.config.get("use_fibonacci_inverse", False):
                logger.info("Application de la pondération Fibonacci inversée...")
                
                # Créer des compteurs à partir des probabilités actuelles
                # Utilise les probabilités existantes pour déterminer l'ordre de pondération
                # Multiplier par 1000 pour donner une base entière pour Counter
                number_counter = Counter({num: int(prob * 1000) for num, prob in number_probs.items()})
                star_counter = Counter({star: int(prob * 1000) for star, prob in star_probs.items()})
                
                # Appliquer la pondération Fibonacci inversée
                # reverse_order=True signifie que les éléments avec les plus petites "fréquences" (probabilités ici)
                # recevront les poids Fibonacci les plus élevés.
                fibonacci_number_weights = apply_inverse_fibonacci_weights(number_counter, reverse_order=True)
                fibonacci_star_weights = apply_inverse_fibonacci_weights(star_counter, reverse_order=True)
                
                # Obtenir le poids de mélange depuis la configuration
                blend_weight = self.config.get("fibonacci_inverse_weight_blend", 0.5) # Par défaut 0.5 si non spécifié
                
                # Combiner avec les probabilités existantes
                for num in range(1, self.config["max_number"] + 1):
                    fib_weight = fibonacci_number_weights.get(num, 0.0) # Assurer que c'est un float
                    # Nouvelle probabilité = (Probabilité actuelle * (1 - poids de mélange)) + (Poids Fibonacci * poids de mélange)
                    number_probs[num] = number_probs.get(num, 0.0) * (1 - blend_weight) + fib_weight * blend_weight
                
                for star in range(1, self.config["max_star"] + 1):
                    fib_weight = fibonacci_star_weights.get(star, 0.0) # Assurer que c'est un float
                    star_probs[star] = star_probs.get(star, 0.0) * (1 - blend_weight) + fib_weight * blend_weight
                
                logger.info(f"Pondération Fibonacci inversée appliquée (poids de mélange: {blend_weight})")
            
            # 6. Sélectionner les numéros et étoiles avec les probabilités les plus élevées
            # Normaliser les probabilités pour qu'elles somment à 1, si nécessaire pour la sélection pondérée
            total_num_prob = sum(number_probs.values())
            if total_num_prob > 0:
                number_probs = {num: prob / total_num_prob for num, prob in number_probs.items()}
            
            total_star_prob = sum(star_probs.values())
            if total_star_prob > 0:
                star_probs = {star: prob / total_star_prob for star, prob in star_probs.items()}

            # Stocker les prédictions (probabilités finales)
            self.number_predictions = number_probs
            self.star_predictions = star_probs

            # Sélection des numéros et étoiles prédits (les plus probables)
            sorted_numbers = sorted(number_probs.items(), key=lambda x: x[1], reverse=True)
            sorted_stars = sorted(star_probs.items(), key=lambda x: x[1], reverse=True)
            
            predicted_numbers = [num for num, _ in sorted_numbers[:self.config["propose_size"]]]
            predicted_stars = [star for star, _ in sorted_stars[:self.config["star_size"]]]
            
            logger.info(f"Prédiction: numéros {predicted_numbers}, étoiles {predicted_stars}")
            
            return predicted_numbers, predicted_stars
            
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction des numéros: {str(e)}")
            logger.debug(traceback.format_exc())
            return [], []
    
    def predict_next_draw(self) -> Tuple[List[int], List[int]]:
        """
        Prédit les numéros et les étoiles pour le prochain tirage.
        Alias pour predict_numbers.
        
        Returns:
            Tuple contenant:
            - Liste des numéros prédits
            - Liste des étoiles prédites
        """
        return self.predict_numbers()
    
    def generate_combinations(self, number_scores: Dict[int, float], star_scores: Dict[int, float]) -> List[Tuple[List[int], List[int]]]:
        """
        Génère plusieurs combinaisons optimisées pour le prochain tirage,
        en utilisant les scores finaux fournis.
        
        Args:
            number_scores (Dict[int, float]): Scores finaux pour les numéros.
            star_scores (Dict[int, float]): Scores finaux pour les étoiles.

        Returns:
            Liste de tuples, chaque tuple contenant:
            - Liste des numéros
            - Liste des étoiles
        """
        logger.info(f"Génération de {self.config['combinations_to_generate']} combinaisons optimisées...")
        
        try:
            combinations = []
            
            # Stratégie 1: Utiliser les numéros chauds et froids
            # S'assurer que les listes hot/cold ne sont pas vides
            hot_numbers = self.hot[:self.config["propose_size"]] if self.hot else []
            cold_numbers = self.cold[:self.config["propose_size"]] if self.cold else []
            
            hot_stars = self.star_hot[:self.config["star_size"]] if self.star_hot else []
            cold_stars = self.star_cold[:self.config["star_size"]] if self.star_cold else []
            
            # Compléter si nécessaire avec des numéros/étoiles aléatoires mais valides
            def complete_list(current_list, max_val, target_size):
                if len(current_list) < target_size:
                    available = list(set(range(1, max_val + 1)) - set(current_list))
                    if len(available) > 0:
                        current_list.extend(random.sample(available, min(target_size - len(current_list), len(available))))
                return current_list

            hot_numbers = complete_list(hot_numbers, self.config["max_number"], self.config["propose_size"])
            cold_numbers = complete_list(cold_numbers, self.config["max_number"], self.config["propose_size"])
            hot_stars = complete_list(hot_stars, self.config["max_star"], self.config["star_size"])
            cold_stars = complete_list(cold_stars, self.config["max_star"], self.config["star_size"])

            # Ajouter les combinaisons de base (si complètes)
            if len(hot_numbers) == self.config["propose_size"] and len(hot_stars) == self.config["star_size"]:
                combinations.append((sorted(hot_numbers), sorted(hot_stars)))
            if len(cold_numbers) == self.config["propose_size"] and len(cold_stars) == self.config["star_size"]:
                combinations.append((sorted(cold_numbers), sorted(cold_stars)))
            
            # Stratégie 2: Mélanger chauds et froids
            if len(hot_numbers) >= 3 and len(cold_numbers) >= 2 and len(hot_stars) >= 1 and len(cold_stars) >= 1:
                mixed_numbers = random.sample(hot_numbers, 3) + random.sample(cold_numbers, 2)
                mixed_stars = random.sample(hot_stars, 1) + random.sample(cold_stars, 1)
                combinations.append((sorted(mixed_numbers), sorted(mixed_stars)))
            
            # Stratégie 3: Utiliser les prédictions directes (déjà pondérées par ML, Fibonacci, etc.)
            predicted_numbers, predicted_stars = self.predict_next_draw() # Utilise predict_next_draw qui est un alias
            if predicted_numbers and predicted_stars:
                combinations.append((sorted(predicted_numbers), sorted(predicted_stars)))
            
            # Stratégie 4: Simulation Monte Carlo avec filtrage
            logger.info("Démarrage de la simulation Monte Carlo pour la génération de combinaisons...")
            monte_carlo_combinations = self._run_monte_carlo_simulation(number_scores, star_scores)
            
            # Ajouter les combinaisons filtrées de Monte Carlo
            combinations.extend(monte_carlo_combinations)
            
            # Stratégie 5: Système réducteur (Wheeling System)
            # Utiliser les scores pour sélectionner les numéros/étoiles à inclure dans le système réducteur
            if self.config.get("use_wheeling_system", True): # Ajouter au config si besoin
                logger.info("Génération de combinaisons via système réducteur...")
                
                num_to_wheel_count = self.config.get("wheeling_num_count", 10)
                star_to_wheel_count = self.config.get("wheeling_star_count", 5)
                
                if number_scores and star_scores:
                    # Sélectionner les numéros/étoiles les plus prometteurs selon les scores combinés
                    numbers_to_wheel = sorted(number_scores, key=number_scores.get, reverse=True)[:num_to_wheel_count]
                    stars_to_wheel = sorted(star_scores, key=star_scores.get, reverse=True)[:star_to_wheel_count]
                    
                    wheeled_combinations = self._generate_wheeling_combinations(numbers_to_wheel, stars_to_wheel)
                    combinations.extend(wheeled_combinations)
                else:
                    logger.warning("Scores non disponibles pour la sélection des numéros/étoiles pour le système réducteur.")
            
            # S'assurer qu'on a le bon nombre de combinaisons uniques
            unique_combinations = []
            seen_combos = set()
            for nums, stars in combinations:
                combo_tuple = (tuple(sorted(nums)), tuple(sorted(stars)))
                if combo_tuple not in seen_combos:
                    unique_combinations.append((sorted(nums), sorted(stars)))
                    seen_combos.add(combo_tuple)
            
            # Si pas assez de combinaisons uniques, compléter avec des aléatoires pondérées (sans filtre avancé)
            # Cette boucle est une sécurité pour atteindre le nombre désiré
            while len(unique_combinations) < self.config["combinations_to_generate"]:
                logger.warning(f"Pas assez de combinaisons uniques ({len(unique_combinations)}), ajout de combinaisons aléatoires pondérées supplémentaires.")
                nums, stars = self._generate_weighted_random_combination(number_scores, star_scores)
                combo_tuple = (tuple(sorted(nums)), tuple(sorted(stars)))
                if combo_tuple not in seen_combos:
                    unique_combinations.append((sorted(nums), sorted(stars)))
                    seen_combos.add(combo_tuple)
                # Ajouter une petite sécurité pour éviter boucle infinie si scores très concentrés
                if len(seen_combos) > self.config["monte_carlo_simulations"] * 2: # Limite d'essais pour la complétion
                    logger.error("Impossible de générer suffisamment de combinaisons uniques, arrêt.")
                    break

            # Sélectionner le nombre final de combinaisons
            final_combinations = unique_combinations[:self.config["combinations_to_generate"]]

            # Enregistrer l'historique des combinaisons générées
            current_date = datetime.now().strftime("%Y-%m-%d") # Ou utiliser la date du dernier tirage + 1 ?
            self.generated_combinations_history.append((current_date, final_combinations))
            # Limiter la taille de l'historique si nécessaire
            max_history = self.config.get("max_combination_history", 100)
            if len(self.generated_combinations_history) > max_history:
                self.generated_combinations_history.pop(0)

            logger.info(f"{len(final_combinations)} combinaisons générées et filtrées")
            
            return final_combinations
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération des combinaisons: {str(e)}")
            logger.debug(traceback.format_exc())
            return []
    
    def run_analysis(self) -> bool:
        """
        Exécute l'analyse complète.
        
        Returns:
            bool: True si l'analyse a réussi, False sinon
        """
        logger.info("Démarrage de l'analyse avancée des tirages d'Euromillions...")
        
        try:
            # 1. Chargement des données
            if not self.load_data():
                return False
            
            # 2. Calcul des statistiques globales
            self.compute_global_stats()
            
            # 3. Calcul des statistiques sur la fenêtre récente
            self.compute_window_stats()
            
            # 4. Identification des numéros chauds et froids
            self.identify_hot_cold()
            
            # 5. Analyse des corrélations
            if self.config["use_correlation"]:
                self.analyze_correlations()
            
            # 6. Analyse des tendances temporelles
            if self.config["use_temporal"]:
                self.analyze_temporal_patterns()
            
            # 7. Analyse des clusters
            if self.config["use_clustering"]:
                self.analyze_clustering()
            
            # 8. Analyses supplémentaires
            if self.config["analyze_parity"]:
                self.analyze_parity()
            
            if self.config["analyze_sum"]:
                self.analyze_sum()
            
            self.analyze_sequences()

            # 9. Calcul des scores d'écart (nécessaire avant compute_scores)
            self.compute_gap_scores()
            
            # 10. Entraînement des modèles ML
            if self.config["use_ml"]:
                self.train_ml_models()
            
            # 11. Prédiction des numéros (génère self.number_predictions et self.star_predictions)
            self.predict_numbers()

            # 12. Calcul des scores finaux combinés
            self.final_number_scores, self.final_star_scores = self.compute_scores()
            
            logger.info("Analyse terminée avec succès")
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse: {str(e)}")
            logger.debug(traceback.format_exc())
            return False
    
    def generate_report(self) -> str:
        """
        Génère un rapport détaillé de l'analyse.
        
        Returns:
            str: Chemin du fichier de rapport
        """
        logger.info("Génération du rapport d'analyse...")
        
        try:
            # Créer le répertoire de sortie s'il n'existe pas
            if not self.output_dir.exists():
                self.output_dir.mkdir(parents=True)
            
            # Nom du fichier de rapport
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = self.output_dir / f"rapport_euromillions_{timestamp}.txt"
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("RAPPORT D'ANALYSE AVANCÉE DES TIRAGES EUROMILLIONS\n")
                f.write("=" * 80 + "\n\n")
                
                f.write(f"Date de l'analyse: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Fichier analysé: {self.config['csv_file']}\n")
                f.write(f"Nombre de tirages: {len(self.df)}\n\n")
                
                # Statistiques globales
                f.write("-" * 80 + "\n")
                f.write("STATISTIQUES GLOBALES\n")
                f.write("-" * 80 + "\n\n")
                
                f.write("Fréquence des numéros principaux:\n")
                if self.freq:
                    for num, count in sorted(self.freq.items()):
                        f.write(f"  {num}: {count} fois ({count/(len(self.df)*5)*100:.2f}%)\n")
                
                f.write("\nFréquence des étoiles:\n")
                if self.star_freq:
                    for star, count in sorted(self.star_freq.items()):
                        f.write(f"  {star}: {count} fois ({count/(len(self.df)*2)*100:.2f}%)\n")
                
                # Numéros chauds et froids
                f.write("\n" + "-" * 80 + "\n")
                f.write("NUMÉROS CHAUDS ET FROIDS\n")
                f.write("-" * 80 + "\n\n")
                
                f.write(f"Numéros chauds (les plus fréquents sur les {self.config['window_draws']} derniers tirages):\n")
                if self.hot:
                    for num in self.hot:
                        count = self.window_counts.get(num, 0)
                        f.write(f"  {num}: {count} fois ({count/(min(self.config['window_draws'], len(self.df))*5)*100:.2f}%)\n")
                
                f.write(f"\nNuméros froids (les moins fréquents sur les {self.config['window_draws']} derniers tirages):\n")
                if self.cold:
                    for num in self.cold:
                        count = self.window_counts.get(num, 0)
                        f.write(f"  {num}: {count} fois ({count/(min(self.config['window_draws'], len(self.df))*5)*100:.2f}%)\n")
                
                f.write(f"\nÉtoiles chaudes (les plus fréquentes sur les {self.config['window_draws']} derniers tirages):\n")
                if self.star_hot:
                    for star in self.star_hot:
                        count = self.star_window_counts.get(star, 0)
                        f.write(f"  {star}: {count} fois ({count/(min(self.config['window_draws'], len(self.df))*2)*100:.2f}%)\n")
                
                f.write(f"\nÉtoiles froides (les moins fréquentes sur les {self.config['window_draws']} derniers tirages):\n")
                if self.star_cold:
                    for star in self.star_cold:
                        count = self.star_window_counts.get(star, 0)
                        f.write(f"  {star}: {count} fois ({count/(min(self.config['window_draws'], len(self.df))*2)*100:.2f}%)\n")
                
                # Analyses supplémentaires
                if self.parity_stats:
                    f.write("\n" + "-" * 80 + "\n")
                    f.write("ANALYSE DE PARITÉ\n")
                    f.write("-" * 80 + "\n\n")
                    
                    f.write("Distribution des numéros pairs/impairs:\n")
                    for pattern, stats in sorted(self.parity_stats.items()):
                        f.write(f"  {pattern}: {stats['count']} fois ({stats['percentage']:.2f}%)\n")
                
                if self.sum_stats:
                    f.write("\n" + "-" * 80 + "\n")
                    f.write("ANALYSE DE SOMME\n")
                    f.write("-" * 80 + "\n\n")
                    
                    f.write(f"Somme minimale: {self.sum_stats['min']}\n")
                    f.write(f"Somme maximale: {self.sum_stats['max']}\n")
                    f.write(f"Somme moyenne: {self.sum_stats['avg']:.2f}\n")
                    f.write(f"Somme médiane: {self.sum_stats['median']:.2f}\n")
                    f.write(f"Écart-type: {self.sum_stats['std']:.2f}\n\n")
                    
                    f.write("Distribution des sommes par plage:\n")
                    for range_key, stats in sorted(self.sum_ranges.items()):
                        f.write(f"  {range_key}: {stats['count']} fois ({stats['percentage']:.2f}%)\n")
                    
                    f.write("\nSommes les plus fréquentes:\n")
                    for sum_val, count in self.most_common_sums:
                        f.write(f"  {sum_val}: {count} fois\n")
                
                if self.sequence_stats:
                    f.write("\n" + "-" * 80 + "\n")
                    f.write("ANALYSE DES SÉQUENCES\n")
                    f.write("-" * 80 + "\n\n")
                    
                    f.write("Distribution des séquences de numéros consécutifs:\n")
                    for seq_count, stats in sorted(self.sequence_stats.items()):
                        f.write(f"  {seq_count} séquence(s): {stats['count']} fois ({stats['percentage']:.2f}%)\n")
                
                # Prédictions
                f.write("\n" + "-" * 80 + "\n")
                f.write("PRÉDICTIONS POUR LE PROCHAIN TIRAGE\n")
                f.write("-" * 80 + "\n\n")
                
                # Utiliser les scores finaux pour la prédiction affichée
                predicted_numbers = sorted(self.final_number_scores, key=self.final_number_scores.get, reverse=True)[:self.config["propose_size"]]
                predicted_stars = sorted(self.final_star_scores, key=self.final_star_scores.get, reverse=True)[:self.config["star_size"]]
                
                f.write(f"Numéros prédits: {', '.join(map(str, predicted_numbers))}\n")
                f.write(f"Étoiles prédites: {', '.join(map(str, predicted_stars))}\n\n")
                
                f.write("Top 10 numéros avec leurs scores finaux:\n")
                if self.final_number_scores:
                    sorted_numbers = sorted(self.final_number_scores.items(), key=lambda x: x[1], reverse=True)[:10]
                    for num, score in sorted_numbers:
                        f.write(f"  {num}: {score:.4f}\n")
                
                f.write("\nTop 5 étoiles avec leurs scores finaux:\n")
                if self.final_star_scores:
                    sorted_stars = sorted(self.final_star_scores.items(), key=lambda x: x[1], reverse=True)[:5]
                    for star, score in sorted_stars:
                        f.write(f"  {star}: {score:.4f}\n")
                
                # Combinaisons optimisées
                f.write("\n" + "-" * 80 + "\n")
                f.write("COMBINAISONS OPTIMISÉES\n")
                f.write("-" * 80 + "\n\n")
                
                # Passer les scores finaux à generate_combinations
                combinations = self.generate_combinations(self.final_number_scores, self.final_star_scores)
                
                for i, (numbers, stars) in enumerate(combinations):
                    f.write(f"Combinaison {i+1}: {' - '.join(map(str, numbers))} | {' - '.join(map(str, stars))}\n")
                
                f.write("\n" + "=" * 80 + "\n")
                f.write("FIN DU RAPPORT\n")
                f.write("=" * 80 + "\n")
            
            logger.info(f"Rapport généré: {report_file}")
            
            return str(report_file)
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération du rapport: {str(e)}")
            logger.debug(traceback.format_exc())
            return ""
    
    def generate_visualizations(self) -> List[str]:
        """
        Génère des visualisations des résultats de l'analyse.
        
        Returns:
            List[str]: Liste des chemins des fichiers de visualisation
        """
        logger.info("Génération des visualisations...")
        
        try:
            # Créer le répertoire de visualisations
            vis_dir = self.output_dir / "visualizations"
            if not vis_dir.exists():
                vis_dir.mkdir(parents=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_files = []
            
            # 1. Distribution des fréquences des numéros
            plt.figure(figsize=(12, 8))
            
            if self.freq:
                nums = list(range(1, self.config["max_number"] + 1))
                freqs = [self.freq.get(num, 0) for num in nums]
                
                plt.bar(nums, freqs, color='royalblue')
                plt.axhline(y=sum(freqs) / len(nums), color='r', linestyle='-', label='Moyenne')
                
                plt.title('Distribution des fréquences des numéros principaux')
                plt.xlabel('Numéro')
                plt.ylabel('Fréquence')
                plt.xticks(nums[::5])
                plt.grid(axis='y', alpha=0.3)
                plt.legend()
                
                plot_path = vis_dir / f"number_frequency_{timestamp}.png"
                plt.tight_layout()
                plt.savefig(plot_path)
                plt.close()
                
                plot_files.append(str(plot_path))
            
            # 2. Distribution des fréquences des étoiles
            plt.figure(figsize=(10, 6))
            
            if self.star_freq:
                stars = list(range(1, self.config["max_star"] + 1))
                star_freqs = [self.star_freq.get(star, 0) for star in stars]
                
                plt.bar(stars, star_freqs, color='gold')
                plt.axhline(y=sum(star_freqs) / len(stars), color='r', linestyle='-', label='Moyenne')
                
                plt.title('Distribution des fréquences des étoiles')
                plt.xlabel('Étoile')
                plt.ylabel('Fréquence')
                plt.xticks(stars)
                plt.grid(axis='y', alpha=0.3)
                plt.legend()
                
                plot_path = vis_dir / f"star_frequency_{timestamp}.png"
                plt.tight_layout()
                plt.savefig(plot_path)
                plt.close()
                
                plot_files.append(str(plot_path))
            
            # 3. Numéros chauds et froids
            plt.figure(figsize=(12, 8))
            
            if self.window_counts:
                all_nums = list(range(1, self.config["max_number"] + 1))
                all_freqs = [self.window_counts.get(num, 0) for num in all_nums]
                
                colors = ['royalblue'] * self.config["max_number"]
                
                # Marquer les numéros chauds en rouge
                if self.hot:
                    for num in self.hot:
                        if 1 <= num <= self.config["max_number"]:
                            colors[num-1] = 'crimson'
                
                # Marquer les numéros froids en bleu clair
                if self.cold:
                    for num in self.cold:
                        if 1 <= num <= self.config["max_number"]:
                            colors[num-1] = 'skyblue'
                
                plt.bar(all_nums, all_freqs, color=colors)
                
                plt.title(f'Numéros chauds et froids (sur les {self.config["window_draws"]} derniers tirages)')
                plt.xlabel('Numéro')
                plt.ylabel('Fréquence')
                plt.xticks(all_nums[::5])
                plt.grid(axis='y', alpha=0.3)
                
                # Légende personnalisée
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='crimson', label='Numéros chauds'),
                    Patch(facecolor='skyblue', label='Numéros froids'),
                    Patch(facecolor='royalblue', label='Autres numéros')
                ]
                plt.legend(handles=legend_elements)
                
                plot_path = vis_dir / f"hot_cold_numbers_{timestamp}.png"
                plt.tight_layout()
                plt.savefig(plot_path)
                plt.close()
                
                plot_files.append(str(plot_path))
            
            # 4. Analyse de parité
            if self.parity_stats:
                plt.figure(figsize=(10, 6))
                
                patterns = list(self.parity_stats.keys())
                counts = [self.parity_stats[p]['count'] for p in patterns]
                
                plt.bar(patterns, counts, color='mediumseagreen')
                
                plt.title('Distribution des combinaisons de parité')
                plt.xlabel('Combinaison (Pairs-Impairs)')
                plt.ylabel('Nombre de tirages')
                plt.grid(axis='y', alpha=0.3)
                
                plot_path = vis_dir / f"parity_distribution_{timestamp}.png"
                plt.tight_layout()
                plt.savefig(plot_path)
                plt.close()
                
                plot_files.append(str(plot_path))
            
            # 5. Analyse de somme
            if self.sum_stats and self.sum_ranges:
                plt.figure(figsize=(12, 6))
                
                ranges = list(self.sum_ranges.keys())
                counts = [self.sum_ranges[r]['count'] for r in ranges]
                
                plt.bar(ranges, counts, color='purple')
                
                plt.title('Distribution des sommes par plage')
                plt.xlabel('Plage de somme')
                plt.ylabel('Nombre de tirages')
                plt.xticks(rotation=45)
                plt.grid(axis='y', alpha=0.3)
                
                plot_path = vis_dir / f"sum_distribution_{timestamp}.png"
                plt.tight_layout()
                plt.savefig(plot_path)
                plt.close()
                
                plot_files.append(str(plot_path))
            
            # 6. Prédictions
            if self.number_predictions and self.star_predictions:
                plt.figure(figsize=(14, 10))
                
                # Sous-graphique pour les numéros
                plt.subplot(2, 1, 1)
                
                sorted_numbers = sorted(self.number_predictions.items(), key=lambda x: x[1], reverse=True)[:15]
                nums = [num for num, _ in sorted_numbers]
                probs = [prob for _, prob in sorted_numbers]
                
                bars = plt.bar(nums, probs, color='royalblue')
                
                # Marquer les numéros prédits
                # Utiliser les scores finaux pour la sélection des numéros prédits
                predicted_numbers = sorted(self.final_number_scores, key=self.final_number_scores.get, reverse=True)[:self.config["propose_size"]]
                for i, num in enumerate(nums):
                    if num in predicted_numbers:
                        bars[i].set_color('crimson')
                
                plt.title('Probabilités des numéros principaux (top 15)')
                plt.xlabel('Numéro')
                plt.ylabel('Probabilité')
                plt.grid(axis='y', alpha=0.3)
                
                # Sous-graphique pour les étoiles
                plt.subplot(2, 1, 2)
                
                sorted_stars = sorted(self.star_predictions.items(), key=lambda x: x[1], reverse=True)
                stars = [star for star, _ in sorted_stars]
                star_probs = [prob for _, prob in sorted_stars]
                
                bars = plt.bar(stars, star_probs, color='gold')
                
                # Marquer les étoiles prédites
                # Utiliser les scores finaux pour la sélection des étoiles prédites
                predicted_stars = sorted(self.final_star_scores, key=self.final_star_scores.get, reverse=True)[:self.config["star_size"]]
                for i, star in enumerate(stars):
                    if star in predicted_stars:
                        bars[i].set_color('crimson')
                
                plt.title('Probabilités des étoiles')
                plt.xlabel('Étoile')
                plt.ylabel('Probabilité')
                plt.grid(axis='y', alpha=0.3)
                
                plot_path = vis_dir / f"predictions_{timestamp}.png"
                plt.tight_layout()
                plt.savefig(plot_path)
                plt.close()
                
                plot_files.append(str(plot_path))
            
            # 7. Évolution temporelle si disponible
            if self.temporal_patterns and 'Date' in self.df.columns:
                plt.figure(figsize=(14, 8))
                
                # Sélectionner quelques numéros représentatifs
                if self.hot and self.cold:
                    selected_nums = self.hot[:3] + self.cold[:2]
                else:
                    selected_nums = list(range(1, 6))
                
                for num in selected_nums:
                    if num in self.temporal_patterns:
                        periods = [str(p) for p, _ in self.temporal_patterns[num]]
                        freqs = [f for _, f in self.temporal_patterns[num]]
                        
                        plt.plot(periods, freqs, marker='o', label=f'Numéro {num}')
                
                plt.title('Évolution temporelle des fréquences')
                plt.xlabel('Période')
                plt.ylabel('Fréquence relative')
                plt.xticks(rotation=45)
                plt.grid(alpha=0.3)
                plt.legend()
                
                plot_path = vis_dir / f"temporal_evolution_{timestamp}.png"
                plt.tight_layout()
                plt.savefig(plot_path)
                plt.close()
                
                plot_files.append(str(plot_path))
            
            logger.info(f"{len(plot_files)} visualisations générées")
            
            return plot_files
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération des visualisations: {str(e)}")
            logger.debug(traceback.format_exc())
            return []
    
    def export_to_excel(self) -> str:
        """
        Exporte les résultats de l'analyse vers un fichier Excel.
        
        Returns:
            str: Chemin du fichier Excel
        """
        if not self.config["export_excel"]:
            return ""
            
        logger.info("Export des résultats vers Excel...")
        
        try:
            # Vérifier si pandas a la fonctionnalité d'export Excel
            if not hasattr(pd.DataFrame, 'to_excel'):
                logger.error("Fonctionnalité d'export Excel non disponible dans pandas")
                return ""
            
            # Créer le répertoire de sortie s'il n'existe pas
            if not self.output_dir.exists():
                self.output_dir.mkdir(parents=True)
            
            # Nom du fichier Excel
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            excel_file = self.output_dir / f"euromillions_analysis_{timestamp}.xlsx"
            
            # Créer un writer Excel
            with pd.ExcelWriter(excel_file) as writer:
                # Feuille 1: Statistiques des numéros
                if self.freq:
                    df_numbers = pd.DataFrame({
                        'Numéro': list(range(1, self.config["max_number"] + 1)),
                        'Fréquence globale': [self.freq.get(num, 0) for num in range(1, self.config["max_number"] + 1)],
                        'Fréquence récente': [self.window_counts.get(num, 0) if self.window_counts else 0 for num in range(1, self.config["max_number"] + 1)],
                        'Score final prédit': [self.final_number_scores.get(num, 0) if hasattr(self, 'final_number_scores') and self.final_number_scores else 0 for num in range(1, self.config["max_number"] + 1)]
                    })
                    
                    # Ajouter des colonnes pour les numéros chauds/froids
                    df_numbers['Est chaud'] = df_numbers['Numéro'].apply(lambda x: 'Oui' if self.hot and x in self.hot else 'Non')
                    df_numbers['Est froid'] = df_numbers['Numéro'].apply(lambda x: 'Oui' if self.cold and x in self.cold else 'Non')
                    
                    df_numbers.to_excel(writer, sheet_name='Statistiques Numéros', index=False)
                
                # Feuille 2: Statistiques des étoiles
                if self.star_freq:
                    df_stars = pd.DataFrame({
                        'Étoile': list(range(1, self.config["max_star"] + 1)),
                        'Fréquence globale': [self.star_freq.get(star, 0) for star in range(1, self.config["max_star"] + 1)],
                        'Fréquence récente': [self.star_window_counts.get(star, 0) if self.star_window_counts else 0 for star in range(1, self.config["max_star"] + 1)],
                        'Score final prédit': [self.final_star_scores.get(star, 0) if hasattr(self, 'final_star_scores') and self.final_star_scores else 0 for star in range(1, self.config["max_star"] + 1)]
                    })
                    
                    df_stars['Est chaude'] = df_stars['Étoile'].apply(lambda x: 'Oui' if self.star_hot and x in self.star_hot else 'Non')
                    df_stars['Est froide'] = df_stars['Étoile'].apply(lambda x: 'Oui' if self.star_cold and x in self.star_cold else 'Non')
                    
                    df_stars.to_excel(writer, sheet_name='Statistiques Étoiles', index=False)
                
                # Feuille 3: Prédictions
                # Utiliser les scores finaux pour les numéros et étoiles prédits
                predicted_numbers = sorted(self.final_number_scores, key=self.final_number_scores.get, reverse=True)[:self.config["propose_size"]]
                predicted_stars = sorted(self.final_star_scores, key=self.final_star_scores.get, reverse=True)[:self.config["star_size"]]

                combinations = self.generate_combinations(self.final_number_scores, self.final_star_scores)
                
                df_predictions = pd.DataFrame({
                    'Numéros prédits': [', '.join(map(str, predicted_numbers))],
                    'Étoiles prédites': [', '.join(map(str, predicted_stars))]
                })
                
                df_predictions.to_excel(writer, sheet_name='Prédictions', index=False)
                
                # Feuille 4: Combinaisons optimisées
                if combinations:
                    df_combinations = pd.DataFrame({
                        'Combinaison': [f"{i+1}" for i in range(len(combinations))],
                        'Numéros': [', '.join(map(str, nums)) for nums, _ in combinations],
                        'Étoiles': [', '.join(map(str, stars)) for _, stars in combinations]
                    })
                    
                    df_combinations.to_excel(writer, sheet_name='Combinaisons', index=False)
                
                # Feuille 5: Analyses supplémentaires
                if self.parity_stats or self.sum_stats or self.sequence_stats:
                    data = {}
                    
                    if self.parity_stats:
                        data['Distribution de parité'] = [f"{pattern}: {stats['count']} ({stats['percentage']:.2f}%)" for pattern, stats in self.parity_stats.items()]
                    
                    if self.sum_stats:
                        data['Statistiques de somme'] = [
                            f"Min: {self.sum_stats['min']}",
                            f"Max: {self.sum_stats['max']}",
                            f"Moyenne: {self.sum_stats['avg']:.2f}",
                            f"Médiane: {self.sum_stats['median']:.2f}",
                            f"Écart-type: {self.sum_stats['std']:.2f}"
                        ]
                        
                        if self.sum_ranges:
                            data['Distribution des sommes'] = [f"{range_key}: {stats['count']} ({stats['percentage']:.2f}%)" for range_key, stats in self.sum_ranges.items()]
                    
                    if self.sequence_stats:
                        data['Distribution des séquences'] = [f"{seq_count} séquence(s): {stats['count']} ({stats['percentage']:.2f}%)" for seq_count, stats in self.sequence_stats.items()]
                    
                    # Trouver la longueur maximale
                    max_len = max(len(values) for values in data.values())
                    
                    # Compléter les listes plus courtes
                    for key in data:
                        data[key] = data[key] + [''] * (max_len - len(data[key]))
                    
                    df_analyses = pd.DataFrame(data)
                    df_analyses.to_excel(writer, sheet_name='Analyses supplémentaires', index=False)
                
                # Feuille 6: Données brutes
                if self.df is not None:
                    self.df.to_excel(writer, sheet_name='Données brutes', index=False)
            
            logger.info(f"Export Excel terminé: {excel_file}")
            
            return str(excel_file)
            
        except Exception as e:
            logger.error(f"Erreur lors de l'export Excel: {str(e)}")
            logger.debug(traceback.format_exc())
            return ""
    
    def backtesting(self, start_idx=None, end_idx=None, step_size=1):
        """
        Effectue un backtesting des prédictions sur les données historiques.
        
        Args:
            start_idx: Indice de début pour le backtesting (None = début des données)
            end_idx: Indice de fin pour le backtesting (None = fin des données)
            step_size: Nombre de tirages à avancer à chaque étape
            
        Returns:
            Dict: Résultats du backtesting
        """
        logger.info("Démarrage du backtesting...")
        
        try:
            if self.df is None or len(self.df) < 10:
                logger.error("Données insuffisantes pour le backtesting")
                return {}
            
            # Définir les indices par défaut
            if start_idx is None:
                start_idx = 0
            
            if end_idx is None:
                end_idx = len(self.df)
            
            # Vérifier les limites
            if start_idx < 0:
                start_idx = 0
            
            if end_idx > len(self.df):
                end_idx = len(self.df)
            
            if start_idx >= end_idx:
                logger.error("Indices de backtesting invalides")
                return {}
            
            # Initialiser les résultats
            results = {
                'correct_numbers': [],
                'correct_stars': [],
                'accuracy_numbers': [],
                'accuracy_stars': [],
                'predictions': []
            }
            
            # Fenêtre minimale pour l'entraînement
            min_window = 20
            
            # Boucle de backtesting
            for i in range(start_idx + min_window, end_idx, step_size):
                logger.info(f"Backtesting: tirage {i+1}/{end_idx}")
                
                # Créer un sous-ensemble des données jusqu'à l'indice i (exclus)
                train_df = self.df.iloc[:i].copy()
                
                # Créer un analyseur temporaire
                temp_config = self.config.copy()
                temp_config["window_draws"] = min(50, len(train_df))
                
                temp_analyzer = EuromillionsAdvancedAnalyzer(temp_config)
                temp_analyzer.df = train_df
                temp_analyzer.number_cols = self.number_cols
                temp_analyzer.star_cols = self.star_cols
                
                # Exécuter l'analyse
                temp_analyzer.compute_global_stats()
                temp_analyzer.compute_window_stats()
                temp_analyzer.identify_hot_cold()
                
                if self.config["use_correlation"]:
                    temp_analyzer.analyze_correlations()
                
                if self.config["use_temporal"]:
                    temp_analyzer.analyze_temporal_patterns()
                
                if self.config["use_clustering"]:
                    temp_analyzer.analyze_clustering()
                
                # Calculer les scores d'écart pour le temp_analyzer
                temp_analyzer.compute_gap_scores()

                if self.config["use_ml"]:
                    temp_analyzer.train_ml_models()
                
                # Prédire le prochain tirage (met à jour number_predictions et star_predictions)
                temp_analyzer.predict_numbers()

                # Calculer les scores finaux combinés pour le temp_analyzer
                temp_number_scores, temp_star_scores = temp_analyzer.compute_scores()

                # Sélectionner les numéros et étoiles prédits basés sur les scores finaux
                predicted_numbers = sorted(temp_number_scores, key=temp_number_scores.get, reverse=True)[:self.config["propose_size"]]
                predicted_stars = sorted(temp_star_scores, key=temp_star_scores.get, reverse=True)[:self.config["star_size"]]
                
                # Comparer avec le tirage réel
                actual_row = self.df.iloc[i]
                actual_numbers = actual_row[self.number_cols].dropna().astype(int).tolist()
                actual_stars = actual_row[self.star_cols].dropna().astype(int).tolist()
                
                # Compter les numéros corrects
                correct_numbers = len(set(predicted_numbers) & set(actual_numbers))
                correct_stars = len(set(predicted_stars) & set(actual_stars))
                
                # Calculer les précisions
                accuracy_numbers = correct_numbers / self.config["propose_size"] if self.config["propose_size"] > 0 else 0
                accuracy_stars = correct_stars / self.config["star_size"] if self.config["star_size"] > 0 else 0
                
                # Enregistrer les résultats
                results['correct_numbers'].append(correct_numbers)
                results['correct_stars'].append(correct_stars)
                results['accuracy_numbers'].append(accuracy_numbers)
                results['accuracy_stars'].append(accuracy_stars)
                
                # Enregistrer la prédiction
                results['predictions'].append({
                    'index': i,
                    'date': actual_row['Date'] if 'Date' in actual_row else None,
                    'predicted_numbers': predicted_numbers,
                    'predicted_stars': predicted_stars,
                    'actual_numbers': actual_numbers,
                    'actual_stars': actual_stars,
                    'correct_numbers': correct_numbers,
                    'correct_stars': correct_stars
                })
            
            # Calculer les statistiques globales
            results['avg_correct_numbers'] = sum(results['correct_numbers']) / len(results['correct_numbers']) if results['correct_numbers'] else 0
            results['avg_correct_stars'] = sum(results['correct_stars']) / len(results['correct_stars']) if results['correct_stars'] else 0
            results['avg_accuracy_numbers'] = sum(results['accuracy_numbers']) / len(results['accuracy_numbers']) if results['accuracy_numbers'] else 0
            results['avg_accuracy_stars'] = sum(results['accuracy_stars']) / len(results['accuracy_stars']) if results['accuracy_stars'] else 0
            
            # Distribution des numéros corrects
            results['distribution_correct_numbers'] = Counter(results['correct_numbers'])
            results['distribution_correct_stars'] = Counter(results['correct_stars'])
            
            logger.info(f"Backtesting terminé: {len(results['predictions'])} prédictions évaluées")
            logger.info(f"Nombre moyen de numéros corrects: {results['avg_correct_numbers']:.2f}")
            logger.info(f"Nombre moyen d'étoiles correctes: {results['avg_correct_stars']:.2f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Erreur lors du backtesting: {str(e)}")
            logger.debug(traceback.format_exc())
            return {}
    
    def plot_backtesting_results(self, results):
        """
        Génère des visualisations des résultats du backtesting.
        
        Args:
            results: Résultats du backtesting
            
        Returns:
            List[str]: Liste des chemins des fichiers de visualisation
        """
        if not results or 'predictions' not in results or not results['predictions']:
            logger.error("Résultats de backtesting invalides ou vides")
            return []
        
        logger.info("Génération des visualisations de backtesting...")
        
        try:
            # Créer le répertoire de visualisations
            vis_dir = self.output_dir / "visualizations"
            if not vis_dir.exists():
                vis_dir.mkdir(parents=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_files = []
            
            # 1. Évolution du nombre de numéros corrects
            plt.figure(figsize=(12, 6))
            
            indices = [p['index'] for p in results['predictions']]
            correct_numbers = [p['correct_numbers'] for p in results['predictions']]
            correct_stars = [p['correct_stars'] for p in results['predictions']]
            
            plt.plot(indices, correct_numbers, marker='o', label='Numéros corrects')
            plt.plot(indices, correct_stars, marker='s', label='Étoiles correctes')
            
            plt.axhline(y=results['avg_correct_numbers'], color='r', linestyle='--', label=f'Moyenne numéros: {results["avg_correct_numbers"]:.2f}')
            plt.axhline(y=results['avg_correct_stars'], color='g', linestyle='--', label=f'Moyenne étoiles: {results["avg_correct_stars"]:.2f}')
            
            plt.title('Évolution du nombre de numéros et étoiles corrects')
            plt.xlabel('Indice du tirage')
            plt.ylabel('Nombre corrects')
            plt.grid(alpha=0.3)
            plt.legend()
            
            plot_path = vis_dir / f"backtesting_evolution_{timestamp}.png"
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            
            plot_files.append(str(plot_path))
            
            # 2. Distribution du nombre de numéros corrects
            plt.figure(figsize=(10, 6))
            
            dist_numbers = results['distribution_correct_numbers']
            nums = sorted(dist_numbers.keys())
            counts = [dist_numbers[n] for n in nums]
            
            plt.bar(nums, counts, color='royalblue')
            
            plt.title('Distribution du nombre de numéros corrects')
            plt.xlabel('Nombre de numéros corrects')
            plt.ylabel('Fréquence')
            plt.xticks(range(self.config["propose_size"] + 1)) # Adapter les ticks à la taille de proposition
            plt.grid(axis='y', alpha=0.3)
            
            plot_path = vis_dir / f"backtesting_dist_numbers_{timestamp}.png"
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            
            plot_files.append(str(plot_path))
            
            # 3. Distribution du nombre d'étoiles correctes
            plt.figure(figsize=(10, 6))
            
            dist_stars = results['distribution_correct_stars']
            stars = sorted(dist_stars.keys())
            counts = [dist_stars[s] for s in stars]
            
            plt.bar(stars, counts, color='gold')
            
            plt.title('Distribution du nombre d\'étoiles correctes')
            plt.xlabel('Nombre d\'étoiles correctes')
            plt.ylabel('Fréquence')
            plt.xticks(range(self.config["star_size"] + 1)) # Adapter les ticks à la taille de proposition
            plt.grid(axis='y', alpha=0.3)
            
            plot_path = vis_dir / f"backtesting_dist_stars_{timestamp}.png"
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            
            plot_files.append(str(plot_path))
            
            logger.info(f"{len(plot_files)} visualisations de backtesting générées")
            
            return plot_files
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération des visualisations de backtesting: {str(e)}")
            logger.debug(traceback.format_exc())
            return []


    def compute_gap_scores(self) -> None:
        """
        Calcule le score d'écart pour chaque numéro et étoile.
        L'écart est le nombre de tirages depuis la dernière apparition.
        Le score est basé sur cet écart (un écart plus grand donne un score plus élevé).
        Les scores sont stockés dans self.number_gap_scores et self.star_gap_scores.
        """
        logger.info("Calcul des scores d'écart (gap scores)...")
        
        self.number_gap_scores = {}
        self.star_gap_scores = {}
        
        if self.df is None or self.df.empty:
            logger.warning("DataFrame vide, impossible de calculer les scores d'écart.")
            return

        total_draws = len(self.df)
        
        # Calcul pour les numéros principaux
        for num in range(1, self.config["max_number"] + 1):
            last_occurrence_index = -1
            # Parcourir les tirages en ordre inverse pour trouver la dernière occurrence
            # Utiliser .values pour un accès plus rapide aux données brutes
            found_in_draw = False
            for i in range(total_draws - 1, -1, -1): # Du plus récent au plus ancien
                row_numbers = self.df.iloc[i][self.number_cols].dropna().astype(int).tolist()
                if num in row_numbers:
                    last_occurrence_index = (total_draws - 1) - i # Gap = nombre de tirages depuis
                    found_in_draw = True
                    break
            
            if found_in_draw:
                gap = last_occurrence_index 
            else:
                gap = total_draws # Si jamais apparu, le gap est le nombre total de tirages
                
            # Score simple basé sur l'écart (un écart plus grand = score plus élevé)
            self.number_gap_scores[num] = gap

        # Calcul pour les étoiles
        for star in range(1, self.config["max_star"] + 1):
            last_occurrence_index = -1
            found_in_draw = False
            for i in range(total_draws - 1, -1, -1):
                row_stars = self.df.iloc[i][self.star_cols].dropna().astype(int).tolist()
                if star in row_stars:
                    last_occurrence_index = (total_draws - 1) - i
                    found_in_draw = True
                    break
            
            if found_in_draw:
                gap = last_occurrence_index
            else:
                gap = total_draws
                
            self.star_gap_scores[star] = gap
            
        logger.info("Calcul des scores d'écart terminé.")



    def compute_scores(self) -> Tuple[Dict[int, float], Dict[int, float]]:
        """
        Calcule les scores finaux pour chaque numéro et étoile en combinant
        les différentes métriques calculées (prédictions ML, scores d'écart, fréquence, etc.).

        Returns:
            Tuple[Dict[int, float], Dict[int, float]]: Dictionnaires des scores finaux
                                                        pour les numéros et les étoiles.
        """
        logger.info("Calcul des scores finaux...")
        
        number_scores: Dict[int, float] = {}
        star_scores: Dict[int, float] = {}

        # --- Calcul des scores pour les numéros --- 
        # Assurer que les dictionnaires de source de score existent
        number_predictions = self.number_predictions if hasattr(self, 'number_predictions') and self.number_predictions else {}
        number_gap_scores = self.number_gap_scores if hasattr(self, 'number_gap_scores') and self.number_gap_scores else {}
        window_counts = self.window_counts if hasattr(self, 'window_counts') and self.window_counts else Counter()

        max_gap_numbers = max(number_gap_scores.values()) if number_gap_scores else 1
        max_freq_numbers = max(window_counts.values()) if window_counts else 1

        for num in range(1, self.config["max_number"] + 1):
            score = 0.0
            weights_sum = 0.0

            # 1. Score basé sur les prédictions (ML, fréquence, Fibonacci, etc.)
            pred_score = number_predictions.get(num, 0.0)
            score += pred_score * self.config.get("score_weight_prediction", 0.5) 
            weights_sum += self.config.get("score_weight_prediction", 0.5)

            # 2. Score basé sur l'écart (gap) - Normalisé
            if max_gap_numbers > 0:
                gap_score = number_gap_scores.get(num, 0) / max_gap_numbers
                score += gap_score * self.config.get("score_weight_gap", 0.3) 
                weights_sum += self.config.get("score_weight_gap", 0.3)

            # 3. Score basé sur la fréquence récente - Normalisé
            if max_freq_numbers > 0:
                freq_score = window_counts.get(num, 0) / max_freq_numbers
                score += freq_score * self.config.get("score_weight_frequency", 0.2) 
                weights_sum += self.config.get("score_weight_frequency", 0.2)

            # Normaliser le score final par la somme des poids utilisés
            final_score = (score / weights_sum) if weights_sum > 0 else 0.0
            number_scores[num] = final_score

        # --- Calcul des scores pour les étoiles --- 
        star_predictions = self.star_predictions if hasattr(self, 'star_predictions') and self.star_predictions else {}
        star_gap_scores = self.star_gap_scores if hasattr(self, 'star_gap_scores') and self.star_gap_scores else {}
        star_window_counts = self.star_window_counts if hasattr(self, 'star_window_counts') and self.star_window_counts else Counter()

        max_gap_stars = max(star_gap_scores.values()) if star_gap_scores else 1
        max_freq_stars = max(star_window_counts.values()) if star_window_counts else 1

        for star in range(1, self.config["max_star"] + 1):
            score = 0.0
            weights_sum = 0.0

            # 1. Score basé sur les prédictions
            pred_score = star_predictions.get(star, 0.0)
            score += pred_score * self.config.get("score_weight_prediction", 0.5)
            weights_sum += self.config.get("score_weight_prediction", 0.5)

            # 2. Score basé sur l'écart (gap) - Normalisé
            if max_gap_stars > 0:
                gap_score = star_gap_scores.get(star, 0) / max_gap_stars
                score += gap_score * self.config.get("score_weight_gap", 0.3)
                weights_sum += self.config.get("score_weight_gap", 0.3)

            # 3. Score basé sur la fréquence récente - Normalisé
            if max_freq_stars > 0:
                freq_score = star_window_counts.get(star, 0) / max_freq_stars
                score += freq_score * self.config.get("score_weight_frequency", 0.2)
                weights_sum += self.config.get("score_weight_frequency", 0.2)

            # Normaliser le score final
            final_score = (score / weights_sum) if weights_sum > 0 else 0.0
            star_scores[star] = final_score
            
        # Normaliser les scores finaux pour qu'ils somment à 1 (si nécessaire pour generate_combinations)
        # Ceci est important pour que np.random.choice puisse utiliser ces scores comme probabilités
        total_num_score = sum(number_scores.values())
        if total_num_score > 0:
             number_scores = {num: score / total_num_score for num, score in number_scores.items()}
             
        total_star_score = sum(star_scores.values())
        if total_star_score > 0:
             star_scores = {star: score / total_star_score for star, score in star_scores.items()}

        logger.info("Calcul des scores finaux terminé.")
        return number_scores, star_scores



    def save_results(self, combinations: List[Tuple[List[int], List[int]]]) -> None:
        """
        Enregistre les combinaisons générées dans un fichier texte.

        Args:
            combinations (List[Tuple[List[int], List[int]]]): Liste des combinaisons générées.
        """
        logger.info("Enregistrement des combinaisons générées...")
        
        try:
            # Créer le répertoire de sortie s'il n'existe pas
            if not self.output_dir.exists():
                self.output_dir.mkdir(parents=True)
            
            # Nom du fichier de résultats
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = self.output_dir / f"combinaisons_euromillions_{timestamp}.txt"
            
            with open(results_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("COMBINAISONS EUROMILLIONS GÉNÉRÉES\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Date de génération: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Nombre de combinaisons: {len(combinations)}\n\n")
                
                for i, (nums, stars) in enumerate(combinations, 1):
                    num_str = ', '.join(map(str, self._convert_to_int_list(nums)))
                    star_str = ', '.join(map(str, self._convert_to_int_list(stars)))
                    f.write(f"Combinaison {i}: Numéros [{num_str}], Étoiles [{star_str}]\n")
            
            logger.info(f"Combinaisons enregistrées dans: {results_file}")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'enregistrement des résultats: {str(e)}")
            logger.debug(traceback.format_exc())



    def _generate_weighted_random_combination(self, number_scores: Dict[int, float], star_scores: Dict[int, float]) -> Tuple[List[int], List[int]]:
        """Génère une seule combinaison aléatoire pondérée par les scores."""
        numbers = []
        stars = []
        
        # Vérifier si les scores sont valides pour la sélection pondérée
        if not number_scores or not star_scores or sum(number_scores.values()) == 0 or sum(star_scores.values()) == 0:
            logger.warning("Scores non valides pour la génération pondérée, utilisation d'une sélection aléatoire simple.")
            while len(numbers) < self.config["propose_size"]:
                num = random.randint(1, self.config["max_number"] + 1)
                if num not in numbers:
                    numbers.append(num)
            while len(stars) < self.config["star_size"]:
                star = random.randint(1, self.config["max_star"] + 1)
                if star not in stars:
                    stars.append(star)
            return sorted(numbers), sorted(stars)

        try:
            # Sélection pondérée des numéros
            # Assurer que les poids sont positifs et somment à 1
            number_items = list(number_scores.items())
            number_values = [num for num, _ in number_items]
            number_weights = [score for _, score in number_items]
            
            # Normaliser les poids pour np.random.choice
            total_num_weight = sum(number_weights)
            if total_num_weight > 0:
                number_weights = [w / total_num_weight for w in number_weights]
            else: # Fallback si tous les poids sont zéro
                number_weights = [1.0 / len(number_values)] * len(number_values)
            
            # Assurer que les poids sont non négatifs et somment à 1 (tolérance pour les flottants)
            if not np.isclose(sum(number_weights), 1.0) or any(w < 0 for w in number_weights):
                logger.warning("Poids des numéros invalides après normalisation, réinitialisation à uniforme.")
                number_weights = [1.0 / len(number_values)] * len(number_values)

            while len(numbers) < self.config["propose_size"]:
                num = np.random.choice(number_values, p=number_weights)
                if num not in numbers:
                    numbers.append(num)

            # Sélection pondérée des étoiles
            star_items = list(star_scores.items())
            star_values = [star for star, _ in star_items]
            star_weights = [score for _, score in star_items]
            
            # Normaliser les poids pour np.random.choice
            total_star_weight = sum(star_weights)
            if total_star_weight > 0:
                star_weights = [w / total_star_weight for w in star_weights]
            else: # Fallback si tous les poids sont zéro
                star_weights = [1.0 / len(star_values)] * len(star_values)

            # Assurer que les poids sont non négatifs et somment à 1 (tolérance pour les flottants)
            if not np.isclose(sum(star_weights), 1.0) or any(w < 0 for w in star_weights):
                logger.warning("Poids des étoiles invalides après normalisation, réinitialisation à uniforme.")
                star_weights = [1.0 / len(star_values)] * len(star_values)

            while len(stars) < self.config["star_size"]:
                star = np.random.choice(star_values, p=star_weights)
                if star not in stars:
                    stars.append(star)
            
            return sorted(numbers), sorted(stars)
        except Exception as e:
            logger.error(f"Erreur dans _generate_weighted_random_combination: {e}")
            logger.debug(traceback.format_exc())
            # Fallback vers aléatoire simple en cas d'erreur
            numbers = random.sample(range(1, self.config["max_number"] + 1), self.config["propose_size"])
            stars = random.sample(range(1, self.config["max_star"] + 1), self.config["star_size"])
            return sorted(numbers), sorted(stars)



    def _is_combination_valid(self, numbers: List[int], stars: List[int]) -> bool:
        """Vérifie si une combinaison respecte certains filtres heuristiques."""
        # Filtre 1: Somme des numéros (utiliser les plages calculées si disponibles)
        if self.sum_stats and self.sum_ranges:
            num_sum = sum(numbers)
            # Vérifier si la somme tombe dans une plage "raisonnable" (ex: éviter les extrêmes)
            # On va considérer les plages qui représentent un certain pourcentage des tirages historiques
            # Par exemple, rejeter si la somme est dans une plage qui représente moins de 1% des tirages
            is_sum_in_valid_range = False
            for range_key, stats in self.sum_ranges.items():
                # Extraire les limites de la plage
                start_str, end_str = range_key.split('-')
                start = int(start_str)
                end = int(end_str)
                
                if start <= num_sum <= end and stats['percentage'] >= 1.0: # Minimum 1% des tirages
                    is_sum_in_valid_range = True
                    break
            if not is_sum_in_valid_range:
                # logger.debug(f"Combinaison rejetée (somme hors plage valide): {numbers}")
                return False

        # Filtre 2: Parité (éviter les extrêmes: tout pair ou tout impair, ou des répartitions très rares)
        even_count = sum(1 for num in numbers if num % 2 == 0)
        odd_count = len(numbers) - even_count
        parity_pattern = f"{even_count}E-{odd_count}O"
        
        if self.parity_stats:
            # Rejeter si la configuration de parité est très rare (ex: moins de 1% des tirages)
            if parity_pattern not in self.parity_stats or self.parity_stats[parity_pattern]['percentage'] < 1.0:
                # logger.debug(f"Combinaison rejetée (parité rare): {numbers}")
                return False
        else: # Fallback si pas de stats de parité, rejeter les extrêmes simples
            if even_count == 0 or even_count == len(numbers):
                # logger.debug(f"Combinaison rejetée (parité extrême - fallback): {numbers}")
                return False


        # Filtre 3: Séquences (éviter trop de numéros consécutifs)
        sequences = 0
        seq_length = 1
        sorted_numbers = sorted(numbers)
        for i in range(1, len(sorted_numbers)):
            if sorted_numbers[i] == sorted_numbers[i-1] + 1:
                seq_length += 1
            else:
                if seq_length >= 3: # Rejeter si 3 numéros consécutifs ou plus
                    sequences += 1
                seq_length = 1
        if seq_length >= 3:
             sequences += 1
        if sequences > 0: # Rejeter si au moins une séquence de 3+ numéros consécutifs
            # logger.debug(f"Combinaison rejetée (séquence >= 3): {numbers}")
            return False
            
        # Filtre 4: Écart entre numéros (éviter les numéros trop "serrés" ou trop "espacés" si cela est rare)
        # Calculer l'écart moyen entre les numéros triés
        if len(numbers) > 1:
            gaps = [sorted_numbers[i] - sorted_numbers[i-1] for i in range(1, len(sorted_numbers))]
            avg_gap = np.mean(gaps)
            std_gap = np.std(gaps) if len(gaps) > 1 else 0

            # Comparer à l'écart moyen historique (si disponible)
            # Pour l'instant, on n'a pas de stats historiques sur l'écart moyen.
            # On peut définir des seuils heuristiques.
            # Par exemple, si l'écart moyen est trop petit (numéros trop groupés) ou trop grand (trop espacés)
            # Ces seuils sont arbitraires et peuvent être ajustés.
            if avg_gap < 5 or avg_gap > 15: # Exemple de seuils
                # logger.debug(f"Combinaison rejetée (écart moyen des numéros hors plage): {numbers} (avg_gap={avg_gap:.2f})")
                pass # Désactivé pour l'instant, nécessite une analyse plus poussée des écarts historiques

        # Filtre 5: Distance par rapport aux tirages précédents (éviter répétition exacte récente)
        # Cela est déjà géré par `seen_combos` dans `generate_combinations` pour l'unicité
        # Si on veut éviter les combinaisons "trop similaires" aux récentes, il faudrait une métrique de similarité
        # et un seuil, ce qui est plus complexe. Pour l'instant, on se concentre sur l'unicité exacte.

        return True # La combinaison a passé tous les filtres

    def _run_monte_carlo_simulation(self, number_scores: Dict[int, float], star_scores: Dict[int, float]) -> List[Tuple[List[int], List[int]]]:
        """Exécute la simulation Monte Carlo pour générer des combinaisons filtrées."""
        valid_combinations = []
        seen_combos = set()
        num_simulations = self.config.get("monte_carlo_simulations", 10000) 
        target_combinations_to_find = self.config["combinations_to_generate"] * 2 # Chercher plus pour avoir du choix
        max_attempts = num_simulations * 5 # Limite pour éviter boucle infinie

        logger.info(f"Lancement de {num_simulations} simulations Monte Carlo...")

        attempts = 0
        while len(valid_combinations) < target_combinations_to_find and attempts < max_attempts: 
            attempts += 1
            # Générer une combinaison candidate pondérée
            candidate_numbers, candidate_stars = self._generate_weighted_random_combination(number_scores, star_scores)
            
            # Vérifier l'unicité
            combo_tuple = (tuple(sorted(candidate_numbers)), tuple(sorted(candidate_stars)))
            if combo_tuple in seen_combos:
                continue
                
            # Appliquer les filtres heuristiques
            if self._is_combination_valid(candidate_numbers, candidate_stars):
                valid_combinations.append((candidate_numbers, candidate_stars))
                seen_combos.add(combo_tuple)
                if attempts % (num_simulations // 10 if num_simulations >= 10 else 1) == 0:
                     logger.debug(f"Monte Carlo: {len(valid_combinations)} combinaisons valides trouvées après {attempts} tentatives.")

        logger.info(f"Simulation Monte Carlo terminée: {len(valid_combinations)} combinaisons valides trouvées après {attempts} tentatives.")
        # Retourner seulement le nombre requis, potentiellement moins si pas assez trouvées
        return valid_combinations[:self.config["combinations_to_generate"]] # Retourne seulement le nombre final désiré


    def _calculate_combination_rank(self, generated_nums: List[int], generated_stars: List[int], actual_nums: List[int], actual_stars: List[int]) -> Optional[str]:
        """Calcule le rang d'une combinaison générée par rapport à un tirage réel."""
        matched_nums = len(set(generated_nums) & set(actual_nums))
        matched_stars = len(set(generated_stars) & set(actual_stars))

        # Définir les rangs (simplifié, peut être ajusté selon les règles officielles)
        if matched_nums == 5 and matched_stars == 2: return "Rang 1 (5+2)"
        if matched_nums == 5 and matched_stars == 1: return "Rang 2 (5+1)"
        if matched_nums == 5 and matched_stars == 0: return "Rang 3 (5+0)"
        if matched_nums == 4 and matched_stars == 2: return "Rang 4 (4+2)"
        if matched_nums == 4 and matched_stars == 1: return "Rang 5 (4+1)"
        if matched_nums == 3 and matched_stars == 2: return "Rang 6 (3+2)"
        if matched_nums == 4 and matched_stars == 0: return "Rang 7 (4+0)"
        if matched_nums == 2 and matched_stars == 2: return "Rang 8 (2+2)"
        if matched_nums == 3 and matched_stars == 1: return "Rang 9 (3+1)"
        if matched_nums == 3 and matched_stars == 0: return "Rang 10 (3+0)"
        if matched_nums == 1 and matched_stars == 2: return "Rang 11 (1+2)"
        if matched_nums == 2 and matched_stars == 1: return "Rang 12 (2+1)"
        if matched_nums == 2 and matched_stars == 0: return "Rang 13 (2+0)"
        
        return None # Aucun gain

    def track_performance(self):
        """Analyse la performance des combinaisons générées historiquement."""
        logger.info("Analyse de la performance historique des combinaisons générées...")
        if not self.generated_combinations_history or self.df is None or 'Date' not in self.df.columns:
            logger.warning("Historique des combinaisons ou données de tirage manquantes pour l'analyse de performance.")
            return

        # S'assurer que la colonne Date est bien au format datetime
        if not pd.api.types.is_datetime64_any_dtype(self.df['Date']):
            try:
                self.df['Date'] = pd.to_datetime(self.df['Date'])
            except Exception as e:
                logger.error(f"Impossible de convertir la colonne 'Date' en datetime pour le suivi de performance: {e}")
                return
        
        # Trier le DataFrame par date au cas où
        df_sorted = self.df.sort_values('Date').reset_index(drop=True) # drop=True pour ne pas ajouter l'ancien index
        
        new_performance_entries = []
        processed_dates = {entry['generation_date'] for entry in self.performance_history} # Pour éviter doublons

        for generation_date_str, combinations in self.generated_combinations_history:
            if generation_date_str in processed_dates:
                continue # Déjà traité

            # Trouver l'index du tirage suivant la date de génération
            next_draw_index = -1
            try:
                generation_date = datetime.strptime(generation_date_str, '%Y-%m-%d')
                # Chercher le premier tirage APRÈS la date de génération
                # Utiliser idxmax pour trouver le premier index où la condition est vraie
                after_generation_draws = df_sorted[df_sorted['Date'] > generation_date]
                if not after_generation_draws.empty:
                    next_draw_index = after_generation_draws.index[0]
            except ValueError:
                 logger.warning(f"Format de date invalide dans l'historique: {generation_date_str}")
                 continue

            if next_draw_index == -1:
                logger.debug(f"Aucun tirage trouvé après la date {generation_date_str} pour évaluer la performance.")
                continue

            # Obtenir le tirage réel
            actual_draw = df_sorted.iloc[next_draw_index]
            actual_nums = actual_draw[self.number_cols].dropna().astype(int).tolist()
            actual_stars = actual_draw[self.star_cols].dropna().astype(int).tolist()
            actual_draw_date_str = actual_draw['Date'].strftime('%Y-%m-%d')

            # Évaluer chaque combinaison générée pour cette date
            results = {'total_combinations': len(combinations), 'wins': {}}
            best_rank = None
            best_rank_num = 99 # Pour trier (plus petit = meilleur)

            for nums, stars in combinations:
                rank = self._calculate_combination_rank(nums, stars, actual_nums, actual_stars)
                if rank:
                    # Extraire le numéro du rang (ex: "Rang 1 (5+2)" -> 1)
                    try:
                        rank_num = int(rank.split(' ')[1].replace('(', ''))
                    except (IndexError, ValueError):
                        rank_num = 99 # Valeur par default si le format n'est pas celui attendu
                    
                    results['wins'][rank] = results['wins'].get(rank, 0) + 1
                    if rank_num < best_rank_num:
                        best_rank_num = rank_num
                        best_rank = rank
            
            performance_entry = {
                'generation_date': generation_date_str,
                'evaluated_draw_date': actual_draw_date_str,
                'best_rank_achieved': best_rank,
                'win_distribution': results['wins']
            }
            new_performance_entries.append(performance_entry)
            processed_dates.add(generation_date_str)

        # Ajouter les nouvelles entrées à l'historique et trier
        self.performance_history.extend(new_performance_entries)
        self.performance_history.sort(key=lambda x: x['generation_date'])
        
        # Limiter la taille de l'historique de performance
        max_perf_history = self.config.get("max_performance_history", 100)
        if len(self.performance_history) > max_perf_history:
             self.performance_history = self.performance_history[-max_perf_history:]

        logger.info(f"Analyse de performance terminée. {len(new_performance_entries)} nouvelles évaluations ajoutées.")

    def load_history(self, history_file="analysis_history.joblib"):
        """Charge l'historique des combinaisons et performances depuis un fichier."""
        history_path = self.output_dir / history_file
        if history_path.exists():
            try:
                data = safe_joblib_load(history_path)
                self.generated_combinations_history = data.get('combinations', [])
                self.performance_history = data.get('performance', [])
                logger.info(f"Historique chargé depuis {history_path}")
            except ModelIntegrityError:
                raise  # mode strict : ne pas charger un historique non vérifié
            except Exception as e:
                logger.error(f"Erreur lors du chargement de l'historique depuis {history_path}: {e}")
        else:
            logger.info("Aucun fichier d'historique trouvé, démarrage avec un historique vide.")

    def save_history(self, history_file="analysis_history.joblib"):
        """Sauvegarde l'historique des combinaisons et performances dans un fichier."""
        history_path = self.output_dir / history_file
        try:
            data = {
                'combinations': self.generated_combinations_history,
                'performance': self.performance_history
            }
            joblib.dump(data, history_path)
            logger.info(f"Historique sauvegardé dans {history_path}")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de l'historique dans {history_path}: {e}")



    def _generate_wheeling_combinations(self, numbers_to_wheel: List[int], stars_to_wheel: List[int], num_guarantee: int = 3, star_guarantee: int = 1) -> List[Tuple[List[int], List[int]]]:
        """
        Génère des combinaisons en utilisant un système réducteur simple (garantie basique).
        Ceci est une implémentation basique, des systèmes plus complexes existent.

        Args:
            numbers_to_wheel (List[int]): Liste des numéros principaux à couvrir.
            stars_to_wheel (List[int]): Liste des étoiles à couvrir.
            num_guarantee (int): Garantie minimale pour les numéros (ex: 3 si 4).
            star_guarantee (int): Garantie minimale pour les étoiles (ex: 1 si 2).

        Returns:
            Liste de combinaisons générées par le système réducteur.
        """
        logger.info(f"Génération de combinaisons via système réducteur (Numéros: {len(numbers_to_wheel)} pour {self.config['propose_size']}, Étoiles: {len(stars_to_wheel)} pour {self.config['star_size']})...")
        wheeled_combinations = []
        
        # Vérifier si les listes sont assez grandes
        if len(numbers_to_wheel) < self.config['propose_size'] or len(stars_to_wheel) < self.config['star_size']:
            logger.warning("Pas assez de numéros/étoiles fournis pour le système réducteur. Fallback à la sélection aléatoire parmi les fournis.")
            # Fallback: générer une combinaison aléatoire à partir des numéros fournis
            # S'assurer qu'on ne demande pas plus d'éléments qu'il n'y en a
            nums_sample_size = min(self.config['propose_size'], len(numbers_to_wheel))
            stars_sample_size = min(self.config['star_size'], len(stars_to_wheel))

            if nums_sample_size > 0:
                 nums = random.sample(numbers_to_wheel, nums_sample_size)
            else:
                 nums = [] # Ou générer totalement aléatoire si numbers_to_wheel est vide
                 
            if stars_sample_size > 0:
                 stars = random.sample(stars_to_wheel, stars_sample_size)
            else:
                 stars = [] # Ou générer totalement aléatoire si stars_to_wheel est vide
            
            # Compléter avec des numéros/étoiles aléatoires si les listes initiales étaient trop petites
            while len(nums) < self.config['propose_size']:
                new_num = random.randint(1, self.config['max_number'])
                if new_num not in nums:
                    nums.append(new_num)
            while len(stars) < self.config['star_size']:
                new_star = random.randint(1, self.config['max_star'])
                if new_star not in stars:
                    stars.append(new_star)

            return [(sorted(nums), sorted(stars))]

        try:
            # Générer toutes les combinaisons possibles de la taille requise à partir des numéros fournis
            # Ceci n'est PAS un vrai système réducteur optimisé, mais une simple génération de toutes les combinaisons
            # Un vrai système réducteur sélectionnerait un sous-ensemble minimal pour garantir la couverture
            num_combos = list(iter_combinations(numbers_to_wheel, self.config['propose_size']))
            star_combos = list(iter_combinations(stars_to_wheel, self.config['star_size']))

            # Limiter le nombre de combinaisons générées pour éviter une explosion combinatoire
            max_wheeled = self.config.get("max_wheeling_combinations", 50)
            count = 0
            # Combiner les numéros et les étoiles (ici, on prend juste les premières combinaisons)
            # Une approche plus sophistiquée est nécessaire pour une vraie garantie
            for n_combo in num_combos:
                for s_combo in star_combos:
                    if count < max_wheeled:
                        wheeled_combinations.append((sorted(list(n_combo)), sorted(list(s_combo))))
                        count += 1
                    else:
                        break
                if count >= max_wheeled:
                    break
            
            logger.info(f"{len(wheeled_combinations)} combinaisons générées par la méthode de couverture (limité à {max_wheeled}).")
            return wheeled_combinations

        except Exception as e:
            logger.error(f"Erreur lors de la génération des combinaisons réductrices: {e}")
            logger.debug(traceback.format_exc())
            return []

