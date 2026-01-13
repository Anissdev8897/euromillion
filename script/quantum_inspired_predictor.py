#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module de Prédiction Inspiré par le Quantique (Quantum-Inspired Machine Learning)
Ce module intègre des concepts quantiques (superposition, intrication) dans le Machine Learning
pour améliorer la détection de patterns complexes dans les tirages EuroMillions.

Approches implémentées:
1. Quantum Neural Networks (QNN) simulés avec PennyLane
2. Quantum Long Short-Term Memory (QLSTM) hybride
3. Recuit Simulé Quantique pour l'optimisation combinatoire
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from datetime import datetime
import traceback

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("QuantumInspiredPredictor")

# ⚠️ CRITIQUE : Import conditionnel des librairies quantiques
try:
    import pennylane as qml
    from pennylane import numpy as pnp
    PENNYLANE_AVAILABLE = True
    logger.info("✅ PennyLane disponible - Système quantique activé")
except ImportError:
    PENNYLANE_AVAILABLE = False
    logger.warning("⚠️ PennyLane non disponible - Installation: pip install pennylane")
    # Créer des stubs pour éviter les erreurs
    qml = None
    pnp = None

# Note: Qiskit n'est pas utilisé dans cette implémentation
# Le système utilise PennyLane pour les QNN et un recuit simulé classique pour l'optimisation
# Qiskit pourrait être ajouté dans le futur pour des fonctionnalités avancées

# Import des modèles classiques pour l'hybridation
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.error("❌ scikit-learn non disponible")

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    logger.info("✅ PyTorch disponible - LSTM classique activé")
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("⚠️ PyTorch non disponible - Installation: pip install torch")


class QuantumNeuralNetwork:
    """
    Réseau de Neurones Quantique (QNN) simulé utilisant PennyLane.
    Implémente un Variational Quantum Circuit (VQC) pour la classification.
    """
    
    def __init__(self, n_qubits: int = 4, n_layers: int = 2, dev_name: str = 'default.qubit'):
        """
        Initialise le QNN.
        
        Args:
            n_qubits: Nombre de qubits dans le circuit quantique
            n_layers: Nombre de couches du circuit
            dev_name: Nom du device PennyLane (default.qubit pour simulation)
        """
        if not PENNYLANE_AVAILABLE:
            raise ImportError("PennyLane n'est pas disponible. Installez-le avec: pip install pennylane")
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device(dev_name, wires=n_qubits)
        self.weights = None
        
        # Créer le circuit quantique
        self.qnode = qml.QNode(self.quantum_circuit, self.dev)
        
        # Initialiser les poids aléatoirement
        self.init_weights()
        
        logger.info(f"✅ QNN initialisé: {n_qubits} qubits, {n_layers} couches")
    
    def quantum_circuit(self, inputs: np.ndarray, weights: np.ndarray) -> float:
        """
        Définit le circuit quantique paramétré (VQC).
        
        Args:
            inputs: Données d'entrée encodées dans l'état quantique
            weights: Paramètres du circuit (angles de rotation)
            
        Returns:
            Valeur de mesure (expectation value)
        """
        # Encoder les données classiques dans l'état quantique
        for i in range(self.n_qubits):
            qml.RY(inputs[i] * np.pi, wires=i)
        
        # Appliquer les couches de rotation paramétrées
        weight_idx = 0
        for layer in range(self.n_layers):
            # Entrelacement (entanglement)
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            
            # Rotations paramétrées
            for i in range(self.n_qubits):
                qml.RY(weights[weight_idx], wires=i)
                weight_idx += 1
                qml.RZ(weights[weight_idx], wires=i)
                weight_idx += 1
        
        # Mesure de l'observable (Z sur le premier qubit)
        return qml.expval(qml.PauliZ(0))
    
    def init_weights(self) -> None:
        """Initialise les poids du circuit quantique."""
        # Nombre de poids = n_layers * n_qubits * 2 (RY + RZ)
        n_weights = self.n_layers * self.n_qubits * 2
        self.weights = pnp.random.uniform(0, 2 * np.pi, size=n_weights, requires_grad=True)
    
    def forward(self, inputs: np.ndarray) -> float:
        """
        Passe avant (forward pass) du QNN.
        
        Args:
            inputs: Données d'entrée (doit être de taille n_qubits)
            
        Returns:
            Prédiction du modèle
        """
        if len(inputs) != self.n_qubits:
            # Redimensionner les inputs si nécessaire
            if len(inputs) > self.n_qubits:
                inputs = inputs[:self.n_qubits]
            else:
                inputs = np.pad(inputs, (0, self.n_qubits - len(inputs)), 'constant')
        
        # Normaliser les inputs entre 0 et 1
        inputs = (inputs - inputs.min()) / (inputs.max() - inputs.min() + 1e-8)
        
        return self.qnode(inputs, self.weights)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Prédit les probabilités pour un ensemble de données.
        
        Args:
            X: Matrice de features (n_samples, n_features)
            
        Returns:
            Probabilités prédites (n_samples,)
        """
        predictions = []
        for x in X:
            prob = self.forward(x)
            # Convertir en probabilité entre 0 et 1
            prob = (prob + 1) / 2  # Normaliser de [-1, 1] à [0, 1]
            predictions.append(prob)
        
        return np.array(predictions)


class QuantumLSTMHybrid:
    """
    Modèle hybride combinant LSTM classique et couche quantique.
    Le LSTM traite les séquences temporelles, puis une couche QNN améliore la classification.
    """
    
    def __init__(self, input_size: int = 32, hidden_size: int = 64, n_qubits: int = 4, n_layers: int = 2):
        """
        Initialise le modèle hybride LSTM-QNN.
        
        Args:
            input_size: Taille des features d'entrée
            hidden_size: Taille de la couche cachée LSTM
            n_qubits: Nombre de qubits pour le QNN
            n_layers: Nombre de couches du circuit quantique
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch n'est pas disponible. Installez-le avec: pip install torch")
        
        if not PENNYLANE_AVAILABLE:
            logger.warning("⚠️ PennyLane non disponible - Utilisation de LSTM classique uniquement")
            self.use_quantum = False
        else:
            self.use_quantum = True
        
        # LSTM classique
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=1)
        self.fc = nn.Linear(hidden_size, n_qubits)  # Sortie vers le QNN
        
        # QNN quantique
        if self.use_quantum:
            self.qnn = QuantumNeuralNetwork(n_qubits=n_qubits, n_layers=n_layers)
        
        self.hidden_size = hidden_size
        logger.info(f"✅ QuantumLSTMHybrid initialisé: LSTM({input_size}→{hidden_size}) + QNN({n_qubits} qubits)")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passe avant du modèle hybride.
        
        Args:
            x: Tenseur d'entrée (batch_size, seq_len, input_size)
            
        Returns:
            Prédictions (batch_size, output_size)
        """
        # LSTM classique
        lstm_out, _ = self.lstm(x)
        # Prendre la dernière sortie de la séquence
        lstm_out = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Couche fully connected vers le QNN
        fc_out = self.fc(lstm_out)  # (batch_size, n_qubits)
        
        if self.use_quantum:
            # Passer par le QNN (nécessite conversion numpy)
            predictions = []
            for i in range(fc_out.shape[0]):
                qnn_input = fc_out[i].detach().numpy()
                qnn_output = self.qnn.forward(qnn_input)
                predictions.append(qnn_output)
            return torch.tensor(predictions, dtype=torch.float32)
        else:
            # Utiliser uniquement la sortie LSTM
            return fc_out


class QuantumAnnealingOptimizer:
    """
    Optimiseur de Recuit Simulé Quantique pour la sélection optimale de combinaisons.
    Utilise un algorithme d'optimisation inspiré par le quantique pour trouver
    la meilleure combinaison de numéros selon une fonction de coût.
    """
    
    def __init__(self, max_number: int = 50, max_star: int = 12, n_numbers: int = 5, n_stars: int = 2):
        """
        Initialise l'optimiseur quantique.
        
        Args:
            max_number: Numéro maximum (50 pour EuroMillions)
            max_star: Étoile maximum (12 pour EuroMillions)
            n_numbers: Nombre de numéros à sélectionner (5)
            n_stars: Nombre d'étoiles à sélectionner (2)
        """
        self.max_number = max_number
        self.max_star = max_star
        self.n_numbers = n_numbers
        self.n_stars = n_stars
        
        logger.info(f"✅ QuantumAnnealingOptimizer initialisé: {n_numbers} numéros, {n_stars} étoiles")
    
    def _calculate_cluster_penalty(self, numbers: List[int]) -> float:
        """
        Calcule une pénalité basée sur la densité des clusters.
        Plus les numéros sont regroupés, plus la pénalité est élevée.
        
        Args:
            numbers: Liste des numéros (doit être triée)
            
        Returns:
            Pénalité cluster (0.0 à 3.0)
        """
        if len(numbers) < 2:
            return 0.0
        
        sorted_nums = sorted(numbers)
        
        # Calculer l'écart entre le min et le max (plage totale)
        range_span = sorted_nums[-1] - sorted_nums[0]
        
        # Calculer les écarts entre numéros consécutifs
        gaps = [sorted_nums[i+1] - sorted_nums[i] for i in range(len(sorted_nums)-1)]
        avg_gap = np.mean(gaps) if gaps else 0
        min_gap = min(gaps) if gaps else 0
        
        # Pénalité basée sur la plage totale (si tous les numéros sont dans une petite plage)
        if range_span < 15:  # Les 5 numéros sont dans une plage de 15
            range_penalty = 2.0
        elif range_span < 25:
            range_penalty = 1.0
        elif range_span < 35:
            range_penalty = 0.5
        else:
            range_penalty = 0.0
        
        # Pénalité basée sur l'écart moyen (si les numéros sont trop proches en moyenne)
        if avg_gap < 5:
            gap_penalty = 1.0
        elif avg_gap < 8:
            gap_penalty = 0.5
        else:
            gap_penalty = 0.0
        
        # Pénalité supplémentaire si plusieurs numéros sont très proches
        if min_gap < 3:
            min_gap_penalty = 0.5
        else:
            min_gap_penalty = 0.0
        
        return range_penalty + gap_penalty + min_gap_penalty
    
    def _calculate_quartile_coverage(self, numbers: List[int]) -> int:
        """
        Vérifie combien de quartiles sont couverts par les numéros.
        Quartiles: [1-12], [13-25], [26-37], [38-50]
        
        Args:
            numbers: Liste des numéros
            
        Returns:
            Nombre de quartiles couverts (0 à 4)
        """
        quartiles = [
            set(range(1, 13)),   # Quartile 1: 1-12
            set(range(13, 26)),  # Quartile 2: 13-25
            set(range(26, 38)),  # Quartile 3: 26-37
            set(range(38, 51))   # Quartile 4: 38-50
        ]
        
        covered = 0
        for quartile in quartiles:
            if any(n in quartile for n in numbers):
                covered += 1
        
        return covered
    
    def _calculate_gradual_similarity(self, numbers: List[int], stars: List[int],
                                     historical_data: pd.DataFrame) -> float:
        """
        Calcule une pénalité de similarité graduelle (non binaire).
        
        Args:
            numbers: Liste des numéros
            stars: Liste des étoiles
            historical_data: Données historiques
            
        Returns:
            Pénalité de similarité (0.0 à 5.0)
        """
        if historical_data is None or len(historical_data) == 0:
            return 0.0
        
        total_penalty = 0.0
        recent = historical_data.tail(10)
        
        for _, row in recent.iterrows():
            try:
                recent_numbers = [int(row[f'N{i}']) for i in range(1, 6) if pd.notna(row.get(f'N{i}'))]
                recent_stars = [int(row[f'E{i}']) for i in range(1, 3) if pd.notna(row.get(f'E{i}'))]
            except (KeyError, ValueError, TypeError):
                continue
            
            # Calculer la similarité graduelle pour les numéros
            number_overlap = len(set(numbers) & set(recent_numbers))
            if number_overlap >= 5:
                total_penalty += 2.5  # Très similaire
            elif number_overlap >= 4:
                total_penalty += 1.5
            elif number_overlap >= 3:
                total_penalty += 0.5
            
            # Calculer la similarité graduelle pour les étoiles
            star_overlap = len(set(stars) & set(recent_stars))
            if star_overlap >= 2:
                total_penalty += 1.0
            elif star_overlap >= 1:
                total_penalty += 0.3
        
        return total_penalty
    
    def cost_function(self, combination: Tuple[List[int], List[int]], 
                     number_probs: Dict[int, float], 
                     star_probs: Dict[int, float],
                     historical_data: pd.DataFrame = None) -> float:
        """
        Fonction de coût améliorée pour évaluer une combinaison.
        Minimise cette fonction pour trouver la meilleure combinaison.
        
        🔬 AMÉLIORATIONS :
        - Pénalité cluster explicite pour réduire le biais (26-36)
        - Similarité graduelle (non binaire)
        - Diversité renforcée avec bonus de couverture de quartiles
        - Poids normalisés pour cohérence
        
        Args:
            combination: Tuple (numéros, étoiles)
            number_probs: Probabilités prédites pour chaque numéro
            star_probs: Probabilités prédites pour chaque étoile
            historical_data: Données historiques pour calculer des pénalités
            
        Returns:
            Coût de la combinaison (plus bas = meilleur)
        """
        numbers, stars = combination
        
        # ===== 1. COÛT DE PROBABILITÉ (Poids: 0.4) =====
        # Maximiser la probabilité totale (inverser pour minimiser)
        number_cost = -sum([number_probs.get(n, 0.0) for n in numbers])
        star_cost = -sum([star_probs.get(s, 0.0) for s in stars])
        prob_cost = (number_cost + star_cost) * 0.4  # Poids normalisé
        
        # ===== 2. PÉNALITÉ CONSÉCUTIFS (Poids: 0.1) =====
        consecutive_penalty = 0
        sorted_numbers = sorted(numbers)
        for i in range(len(sorted_numbers) - 1):
            if sorted_numbers[i + 1] - sorted_numbers[i] == 1:
                consecutive_penalty += 0.5
        consecutive_penalty *= 0.1  # Poids normalisé
        
        # ===== 3. PÉNALITÉ CLUSTER (Poids: 0.2) - NOUVEAU =====
        # Réduire le biais vers les clusters (26-36)
        cluster_penalty = self._calculate_cluster_penalty(numbers) * 0.2
        
        # Pénalité supplémentaire si moins de 3 quartiles sont couverts
        quartile_coverage = self._calculate_quartile_coverage(numbers)
        if quartile_coverage < 3:
            cluster_penalty += 1.5 * 0.2  # Pénalité pour mauvaise répartition
        
        # ===== 4. PÉNALITÉ SIMILARITÉ GRADUELLE (Poids: 0.2) - AMÉLIORÉ =====
        similarity_penalty = self._calculate_gradual_similarity(
            numbers, stars, historical_data
        ) * 0.2
        
        # ===== 5. RÉCOMPENSE DIVERSITÉ RENFORCÉE (Poids: 0.1) - AMÉLIORÉ =====
        # Diversité basée sur l'écart-type (augmenté de 0.1 à 0.3)
        diversity_std = np.std(numbers) * 0.3
        
        # Bonus pour couvrir toute la plage (1-50)
        range_coverage = (sorted_numbers[-1] - sorted_numbers[0]) / 49.0  # Normalisé [0, 1]
        range_bonus = range_coverage * 0.2
        
        # Bonus pour couvrir plusieurs quartiles
        quartile_bonus = (quartile_coverage / 4.0) * 0.1
        
        diversity_reward = -(diversity_std + range_bonus + quartile_bonus) * 0.1
        
        # ===== COÛT TOTAL =====
        total_cost = (
            prob_cost +
            consecutive_penalty +
            cluster_penalty +
            similarity_penalty -
            diversity_reward
        )
        
        return total_cost
    
    def quantum_annealing(self, number_probs: Dict[int, float], 
                         star_probs: Dict[int, float],
                         historical_data: pd.DataFrame = None,
                         n_iterations: int = 1000,
                         initial_temp: float = 100.0,
                         cooling_rate: float = 0.95) -> Tuple[List[int], List[int]]:
        """
        Algorithme de Recuit Simulé Quantique pour optimiser la sélection.
        
        Args:
            number_probs: Probabilités prédites pour chaque numéro
            star_probs: Probabilités prédites pour chaque étoile
            historical_data: Données historiques
            n_iterations: Nombre d'itérations
            initial_temp: Température initiale
            cooling_rate: Taux de refroidissement
            
        Returns:
            Meilleure combinaison trouvée (numéros, étoiles)
        """
        # Initialisation aléatoire
        current_numbers = sorted(np.random.choice(
            list(range(1, self.max_number + 1)), 
            size=self.n_numbers, 
            replace=False,
            p=[number_probs.get(i, 0.01) for i in range(1, self.max_number + 1)]
        ))
        current_stars = sorted(np.random.choice(
            list(range(1, self.max_star + 1)), 
            size=self.n_stars, 
            replace=False,
            p=[star_probs.get(i, 0.01) for i in range(1, self.max_star + 1)]
        ))
        
        current_cost = self.cost_function(
            (current_numbers, current_stars), 
            number_probs, 
            star_probs, 
            historical_data
        )
        
        best_numbers = current_numbers.copy()
        best_stars = current_stars.copy()
        best_cost = current_cost
        
        temperature = initial_temp
        
        # 🔬 AMÉLIORATION : Suivi pour refroidissement adaptatif
        no_improvement_count = 0
        last_improvement_iter = 0
        
        logger.info(f"Démarrage du recuit simulé quantique: {n_iterations} itérations")
        
        for iteration in range(n_iterations):
            # Générer une nouvelle solution voisine
            new_numbers = current_numbers.copy()
            new_stars = current_stars.copy()
            
            # Mutation quantique: échanger un numéro avec probabilité basée sur les probabilités quantiques
            if np.random.random() < 0.5:
                # Changer un numéro
                idx = np.random.randint(0, len(new_numbers))
                old_num = new_numbers[idx]
                
                # Sélectionner un nouveau numéro avec probabilité quantique
                candidates = [i for i in range(1, self.max_number + 1) if i not in new_numbers]
                probs = [number_probs.get(i, 0.01) for i in candidates]
                probs = np.array(probs)
                probs = probs / probs.sum()  # Normaliser
                
                new_num = np.random.choice(candidates, p=probs)
                new_numbers[idx] = new_num
                new_numbers = sorted(new_numbers)
            else:
                # Changer une étoile
                idx = np.random.randint(0, len(new_stars))
                old_star = new_stars[idx]
                
                candidates = [i for i in range(1, self.max_star + 1) if i not in new_stars]
                probs = [star_probs.get(i, 0.01) for i in candidates]
                probs = np.array(probs)
                probs = probs / probs.sum()
                
                new_star = np.random.choice(candidates, p=probs)
                new_stars[idx] = new_star
                new_stars = sorted(new_stars)
            
            # Calculer le coût de la nouvelle solution
            new_cost = self.cost_function(
                (new_numbers, new_stars), 
                number_probs, 
                star_probs, 
                historical_data
            )
            
            # Accepter ou rejeter selon le critère de Metropolis (avec effet quantique)
            delta = new_cost - current_cost
            
            # 🔬 AMÉLIORATION : Tunneling quantique amélioré (coefficient 0.05 au lieu de 0.1)
            # Permet un tunneling plus agressif pour échapper aux minima locaux
            if delta < 0:
                # Amélioration: toujours accepter
                accept = True
                no_improvement_count = 0
                last_improvement_iter = iteration
            else:
                # Probabilité d'acceptation avec effet de tunneling quantique amélioré
                quantum_tunneling = np.exp(-delta / temperature) * (1 + np.exp(-delta / (temperature * 0.05)))
                accept = np.random.random() < quantum_tunneling
                if not accept:
                    no_improvement_count += 1
            
            if accept:
                current_numbers = new_numbers
                current_stars = new_stars
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_numbers = current_numbers.copy()
                    best_stars = new_stars.copy()
                    best_cost = current_cost
                    no_improvement_count = 0
                    last_improvement_iter = iteration
            
            # 🔬 AMÉLIORATION : Refroidissement adaptatif
            # Si pas d'amélioration depuis 50 itérations, réchauffer légèrement
            if no_improvement_count > 50 and temperature < initial_temp * 0.5:
                temperature *= 1.05  # Réchauffement local pour échapper aux minima locaux
                no_improvement_count = 0  # Réinitialiser le compteur
            else:
                # Refroidissement normal
                temperature *= cooling_rate
            
            if (iteration + 1) % 100 == 0:
                logger.debug(f"Itération {iteration + 1}/{n_iterations}: Coût = {best_cost:.4f}, Temp = {temperature:.2f}, Amélioration itération {last_improvement_iter}")
        
        logger.info(f"✅ Recuit simulé terminé: Meilleur coût = {best_cost:.4f}")
        
        return best_numbers, best_stars


class QuantumInspiredPredictor:
    """
    Prédicteur principal inspiré par le quantique.
    Combine QNN, QLSTM et optimisation quantique pour générer des prédictions.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialise le prédicteur quantique.
        
        Args:
            config: Configuration du prédicteur
        """
        if config is None:
            config = {}
        
        self.config = {
            'max_number': config.get('max_number', 50),
            'max_star': config.get('max_star', 12),
            'n_numbers': config.get('n_numbers', 5),
            'n_stars': config.get('n_stars', 2),
            'use_qnn': config.get('use_qnn', True) and PENNYLANE_AVAILABLE,
            'use_qlstm': config.get('use_qlstm', True) and TORCH_AVAILABLE and PENNYLANE_AVAILABLE,
            'use_quantum_annealing': config.get('use_quantum_annealing', True),
        }
        
        self.qnn = None
        self.qlstm = None
        self.optimizer = QuantumAnnealingOptimizer(
            max_number=self.config['max_number'],
            max_star=self.config['max_star'],
            n_numbers=self.config['n_numbers'],
            n_stars=self.config['n_stars']
        )
        
        logger.info("✅ QuantumInspiredPredictor initialisé")
        logger.info(f"   - QNN: {'Activé' if self.config['use_qnn'] else 'Désactivé'}")
        logger.info(f"   - QLSTM: {'Activé' if self.config['use_qlstm'] else 'Désactivé'}")
        logger.info(f"   - Quantum Annealing: {'Activé' if self.config['use_quantum_annealing'] else 'Désactivé'}")
    
    def predict(self, features: np.ndarray, 
                historical_data: pd.DataFrame = None,
                number_probs: Dict[int, float] = None,
                star_probs: Dict[int, float] = None) -> Tuple[List[int], List[int]]:
        """
        Génère une prédiction quantique.
        
        Args:
            features: Features préparées pour le modèle
            historical_data: Données historiques
            number_probs: Probabilités de base pour les numéros
            star_probs: Probabilités de base pour les étoiles
            
        Returns:
            Prédiction (numéros, étoiles)
        """
        if number_probs is None:
            number_probs = {i: 1.0 / self.config['max_number'] for i in range(1, self.config['max_number'] + 1)}
        
        if star_probs is None:
            star_probs = {i: 1.0 / self.config['max_star'] for i in range(1, self.config['max_star'] + 1)}
        
        # Utiliser l'optimiseur quantique pour trouver la meilleure combinaison
        if self.config['use_quantum_annealing']:
            numbers, stars = self.optimizer.quantum_annealing(
                number_probs, 
                star_probs, 
                historical_data,
                n_iterations=500  # Réduire pour la vitesse
            )
        else:
            # Sélection simple basée sur les probabilités
            numbers = sorted(np.random.choice(
                list(range(1, self.config['max_number'] + 1)),
                size=self.config['n_numbers'],
                replace=False,
                p=[number_probs.get(i, 0.01) for i in range(1, self.config['max_number'] + 1)]
            ))
            stars = sorted(np.random.choice(
                list(range(1, self.config['max_star'] + 1)),
                size=self.config['n_stars'],
                replace=False,
                p=[star_probs.get(i, 0.01) for i in range(1, self.config['max_star'] + 1)]
            ))
        
        return numbers, stars


if __name__ == "__main__":
    # Test du module
    logger.info("Test du module QuantumInspiredPredictor...")
    
    # Test QNN
    if PENNYLANE_AVAILABLE:
        try:
            qnn = QuantumNeuralNetwork(n_qubits=4, n_layers=2)
            test_input = np.random.rand(4)
            output = qnn.forward(test_input)
            logger.info(f"✅ QNN test réussi: output = {output:.4f}")
        except Exception as e:
            logger.error(f"❌ Erreur QNN: {str(e)}")
    
    # Test Optimizer
    try:
        optimizer = QuantumAnnealingOptimizer()
        number_probs = {i: np.random.random() for i in range(1, 51)}
        star_probs = {i: np.random.random() for i in range(1, 13)}
        numbers, stars = optimizer.quantum_annealing(number_probs, star_probs, n_iterations=100)
        logger.info(f"✅ Optimizer test réussi: {numbers}, {stars}")
    except Exception as e:
        logger.error(f"❌ Erreur Optimizer: {str(e)}")
        logger.debug(traceback.format_exc())
    
    logger.info("Tests terminés")

