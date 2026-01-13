#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module pour la pondération basée sur la série de Fibonacci.
"""

import numpy as np
from collections import Counter
from typing import Dict, List, Counter as CounterType # Utiliser CounterType pour éviter les conflits de nom

def get_fibonacci_sequence(n: int) -> List[int]:
    """
    Génère les n premiers nombres de la séquence de Fibonacci.
    """
    if n <= 0:
        return []
    elif n == 1:
        return [1]
    sequence = [1, 1]
    while len(sequence) < n:
        sequence.append(sequence[-1] + sequence[-2])
    return sequence

def apply_inverse_fibonacci_weights(
    frequency_counter: CounterType[int], # Utiliser CounterType ici
    reverse_order: bool = True
) -> Dict[int, float]:
    """
    Applique une pondération basée sur la série de Fibonacci inversée aux éléments.
    Par défaut (reverse_order=True), les éléments moins fréquents reçoivent des poids plus élevés.

    Args:
        frequency_counter (Counter): Un objet Counter contenant les fréquences des éléments.
        reverse_order (bool): Si True, les poids Fibonacci sont assignés dans l'ordre inverse
                              (les moins fréquents obtiennent les poids les plus élevés).
                              Si False, les plus fréquents obtiennent les poids les plus élevés.

    Returns:
        Dict[int, float]: Un dictionnaire où les clés sont les éléments et les valeurs sont leurs poids Fibonacci (normalisés).
    """
    if not frequency_counter:
        return {}

    # Trier les éléments par fréquence (du moins fréquent au plus fréquent par défaut)
    sorted_items = sorted(frequency_counter.items(), key=lambda item: item[1])

    # Générer la séquence de Fibonacci de la taille nécessaire
    n_items = len(sorted_items)
    fib_sequence = get_fibonacci_sequence(n_items)

    if not fib_sequence:
        # Cela devrait être géré par get_fibonacci_sequence, mais sécurité supplémentaire
        return {item: 0.0 for item, _ in sorted_items}

    # Assigner les poids Fibonacci
    weights = {}
    if reverse_order:
        # Les moins fréquents obtiennent les poids Fibonacci les plus élevés
        for i, (item, _) in enumerate(sorted_items):
            # Assurez-vous que l'index est valide pour fib_sequence
            weights[item] = float(fib_sequence[n_items - 1 - i])
    else:
        # Les plus fréquents obtiennent les poids Fibonacci les plus élevés
        for i, (item, _) in enumerate(sorted_items):
            weights[item] = float(fib_sequence[i])

    # Normaliser les poids pour qu'ils somment à 1
    total_weight = sum(weights.values())
    if total_weight > 0:
        normalized_weights = {item: weight / total_weight for item, weight in weights.items()}
        return normalized_weights
    else:
        # Si tous les poids sont nuls (cas improbable mais possible si fib_sequence retourne des zéros ou n_items est 0),
        # assigner un poids égal pour éviter la division par zéro.
        equal_weight = 1.0 / n_items if n_items > 0 else 0.0
        return {item: equal_weight for item, _ in sorted_items}

