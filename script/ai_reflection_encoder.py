#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Système de réflexion IA pour l'encodeur avancé EuroMillions
Utilise des LLMs pour améliorer les features et récompenser les meilleures réflexions
"""

import os
import sys
import logging
import json
import requests
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

logger = logging.getLogger("AIReflectionEncoder")

class AIReflectionEncoder:
    """Système de réflexion IA pour améliorer l'encodage"""
    
    # Configurations LLM disponibles
    LLM_CONFIGS = {
        'openai': {
            'api_type': 'openai',
            'model': 'gpt-4o-mini-2024-07-18',
            'base_url': 'https://api.openai.com/v1',
            'api_key': '',
            'max_tokens': 8096,
            'temperature': 0.0
        },
        'grok-4-latest': {
            'api_type': 'grok-4-latest',
            'model': 'x-ai/grok-beta',
            'base_url': 'https://openrouter.ai/api/v1',
            'api_key': '',
            'max_tokens': 4096,
            'temperature': 0.0
        },
        'claude-opus-4.1': {
            'api_type': 'claude-opus-4.1',
            'model': 'anthropic/claude-opus-4.1',
            'base_url': 'https://openrouter.ai/api/v1',
            'api_key': '',
            'max_tokens': 4096,
            'temperature': 0.0
        }
    }
    
    def __init__(self, llm_config: str = 'openai', enable_reflection: bool = True):
        """
        Initialise le système de réflexion IA.
        
        Args:
            llm_config: Nom de la configuration LLM à utiliser
            enable_reflection: Activer la réflexion IA
        """
        self.llm_config_name = llm_config
        self.enable_reflection = enable_reflection
        self.reward_history = []  # Historique des récompenses
        self.best_reflections = []  # Meilleures réflexions
        
        if llm_config in self.LLM_CONFIGS:
            self.llm_config = self.LLM_CONFIGS[llm_config].copy()
            logger.info(f"Configuration LLM chargée: {llm_config}")
        else:
            logger.warning(f"Configuration LLM '{llm_config}' non trouvée, utilisation de 'openai'")
            self.llm_config = self.LLM_CONFIGS['openai'].copy()
        
        # Répertoire pour sauvegarder les réflexions
        self.reflection_dir = Path("reflections_euromillions")
        if not self.reflection_dir.exists():
            self.reflection_dir.mkdir(parents=True, exist_ok=True)
    
    def call_llm(self, prompt: str, system_prompt: str = None) -> Optional[str]:
        """
        Appelle le LLM avec un prompt.
        
        Args:
            prompt: Prompt utilisateur
            system_prompt: Prompt système (optionnel)
            
        Returns:
            Réponse du LLM ou None en cas d'erreur
        """
        if not self.enable_reflection:
            return None
        
        # Vérifier que la clé API est présente et valide
        if not self.llm_config.get('api_key') or self.llm_config['api_key'].strip() == '':
            logger.warning("Clé API LLM non configurée. Réflexion IA désactivée.")
            return None
        
        try:
            headers = {
                "Authorization": f"Bearer {self.llm_config['api_key']}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/euromillions-ai",
                "X-Title": "EuroMillions AI Encoder"
            }
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            # Construire le payload selon le type d'API
            if self.llm_config['api_type'] == 'openai':
                # API OpenAI directe
                payload = {
                    "model": self.llm_config['model'],
                    "messages": messages,
                    "max_tokens": self.llm_config['max_tokens'],
                    "temperature": self.llm_config['temperature']
                }
                api_url = f"{self.llm_config['base_url']}/chat/completions"
            else:
                # API OpenRouter (Grok, Claude, etc.)
                payload = {
                    "model": self.llm_config['model'],
                    "messages": messages,
                    "max_tokens": self.llm_config['max_tokens'],
                    "temperature": self.llm_config['temperature']
                }
                api_url = f"{self.llm_config['base_url']}/chat/completions"
            
            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    return result['choices'][0]['message']['content']
            elif response.status_code == 401:
                logger.warning("Erreur d'authentification LLM (401). Vérifiez votre clé API.")
                logger.warning("L'entraînement continuera sans réflexion IA.")
                return None
            elif response.status_code == 429:
                logger.warning("Limite de taux LLM atteinte (429). Réflexion IA temporairement désactivée.")
                return None
            else:
                logger.warning(f"Erreur LLM: {response.status_code} - {response.text[:200]}")
                logger.warning("L'entraînement continuera sans réflexion IA.")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.warning(f"Erreur réseau lors de l'appel LLM: {str(e)}")
            logger.warning("L'entraînement continuera sans réflexion IA.")
            return None
        except Exception as e:
            logger.warning(f"Erreur lors de l'appel LLM: {str(e)}")
            logger.warning("L'entraînement continuera sans réflexion IA.")
            return None
    
    def reflect_on_features(self, df: pd.DataFrame, feature_summary: Dict[str, Any]) -> Optional[str]:
        """
        Demande à l'IA de réfléchir sur les features générées.
        
        Args:
            df: DataFrame avec les données
            feature_summary: Résumé des features générées
            
        Returns:
            Réflexion de l'IA ou None
        """
        if not self.enable_reflection:
            return None
        
        system_prompt = """Tu es un expert en Machine Learning spécialisé dans l'analyse de données de loterie.
Ton rôle est d'analyser les features générées pour les tirages EuroMillions et de proposer des améliorations.
Réfléchis aux patterns, aux corrélations, et aux features qui pourraient améliorer la précision des prédictions."""
        
        prompt = f"""Analyse les features suivantes générées pour {len(df)} tirages EuroMillions:

Résumé des features:
- Features temporelles: {feature_summary.get('temporal_count', 0)}
- Features numériques: {feature_summary.get('numerical_count', 0)}
- Features de séquence: {feature_summary.get('sequence_count', 0)}
- Total: {feature_summary.get('total_count', 0)} features

Statistiques des données:
- Nombre de tirages: {len(df)}
- Période: {df['Date'].min() if 'Date' in df.columns else 'N/A'} à {df['Date'].max() if 'Date' in df.columns else 'N/A'}

Réfléchis et propose:
1. Quelles features sont les plus importantes pour prédire les numéros?
2. Y a-t-il des patterns temporels à exploiter?
3. Quelles améliorations pourraient être apportées à l'encodage?
4. Des features manquantes qui pourraient améliorer la précision?

Fournis une réflexion structurée et actionnable."""
        
        reflection = self.call_llm(prompt, system_prompt)
        
        if reflection:
            logger.info("Réflexion IA générée avec succès")
            self._save_reflection(reflection, feature_summary)
            return reflection
        
        return None
    
    def generate_reflection(self, df: pd.DataFrame) -> Optional[str]:
        """
        Génère une réflexion IA sur les features (alias pour reflect_on_features).
        Cette méthode est appelée depuis advanced_encoder.py pour compatibilité.
        
        Args:
            df: DataFrame avec les données encodées
            
        Returns:
            Réflexion de l'IA ou None
        """
        if not self.enable_reflection:
            return None
        
        # Créer un résumé simple des features
        feature_summary = {
            'temporal_count': len([col for col in df.columns if any(x in col.lower() for x in ['day', 'month', 'week', 'year', 'temporal'])]),
            'numerical_count': len([col for col in df.columns if df[col].dtype in ['int64', 'float64']]),
            'sequence_count': len([col for col in df.columns if any(x in col.lower() for x in ['sequence', 'gap', 'trend'])]),
            'total_count': len(df.columns)
        }
        
        # Utiliser la méthode existante
        return self.reflect_on_features(df, feature_summary)
    
    def reward_best_reflection(self, reflection: str, performance_metrics: Dict[str, float]) -> None:
        """
        Récompense la meilleure réflexion basée sur les métriques de performance.
        
        Args:
            reflection: Réflexion à évaluer
            performance_metrics: Métriques de performance (accuracy, f1_score, etc.)
        """
        if not reflection:
            return
        
        # Calculer un score de récompense
        reward_score = 0.0
        
        if 'accuracy' in performance_metrics:
            reward_score += performance_metrics['accuracy'] * 0.3
        if 'f1_score' in performance_metrics:
            reward_score += performance_metrics['f1_score'] * 0.4
        if 'precision' in performance_metrics:
            reward_score += performance_metrics['precision'] * 0.2
        if 'recall' in performance_metrics:
            reward_score += performance_metrics['recall'] * 0.1
        
        reward_entry = {
            'timestamp': datetime.now().isoformat(),
            'reflection': reflection[:500],  # Limiter la taille
            'reward_score': reward_score,
            'metrics': performance_metrics
        }
        
        self.reward_history.append(reward_entry)
        
        # Garder seulement les meilleures réflexions
        if reward_score > 0.7:  # Seuil pour "meilleure réflexion"
            self.best_reflections.append(reward_entry)
            self.best_reflections.sort(key=lambda x: x['reward_score'], reverse=True)
            self.best_reflections = self.best_reflections[:10]  # Garder les 10 meilleures
            
            logger.info(f"✅ Meilleure réflexion récompensée! Score: {reward_score:.3f}")
            logger.info(f"Merci de votre réflexion! La meilleure réflexion est: {reflection[:200]}...")
        
        # Sauvegarder l'historique
        self._save_reward_history()
    
    def get_best_reflection(self) -> Optional[str]:
        """
        Récupère la meilleure réflexion.
        
        Returns:
            Meilleure réflexion ou None
        """
        if self.best_reflections:
            return self.best_reflections[0]['reflection']
        return None
    
    def improve_features_with_reflection(self, df: pd.DataFrame, current_features: List[str]) -> List[str]:
        """
        Améliore les features basées sur les meilleures réflexions.
        
        Args:
            df: DataFrame avec les données
            current_features: Liste des features actuelles
            
        Returns:
            Liste améliorée de features
        """
        if not self.enable_reflection or not self.best_reflections:
            return current_features
        
        best_reflection = self.get_best_reflection()
        if not best_reflection:
            return current_features
        
        system_prompt = """Tu es un expert en feature engineering pour le Machine Learning.
Analyse la meilleure réflexion précédente et propose des features supplémentaires à ajouter."""
        
        prompt = f"""Basé sur cette réflexion précédente (qui a obtenu un score élevé):
{best_reflection}

Et les features actuelles:
{', '.join(current_features[:20])}...

Propose des features supplémentaires spécifiques à ajouter pour améliorer la précision.
Réponds uniquement avec une liste de noms de features, une par ligne, sans explication."""
        
        response = self.call_llm(prompt, system_prompt)
        
        if response:
            # Parser la réponse pour extraire les noms de features
            new_features = [line.strip() for line in response.split('\n') 
                          if line.strip() and not line.strip().startswith('#')]
            
            logger.info(f"Features supplémentaires proposées par l'IA: {len(new_features)}")
            return current_features + new_features
        
        return current_features
    
    def _save_reflection(self, reflection: str, feature_summary: Dict[str, Any]) -> None:
        """Sauvegarde une réflexion dans un fichier."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.reflection_dir / f"reflection_{timestamp}.json"
        
        data = {
            'timestamp': timestamp,
            'reflection': reflection,
            'feature_summary': feature_summary,
            'llm_config': self.llm_config_name
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.debug(f"Réflexion sauvegardée: {filename}")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de la réflexion: {str(e)}")
    
    def _save_reward_history(self) -> None:
        """Sauvegarde l'historique des récompenses."""
        filename = self.reflection_dir / "reward_history.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump({
                    'reward_history': self.reward_history,
                    'best_reflections': self.best_reflections
                }, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de l'historique: {str(e)}")
    
    def load_reward_history(self) -> None:
        """Charge l'historique des récompenses depuis le fichier."""
        filename = self.reflection_dir / "reward_history.json"
        
        if filename.exists():
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.reward_history = data.get('reward_history', [])
                    self.best_reflections = data.get('best_reflections', [])
                logger.info(f"Historique des récompenses chargé: {len(self.best_reflections)} meilleures réflexions")
            except Exception as e:
                logger.error(f"Erreur lors du chargement de l'historique: {str(e)}")

