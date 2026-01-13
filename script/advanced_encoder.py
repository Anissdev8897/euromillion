#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Encodeur avancé pour améliorer la précision des prédictions EuroMillions
Ajoute des features encodées sophistiquées pour renforcer les modèles ML
VERSION MODIFIÉE: Intégration des features vidéo
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import logging

logger = logging.getLogger("AdvancedEncoder")

# Import du système de réflexion IA
try:
    from ai_reflection_encoder import AIReflectionEncoder
    AI_REFLECTION_AVAILABLE = True
except ImportError:
    AI_REFLECTION_AVAILABLE = False
    logger.warning("Système de réflexion IA non disponible.")

class AdvancedEuromillionsEncoder:
    """Encodeur avancé pour les features EuroMillions avec support vidéo"""
    
    def __init__(self, enable_ai_reflection: bool = True, llm_config: str = 'openai'):
        """
        Initialise l'encodeur.
        
        Args:
            enable_ai_reflection: Activer la réflexion IA
            llm_config: Configuration LLM à utiliser
        """
        self.scaler = StandardScaler()
        self.pca = None
        self.feature_names = []
        self.numeric_columns = []
        
        # Initialiser le système de réflexion IA
        self.ai_reflection = None
        if AI_REFLECTION_AVAILABLE and enable_ai_reflection:
            try:
                self.ai_reflection = AIReflectionEncoder(llm_config=llm_config, enable_reflection=enable_ai_reflection)
                self.ai_reflection.load_reward_history()
                logger.info("✅ Système de réflexion IA initialisé - Amélioration continue activée")
            except Exception as e:
                logger.warning(f"Erreur lors de l'initialisation de la réflexion IA: {str(e)}")
                self.ai_reflection = None
        
    def encode_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode les features temporelles.
        
        Args:
            df: DataFrame avec colonne Date ou date_de_tirage
            
        Returns:
            DataFrame avec features temporelles ajoutées
        """
        df_encoded = df.copy()
        
        # Déterminer la colonne de date
        date_col = None
        if 'Date' in df.columns:
            date_col = 'Date'
        elif 'date_de_tirage' in df.columns:
            date_col = 'date_de_tirage'
        
        if date_col:
            df_encoded[date_col] = pd.to_datetime(df_encoded[date_col])
            
            # Jour de la semaine (0=lundi, 6=dimanche)
            df_encoded['day_of_week'] = df_encoded[date_col].dt.dayofweek
            
            # Jour du mois
            df_encoded['day_of_month'] = df_encoded[date_col].dt.day
            
            # Mois
            df_encoded['month'] = df_encoded[date_col].dt.month
            
            # Semaine de l'année
            df_encoded['week_of_year'] = df_encoded[date_col].dt.isocalendar().week
            
            # Encodage cyclique pour les features temporelles
            df_encoded['day_of_week_sin'] = np.sin(2 * np.pi * df_encoded['day_of_week'] / 7)
            df_encoded['day_of_week_cos'] = np.cos(2 * np.pi * df_encoded['day_of_week'] / 7)
            df_encoded['month_sin'] = np.sin(2 * np.pi * df_encoded['month'] / 12)
            df_encoded['month_cos'] = np.cos(2 * np.pi * df_encoded['month'] / 12)
        
        return df_encoded
    
    def encode_number_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode les features des numéros.
        
        Args:
            df: DataFrame avec colonnes N1-N5, E1-E2
            
        Returns:
            DataFrame avec features numériques ajoutées
        """
        df_encoded = df.copy()
        
        number_cols = ['N1', 'N2', 'N3', 'N4', 'N5']
        star_cols = ['E1', 'E2']
        
        if all(col in df.columns for col in number_cols):
            # Somme des numéros
            df_encoded['sum_numbers'] = df[number_cols].sum(axis=1)
            
            # Moyenne des numéros
            df_encoded['mean_numbers'] = df[number_cols].mean(axis=1)
            
            # Écart-type des numéros
            df_encoded['std_numbers'] = df[number_cols].std(axis=1)
            
            # Nombre de numéros pairs
            df_encoded['count_even'] = df[number_cols].apply(lambda x: (x % 2 == 0).sum(), axis=1)
            
            # Nombre de numéros impairs
            df_encoded['count_odd'] = df[number_cols].apply(lambda x: (x % 2 != 0).sum(), axis=1)
            
            # Écart entre min et max
            df_encoded['range_numbers'] = df[number_cols].max(axis=1) - df[number_cols].min(axis=1)
        
        if all(col in df.columns for col in star_cols):
            # Somme des étoiles
            df_encoded['sum_stars'] = df[star_cols].sum(axis=1)
            
            # Écart entre les étoiles
            df_encoded['diff_stars'] = abs(df[star_cols[0]] - df[star_cols[1]])
        
        return df_encoded
    
    def add_video_features(self, df: pd.DataFrame, video_embeddings: dict) -> pd.DataFrame:
        """
        🎥 NOUVEAU: Ajoute les features vidéo aux données.
        
        Args:
            df: DataFrame avec les données de tirages
            video_embeddings: Dict {video_name: embedding_dict}
            
        Returns:
            DataFrame avec features vidéo ajoutées
        """
        if not video_embeddings or len(video_embeddings) == 0:
            logger.warning("Aucun embedding vidéo fourni")
            return df
        
        logger.info(f"🎥 Intégration de {len(video_embeddings)} embeddings vidéo...")
        
        # Vérifier si les features vidéo sont déjà présentes (éviter les doublons)
        existing_video_cols = [col for col in df.columns if col.startswith('video_feat_')]
        if len(existing_video_cols) > 0:
            logger.info(f"⚠️ Features vidéo déjà présentes ({len(existing_video_cols)} colonnes) - Pas d'ajout en double")
            return df
        
        df_with_video = df.copy()
        
        # Déterminer la dimension des features vidéo depuis le premier embedding
        first_embedding = next(iter(video_embeddings.values()))
        if 'mean_features' in first_embedding:
            video_feature_dim = len(first_embedding['mean_features'])
            logger.info(f"📐 Dimension des features vidéo détectée: {video_feature_dim}D")
        else:
            # Fallback: ResNet50 = 2048D
            video_feature_dim = 2048
            logger.warning(f"⚠️ Dimension non détectée, utilisation par défaut: {video_feature_dim}D")
        
        video_features_list = []
        matched_count = 0
        unmatched_dates = []
        
        # Déterminer la colonne de date
        date_col = None
        if 'date_de_tirage' in df.columns:
            date_col = 'date_de_tirage'
        elif 'Date' in df.columns:
            date_col = 'Date'
        
        if not date_col:
            logger.error("Aucune colonne de date trouvée (date_de_tirage ou Date)")
            return df
        
        # Convertir la colonne Date en datetime si nécessaire
        if df[date_col].dtype != 'datetime64[ns]':
            df_with_video[date_col] = pd.to_datetime(df_with_video[date_col], errors='coerce')
        
        for idx, row in df_with_video.iterrows():
            # Extraire la date du tirage
            date = row[date_col]
            
            # Gérer les dates NaT
            if pd.isna(date):
                video_features_list.append(np.zeros(video_feature_dim))
                continue
            
            date_str = date.strftime('%Y%m%d')
            
            # Chercher l'embedding correspondant (chercher la date dans le nom de la vidéo)
            embedding_found = False
            for video_name, embedding in video_embeddings.items():
                # Le nom de la vidéo contient généralement la date au format YYYYMMDD
                if date_str in video_name:
                    # Vérifier que l'embedding a les features nécessaires
                    if 'mean_features' in embedding:
                        mean_features = embedding['mean_features']
                        # S'assurer que la dimension correspond
                        if len(mean_features) == video_feature_dim:
                            video_features_list.append(mean_features)
                        else:
                            # Ajuster la dimension si nécessaire
                            if len(mean_features) > video_feature_dim:
                                video_features_list.append(mean_features[:video_feature_dim])
                            else:
                                padded = np.zeros(video_feature_dim)
                                padded[:len(mean_features)] = mean_features
                                video_features_list.append(padded)
                        embedding_found = True
                        matched_count += 1
                        break
            
            if not embedding_found:
                # Padding avec des zéros si pas d'embedding
                video_features_list.append(np.zeros(video_feature_dim))
                unmatched_dates.append(date_str)
        
        # Convertir en array numpy
        if len(video_features_list) > 0:
            video_features_array = np.array(video_features_list)
            
            # Vérifier la cohérence des dimensions
            if video_features_array.shape[0] != len(df_with_video):
                logger.error(f"❌ Incohérence: {video_features_array.shape[0]} features pour {len(df_with_video)} lignes")
                return df
            
            # Ajouter les features vidéo au DataFrame
            for i in range(video_feature_dim):
                df_with_video[f'video_feat_{i}'] = video_features_array[:, i]
            
            logger.info(f"✅ Features vidéo ajoutées: {video_feature_dim} colonnes")
            logger.info(f"✅ {matched_count}/{len(df)} tirages ont des features vidéo correspondantes")
            
            if unmatched_dates and len(unmatched_dates) <= 10:
                logger.info(f"📅 Dates sans embedding: {', '.join(unmatched_dates[:10])}")
            elif len(unmatched_dates) > 10:
                logger.info(f"📅 {len(unmatched_dates)} dates sans embedding (exemples: {', '.join(unmatched_dates[:5])}...)")
        else:
            logger.warning("⚠️ Aucune feature vidéo générée")
        
        return df_with_video
    
    def encode_features(self, df: pd.DataFrame, video_embeddings: dict = None) -> pd.DataFrame:
        """
        Encode toutes les features incluant les vidéos.
        
        Args:
            df: DataFrame avec les données brutes
            video_embeddings: Dict optionnel avec les embeddings vidéo
            
        Returns:
            DataFrame avec toutes les features encodées
        """
        logger.info("Encodage des features avec réflexion IA...")
        
        # Encoder les features temporelles
        df_encoded = self.encode_temporal_features(df)
        
        # Encoder les features numériques
        df_encoded = self.encode_number_features(df_encoded)
        
        # 🎥 NOUVEAU: Ajouter les features vidéo
        if video_embeddings is not None and len(video_embeddings) > 0:
            df_encoded = self.add_video_features(df_encoded, video_embeddings)
        else:
            logger.warning("⚠️ Aucune feature vidéo ajoutée - video_embeddings est vide ou None")
        
        # Appliquer la réflexion IA si disponible
        if self.ai_reflection is not None:
            try:
                # Générer une réflexion IA sur les features encodées
                reflection = self.ai_reflection.generate_reflection(df_encoded)
                if reflection:
                    logger.info("💡 Réflexion IA reçue pour améliorer les features")
                    logger.info(f"Merci de votre réflexion! La meilleure réflexion est: {reflection[:200]}...")
            except Exception as e:
                logger.warning(f"Erreur lors de la réflexion IA: {str(e)}")
        
        logger.info(f"Features encodées: {len(df_encoded.columns)} colonnes")
        
        return df_encoded
    
    def prepare_ml_features(self, df: pd.DataFrame, fit: bool = True, use_scaler: bool = True, video_embeddings: dict = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prépare les features pour le ML en incluant les vidéos.
        
        Args:
            df: DataFrame avec features encodées
            fit: Si True, fit le scaler
            use_scaler: Si True, applique la normalisation (compatibilité avec l'ancien code)
            video_embeddings: Dictionnaire des embeddings vidéo {date: embedding}
            
        Returns:
            Tuple (X, y) avec features normalisées et targets
        """
        # 🎥 NOUVEAU: Ajouter les features vidéo si disponibles ET si elles ne sont pas déjà présentes
        df_with_video = df.copy()
        
        # Vérifier si les features vidéo sont déjà présentes
        existing_video_cols = [col for col in df.columns if col.startswith('video_feat_')]
        
        if len(existing_video_cols) == 0 and video_embeddings is not None and len(video_embeddings) > 0:
            # Les features vidéo ne sont pas présentes, les ajouter
            df_with_video = self.add_video_features(df_with_video, video_embeddings)
        elif len(existing_video_cols) > 0:
            logger.info(f"✅ Features vidéo déjà présentes dans le DataFrame ({len(existing_video_cols)} colonnes)")
        
        # Sélectionner uniquement les colonnes numériques
        numeric_cols = df_with_video.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclure les colonnes cibles
        exclude_cols = ['N1', 'N2', 'N3', 'N4', 'N5', 'E1', 'E2']
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Compter les features vidéo
        video_cols = [col for col in numeric_cols if col.startswith('video_feat_')]
        
        if len(video_cols) > 0:
            logger.info(f"✅ {len(video_cols)} features vidéo détectées et incluses dans le ML")
        else:
            logger.warning("⚠️ Aucune feature vidéo détectée dans les colonnes")
        
        logger.info(f"✅ {len(numeric_cols)} colonnes numériques sélectionnées pour les features ML")
        logger.info(f"   - Features classiques: {len(numeric_cols) - len(video_cols)}")
        logger.info(f"   - Features vidéo: {len(video_cols)}")
        
        self.numeric_columns = numeric_cols
        
        X = df_with_video[numeric_cols].values
        
        # Extraire les targets (y)
        target_cols = ['N1', 'N2', 'N3', 'N4', 'N5', 'E1', 'E2']
        if all(col in df_with_video.columns for col in target_cols):
            y = df_with_video[target_cols].values
        else:
            # Si pas de targets, retourner un array vide
            y = np.array([])
        
        # Normaliser si demandé
        if use_scaler:
            if fit:
                X = self.scaler.fit_transform(X)
            else:
                X = self.scaler.transform(X)
        
        return X, y
