#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PATCH pour intégrer les features vidéo dans le système EuroMillions
À ajouter dans advanced_encoder.py
"""

# ============================================================================
# ÉTAPE 1: Ajouter cette méthode dans la classe AdvancedEuromillionsEncoder
# ============================================================================

def add_video_features(self, df: pd.DataFrame, video_embeddings: dict) -> pd.DataFrame:
    """
    Ajoute les features vidéo aux données.
    
    Args:
        df: DataFrame avec les données de tirages
        video_embeddings: Dict {video_name: embedding_dict}
        
    Returns:
        DataFrame avec features vidéo ajoutées
    """
    import logging
    logger = logging.getLogger("AdvancedEncoder")
    
    if not video_embeddings or len(video_embeddings) == 0:
        logger.warning("Aucun embedding vidéo fourni")
        return df
    
    logger.info(f"🎥 Intégration de {len(video_embeddings)} embeddings vidéo...")
    
    df_with_video = df.copy()
    
    # Préparer les colonnes pour les features vidéo
    # On utilise les mean_features (2048D) de chaque embedding
    video_feature_dim = 2048
    video_features_list = []
    matched_count = 0
    
    for idx, row in df.iterrows():
        # Extraire la date du tirage
        if 'date_de_tirage' in row:
            date = pd.to_datetime(row['date_de_tirage'])
            date_str = date.strftime('%Y%m%d')
        elif 'Date' in row:
            date = pd.to_datetime(row['Date'])
            date_str = date.strftime('%Y%m%d')
        else:
            logger.warning(f"Ligne {idx}: Pas de colonne de date trouvée")
            video_features_list.append(np.zeros(video_feature_dim))
            continue
        
        # Chercher l'embedding correspondant
        embedding_found = False
        for video_name, embedding in video_embeddings.items():
            if date_str in video_name:
                # Utiliser les features moyennes
                video_features_list.append(embedding['mean_features'])
                embedding_found = True
                matched_count += 1
                break
        
        if not embedding_found:
            # Padding avec des zéros si pas d'embedding
            video_features_list.append(np.zeros(video_feature_dim))
    
    # Convertir en array numpy
    video_features_array = np.array(video_features_list)
    
    # Ajouter les features vidéo au DataFrame
    for i in range(video_feature_dim):
        df_with_video[f'video_feat_{i}'] = video_features_array[:, i]
    
    logger.info(f"✅ Features vidéo ajoutées: {video_feature_dim} colonnes")
    logger.info(f"✅ {matched_count}/{len(df)} tirages ont des features vidéo")
    
    return df_with_video


# ============================================================================
# ÉTAPE 2: Modifier la méthode encode_features pour inclure les vidéos
# ============================================================================

def encode_features_with_video(self, df: pd.DataFrame, video_embeddings: dict = None) -> pd.DataFrame:
    """
    Encode toutes les features incluant les vidéos.
    
    Args:
        df: DataFrame avec les données brutes
        video_embeddings: Dict optionnel avec les embeddings vidéo
        
    Returns:
        DataFrame avec toutes les features encodées
    """
    import logging
    logger = logging.getLogger("AdvancedEncoder")
    
    logger.info("Encodage des features avec réflexion IA...")
    
    # Encoder les features temporelles
    df_encoded = self.encode_temporal_features(df)
    
    # Encoder les features numériques
    df_encoded = self.encode_number_features(df_encoded)
    
    # NOUVEAU: Ajouter les features vidéo
    if video_embeddings is not None and len(video_embeddings) > 0:
        df_encoded = self.add_video_features(df_encoded, video_embeddings)
    else:
        logger.warning("⚠️ Aucune feature vidéo ajoutée - video_embeddings est vide ou None")
    
    # Appliquer la réflexion IA si disponible
    if self.ai_reflection is not None:
        try:
            reflection = self.ai_reflection.generate_reflection(df_encoded)
            if reflection:
                logger.info("💡 Réflexion IA reçue pour améliorer les features")
                logger.info(f"Merci de votre réflexion! La meilleure réflexion est: {reflection[:200]}...")
        except Exception as e:
            logger.warning(f"Erreur lors de la réflexion IA: {str(e)}")
    
    logger.info(f"Features encodées: {len(df_encoded.columns)} colonnes")
    
    return df_encoded


# ============================================================================
# ÉTAPE 3: Modifier prepare_ml_features pour gérer les features vidéo
# ============================================================================

def prepare_ml_features_with_video(self, df: pd.DataFrame, fit: bool = True) -> np.ndarray:
    """
    Prépare les features pour le ML en incluant les vidéos.
    
    Args:
        df: DataFrame avec features encodées
        fit: Si True, fit le scaler
        
    Returns:
        Array numpy avec features normalisées
    """
    import logging
    logger = logging.getLogger("AdvancedEncoder")
    
    # Sélectionner uniquement les colonnes numériques
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
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
    
    X = df[numeric_cols].values
    
    # Normaliser
    if fit:
        X = self.scaler.fit_transform(X)
    else:
        X = self.scaler.transform(X)
    
    return X


# ============================================================================
# INSTRUCTIONS D'INTÉGRATION
# ============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                   PATCH D'INTÉGRATION DES FEATURES VIDÉO                 ║
╚══════════════════════════════════════════════════════════════════════════╝

📋 ÉTAPES D'INTÉGRATION:

1. Ouvrez le fichier: script/advanced_encoder.py

2. Ajoutez la méthode add_video_features() dans la classe AdvancedEuromillionsEncoder
   (après la méthode encode_number_features)

3. Modifiez la méthode encode_features() pour ajouter:
   
   # AJOUTER CETTE LIGNE après encode_number_features:
   if video_embeddings is not None and len(video_embeddings) > 0:
       df_encoded = self.add_video_features(df_encoded, video_embeddings)

4. Modifiez la signature de encode_features():
   
   def encode_features(self, df: pd.DataFrame, video_embeddings: dict = None) -> pd.DataFrame:

5. Dans prepare_ml_features(), ajoutez le logging des features vidéo:
   
   video_cols = [col for col in numeric_cols if col.startswith('video_feat_')]
   if len(video_cols) > 0:
       logger.info(f"✅ {len(video_cols)} features vidéo détectées")

6. Dans le fichier principal (euromillions_trainer.py ou check_and_train.py):
   
   # Charger les embeddings vidéo
   from video_deep_analyzer import VideoDeepAnalyzer
   from pathlib import Path
   import pickle
   
   embeddings = {}
   for pkl_file in Path("encoded_videos").glob("*_embedding.pkl"):
       video_name = pkl_file.stem.replace("_embedding", "")
       with open(pkl_file, 'rb') as f:
           embeddings[video_name] = pickle.load(f)
   
   # Passer aux analyseurs
   analyzer = EuromillionsAdvancedAnalyzer(
       csv_file="tirage_euromillions_complet.csv",
       video_embeddings=embeddings  # AJOUTER CE PARAMÈTRE
   )

╔══════════════════════════════════════════════════════════════════════════╗
║  Après ces modifications, relancez l'entraînement et vous verrez:       ║
║  ✅ Features fusionnées: 2096 colonnes (48 + 2048 vidéo)               ║
╚══════════════════════════════════════════════════════════════════════════╝
""")
