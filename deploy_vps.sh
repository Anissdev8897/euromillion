#!/bin/bash
# Script de déploiement sur le VPS pour le serveur API EuroMillions/Loto

# ⚠️ CRITIQUE : Type de jeu (euromillions ou loto) - peut être défini via variable d'environnement
GAME_TYPE=${GAME_TYPE:-euromillions}  # Par défaut: euromillions

echo "=========================================="
echo "Déploiement du serveur API ${GAME_TYPE^}"
echo "=========================================="
echo ""

# Vérifier que Python est installé
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 n'est pas installé"
    exit 1
fi

echo "✅ Python3 trouvé: $(python3 --version)"
echo "✅ Type de jeu: $GAME_TYPE"
echo ""

# Créer un environnement virtuel si nécessaire
if [ ! -d "venv" ]; then
    echo "📦 Création de l'environnement virtuel..."
    python3 -m venv venv
fi

# Activer l'environnement virtuel
echo "🔧 Activation de l'environnement virtuel..."
source venv/bin/activate

# Installer les dépendances
echo "📥 Installation des dépendances..."
pip install --upgrade pip
pip install -r requirements_api.txt

echo ""
echo "✅ Dépendances installées"
echo ""

# Vérifier que le fichier CSV existe
CSV_FILE="tirage_${GAME_TYPE}_complet.csv"
if [ ! -f "$CSV_FILE" ]; then
    echo "⚠️  ATTENTION: Le fichier CSV n'existe pas: $CSV_FILE"
    echo "   Le serveur fonctionnera mais certaines fonctionnalités peuvent être limitées."
    echo "   Génération du fichier de cycles..."
    python3 script/cycle_data_generator.py --csv "$CSV_FILE" --generate 2>/dev/null || echo "   ⚠️  Impossible de générer le fichier de cycles (fichier CSV manquant)"
    echo ""
fi

# Vérifier le fichier de cycles
CYCLE_FILE="tirage_${GAME_TYPE}_complet_cycles.csv"
if [ ! -f "$CYCLE_FILE" ]; then
    echo "⚠️  Fichier de cycles non trouvé: $CYCLE_FILE"
    echo "   Tentative de génération..."
    if [ -f "$CSV_FILE" ]; then
        python3 script/cycle_data_generator.py --csv "$CSV_FILE" --generate || echo "   ⚠️  Erreur lors de la génération"
    fi
    echo ""
fi

# Créer les répertoires nécessaires
echo "📁 Création des répertoires..."
mkdir -p "resultats_${GAME_TYPE}"
mkdir -p "models_${GAME_TYPE}"
mkdir -p "reflections_${GAME_TYPE}"

echo "✅ Répertoires créés"
echo ""

# Vérifier que les modèles existent
MODEL_DIR="models_${GAME_TYPE}"
if [ ! -d "$MODEL_DIR" ] || [ -z "$(ls -A $MODEL_DIR 2>/dev/null)" ]; then
    echo "⚠️  ATTENTION: Aucun modèle trouvé dans $MODEL_DIR/"
    echo "   Les modèles doivent être entraînés sur le PC local et transférés sur le VPS."
    echo "   Voir README_LOTO.md ou GUIDE_ADAPTATION_LOTO.md pour plus d'informations."
    echo ""
fi

echo "=========================================="
echo "Configuration terminée"
echo "=========================================="
echo ""
echo "Pour démarrer le serveur:"
echo "  export GAME_TYPE=$GAME_TYPE  # Optionnel, euromillions par défaut"
echo "  ./start_api_vps.sh"
echo ""
echo "Ou utiliser systemd pour un service permanent:"
echo "  sudo systemctl start euromillions-api"
echo ""
echo "Pour changer le type de jeu:"
echo "  export GAME_TYPE=loto"
echo "  ./start_api_vps.sh"
echo ""

