#!/bin/bash
# Script de démarrage du serveur API sur le VPS
# Usage: ./start_api_vps.sh [--port PORT] [--host HOST]

set -e  # Arrêter en cas d'erreur

# Configuration par défaut
PORT=${PORT:-5000}
HOST=${HOST:-0.0.0.0}
GAME_TYPE=${GAME_TYPE:-euromillions}  # ou "loto"

# Couleurs pour les messages
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Serveur API ${GAME_TYPE^} - VPS${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Vérifier que Python est installé
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 n'est pas installé${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Python3 trouvé: $(python3 --version)${NC}"

# Activer l'environnement virtuel
if [ -d "venv" ]; then
    echo -e "${GREEN}🔧 Activation de l'environnement virtuel...${NC}"
    source venv/bin/activate
else
    echo -e "${YELLOW}⚠️  Environnement virtuel non trouvé. Création...${NC}"
    python3 -m venv venv
    source venv/bin/activate
    echo -e "${GREEN}✅ Environnement virtuel créé${NC}"
fi

# Vérifier et installer les dépendances
echo -e "${GREEN}📦 Vérification des dépendances...${NC}"
if ! python3 -c "import flask" 2>/dev/null; then
    echo -e "${YELLOW}📥 Installation des dépendances...${NC}"
    pip install --upgrade pip
    pip install -r requirements_api.txt
    echo -e "${GREEN}✅ Dépendances installées${NC}"
else
    echo -e "${GREEN}✅ Dépendances déjà installées${NC}"
fi

# Vérifier le fichier CSV
CSV_FILE="tirage_${GAME_TYPE}_complet.csv"
if [ ! -f "$CSV_FILE" ]; then
    echo -e "${YELLOW}⚠️  Fichier CSV non trouvé: $CSV_FILE${NC}"
    echo -e "${YELLOW}   Le serveur peut démarrer mais les prédictions peuvent échouer${NC}"
else
    echo -e "${GREEN}✅ Fichier CSV trouvé: $CSV_FILE${NC}"
fi

# Vérifier le fichier de cycles
CYCLE_FILE="tirage_${GAME_TYPE}_complet_cycles.csv"
if [ ! -f "$CYCLE_FILE" ]; then
    echo -e "${YELLOW}⚠️  Fichier de cycles non trouvé: $CYCLE_FILE${NC}"
    echo -e "${YELLOW}   Génération du fichier de cycles...${NC}"
    python3 script/cycle_data_generator.py --csv "$CSV_FILE" --generate || echo -e "${RED}❌ Erreur lors de la génération du fichier de cycles${NC}"
fi

# Vérifier les modèles
MODEL_DIR="models_${GAME_TYPE}"
if [ ! -d "$MODEL_DIR" ] || [ -z "$(ls -A $MODEL_DIR 2>/dev/null)" ]; then
    echo -e "${YELLOW}⚠️  Aucun modèle trouvé dans $MODEL_DIR/${NC}"
    echo -e "${YELLOW}   Les modèles doivent être entraînés sur le PC local et transférés${NC}"
else
    echo -e "${GREEN}✅ Modèles trouvés dans $MODEL_DIR/${NC}"
fi

# Créer les répertoires nécessaires
mkdir -p "resultats_${GAME_TYPE}"
mkdir -p "$MODEL_DIR"
mkdir -p "reflections_${GAME_TYPE}"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Démarrage du serveur API${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${GREEN}Serveur accessible sur: http://${HOST}:${PORT}${NC}"
echo -e "${GREEN}Type de jeu: ${GAME_TYPE}${NC}"
echo -e "${YELLOW}Appuyez sur Ctrl+C pour arrêter le serveur${NC}"
echo ""

# Exporter les variables d'environnement
export FLASK_APP=api_server.py
export FLASK_ENV=production
export PORT=$PORT
export HOST=$HOST
export GAME_TYPE=$GAME_TYPE

# Démarrer le serveur
python3 api_server.py --host $HOST --port $PORT

