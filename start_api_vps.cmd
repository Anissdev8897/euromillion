@echo off
REM Script de démarrage du serveur API sur Windows Server VPS
REM Usage: start_api_vps.cmd [--port PORT] [--host HOST]

chcp 65001 >nul
setlocal enabledelayedexpansion

REM Configuration par défaut
if "%PORT%"=="" set PORT=5002
if "%HOST%"=="" set HOST=0.0.0.0
if "%GAME_TYPE%"=="" set GAME_TYPE=euromillions

echo ========================================
echo   Serveur API %GAME_TYPE% - Windows Server VPS
echo ========================================
echo.

REM Vérifier que Python est installé
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERREUR] Python n'est pas installé ou n'est pas dans le PATH
    echo Veuillez installer Python depuis https://www.python.org/
    pause
    exit /b 1
)

echo [OK] Python détecté
python --version
echo [OK] Type de jeu: %GAME_TYPE%
echo.

REM Activer l'environnement virtuel si disponible
set VENV_ACTIVATED=0
if exist "venv\Scripts\activate.bat" goto activate_venv
if exist "env_local\Scripts\activate.bat" goto activate_env_local
goto create_venv

:activate_venv
echo [INFO] Activation de l'environnement virtuel...
call venv\Scripts\activate.bat
echo [OK] Environnement virtuel activé
set VENV_ACTIVATED=1
goto venv_done

:activate_env_local
echo [INFO] Activation de l'environnement virtuel (env_local)...
call env_local\Scripts\activate.bat
echo [OK] Environnement virtuel activé
set VENV_ACTIVATED=1
goto venv_done

:create_venv
echo [INFO] Aucun environnement virtuel trouvé. Création...
python -m venv venv
call venv\Scripts\activate.bat
echo [OK] Environnement virtuel créé
set VENV_ACTIVATED=1

:venv_done

REM Vérifier et installer les dépendances
echo.
echo [INFO] Vérification complète des dépendances...
echo.

set NEED_INSTALL=0

REM Vérifier Flask
python -c "import flask" 2>nul
if errorlevel 1 (
    echo [MANQUANT] Flask
    set NEED_INSTALL=1
) else (
    echo [OK] Flask
)

REM Vérifier pandas
python -c "import pandas" 2>nul
if errorlevel 1 (
    echo [MANQUANT] Pandas
    set NEED_INSTALL=1
) else (
    echo [OK] Pandas
)

REM Vérifier numpy
python -c "import numpy" 2>nul
if errorlevel 1 (
    echo [MANQUANT] NumPy
    set NEED_INSTALL=1
) else (
    echo [OK] NumPy
)

REM Vérifier scikit-learn
python -c "import sklearn" 2>nul
if errorlevel 1 (
    echo [MANQUANT] scikit-learn
    set NEED_INSTALL=1
) else (
    echo [OK] scikit-learn
)

REM Vérifier flask-cors
python -c "import flask_cors" 2>nul
if errorlevel 1 (
    echo [MANQUANT] flask-cors
    set NEED_INSTALL=1
) else (
    echo [OK] flask-cors
)

REM Vérifier joblib
python -c "import joblib" 2>nul
if errorlevel 1 (
    echo [MANQUANT] joblib
    set NEED_INSTALL=1
) else (
    echo [OK] joblib
)

REM Vérifier requests
python -c "import requests" 2>nul
if errorlevel 1 (
    echo [MANQUANT] requests
    set NEED_INSTALL=1
) else (
    echo [OK] requests
)

REM Vérifier beautifulsoup4
python -c "import bs4" 2>nul
if errorlevel 1 (
    echo [MANQUANT] beautifulsoup4
    set NEED_INSTALL=1
) else (
    echo [OK] beautifulsoup4
)

REM Vérifier pennylane (optionnel - pour le système quantique)
python -c "import pennylane" 2>nul
if errorlevel 1 (
    echo [OPTIONNEL] PennyLane non trouvé - Système quantique limité
) else (
    echo [OK] PennyLane - Système quantique disponible
)

REM Vérifier torch (optionnel - pour le système quantique)
python -c "import torch" 2>nul
if errorlevel 1 (
    echo [OPTIONNEL] PyTorch non trouvé - QLSTM limité
) else (
    echo [OK] PyTorch - QLSTM disponible
)

echo.
if "%NEED_INSTALL%"=="1" (
    echo [INSTALLATION] Modules manquants - Installation en cours...
    echo.
    python -m pip install --upgrade pip
    if errorlevel 1 (
        echo [ERREUR] Échec de la mise à jour de pip
        pause
        exit /b 1
    )
    
    python -m pip install -r requirements_api.txt
    if errorlevel 1 (
        echo [ERREUR] Échec de l'installation des dépendances
        echo Vérifiez le fichier requirements_api.txt et votre connexion Internet
        pause
        exit /b 1
    )
    
    echo [OK] Dépendances installées avec succès
    echo.
    
    REM Vérifier à nouveau après installation
    echo [INFO] Vérification post-installation...
    python -c "import flask, pandas, numpy, sklearn, flask_cors, joblib, requests, bs4" 2>nul
    if errorlevel 1 (
        echo [ATTENTION] Certaines dépendances peuvent ne pas être correctement installées
        echo Le serveur peut démarrer mais certaines fonctionnalités peuvent être limitées
    ) else (
        echo [OK] Toutes les dépendances critiques sont installées
    )
) else (
    echo [OK] Toutes les dépendances critiques sont déjà installées
)

echo.
echo [INFO] Vérification du fichier CSV...
set CSV_FILE=tirage_%GAME_TYPE%_complet.csv
if exist "%CSV_FILE%" (
    echo [OK] Fichier CSV trouvé: %CSV_FILE%
) else (
    echo [ATTENTION] Fichier CSV non trouvé: %CSV_FILE%
    echo Le serveur peut démarrer mais les prédictions peuvent échouer
)

echo.
echo [INFO] Vérification du fichier de cycles...
set CYCLE_FILE=tirage_%GAME_TYPE%_complet_cycles.csv
if exist "%CYCLE_FILE%" (
    echo [OK] Fichier de cycles trouvé: %CYCLE_FILE%
) else (
    echo [ATTENTION] Fichier de cycles non trouvé: %CYCLE_FILE%
    echo Tentative de génération...
    if exist "%CSV_FILE%" (
        python script\cycle_data_generator.py --csv "%CSV_FILE%" --generate
        if errorlevel 1 (
            echo [ERREUR] Erreur lors de la génération du fichier de cycles
        ) else (
            echo [OK] Fichier de cycles généré
        )
    )
)

echo.
echo [INFO] Vérification des modèles...
set MODEL_DIR=models_%GAME_TYPE%
if exist "%MODEL_DIR%" (
    echo [OK] Répertoire de modèles trouvé: %MODEL_DIR%
) else (
    echo [ATTENTION] Répertoire de modèles non trouvé: %MODEL_DIR%
    echo Les modèles doivent être entraînés sur le PC local et transférés
    mkdir "%MODEL_DIR%" 2>nul
)

REM Créer les répertoires nécessaires
if not exist "resultats_%GAME_TYPE%" mkdir "resultats_%GAME_TYPE%"
if not exist "%MODEL_DIR%" mkdir "%MODEL_DIR%"
if not exist "reflections_%GAME_TYPE%" mkdir "reflections_%GAME_TYPE%"

echo.
echo ========================================
echo   Démarrage du serveur API
echo ========================================
echo.
echo Serveur accessible sur: http://%HOST%:%PORT%
echo Type de jeu: %GAME_TYPE%
echo Appuyez sur Ctrl+C pour arrêter le serveur
echo.

REM Exporter les variables d'environnement
set FLASK_APP=api_server.py
set FLASK_ENV=production
set PORT=%PORT%
set HOST=%HOST%
set GAME_TYPE=%GAME_TYPE%

REM Démarrer le serveur
python api_server.py --host %HOST% --port %PORT%

if errorlevel 1 (
    echo.
    echo [ERREUR] Le serveur a rencontré une erreur
    echo Vérifiez les messages d'erreur ci-dessus
)

echo.
pause

