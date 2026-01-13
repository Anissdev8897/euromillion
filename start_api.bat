@echo off
REM Script de demarrage du serveur API - Version sans BOM
REM Usage: start_api.bat

chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion

REM Configuration par defaut
if "%PORT%"=="" set PORT=5002
if "%HOST%"=="" set HOST=107.189.17.46
if "%GAME_TYPE%"=="" set GAME_TYPE=euromillions

echo ========================================
echo   Serveur API %GAME_TYPE% - Windows Server VPS
echo ========================================
echo.

REM Verifier que Python est installe
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERREUR] Python n est pas installe ou n est pas dans le PATH
    echo Veuillez installer Python depuis https://www.python.org/
    pause
    exit /b 1
)

echo [OK] Python detecte
python --version
echo [OK] Type de jeu: %GAME_TYPE%
echo.

REM Activer l environnement virtuel si disponible
set VENV_ACTIVATED=0
if exist "venv\Scripts\activate.bat" goto activate_venv
if exist "env_local\Scripts\activate.bat" goto activate_env_local
goto create_venv

:activate_venv
echo [INFO] Activation de l environnement virtuel...
call venv\Scripts\activate.bat
echo [OK] Environnement virtuel active
set VENV_ACTIVATED=1
goto venv_done

:activate_env_local
echo [INFO] Activation de l environnement virtuel (env_local)...
call env_local\Scripts\activate.bat
echo [OK] Environnement virtuel active
set VENV_ACTIVATED=1
goto venv_done

:create_venv
echo [INFO] Aucun environnement virtuel trouve. Creation...
python -m venv venv
call venv\Scripts\activate.bat
echo [OK] Environnement virtuel cree
set VENV_ACTIVATED=1

:venv_done

REM Verifier et installer les dependances
echo.
echo [INFO] Verification complete des dependances...
echo.

set NEED_INSTALL=0

REM Verifier Flask
python -c "import flask" 2>nul
if errorlevel 1 (
    echo [MANQUANT] Flask
    set NEED_INSTALL=1
) else (
    echo [OK] Flask
)

REM Verifier pandas
python -c "import pandas" 2>nul
if errorlevel 1 (
    echo [MANQUANT] Pandas
    set NEED_INSTALL=1
) else (
    echo [OK] Pandas
)

REM Verifier numpy
python -c "import numpy" 2>nul
if errorlevel 1 (
    echo [MANQUANT] NumPy
    set NEED_INSTALL=1
) else (
    echo [OK] NumPy
)

REM Verifier scikit-learn
python -c "import sklearn" 2>nul
if errorlevel 1 (
    echo [MANQUANT] scikit-learn
    set NEED_INSTALL=1
) else (
    echo [OK] scikit-learn
)

REM Verifier flask-cors
python -c "import flask_cors" 2>nul
if errorlevel 1 (
    echo [MANQUANT] flask-cors
    set NEED_INSTALL=1
) else (
    echo [OK] flask-cors
)

REM Verifier joblib
python -c "import joblib" 2>nul
if errorlevel 1 (
    echo [MANQUANT] joblib
    set NEED_INSTALL=1
) else (
    echo [OK] joblib
)

REM Verifier requests
python -c "import requests" 2>nul
if errorlevel 1 (
    echo [MANQUANT] requests
    set NEED_INSTALL=1
) else (
    echo [OK] requests
)

REM Verifier beautifulsoup4
python -c "import bs4" 2>nul
if errorlevel 1 (
    echo [MANQUANT] beautifulsoup4
    set NEED_INSTALL=1
) else (
    echo [OK] beautifulsoup4
)

REM Verifier pennylane (optionnel)
python -c "import pennylane" 2>nul
if errorlevel 1 (
    echo [OPTIONNEL] PennyLane non trouve - Systeme quantique limite
) else (
    echo [OK] PennyLane - Systeme quantique disponible
)

REM Verifier torch (optionnel)
python -c "import torch" 2>nul
if errorlevel 1 (
    echo [OPTIONNEL] PyTorch non trouve - QLSTM limite
) else (
    echo [OK] PyTorch - QLSTM disponible
)

echo.
if "%NEED_INSTALL%"=="1" (
    echo [INSTALLATION] Modules manquants - Installation en cours...
    echo.
    python -m pip install --upgrade pip
    if errorlevel 1 (
        echo [ERREUR] Echec de la mise a jour de pip
        pause
        exit /b 1
    )
    
    python -m pip install -r requirements_api.txt
    if errorlevel 1 (
        echo [ERREUR] Echec de l installation des dependances
        echo Verifiez le fichier requirements_api.txt et votre connexion Internet
        pause
        exit /b 1
    )
    
    echo [OK] Dependances installees avec succes
    echo.
    
    REM Verifier a nouveau apres installation
    echo [INFO] Verification post-installation...
    python -c "import flask, pandas, numpy, sklearn, flask_cors, joblib, requests, bs4" 2>nul
    if errorlevel 1 (
        echo [ATTENTION] Certaines dependances peuvent ne pas etre correctement installees
        echo Le serveur peut demarrer mais certaines fonctionnalites peuvent etre limitees
    ) else (
        echo [OK] Toutes les dependances critiques sont installees
    )
) else (
    echo [OK] Toutes les dependances critiques sont deja installees
)

echo.
echo [INFO] Verification du fichier CSV...
set CSV_FILE=tirage_%GAME_TYPE%_complet.csv
if exist "%CSV_FILE%" (
    echo [OK] Fichier CSV trouve: %CSV_FILE%
) else (
    echo [ATTENTION] Fichier CSV non trouve: %CSV_FILE%
    echo Le serveur peut demarrer mais les predictions peuvent echouer
)

echo.
echo [INFO] Verification du fichier de cycles...
set CYCLE_FILE=tirage_%GAME_TYPE%_complet_cycles.csv
if exist "%CYCLE_FILE%" (
    echo [OK] Fichier de cycles trouve: %CYCLE_FILE%
) else (
    echo [ATTENTION] Fichier de cycles non trouve: %CYCLE_FILE%
    echo Tentative de generation...
    if exist "%CSV_FILE%" (
        python script\cycle_data_generator.py --csv "%CSV_FILE%" --generate
        if errorlevel 1 (
            echo [ERREUR] Erreur lors de la generation du fichier de cycles
        ) else (
            echo [OK] Fichier de cycles genere
        )
    )
)

echo.
echo [INFO] Verification des modeles...
set MODEL_DIR=models_%GAME_TYPE%
if exist "%MODEL_DIR%" (
    echo [OK] Repertoire de modeles trouve: %MODEL_DIR%
) else (
    echo [ATTENTION] Repertoire de modeles non trouve: %MODEL_DIR%
    echo Les modeles doivent etre entraines sur le PC local et transferes
    mkdir "%MODEL_DIR%" 2>nul
)

REM Creer les repertoires necessaires
if not exist "resultats_%GAME_TYPE%" mkdir "resultats_%GAME_TYPE%"
if not exist "%MODEL_DIR%" mkdir "%MODEL_DIR%"
if not exist "reflections_%GAME_TYPE%" mkdir "reflections_%GAME_TYPE%"

echo.
echo ========================================
echo   Demarrage du serveur API
echo ========================================
echo.
echo Serveur accessible sur: http://%HOST%:%PORT%
echo Type de jeu: %GAME_TYPE%
echo Appuyez sur Ctrl+C pour arreter le serveur
echo.

REM Exporter les variables d environnement
set FLASK_APP=api_server.py
set FLASK_ENV=production
set PORT=%PORT%
set HOST=%HOST%
set GAME_TYPE=%GAME_TYPE%

REM Demarrer le serveur
python api_server.py --host %HOST% --port %PORT%

if errorlevel 1 (
    echo.
    echo [ERREUR] Le serveur a rencontre une erreur
    echo Verifiez les messages d erreur ci-dessus
)

echo.
pause
