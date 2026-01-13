@echo off
REM Script de déploiement sur Windows Server VPS pour le serveur API EuroMillions/Loto

chcp 65001 >nul
setlocal enabledelayedexpansion

REM ⚠️ CRITIQUE : Type de jeu (euromillions ou loto) - peut être défini via variable d'environnement
if "%GAME_TYPE%"=="" set GAME_TYPE=euromillions

echo ========================================
echo Déploiement du serveur API %GAME_TYPE%
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

echo [OK] Python trouvé
python --version
echo [OK] Type de jeu: %GAME_TYPE%
echo.

REM Créer un environnement virtuel si nécessaire
if not exist "venv" (
    echo [INFO] Création de l'environnement virtuel...
    python -m venv venv
    if errorlevel 1 (
        echo [ERREUR] Échec de la création de l'environnement virtuel
        pause
        exit /b 1
    )
)

REM Activer l'environnement virtuel
echo [INFO] Activation de l'environnement virtuel...
call venv\Scripts\activate.bat

REM Installer les dépendances
echo [INFO] Installation des dépendances...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo [ERREUR] Échec de la mise à jour de pip
    pause
    exit /b 1
)

python -m pip install -r requirements_api.txt
if errorlevel 1 (
    echo [ERREUR] Échec de l'installation des dépendances
    pause
    exit /b 1
)

echo.
echo [OK] Dépendances installées
echo.

REM Vérifier que le fichier CSV existe
set CSV_FILE=tirage_%GAME_TYPE%_complet.csv
if not exist "%CSV_FILE%" (
    echo [ATTENTION] Le fichier CSV n'existe pas: %CSV_FILE%
    echo Le serveur fonctionnera mais certaines fonctionnalités peuvent être limitées.
    echo.
) else (
    echo [OK] Fichier CSV trouvé: %CSV_FILE%
)

REM Vérifier le fichier de cycles
set CYCLE_FILE=tirage_%GAME_TYPE%_complet_cycles.csv
if not exist "%CYCLE_FILE%" (
    echo [ATTENTION] Fichier de cycles non trouvé: %CYCLE_FILE%
    echo Tentative de génération...
    if exist "%CSV_FILE%" (
        python script\cycle_data_generator.py --csv "%CSV_FILE%" --generate
        if errorlevel 1 (
            echo [ERREUR] Erreur lors de la génération du fichier de cycles
        ) else (
            echo [OK] Fichier de cycles généré
        )
    ) else (
        echo [INFO] Impossible de générer le fichier de cycles (fichier CSV manquant)
    )
    echo.
) else (
    echo [OK] Fichier de cycles trouvé: %CYCLE_FILE%
)

REM Créer les répertoires nécessaires
echo [INFO] Création des répertoires...
if not exist "resultats_%GAME_TYPE%" mkdir "resultats_%GAME_TYPE%"
if not exist "models_%GAME_TYPE%" mkdir "models_%GAME_TYPE%"
if not exist "reflections_%GAME_TYPE%" mkdir "reflections_%GAME_TYPE%"
if not exist "logs" mkdir "logs"

echo [OK] Répertoires créés
echo.

REM Vérifier que les modèles existent
set MODEL_DIR=models_%GAME_TYPE%
if not exist "%MODEL_DIR%" (
    echo [ATTENTION] Répertoire de modèles non trouvé: %MODEL_DIR%
    echo Les modèles doivent être entraînés sur le PC local et transférés sur le VPS.
    echo Voir README_LOTO.md ou GUIDE_ADAPTATION_LOTO.md pour plus d'informations.
    echo.
) else (
    dir /b "%MODEL_DIR%" >nul 2>&1
    if errorlevel 1 (
        echo [ATTENTION] Répertoire de modèles vide: %MODEL_DIR%
        echo Les modèles doivent être entraînés sur le PC local et transférés sur le VPS.
        echo.
    ) else (
        echo [OK] Modèles trouvés dans %MODEL_DIR%\
    )
)

echo ========================================
echo Configuration terminée
echo ========================================
echo.
echo Pour démarrer le serveur:
echo   set GAME_TYPE=%GAME_TYPE%
echo   start_api_vps.bat
echo.
echo Ou installer comme service Windows:
echo   set GAME_TYPE=%GAME_TYPE%
echo   install_windows_service.bat
echo.
echo Pour changer le type de jeu:
echo   set GAME_TYPE=loto
echo   start_api_vps.bat
echo.
pause

