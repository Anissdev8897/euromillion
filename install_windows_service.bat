@echo off
REM Script d'installation du service Windows pour le serveur API
REM Nécessite NSSM (Non-Sucking Service Manager) ou Task Scheduler

chcp 65001 >nul
setlocal enabledelayedexpansion

if "%GAME_TYPE%"=="" set GAME_TYPE=euromillions
if "%SERVICE_NAME%"=="" set SERVICE_NAME=euromillions-api

echo ========================================
echo   Installation du Service Windows
echo   Service: %SERVICE_NAME%
echo   Type de jeu: %GAME_TYPE%
echo ========================================
echo.

REM Vérifier les privilèges administrateur
net session >nul 2>&1
if errorlevel 1 (
    echo [ERREUR] Ce script doit être exécuté en tant qu'administrateur
    echo Clic droit sur le fichier -^> Exécuter en tant qu'administrateur
    pause
    exit /b 1
)

echo [OK] Privilèges administrateur confirmés
echo.

REM Obtenir le chemin absolu du script
set SCRIPT_DIR=%~dp0
set SCRIPT_DIR=%SCRIPT_DIR:~0,-1%

echo [INFO] Répertoire du script: %SCRIPT_DIR%
echo.

REM Vérifier si NSSM est disponible
where nssm >nul 2>&1
if errorlevel 1 (
    echo [INFO] NSSM non trouvé. Utilisation de Task Scheduler...
    echo.
    goto :use_task_scheduler
) else (
    echo [OK] NSSM trouvé
    echo.
    goto :use_nssm
)

:use_nssm
echo [INFO] Installation avec NSSM...
echo.

REM Arrêter le service s'il existe déjà
sc query "%SERVICE_NAME%" >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Arrêt du service existant...
    nssm stop "%SERVICE_NAME%"
    nssm remove "%SERVICE_NAME%" confirm
    timeout /t 2 >nul
)

REM Installer le service
echo [INFO] Installation du service...
nssm install "%SERVICE_NAME%" "%SCRIPT_DIR%\venv\Scripts\python.exe" "%SCRIPT_DIR%\api_server.py"
if errorlevel 1 (
    echo [ERREUR] Échec de l'installation du service
    pause
    exit /b 1
)

REM Configurer les paramètres
nssm set "%SERVICE_NAME%" AppDirectory "%SCRIPT_DIR%"
nssm set "%SERVICE_NAME%" AppEnvironmentExtra "FLASK_APP=api_server.py" "FLASK_ENV=production" "GAME_TYPE=%GAME_TYPE%" "PORT=5000" "HOST=0.0.0.0"
nssm set "%SERVICE_NAME%" DisplayName "API %GAME_TYPE^% Prédictions IA"
nssm set "%SERVICE_NAME%" Description "Serveur API pour les prédictions %GAME_type% avec IA et Machine Learning Quantique"
nssm set "%SERVICE_NAME%" Start SERVICE_AUTO_START
nssm set "%SERVICE_NAME%" AppStdout "%SCRIPT_DIR%\logs\service_stdout.log"
nssm set "%SERVICE_NAME%" AppStderr "%SCRIPT_DIR%\logs\service_stderr.log"

REM Créer le répertoire de logs
if not exist "%SCRIPT_DIR%\logs" mkdir "%SCRIPT_DIR%\logs"

echo [OK] Service installé avec succès
echo.
echo [INFO] Démarrage du service...
nssm start "%SERVICE_NAME%"
if errorlevel 1 (
    echo [ERREUR] Échec du démarrage du service
    echo Vérifiez les logs dans %SCRIPT_DIR%\logs\
) else (
    echo [OK] Service démarré avec succès
)

echo.
echo ========================================
echo   Service installé
echo ========================================
echo.
echo Commandes utiles:
echo   - Démarrer: nssm start %SERVICE_NAME%
echo   - Arrêter: nssm stop %SERVICE_NAME%
echo   - Redémarrer: nssm restart %SERVICE_NAME%
echo   - Statut: sc query %SERVICE_NAME%
echo   - Logs: type %SCRIPT_DIR%\logs\service_stdout.log
echo.
goto :end

:use_task_scheduler
echo [INFO] Installation avec Task Scheduler...
echo.

REM Créer une tâche planifiée
set TASK_NAME=%SERVICE_NAME%
set TASK_DESCRIPTION=API %GAME_TYPE% Prédictions IA

REM Supprimer la tâche si elle existe déjà
schtasks /delete /tn "%TASK_NAME%" /f >nul 2>&1

REM Créer la tâche
echo [INFO] Création de la tâche planifiée...
schtasks /create /tn "%TASK_NAME%" /tr "\"%SCRIPT_DIR%\venv\Scripts\python.exe\" \"%SCRIPT_DIR%\api_server.py\"" /sc onstart /ru SYSTEM /rl HIGHEST /f
if errorlevel 1 (
    echo [ERREUR] Échec de la création de la tâche
    pause
    exit /b 1
)

echo [OK] Tâche planifiée créée avec succès
echo.
echo [INFO] Démarrage de la tâche...
schtasks /run /tn "%TASK_NAME%"
if errorlevel 1 (
    echo [ERREUR] Échec du démarrage de la tâche
) else (
    echo [OK] Tâche démarrée avec succès
)

echo.
echo ========================================
echo   Tâche planifiée installée
echo ========================================
echo.
echo Commandes utiles:
echo   - Démarrer: schtasks /run /tn "%TASK_NAME%"
echo   - Arrêter: taskkill /FI "WINDOWTITLE eq %TASK_NAME%*" /F
echo   - Statut: schtasks /query /tn "%TASK_NAME%"
echo   - Supprimer: schtasks /delete /tn "%TASK_NAME%" /f
echo.

:end
pause

