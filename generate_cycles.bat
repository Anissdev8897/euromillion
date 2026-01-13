@echo off
chcp 65001 >nul
echo ========================================
echo   Generation du Fichier de Cycles
echo   Avec Dates Automatiques
echo ========================================
echo.

REM Activer l'environnement virtuel si disponible
if exist "env_local\Scripts\activate.bat" (
    echo [INFO] Activation de l'environnement virtuel...
    call env_local\Scripts\activate.bat
    echo [OK] Environnement virtuel active
    goto :check_python
)
if exist "venv\Scripts\activate.bat" (
    echo [INFO] Activation de l'environnement virtuel (venv)...
    call venv\Scripts\activate.bat
    echo [OK] Environnement virtuel active
    goto :check_python
)
if exist "env\Scripts\activate.bat" (
    echo [INFO] Activation de l'environnement virtuel (env)...
    call env\Scripts\activate.bat
    echo [OK] Environnement virtuel active
    goto :check_python
)

:check_python
REM Verifier que Python est installe
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERREUR] Python n'est pas installe ou n'est pas dans le PATH
    pause
    exit /b 1
)

echo.
echo [INFO] Generation du fichier de cycles...
echo [INFO] Fichier source: tirage_euromillions_complet.csv
echo [INFO] Fichier cible: tirage_euromillions_complet_cycles.csv
echo.

python script\cycle_data_generator.py --csv tirage_euromillions_complet.csv --generate

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo [OK] Generation du fichier de cycles terminee avec succes!
    echo ========================================
    echo.
    echo Le fichier tirage_euromillions_complet_cycles.csv contient:
    echo   - Tous les cycles depuis le premier tirage
    echo   - Dates automatiques si manquantes
    echo   - Cycles de tirages (semaines)
    echo   - Cycles lunaires (phases, illumination)
    echo   - Cycles mensuels et annuels
    echo   - Informations temporelles completes
    echo.
) else (
    echo.
    echo ========================================
    echo [ERREUR] Echec de la generation du fichier de cycles
    echo ========================================
    echo.
)

pause

