@echo off
chcp 65001 >nul
echo ========================================
echo   Entrainement Local EuroMillions
echo   Avec Encodeur Avance et Reflexion IA
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
REM Vérifier que Python est installé
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERREUR] Python n'est pas installe ou n'est pas dans le PATH
    pause
    exit /b 1
)

echo.
echo [INFO] Configuration de l'entrainement:
echo   - Encodeur avance: ACTIVE
echo   - Reflexion IA: ACTIVEE
echo   - Toutes les methodes: ACTIVEES
echo   - Amelioration continue: ACTIVEE
echo.

echo [INFO] Verification des nouveaux tirages avant l'entrainement...
echo.

REM Vérifier et mettre à jour les tirages, puis entraîner avec toutes les méthodes
python check_and_train.py --csv tirage_euromillions_complet.csv --output resultats_euromillions --model-dir models_euromillions --method all --force

if errorlevel 1 (
    echo.
    echo [ERREUR] Le processus a echoue
    echo.
    echo [INFO] Verifiez:
    echo   - Que le fichier CSV existe
    echo   - Que toutes les dependances sont installees
    echo   - Les logs ci-dessus pour plus de details
    pause
    exit /b 1
)

echo.
echo ========================================
echo [OK] Entrainement termine avec succes!
echo ========================================
echo.
echo [INFO] Modeles sauvegardes dans: models_euromillions\
echo [INFO] Resultats sauvegardes dans: resultats_euromillions\
echo.
echo [INFO] Pour transferer les modeles sur le VPS:
echo   scp -r models_euromillions\ user@107.189.17.46:/path/to/euromillions/
echo.
pause
