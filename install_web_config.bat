@echo off
REM Script pour installer web.config au bon endroit pour le reverse proxy IIS

echo ========================================
echo Installation de web.config pour IIS
echo ========================================
echo.

set SOURCE_FILE=%~dp0web.config
set TARGET_DIR=C:\inetpub\wwwroot\bottrading_website
set TARGET_FILE=%TARGET_DIR%\web.config

echo Source: %SOURCE_FILE%
echo Cible: %TARGET_FILE%
echo.

REM Verifier que le fichier source existe
if not exist "%SOURCE_FILE%" (
    echo [ERREUR] Fichier source non trouve: %SOURCE_FILE%
    pause
    exit /b 1
)

REM Creer le repertoire cible si necessaire
if not exist "%TARGET_DIR%" (
    echo [INFO] Creation du repertoire: %TARGET_DIR%
    mkdir "%TARGET_DIR%"
)

REM Copier le fichier
echo [INFO] Copie de web.config...
copy /Y "%SOURCE_FILE%" "%TARGET_FILE%"
if errorlevel 1 (
    echo [ERREUR] Echec de la copie
    pause
    exit /b 1
) else (
    echo [OK] web.config copie avec succes
)

echo.
echo ========================================
echo Prochaines etapes:
echo ========================================
echo.
echo 1. Installer URL Rewrite Module:
echo    https://www.iis.net/downloads/microsoft/url-rewrite
echo.
echo 2. Installer Application Request Routing:
echo    https://www.iis.net/downloads/microsoft/application-request-routing
echo.
echo 3. Activer le proxy dans ARR:
echo    IIS Manager -^> Serveur -^> ARR -^> Server Proxy Settings -^> Enable proxy
echo.
echo 4. Redemarrer IIS:
echo    iisreset
echo.
echo 5. Tester:
echo    https://kenopredictionia.fr/api/test
echo.
pause

