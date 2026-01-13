# Script PowerShell pour installer web.config au bon endroit

$sourceFile = Join-Path $PSScriptRoot "web.config"
$targetDir = "C:\inetpub\wwwroot\bottrading_website"
$targetFile = Join-Path $targetDir "web.config"

Write-Host "=========================================="
Write-Host "Installation de web.config pour IIS"
Write-Host "=========================================="
Write-Host ""

# Verifier que le fichier source existe
if (-not (Test-Path $sourceFile)) {
    Write-Host "ERREUR: Fichier source non trouve: $sourceFile" -ForegroundColor Red
    exit 1
}

Write-Host "Source: $sourceFile"
Write-Host "Cible: $targetFile"
Write-Host ""

# Creer le repertoire cible si necessaire
if (-not (Test-Path $targetDir)) {
    Write-Host "Creation du repertoire: $targetDir" -ForegroundColor Yellow
    try {
        New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
        Write-Host "OK: Repertoire cree" -ForegroundColor Green
    } catch {
        Write-Host "ERREUR lors de la creation du repertoire: $_" -ForegroundColor Red
        exit 1
    }
}

# Copier le fichier
Write-Host "Copie de web.config..." -ForegroundColor Cyan
try {
    Copy-Item -Path $sourceFile -Destination $targetFile -Force
    Write-Host "OK: web.config copie avec succes" -ForegroundColor Green
    Write-Host ""
    Write-Host "Fichier installe dans: $targetFile" -ForegroundColor Green
} catch {
    Write-Host "ERREUR lors de la copie: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "=========================================="
Write-Host "Prochaines etapes:"
Write-Host "=========================================="
Write-Host ""
Write-Host "1. Installer URL Rewrite Module"
Write-Host "2. Installer Application Request Routing"
Write-Host "3. Activer le proxy dans ARR (IIS Manager)"
Write-Host "4. Redemarrer IIS: iisreset"
Write-Host "5. Tester: https://kenopredictionia.fr/api/test"
Write-Host ""
