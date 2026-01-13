# Script de test pour vérifier la configuration du reverse proxy IIS

Write-Host "=========================================="
Write-Host "Test de configuration Reverse Proxy IIS"
Write-Host "=========================================="
Write-Host ""

# Vérifier si le fichier web.config existe
$webConfigPath = "C:\inetpub\wwwroot\bottrading_website\web.config"
if (Test-Path $webConfigPath) {
    Write-Host "✅ Fichier web.config trouvé: $webConfigPath" -ForegroundColor Green
} else {
    Write-Host "❌ Fichier web.config NON trouvé: $webConfigPath" -ForegroundColor Red
    Write-Host "   → Copiez le fichier web.config dans ce répertoire" -ForegroundColor Yellow
}

Write-Host ""

# Vérifier si URL Rewrite est installé
$rewriteModule = Get-WindowsFeature -Name IIS-URLRewrite
if ($rewriteModule) {
    if ($rewriteModule.Installed) {
        Write-Host "✅ URL Rewrite Module installé" -ForegroundColor Green
    } else {
        Write-Host "❌ URL Rewrite Module NON installé" -ForegroundColor Red
        Write-Host "   → Téléchargez depuis: https://www.iis.net/downloads/microsoft/url-rewrite" -ForegroundColor Yellow
    }
} else {
    Write-Host "⚠️ Impossible de vérifier URL Rewrite Module" -ForegroundColor Yellow
}

Write-Host ""

# Vérifier si ARR est installé
$arrModule = Get-WindowsFeature | Where-Object { $_.Name -like "*ARR*" -or $_.Name -like "*ApplicationRequestRouting*" }
if ($arrModule) {
    if ($arrModule.Installed) {
        Write-Host "✅ Application Request Routing installé" -ForegroundColor Green
    } else {
        Write-Host "❌ Application Request Routing NON installé" -ForegroundColor Red
        Write-Host "   → Téléchargez depuis: https://www.iis.net/downloads/microsoft/application-request-routing" -ForegroundColor Yellow
    }
} else {
    Write-Host "⚠️ Impossible de vérifier ARR (peut nécessiter une vérification manuelle)" -ForegroundColor Yellow
}

Write-Host ""

# Tester la connexion à l'API directe
Write-Host "Test de connexion à l'API directe (HTTP)..." -ForegroundColor Cyan
try {
    $response = Invoke-WebRequest -Uri "http://107.189.17.46:5002/api/test" -Method GET -TimeoutSec 5 -UseBasicParsing
    Write-Host "✅ API directe accessible: $($response.StatusCode)" -ForegroundColor Green
    Write-Host "   Réponse: $($response.Content)" -ForegroundColor Gray
} catch {
    Write-Host "❌ API directe NON accessible: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "   → Vérifiez que le serveur Flask est démarré" -ForegroundColor Yellow
}

Write-Host ""

# Tester la connexion via HTTPS (reverse proxy)
Write-Host "Test de connexion via HTTPS (reverse proxy)..." -ForegroundColor Cyan
try {
    $response = Invoke-WebRequest -Uri "https://kenopredictionia.fr/api/test" -Method GET -TimeoutSec 5 -UseBasicParsing
    Write-Host "✅ Reverse proxy HTTPS fonctionne: $($response.StatusCode)" -ForegroundColor Green
    Write-Host "   Réponse: $($response.Content)" -ForegroundColor Gray
} catch {
    Write-Host "❌ Reverse proxy HTTPS NON accessible: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "   → Vérifiez la configuration du reverse proxy dans IIS" -ForegroundColor Yellow
    Write-Host "   → Vérifiez que web.config est au bon endroit" -ForegroundColor Yellow
    Write-Host "   → Vérifiez que ARR est activé (Server Proxy Settings)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=========================================="
Write-Host "Instructions:"
Write-Host "1. Si web.config manquant: Copiez web.config dans C:\inetpub\wwwroot\bottrading_website\"
Write-Host "2. Si modules manquants: Installez URL Rewrite et ARR"
Write-Host "3. Activer ARR: IIS Manager → Serveur → ARR → Server Proxy Settings → Enable proxy"
Write-Host "4. Redémarrer IIS: iisreset"
Write-Host "=========================================="

