# Script PowerShell pour supprimer le BOM UTF-8 des fichiers batch
$files = @("start_api_vps.bat", "start_api_vps.cmd")

foreach ($file in $files) {
    if (Test-Path $file) {
        Write-Host "Traitement de $file..."
        
        # Lire le contenu en tant qu'octets
        $bytes = [System.IO.File]::ReadAllBytes($file)
        
        # Supprimer le BOM UTF-8 (EF BB BF)
        if ($bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF) {
            Write-Host "  BOM UTF-8 détecté et supprimé"
            $bytes = $bytes[3..($bytes.Length - 1)]
        }
        
        # Lire le contenu comme texte UTF-8 sans BOM
        $content = [System.Text.Encoding]::UTF8.GetString($bytes)
        
        # Réécrire le fichier sans BOM
        $utf8NoBom = New-Object System.Text.UTF8Encoding $false
        [System.IO.File]::WriteAllText($file, $content, $utf8NoBom)
        
        Write-Host "  $file corrigé avec succès"
    } else {
        Write-Host "  $file non trouvé"
    }
}

Write-Host "Terminé !"

