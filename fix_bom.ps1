# Script PowerShell pour supprimer le BOM UTF-8 du fichier batch
$file = "start_api_vps.bat"
$content = Get-Content $file -Raw -Encoding UTF8
$utf8NoBom = New-Object System.Text.UTF8Encoding $false
[System.IO.File]::WriteAllText($file, $content, $utf8NoBom)
Write-Host "BOM supprimé du fichier $file"

