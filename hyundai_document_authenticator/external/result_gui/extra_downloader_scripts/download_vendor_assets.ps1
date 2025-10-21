# download_vendor_assets.ps1
# PowerShell script to vendor Bootstrap, Bootstrap Icons, and XLSX for offline use

$basePath = "d:\frm_git\hyundai_document_authenticator\hyundai_document_authenticator\external\result_gui\static\vendor"

# Create directories
$dirs = @(
    "$basePath\bootstrap\css",
    "$basePath\bootstrap\js",
    "$basePath\bootstrap-icons\font\fonts",
    "$basePath\xlsx"
)

foreach ($d in $dirs) {
    if (-not (Test-Path $d)) {
        New-Item -ItemType Directory -Force -Path $d | Out-Null
    }
}

# Download files
Invoke-WebRequest -UseBasicParsing -Uri "https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" -OutFile "$basePath\bootstrap\css\bootstrap.min.css"
Invoke-WebRequest -UseBasicParsing -Uri "https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js" -OutFile "$basePath\bootstrap\js\bootstrap.bundle.min.js"
Invoke-WebRequest -UseBasicParsing -Uri "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css" -OutFile "$basePath\bootstrap-icons\font\bootstrap-icons.min.css"
Invoke-WebRequest -UseBasicParsing -Uri "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/fonts/bootstrap-icons.woff2" -OutFile "$basePath\bootstrap-icons\font\fonts\bootstrap-icons.woff2"
Invoke-WebRequest -UseBasicParsing -Uri "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/fonts/bootstrap-icons.woff" -OutFile "$basePath\bootstrap-icons\font\fonts\bootstrap-icons.woff"
Invoke-WebRequest -UseBasicParsing -Uri "https://cdn.jsdelivr.net/npm/xlsx@0.18.5/dist/xlsx.full.min.js" -OutFile "$basePath\xlsx\xlsx.full.min.js"

Write-Host "✅ Vendor assets downloaded successfully to $basePath"
