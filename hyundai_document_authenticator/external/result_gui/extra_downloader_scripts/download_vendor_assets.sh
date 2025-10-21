#!/bin/bash
# download_vendor_assets.sh
# Bash script to vendor Bootstrap, Bootstrap Icons, and XLSX for offline use

BASE_PATH="d:/frm_git/hyundai_document_authenticator/hyundai_document_authenticator/external/result_gui/static/vendor"

# If running on Linux, convert Windows-style path if mounted
BASE_PATH=$(echo "$BASE_PATH" | sed 's#\\#/#g')

# Create directories
mkdir -p "$BASE_PATH/bootstrap/css"
mkdir -p "$BASE_PATH/bootstrap/js"
mkdir -p "$BASE_PATH/bootstrap-icons/font/fonts"
mkdir -p "$BASE_PATH/xlsx"

# Download files
curl -L "https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" -o "$BASE_PATH/bootstrap/css/bootstrap.min.css"
curl -L "https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js" -o "$BASE_PATH/bootstrap/js/bootstrap.bundle.min.js"
curl -L "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css" -o "$BASE_PATH/bootstrap-icons/font/bootstrap-icons.min.css"
curl -L "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/fonts/bootstrap-icons.woff2" -o "$BASE_PATH/bootstrap-icons/font/fonts/bootstrap-icons.woff2"
curl -L "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/fonts/bootstrap-icons.woff" -o "$BASE_PATH/bootstrap-icons/font/fonts/bootstrap-icons.woff"
curl -L "https://cdn.jsdelivr.net/npm/xlsx@0.18.5/dist/xlsx.full.min.js" -o "$BASE_PATH/xlsx/xlsx.full.min.js"

echo "✅ Vendor assets downloaded successfully to $BASE_PATH"
