#!/bin/bash
# ==============================================================================
# OnMyChat Board Update Script
# ==============================================================================
# This script runs directly on the board to update OnMyChat code from Git (GitHub),
# update dependencies, and restart services.
# ==============================================================================

set -euo pipefail

# Colors for logging
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO] $1${NC}"
}

log_success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

log_error() {
    echo -e "${RED}[ERROR] $1${NC}" >&2
}

# Ensure script is run with root privileges
if [ "$EUID" -ne 0 ]; then
    log_error "Please run this script as root (e.g., sudo ./update.sh)"
    exit 1
fi

ONMYCHAT_DIR="/opt/onmychat"

if [ -d "$ONMYCHAT_DIR" ]; then
    log_info "Updating OnMyChat code from Git (GitHub)..."
    cd "$ONMYCHAT_DIR"
    
    # Detect owner of the directory
    RUN_USER=$(stat -c '%U' "$ONMYCHAT_DIR")
    RUN_GROUP=$(stat -c '%G' "$ONMYCHAT_DIR")
    
    # Configure remote to pull from GitHub HTTPS anonymously
    sudo -u "$RUN_USER" git remote set-url origin https://github.com/alexeyvolkoff/onmychat.git || sudo -u "$RUN_USER" git remote add origin https://github.com/alexeyvolkoff/onmychat.git
    sudo -u "$RUN_USER" git pull origin main
    log_success "OnMyChat code updated."

    # --- Step 2: Update Python Dependencies ---
    log_info "Updating Python dependencies in virtualenv..."
    sudo -u "$RUN_USER" ./venv/bin/pip install --upgrade pip
    
    # Explicitly install CPU-only PyTorch to avoid massive CUDA library downloads on ARM
    ARCH=$(uname -m)
    if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
        sudo -u "$RUN_USER" ./venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    fi
    
    # Install other requirements
    sudo -u "$RUN_USER" ./venv/bin/pip install -r requirements.txt
    # Ensure NumPy is pinned below 2.0 to prevent compatibility issues
    sudo -u "$RUN_USER" ./venv/bin/pip install 'numpy<2.0'
    log_success "Python dependencies updated."
    
    # Ensure correct permissions
    chown -R "$RUN_USER:$RUN_GROUP" "$ONMYCHAT_DIR"
else
    log_error "OnMyChat directory not found at $ONMYCHAT_DIR."
    exit 1
fi

# --- Step 3: Update onmydisk-connector ---
log_info "Updating onmydisk-connector..."
CONNECTOR_URL="https://forge.bineon.team/repo/Ubuntu/focal/onmydisk-connector-arm64.deb"
CONNECTOR_DEB="/tmp/onmydisk-connector-arm64.deb"
curl -fsSL "$CONNECTOR_URL" -o "$CONNECTOR_DEB"
dpkg -i "$CONNECTOR_DEB" || true
apt-get install -f -y
rm -f "$CONNECTOR_DEB"
log_success "onmydisk-connector updated."

# --- Step 4: Restart Services ---
log_info "Restarting services..."
if systemctl is-active --quiet rkllama; then
    echo "-> Restarting rkllama service..."
    sudo systemctl restart rkllama
fi

if systemctl is-active --quiet onmychat; then
    echo "-> Restarting onmychat service..."
    sudo systemctl restart onmychat
fi

if systemctl is-active --quiet onmydisk; then
    echo "-> Restarting onmydisk service..."
    sudo systemctl restart onmydisk
fi

log_success "All updates applied and services restarted successfully!"
