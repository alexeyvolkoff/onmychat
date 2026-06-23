#!/bin/bash
# ==============================================================================
# OnMyChat & AI Node Installer Script
# ==============================================================================
# Installs OnMyChat and the chosen inference engine (Ollama or RKLLAMA) to /opt
# and sets up systemd services.
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
    log_error "Please run this script as root (e.g., sudo ./install.sh)"
    exit 1
fi

# Detect system user to run services under
RUN_USER="${SUDO_USER:-$USER}"
if [ "$RUN_USER" = "root" ]; then
    if id "firefly" &>/dev/null; then
        RUN_USER="firefly"
    elif id "orange" &>/dev/null; then
        RUN_USER="orange"
    elif id "ubuntu" &>/dev/null; then
        RUN_USER="ubuntu"
    fi
fi

RUN_USER_HOME=$(eval echo "~$RUN_USER")
log_info "Configuring installation for user: $RUN_USER (Home: $RUN_USER_HOME)"

# Parse or ask for Inference Engine
ENGINE=""
if [ $# -ge 1 ]; then
    ENGINE=$(echo "$1" | tr '[:upper:]' '[:lower:]')
fi

while [ "$ENGINE" != "ollama" ] && [ "$ENGINE" != "rkllama" ]; do
    echo "Please choose the AI inference engine to install:"
    echo "  1) ollama  - Default for x86_64, CUDA, or non-Rockchip systems"
    echo "  2) rkllama - Hardware-accelerated NPU engine for Rockchip boards (RK3588, RK3576, etc.)"
    read -p "Enter choice [1 or 2]: " choice
    case "$choice" in
        1|[Oo]llama) ENGINE="ollama" ;;
        2|[Rr]kllama) ENGINE="rkllama" ;;
        *) echo "Invalid choice. Please select 1 or 2." ;;
    esac
done

log_info "Selected inference engine: $ENGINE"

# --- Step 1: Install OnMyChat to /opt/onmychat ---
log_info "Setting up OnMyChat in /opt/onmychat..."
mkdir -p /opt/onmychat

# Copy files from current directory
cp -rp ./* /opt/onmychat/
# Ensure hidden files like .gitignore and .git are copied if needed
[ -f .gitignore ] && cp .gitignore /opt/onmychat/
[ -d .git ] && cp -rp .git /opt/onmychat/

# Create necessary directories
mkdir -p /opt/onmychat/user_data
mkdir -p /opt/onmychat/memory_index
mkdir -p /opt/onmychat/history

# Create config.ini if it does not exist
if [ ! -f /opt/onmychat/config.ini ]; then
    log_info "Creating config.ini from config.example.ini..."
    cp /opt/onmychat/config.example.ini /opt/onmychat/config.ini
    
    # Adjust paths in config.ini
    sed -i "s|STORAGE_ROOT = .*|STORAGE_ROOT = $RUN_USER_HOME|g" /opt/onmychat/config.ini
    sed -i "s|APP_ROOT_DIR = .*|APP_ROOT_DIR = /opt/onmychat|g" /opt/onmychat/config.ini
    
    if [ "$ENGINE" = "rkllama" ]; then
        sed -i "s|OLLAMA_URL = .*|OLLAMA_URL = http://localhost:8080|g" /opt/onmychat/config.ini
    fi
fi

# Ensure correct ownership
chown -R "$RUN_USER:$RUN_USER" /opt/onmychat

# --- Step 2: Setup Python Virtual Environment ---
log_info "Setting up Python virtual environment in /opt/onmychat/venv..."
sudo -u "$RUN_USER" python3 -m venv /opt/onmychat/venv
sudo -u "$RUN_USER" /opt/onmychat/venv/bin/pip install --upgrade pip

# Detect architecture and install CPU torch on Arm to avoid CUDA overhead
ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
    log_info "ARM64 architecture detected. Installing CPU-only PyTorch..."
    sudo -u "$RUN_USER" /opt/onmychat/venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

log_info "Installing dependencies from requirements.txt..."
sudo -u "$RUN_USER" /opt/onmychat/venv/bin/pip install -r /opt/onmychat/requirements.txt
sudo -u "$RUN_USER" /opt/onmychat/venv/bin/pip install 'numpy<2.0'

# --- Step 3: Install Inference Engine ---
if [ "$ENGINE" = "ollama" ]; then
    log_info "Installing Ollama..."
    if ! command -v ollama &>/dev/null; then
        curl -fsSL https://ollama.com/install.sh | sh
    else
        log_success "Ollama is already installed."
    fi
elif [ "$ENGINE" = "rkllama" ]; then
    log_info "Installing RKLLAMA in /opt/rkllama..."
    mkdir -p /opt/rkllama
    if [ ! -d "/opt/rkllama/.git" ]; then
        log_info "Cloning RKLLAMA repository..."
        sudo -u "$RUN_USER" git clone https://github.com/NotPunchnox/rkllama.git /opt/rkllama
    else
        log_info "RKLLAMA repository already cloned, pulling latest..."
        cd /opt/rkllama && sudo -u "$RUN_USER" git pull && cd -
    fi
    
    mkdir -p /opt/rkllama/models
    chown -R "$RUN_USER:$RUN_USER" /opt/rkllama
    
    # Try to install in Miniconda if available, otherwise fallback to system pip
    if [ -d "$RUN_USER_HOME/miniconda3" ]; then
        log_info "Installing rkllama into Miniconda environment..."
        sudo -u "$RUN_USER" "$RUN_USER_HOME/miniconda3/bin/pip" install -e /opt/rkllama/
    else
        log_warning "Miniconda not found at $RUN_USER_HOME/miniconda3, installing globally..."
        pip3 install -e /opt/rkllama/
    fi
fi

# --- Step 4: Configure systemd Services ---
log_info "Configuring systemd services..."

# Remove old services if present
if [ -f /etc/systemd/system/onmychat.service ]; then
    log_warning "Removing old onmychat service configuration..."
    systemctl stop onmychat || true
    systemctl disable onmychat || true
fi

# Write new onmychat service
cat <<EOF > /etc/systemd/system/onmychat.service
[Unit]
Description=OnMyChat AI Service
After=network.target rkllama.service ollama.service

[Service]
Type=simple
User=$RUN_USER
WorkingDirectory=/opt/onmychat
Environment="PATH=/opt/onmychat/venv/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=/opt/onmychat/venv/bin/uvicorn api:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

if [ "$ENGINE" = "rkllama" ]; then
    if [ -f /etc/systemd/system/rkllama.service ]; then
        systemctl stop rkllama || true
        systemctl disable rkllama || true
    fi
    
    # Determine rkllama_server path
    RKLLAMA_SERVER_CMD="rkllama_server"
    if [ -f "$RUN_USER_HOME/miniconda3/bin/rkllama_server" ]; then
        RKLLAMA_SERVER_CMD="$RUN_USER_HOME/miniconda3/bin/rkllama_server"
    fi
    
    # Write rkllama service
    cat <<EOF > /etc/systemd/system/rkllama.service
[Unit]
Description=RKLLAMA Server
After=network.target

[Service]
Type=simple
WorkingDirectory=/opt/rkllama
Environment=HOME=$RUN_USER_HOME
User=$RUN_USER
ExecStart=$RKLLAMA_SERVER_CMD --processor rk3588 --port 8080 --models /opt/rkllama/models
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable rkllama
    systemctl start rkllama
    log_success "RKLLAMA service started and enabled."
elif [ "$ENGINE" = "ollama" ]; then
    systemctl daemon-reload
    systemctl enable ollama || true
    systemctl restart ollama || true
    log_success "Ollama service started and enabled."
fi

systemctl daemon-reload
systemctl enable onmychat
systemctl start onmychat
log_success "OnMyChat service started and enabled."

log_success "=============================================================================="
log_success "Installation completed successfully!"
log_success "OnMyChat is running at: http://localhost:8000"
if [ "$ENGINE" = "rkllama" ]; then
    log_success "Please put your .rkllm models under /opt/rkllama/models/<model-name>/"
fi
log_success "=============================================================================="
