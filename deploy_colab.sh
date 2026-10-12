#!/usr/bin/env bash
set -e

echo "================================================================="
echo "🚀 OnMyDisk AI Node — Автоматический развертыватель (Colab A100)"
echo "================================================================="

NODE_USER="omduser"
NODE_PASSWORD="colab123"
DEVICE_NAME="Colab-AI-Node"
WORK_MODEL="gemma4:31b"
FUN_MODEL="hf.co/mradermacher/Qwen2.5-32B-Instruct-abliterated-GGUF:Q4_K_M"
AI_TOKEN="colab-ai-secret-token"

echo "[1/8] Создание системного пользователя ${NODE_USER} и групп..."
echo "root:${NODE_PASSWORD}" | chpasswd
id -u "${NODE_USER}" >/dev/null 2>&1 || useradd -m -s /bin/bash -G sudo,video "${NODE_USER}"
echo "${NODE_USER}:${NODE_PASSWORD}" | chpasswd
mkdir -p "/home/${NODE_USER}/Data"
chown -R "${NODE_USER}:${NODE_USER}" "/home/${NODE_USER}"

echo "[2/8] Проверка и монтирование Google Drive..."
python3 -c "
from google.colab import drive
import os
if not os.path.exists('/content/drive/MyDrive'):
    drive.mount('/content/drive')
" || true

NODE_DATA_DIR="/content/drive/MyDrive/onmydisk-node"
mkdir -p "${NODE_DATA_DIR}/ollama_models" \
         "${NODE_DATA_DIR}/etc_onmydisk" \
         "${NODE_DATA_DIR}/onmychat_data" \
         "${NODE_DATA_DIR}/unified_index" \
         "${NODE_DATA_DIR}/huggingface_hub"

echo "[3/8] Установка и запуск Ollama..."
apt-get update -qq && apt-get install -y -qq zstd curl
which ollama >/dev/null 2>&1 || curl -fsSL https://ollama.com/install.sh | sh

LOCAL_OLLAMA="/root/.ollama/models"
mkdir -p "${LOCAL_OLLAMA}"
if [ -d "${NODE_DATA_DIR}/ollama_models/manifests" ]; then
    echo "Копируем веса моделей с Google Drive на NVMe SSD..."
    cp -ru "${NODE_DATA_DIR}/ollama_models/"* "${LOCAL_OLLAMA}/" 2>/dev/null || true
fi

pkill -f "ollama serve" 2>/dev/null || true
OLLAMA_MODELS="${LOCAL_OLLAMA}" OLLAMA_HOST="127.0.0.1:11434" setsid ollama serve >/tmp/ollama.log 2>&1 &
sleep 3

echo "Загрузка WORK модели (${WORK_MODEL})..."
ollama pull "${WORK_MODEL}"
echo "Загрузка FUN модели (${FUN_MODEL})..."
ollama pull "${FUN_MODEL}"

echo "Синхронизация моделей с Google Drive..."
cp -ru "${LOCAL_OLLAMA}/"* "${NODE_DATA_DIR}/ollama_models/" 2>/dev/null || true

echo "[4/8] Установка OnMyDisk Connector..."
apt-get install -y -qq samba udisks2 libglu1-mesa libtag1v5 ruby-full poppler-utils \
    pandoc build-essential zlib1g-dev ffmpeg socat \
    libqt5core5a libqt5network5 libqt5websockets5 libqt5widgets5 libqt5sql5 libqt5webchannel5

if [ ! -f /usr/lib/x86_64-linux-gnu/libssl.so.1.1 ]; then
    wget -q -nc http://security.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2.24_amd64.deb -O /tmp/libssl1.1.deb
    dpkg -i /tmp/libssl1.1.deb || apt-get install -f -y -qq
fi
ln -sf /lib/x86_64-linux-gnu/libcrypt.so.1 /lib/x86_64-linux-gnu/libcrypt.so.2 2>/dev/null || true
ln -sf /usr/lib/x86_64-linux-gnu/libcrypt.so.1 /usr/lib/x86_64-linux-gnu/libcrypt.so.2 2>/dev/null || true
ldconfig

if [ ! -f /usr/bin/onmydisk ]; then
    wget -q -O /tmp/onmydisk-connector.deb "https://forge.bineon.team/repo/OnMyDisk/Connector/onmydisk-connector-amd64.deb"
    apt-get install -y -qq /tmp/onmydisk-connector.deb || (dpkg -i /tmp/onmydisk-connector.deb && apt-get install -f -y -qq)
fi

mkdir -p /etc/onmydisk
if [ -f "${NODE_DATA_DIR}/etc_onmydisk/onmydisk.conf" ]; then
    echo "Восстановление конфигурации OnMyDisk с Google Drive..."
    cp -ru "${NODE_DATA_DIR}/etc_onmydisk/"* /etc/onmydisk/ 2>/dev/null || true
fi

for cfg in "/etc/onmydisk/onmydisk.conf" "/etc/onmydisk/onmydisk.conf.user"; do
    touch "${cfg}"
    grep -q "\[FileNode\]" "${cfg}" || echo -e "\n[FileNode]" >> "${cfg}"
    sed -i "s/^User=.*/User=${NODE_USER}/" "${cfg}" 2>/dev/null || true
    grep -q "User=${NODE_USER}" "${cfg}" || sed -i "/\[FileNode\]/a User=${NODE_USER}\nGroup=${NODE_USER}" "${cfg}"
    sed -i "s/^Group=.*/Group=${NODE_USER}/" "${cfg}" 2>/dev/null || true
    grep -q "\[Settings\]" "${cfg}" || echo -e "\n[Settings]" >> "${cfg}"
    sed -i "s/^hostname=.*/hostname=${DEVICE_NAME}/" "${cfg}" 2>/dev/null || true
    grep -q "hostname=${DEVICE_NAME}" "${cfg}" || sed -i "/\[Settings\]/a hostname=${DEVICE_NAME}" "${cfg}"
done
chown -R "${NODE_USER}:${NODE_USER}" /etc/onmydisk "/home/${NODE_USER}" /var/onmydisk 2>/dev/null || true

echo "[5/8] Настройка OnMyChat..."
if [ ! -d "/content/onmychat" ]; then
    git clone https://github.com/alexeyvolkoff/onmychat.git /content/onmychat
else
    cd /content/onmychat && git pull
fi

pip install -q "opentelemetry-api<=1.42.1" "opentelemetry-sdk<=1.42.1" fastapi uvicorn requests pillow pydantic chromadb numpy sentence-transformers aiortc

mkdir -p /content/onmychat/memory_index/unified_index
if [ -f "${NODE_DATA_DIR}/unified_index/chroma.sqlite3" ]; then
    echo "Восстановление базы знаний ChromaDB с Google Drive..."
    cp -ru "${NODE_DATA_DIR}/unified_index/"* /content/onmychat/memory_index/unified_index/
else
    echo "Скачивание базового архива знаний OMD..."
    wget -q -O /tmp/chroma_omd.tar.gz https://forge.bineon.team/repo/OnMyChat/chroma_omd.tar.gz
    tar -xzf /tmp/chroma_omd.tar.gz -C /content/onmychat/memory_index/unified_index
    rm -f /tmp/chroma_omd.tar.gz
    cp -ru /content/onmychat/memory_index/unified_index/* "${NODE_DATA_DIR}/unified_index/" 2>/dev/null || true
fi

HF_USER_HUB="/home/${NODE_USER}/.cache/huggingface/hub"
mkdir -p "${HF_USER_HUB}"
if [ -d "${NODE_DATA_DIR}/huggingface_hub/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2" ]; then
    echo "Восстановление модели эмбеддингов с Google Drive..."
    cp -ru "${NODE_DATA_DIR}/huggingface_hub/"* "${HF_USER_HUB}/"
else
    echo "Скачивание модели эмбеддингов..."
    wget -q -O /tmp/sentence_model.tar.gz https://forge.bineon.team/repo/OnMyChat/sentence_model.tar.gz
    tar -xzf /tmp/sentence_model.tar.gz -C "${HF_USER_HUB}"
    rm -f /tmp/sentence_model.tar.gz
    cp -ru "${HF_USER_HUB}/"* "${NODE_DATA_DIR}/huggingface_hub/" 2>/dev/null || true
fi

cat <<EOF > /content/onmychat/config.ini
[settings]
DEFAULT_SYSTEM_PROMPT_FILE = default.txt
DEFAULT_ASSISTANT_NAME = June
DEFAULT_ASSISTANT_TITLE = Junior assistant

GATEWAY_URL = https://onmydisk.net
AI_TOKEN = ${AI_TOKEN}

OLLAMA_URL = http://127.0.0.1:11434
DEFAULT_MODEL = ${WORK_MODEL}
VISION_MODEL = ${WORK_MODEL}
WORK_MODEL = ${WORK_MODEL}
FUN_MODEL = ${FUN_MODEL}
FUN_TEMPERATURE = 0.75
FUN_TOP_P = 0.85
AI_PROVIDER = local

LLM_NUM_CTX = 32768
LLM_NUM_PREDICT = 2048
LLM_KEEP_ALIVE = 300
LLM_STREAM_TIMEOUT = 180
LLM_REQUEST_TIMEOUT = 120

STORAGE_ROOT = /content
COMFY_API_URL = http://127.0.0.1:8188
COMFY_TIMEOUT = 300
WORKFLOW_PATH = flow.json
COMFY_OUTPUT_DIR = /content/ComfyUI/output
COMFY_INPUT_DIR = /content/ComfyUI/input
AVATAR_DIR = avatars
APP_ROOT_DIR = /content/onmychat

HISTORY_LIMIT = 300
HISTORY_DIR = history
SUMMARY_THRESHOLD = 20

REALISTIC_MODEL = CyberRealisticXLPlay_V10.0_FP16.safetensors
PERFECT_MODEL = cyberrealisticPony_v180Coreshift.safetensors
FANTASY_MODEL = CyberIllustrious_V12_FP16.safetensors
TOONED_MODEL = CyberRealistic_PonySemi_V5.safetensors

SEARCH_THRESHOLD = 0.75
RAG_THRESHOLD = 0.75
RAG_RELATIVE_RATIO = 0.65
RAG_TOP_K = 8
SEARCH_TOP_K = 20
TTS_ENABLED = false
EOF

chown -R "${NODE_USER}:${NODE_USER}" /content/onmychat "/home/${NODE_USER}"

echo "[6/8] Установка и подготовка ComfyUI..."
if [ ! -d "/content/ComfyUI" ]; then
    git clone https://github.com/comfyanonymous/ComfyUI /content/ComfyUI
    cd /content/ComfyUI && pip install -q -r requirements.txt
fi

mkdir -p /content/ComfyUI/custom_nodes
if [ ! -d "/content/ComfyUI/custom_nodes/rgthree-comfy" ]; then
    git clone https://github.com/rgthree/rgthree-comfy /content/ComfyUI/custom_nodes/rgthree-comfy
    pip install -q -r /content/ComfyUI/custom_nodes/rgthree-comfy/requirements.txt 2>/dev/null || true
fi
if [ ! -d "/content/ComfyUI/custom_nodes/comfyui_essentials" ]; then
    git clone https://github.com/cubiq/ComfyUI_essentials /content/ComfyUI/custom_nodes/comfyui_essentials
    pip install -q -r /content/ComfyUI/custom_nodes/comfyui_essentials/requirements.txt 2>/dev/null || true
fi

mkdir -p /content/ComfyUI/models/checkpoints /content/ComfyUI/models/loras /content/ComfyUI/output /content/ComfyUI/input
if [ -d "/content/drive/MyDrive/ComfyUI-models" ]; then
    echo "Копирование чекпоинтов на локальный NVMe SSD (fast_disk)..."
    cp -ru /content/drive/MyDrive/ComfyUI-models/checkpoints/* /content/ComfyUI/models/checkpoints/ 2>/dev/null || true
    ln -sf /content/drive/MyDrive/ComfyUI-models/loras/* /content/ComfyUI/models/loras/ 2>/dev/null || true
    find /content/drive/MyDrive/ComfyUI-models -maxdepth 1 -name "*.safetensors" -exec ln -sf {} /content/ComfyUI/models/checkpoints/ \; 2>/dev/null || true
fi
chown -R "${NODE_USER}:${NODE_USER}" /content/ComfyUI/output /content/ComfyUI/input
chmod 777 /content/ComfyUI/output /content/ComfyUI/input

pkill -f "main.py.*8188" 2>/dev/null || true
cd /content/ComfyUI
setsid python3 main.py --listen 127.0.0.1 --port 8188 --preview-method auto >/tmp/comfyui.log 2>&1 &

echo "[7/8] Запуск OnMyChat (от пользователя ${NODE_USER}) и OnMyDisk..."
pkill -f "api.py" 2>/dev/null || true
killall onmydisk 2>/dev/null || true
killall uvicorn 2>/dev/null || true
rm -f /tmp/onmychat.log /tmp/onmydisk.log
touch /tmp/onmychat.log /tmp/onmydisk.log
chown "${NODE_USER}:${NODE_USER}" /tmp/onmychat.log /tmp/onmydisk.log

# Запуск OnMyChat от omduser
su - "${NODE_USER}" -c "cd /content/onmychat && PYTHONUNBUFFERED=1 HF_HOME=/home/${NODE_USER}/.cache/huggingface nohup python3 -u api.py 4000 >>/tmp/onmychat.log 2>&1 &"

# Запуск OnMyDisk Connector
setsid /usr/bin/onmydisk >/tmp/onmydisk.log 2>&1 &

echo "Ожидание готовности OnMyChat API..."
for i in {1..30}; do
    if curl -s http://127.0.0.1:4000/docs >/dev/null; then
        echo "✅ OnMyChat готов на порту 4000!"
        break
    fi
    sleep 1
done

sleep 3
cp -ru /etc/onmydisk/* "${NODE_DATA_DIR}/etc_onmydisk/" 2>/dev/null || true

DEVICE_ID=$(grep -oP 'Certificate loaded for\s+"\K[^"]+' /tmp/onmydisk.log | tail -n1 || echo "")

echo ""
echo "================================================================="
echo "🎉 ВСЕ СЕРВИСЫ ЗАПУЩЕНЫ И РАБОТАЮТ!"
echo "================================================================="
echo "  • ID устройства:   ${DEVICE_ID}"
echo "  • Имя устройства:  ${DEVICE_NAME}"
echo "  • Пользователь:    ${NODE_USER}"
echo "  • Пароль:          ${NODE_PASSWORD}"
echo "  • WORK Модель:     ${WORK_MODEL}"
echo "  • FUN Модель:      ${FUN_MODEL}"
echo "================================================================="
echo "Подключение в браузере: https://direct.onmydisk.net -> Добавить устройство"
echo "Переход в режим Watchdog..."
echo ""

while true; do
    pgrep -f "ollama serve" >/dev/null || { echo "Restarting Ollama..."; OLLAMA_MODELS="${LOCAL_OLLAMA}" setsid ollama serve >>/tmp/ollama.log 2>&1 & }
    pgrep -f "main.py.*8188" >/dev/null || { echo "Restarting ComfyUI..."; cd /content/ComfyUI && setsid python3 main.py --listen 127.0.0.1 --port 8188 --preview-method auto >>/tmp/comfyui.log 2>&1 & }
    pgrep -f "api.py" >/dev/null || { echo "Restarting OnMyChat as ${NODE_USER}..."; su - "${NODE_USER}" -c "cd /content/onmychat && PYTHONUNBUFFERED=1 HF_HOME=/home/${NODE_USER}/.cache/huggingface nohup python3 -u api.py 4000 >>/tmp/onmychat.log 2>&1 &"; }
    pgrep -x "onmydisk" >/dev/null || { echo "Restarting OnMyDisk..."; setsid /usr/bin/onmydisk >>/tmp/onmydisk.log 2>&1 & }
    cp -ru /etc/onmydisk/* "${NODE_DATA_DIR}/etc_onmydisk/" 2>/dev/null || true
    sleep 30
done
