#!/usr/bin/env bash
# ==============================================================================
# OnMyDisk + Krea 2 DiT Recovery & Provisioning Script
# Восстанавливает окружение Krea Studio после перезапуска Colab / разрыва сессии
# ==============================================================================
set -euo pipefail

echo "=========================================================="
echo "🚀 [1/6] Проверка и подготовка OnMyDisk Data & ComfyUI..."
echo "=========================================================="

COMFY_ROOT="${COMFY_ROOT:-/content/ComfyUI}"
NODE_USER="${NODE_USER:-omduser}"
GEN_OUTPUT="${COMFY_ROOT}/output/generated"
GD_MODELS="/content/drive/MyDrive/ComfyUI-models"

# Папки для генераций и персон
mkdir -p "${GEN_OUTPUT}"
mkdir -p "/home/${NODE_USER}/Data/personas"
rm -rf "/home/${NODE_USER}/Data/generated"
ln -sfn "${GEN_OUTPUT}" "/home/${NODE_USER}/Data/generated"

# Обратная совместимость для Krea2
mkdir -p "${COMFY_ROOT}/output/Krea2"
rm -rf "/home/${NODE_USER}/Data/Krea2"
ln -sfn "${COMFY_ROOT}/output/Krea2" "/home/${NODE_USER}/Data/Krea2"

# Права доступа для WebRTC Drive и OnMyDisk
chown -R "${NODE_USER}:${NODE_USER}" "${COMFY_ROOT}/output" "/home/${NODE_USER}/Data" 2>/dev/null || true
chmod -R 777 "${COMFY_ROOT}/output" 2>/dev/null || true

echo "✅ Папки и симлинки данных настроены."

echo "=========================================================="
echo "📂 [2/6] Подключение и симлинки моделей Krea 2..."
echo "=========================================================="

mkdir -p "${COMFY_ROOT}/models/diffusion_models"
mkdir -p "${COMFY_ROOT}/models/text_encoders"
mkdir -p "${COMFY_ROOT}/models/clip"
mkdir -p "${COMFY_ROOT}/models/vae"
mkdir -p "${COMFY_ROOT}/models/loras"

# 1. Симлинкуем всё, что сохранено на Google Drive
if [ -d "${GD_MODELS}" ]; then
    echo "Симлинкуем модели из ${GD_MODELS}..."
    [ -d "${GD_MODELS}/diffusion_models" ] && ln -sf "${GD_MODELS}/diffusion_models"/* "${COMFY_ROOT}/models/diffusion_models/" 2>/dev/null || true
    [ -d "${GD_MODELS}/text_encoders" ] && ln -sf "${GD_MODELS}/text_encoders"/* "${COMFY_ROOT}/models/text_encoders/" 2>/dev/null || true
    [ -d "${GD_MODELS}/text_encoders" ] && ln -sf "${GD_MODELS}/text_encoders"/* "${COMFY_ROOT}/models/clip/" 2>/dev/null || true
    [ -d "${GD_MODELS}/vae" ] && ln -sf "${GD_MODELS}/vae"/* "${COMFY_ROOT}/models/vae/" 2>/dev/null || true
    [ -d "${GD_MODELS}/loras" ] && ln -sf "${GD_MODELS}/loras"/* "${COMFY_ROOT}/models/loras/" 2>/dev/null || true
fi

# 2. Проверка VAE (qwen_image_vae.safetensors ~240MB)
VAE_FILE="${COMFY_ROOT}/models/vae/qwen_image_vae.safetensors"
if [ ! -f "${VAE_FILE}" ] || [ ! -s "${VAE_FILE}" ]; then
    echo "Загрузка VAE qwen_image_vae.safetensors..."
    which aria2c >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq aria2)
    aria2c -x 16 -s 16 -k 1M -c -U "Mozilla/5.0" \
        "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/vae/qwen_image_vae.safetensors" \
        -d "${COMFY_ROOT}/models/vae" -o "qwen_image_vae.safetensors"
fi

# 3. Проверка Text Encoder (qwen3-vl-4b-heretic или qwen3vl_4b_bf16)
TE_FILE="${COMFY_ROOT}/models/text_encoders/qwen3-vl-4b-heretic.safetensors"
TE_ALT="${COMFY_ROOT}/models/text_encoders/qwen3vl_4b_bf16.safetensors"
if [ ! -f "${TE_FILE}" ] && [ ! -f "${TE_ALT}" ]; then
    echo "Загрузка qwen3-vl-4b-heretic.safetensors..."
    which aria2c >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq aria2)
    aria2c -x 16 -s 16 -k 1M -c -U "Mozilla/5.0" \
        "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/text_encoders/qwen3vl_4b_bf16.safetensors" \
        -d "${COMFY_ROOT}/models/text_encoders" -o "qwen3vl_4b_bf16.safetensors"
fi
# Симлинкуем в clip/ для совместимости
ln -sf "${COMFY_ROOT}/models/text_encoders"/*.safetensors "${COMFY_ROOT}/models/clip/" 2>/dev/null || true

# 4. Проверка DiT модели CyberRealistic Krea 2
DIT_FILE=$(find "${COMFY_ROOT}/models/diffusion_models" -name "*Krea2*.safetensors" 2>/dev/null | head -n 1 || true)
if [ -z "${DIT_FILE}" ]; then
    echo "⚠️ Модель Krea 2 DiT не найдена в diffusion_models!"
    echo "Если файл на Google Drive, проверьте ${GD_MODELS}/diffusion_models."
fi

echo "=========================================================="
echo "⚙️ [3/6] Проверка и обновление репозитория OnMyChat..."
echo "=========================================================="

git config --global --add safe.directory /content/onmychat 2>/dev/null || true
git -C /content/onmychat pull || echo "Git pull warning"

CONFIG_PATH="/content/onmychat/config.ini"
if [ -f "${CONFIG_PATH}" ]; then
    if ! grep -q "KREA_MODEL" "${CONFIG_PATH}"; then
        echo "Добавляем параметры Krea 2 в config.ini..."
        cat << 'EOF' >> "${CONFIG_PATH}"

# Krea 2 Settings
KREA_MODEL = CyberRealistic_Krea2_V4.0_BF16.safetensors
KREA_CLIP = qwen3-vl-4b-heretic.safetensors
KREA_VAE = qwen_image_vae.safetensors
KREA_LORA = krea2_identity_edit_v1_2.safetensors
EOF
    fi
fi

echo "=========================================================="
echo "🔄 [4/6] Проверка работы ComfyUI..."
echo "=========================================================="

if ! curl -s "http://127.0.0.1:8188/system_stats" >/dev/null 2>&1; then
    echo "ComfyUI не отвечает, запускаем..."
    pkill -f "main.py.*8188" 2>/dev/null || true
    nohup python3 "${COMFY_ROOT}/main.py" --listen 127.0.0.1 --port 8188 --preview-method auto > /tmp/comfyui.log 2>&1 &
    sleep 5
fi

echo "=========================================================="
echo "🚀 [5/6] Перезапуск OnMyChat API (порт 4000)..."
echo "=========================================================="

pkill -9 -f "api.py.*4000" 2>/dev/null || true
sleep 1

su - "${NODE_USER}" -c "cd /content/onmychat && python3 -u api.py 4000" > /tmp/onmychat.log 2>&1 &
echo "OnMyChat запущен от пользователя ${NODE_USER}."

echo "=========================================================="
echo "🩺 [6/6] Проверка статуса (Healthcheck)..."
echo "=========================================================="

sleep 4
python3 - << 'PYEOF'
import urllib.request, json, sys, time

for attempt in range(8):
    try:
        req = urllib.request.urlopen("http://127.0.0.1:4000/assistant", timeout=3)
        data = json.loads(req.read().decode())
        caps = data.get("capabilities", {})
        print("✅ OnMyChat API успешно запущен!")
        print(f" - ComfyUI доступен: {caps.get('image_generation')}")
        print(f" - Krea 2 DiT доступен: {caps.get('krea2')}")
        if caps.get("krea2"):
            print("🎉 Krea Studio полностью готова к работе (~4 сек рендер)!")
        else:
            print("ℹ️ Режим Krea 2 не активен (проверьте наличие DiT модели в models/diffusion_models).")
        sys.exit(0)
    except Exception as e:
        time.sleep(2)

print("⚠️ OnMyChat еще инициализируется, проверьте /tmp/onmychat.log")
PYEOF
