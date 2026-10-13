#!/usr/bin/env bash
set -e

COMFY_ROOT="${COMFY_ROOT:-/content/ComfyUI}"
CUSTOM_NODES="${COMFY_ROOT}/custom_nodes"
MODELS_DIR="${COMFY_ROOT}/models"

echo "=== [1/5] Установка зависимостей и CLI ==="
apt-get update -qq && apt-get install -y -qq aria2
pip install -q --upgrade 'transferit-py[cli]' huggingface_hub

echo "=== [2/5] Подготовка директорий и привязка к OnMyDisk ==="
mkdir -p "${COMFY_ROOT}/output/Krea2"
mkdir -p "/home/omduser/Data"
rm -rf "/home/omduser/Data/Krea2"
ln -sfn "${COMFY_ROOT}/output/Krea2" "/home/omduser/Data/Krea2"
chown -R omduser:omduser "${COMFY_ROOT}/output" "/home/omduser/Data" 2>/dev/null || true
chmod -R 777 "${COMFY_ROOT}/output" 2>/dev/null || true

echo "=== [3/5] Создание директорий для моделей ==="
mkdir -p "${MODELS_DIR}/diffusion_models"
mkdir -p "${MODELS_DIR}/text_encoders"
mkdir -p "${MODELS_DIR}/vae"
mkdir -p "${MODELS_DIR}/loras"

echo "=== [4/5] Загрузка VAE и текстового энкодера Qwen3-VL ==="
# VAE (qwen_image_vae.safetensors ~240MB)
if [ ! -f "${MODELS_DIR}/vae/qwen_image_vae.safetensors" ]; then
    echo "Скачивание qwen_image_vae.safetensors..."
    aria2c -x 16 -s 16 -k 1M -c -U "Mozilla/5.0" \
        "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/vae/qwen_image_vae.safetensors" \
        -d "${MODELS_DIR}/vae" -o "qwen_image_vae.safetensors"
else
    echo "VAE уже существует, пропускаем."
fi

# Text Encoder (qwen3vl_4b_bf16.safetensors ~8.3GB)
if [ ! -f "${MODELS_DIR}/text_encoders/qwen3vl_4b_bf16.safetensors" ]; then
    echo "Скачивание qwen3vl_4b_bf16.safetensors (в 16 потоков)..."
    aria2c -x 16 -s 16 -k 1M -c -U "Mozilla/5.0" \
        "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/text_encoders/qwen3vl_4b_bf16.safetensors" \
        -d "${MODELS_DIR}/text_encoders" -o "qwen3vl_4b_bf16.safetensors"
else
    echo "Text encoder уже существует, пропускаем."
fi

echo "=== [5/5] Загрузка модели CyberRealistic Krea2 с transfer.it ==="
URL="${1:-}"
PASS="${2:-}"

if [ -z "$URL" ]; then
    read -rp "Введите ссылку на transfer.it: " URL
fi

if [ -z "$PASS" ]; then
    read -rp "Введите пароль: " PASS
fi

echo "Скачивание модели в ${MODELS_DIR}/diffusion_models/ ..."
cd "${MODELS_DIR}/diffusion_models"
transferit download "$URL" --password "$PASS"

echo "=== Готово! Модели и ноды установлены. ==="
