#!/usr/bin/env bash
set -e

COMFY_ROOT="${COMFY_ROOT:-/content/ComfyUI}"
MODELS_DIR="${COMFY_ROOT}/models"
OUTPUT_DIR="${COMFY_ROOT}/output/Anima"
OMD_DIR="/home/omduser/Data/Anima"

echo "=== [1/4] Подготовка директорий для Anima ==="
mkdir -p "${MODELS_DIR}/diffusion_models"
mkdir -p "${MODELS_DIR}/text_encoders"
mkdir -p "${MODELS_DIR}/vae"
mkdir -p "${OUTPUT_DIR}"
chmod -R 777 "${OUTPUT_DIR}"

# Симлинк для OnMyDisk
if [ -L "${OMD_DIR}" ]; then
    rm -f "${OMD_DIR}"
elif [ -d "${OMD_DIR}" ]; then
    rm -rf "${OMD_DIR}"
fi
ln -sf "${OUTPUT_DIR}" "${OMD_DIR}"
chown -hR omduser:omduser /home/omduser/Data 2>/dev/null || true

echo "=== [2/4] Загрузка VAE (если нет) ==="
if [ ! -f "${MODELS_DIR}/vae/qwen_image_vae.safetensors" ]; then
    aria2c -x 16 -s 16 -k 1M -c \
        "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/vae/qwen_image_vae.safetensors" \
        -d "${MODELS_DIR}/vae" -o "qwen_image_vae.safetensors"
else
    echo "VAE qwen_image_vae.safetensors уже на месте."
fi

echo "=== [3/4] Загрузка Text Encoder Qwen-3 0.6B (~1.2 GB) ==="
if [ ! -f "${MODELS_DIR}/text_encoders/qwen_3_06b_base.safetensors" ]; then
    aria2c -x 16 -s 16 -k 1M -c \
        "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/text_encoders/qwen_3_06b_base.safetensors" \
        -d "${MODELS_DIR}/text_encoders" -o "qwen_3_06b_base.safetensors"
else
    echo "Text encoder qwen_3_06b_base.safetensors уже на месте."
fi

echo "=== [4/4] Загрузка Diffusion Model Anima Turbo v1.1 (~4.2 GB) ==="
if [ ! -f "${MODELS_DIR}/diffusion_models/anima-turbo-v1.1.safetensors" ]; then
    aria2c -x 16 -s 16 -k 1M -c \
        "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/diffusion_models/anima-turbo-v1.1.safetensors" \
        -d "${MODELS_DIR}/diffusion_models" -o "anima-turbo-v1.1.safetensors"
else
    echo "Diffusion model anima-turbo-v1.1.safetensors уже на месте."
fi

echo "=== Готово! Модели Anima установлены. ==="
