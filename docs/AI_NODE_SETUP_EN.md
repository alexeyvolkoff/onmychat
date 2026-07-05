# Personal AI Node Setup Guide

> Run your own AI infrastructure: private, uncensored, unlimited.

This guide walks you through deploying a personal **OnMyChat** AI node on your own hardware and connecting it to your **OnMyDisk** account via the **OnMyDisk Connector**. Once set up, you get:

- **Private RAG** — your files indexed locally, no data leaves your network
- **Semantic Search** — find documents by meaning, not just filenames
- **Unlimited Chat** — no rate limits, no content filters
- **Image Generation** — ComfyUI on your GPU
- **Full Privacy** — inference runs on your hardware

---

## Architecture Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Web Browser    │────>│  OnMyDisk Gateway│────>│  Your Device     │
│  (onmydisk.net) │     │  (cloud proxy)   │     │  (Connector)     │
└─────────────────┘     └──────────────────┘     └──────────────────┘
                                                         │
                                                         v
                                                  ┌──────────────────┐
                                                  │  OnMyChat AI     │
                                                  │  Files (local)   │
                                                  ├──────────────────┤
                                                  │  Ollama (LLM)    │
                                                  │  ChromaDB (RAG)  │
                                                  │  ComfyUI (Image) │
                                                  └──────────────────┘
```

The **Connector** (`onmydisk-connector`) provides access to your files over gateway and announces your local AI node.
The **AI Node** (`onmychat`) handles inference, search, and RAG — all behind your NAT.

---

## Prerequisites

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 8 GB | 32 GB |
| Storage | 20 GB free | 100+ GB (SSD) |
| GPU (optional) | — | NVIDIA GTX 1060 6GB+ |
| OS | Ubuntu 22.04 / 24.04 | Ubuntu 24.04 LTS |
| Network | Internet access | Installed **OnMyDisk** Connector |

---

## Step 1: Install the **OnMyDisk** Connector

The Connector is an **advanced client** of **On My Disk** designed for installation on various storage devices. It runs as a system service, shares your local folders, and hosts the local AI assistant.

### Download the Package

Pre-built `.deb` packages are available for **amd64** and **arm64**:

| Architecture | Download URL |
|-------------|--------------|
| x86_64 (amd64) | `https://forge.bineon.team/repo/Ubuntu/focal/onmydisk-connector-amd64.deb` |
| ARM64 | `https://forge.bineon.team/repo/Ubuntu/focal/onmydisk-connector-arm64.deb` |
| ARMHF (armv7) | `https://forge.bineon.team/repo/Ubuntu/focal/onmydisk-connector-armhf.deb` |

You can also build from source (see [Developer's Guide](DEVELOPERS_GUIDE.md)).

### Install

```bash
sudo apt update
sudo apt install samba udisks2 libqt5core5a libqt5network5 \
    libqt5websockets5 libqt5widgets5 libqt5sql5 \
    libqt5webchannel5 ffmpeg ruby-full build-essential zlib1g-dev

# For x86_64 (amd64)
sudo dpkg -i onmydisk-connector-amd64.deb
# For ARM64 (Raspberry Pi 4/5, Orange Pi, Firefly RK3588, etc. running a 64-bit OS)
# sudo dpkg -i onmydisk-connector-arm64.deb
# For ARMHF (armv7) (Raspberry Pi 3, Orange Pi, etc. running a 32-bit OS)
# sudo dpkg -i onmydisk-connector-armhf.deb


sudo apt --fix-broken install   # if any dependencies are missing
```
### Post-Installation Setup
Configure the client to run under your system user to ensure correct access permissions to your home directory. Edit `/etc/onmydisk/onmydisk.conf` and add your username and group under the `[FileNode]` section:

```ini
[FileNode]
Port=80
User=firefly  # replace with your system username
Group=firefly # replace with your system group name
```

Save the configuration and restart the service:

```bash
sudo systemctl restart onmydisk
```

### Verify Installation

```bash
systemctl status onmydisk
# Should show: active (running)
```

The Connector listens on **port 80** by default. Open `http://<your-device-ip-or-name>` to access the Web UI.

---

## Step 2: Register and Connect to **On My Disk**

Initially, your device will run in anonymous mode, not linked to any **On My Disk** account, accessible only within your local network. To access your device over the Internet, from the mobile app, and share files and the AI assistant with other users, follow these steps:

1. Create an **On My Disk** account or sign in using Google at the gateway interface: [**onmydisk**.net](https://onmydisk.net).
2. Open the Connector Web UI: `http://<ip-or-device-name>` and navigate to profile settings (**Settings → Profile**).
3. Click "Link device to account". In the pop-up window, log in with your account.
4. Verify in the gateway web interface [**onmydisk**.net](https://onmydisk.net) - your device should appear in the **My Devices** section. Now you can use your device remotely and share files and the AI assistant with other users in your group.

---

## Step 3: Install and Configure **OnMyChat** (AI Node)

**OnMyChat** is an additional service that orchestrates the LLM, RAG, and search.

```bash
# Clone the repository
git clone <repository-url> /opt/onmychat
cd /opt/onmychat

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Optional: for web search
# playwright install
```

### Configure

```bash
cp config.example.ini config.ini
```

Edit `config.ini`:

```ini
[settings]
GATEWAY_URL = https://onmydisk.net
AI_TOKEN = <create-your-own-ai-token>

OLLAMA_URL = http://localhost:11434
DEFAULT_MODEL = gemma4:12b

APP_ROOT_DIR = /opt/onmychat
```

After that, open the Connector Web UI (`http://<ip-or-device-name>`), go to **Settings → Integrations**, enable **On My Chat**, specify **Base URL**: `http://localhost:8000` and enter the same **AI Token** as in the config above. Connector will start proxying AI requests from the **OnMyDisk** gateway to your local **OnMyChat**.

### Run

```bash
source venv/bin/activate
python3 api.py
```

The AI node listens on **port 8000** by default.

---

## Step 4: Install and Configure Ollama

Ollama runs the LLM locally on your hardware.

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Pull Models

```bash
# Main chat model
ollama pull gemma4:12b

# Lightweight model for utility tasks
ollama pull gemma4:4b

# (Optional) uncensored model
ollama pull igorls/gemma-4-E4B-it-heretic-GGUF:latest
```

### Verify

```bash
ollama list
curl http://localhost:11434/api/tags
```

> [!IMPORTANT]
> For GPU acceleration, install the NVIDIA Container Toolkit or ROCm. On a CPU-only system, choose smaller models (e.g., `gemma4:4b` or `qwen2.5:7b`).

### Rockchip NPU Boards (RK3588 / RK3576, e.g., Firefly, Orange Pi 5, etc.)

If you are setting up the node on a board powered by a Rockchip SoC, standard Ollama is not supported. You must use **RKLLAMA** to run models on the NPU.

Please follow the detailed setup instructions in the [Rockchip NPU Setup Guide](RKLLAMA_SETUP.md).

---


## RAG and Semantic Search

The built-in ChromaDB provides semantic search and RAG capabilities.

Knowledge collections imported via `/learn <file_path_or_URL> <collection>` are stored in `/opt/onmychat/data/chroma_db/<collection>`.
After importing a document or link, verify:

```bash
ls /opt/onmychat/memory_index/chroma_db/
# Example output: 'omd'
```
This way you can organize scattered documents and external sources into logically connected knowledge collections. By referencing a collection in your query (`/explain <collection> <your_question>`), you will significantly improve the quality and relevance of the assistant's responses.

Additionally, the system periodically auto-indexes files in your storage. The semantic search index is saved in search_index:

```bash
ls /opt/onmychat/memory_index/search_index/
# Should contain the 'omd_search' collection
```

---

## AI Node Operating Modes

An AI node connected to the **OnMyDisk** gateway can operate in two modes, determined by the user's authentication on the gateway:

- **Public Mode** — the user is a guest. The node checks the token balance and applies limits on import and generation.
- **Private Mode** — the user is the device owner or is in the device's access list. Token checks and limits are not applied.

We are working on enabling node owners who make their nodes publicly available to earn from the use of their hardware.
---

## Advanced: Image Generation (ComfyUI)

1. Install ComfyUI:
   ```bash
   git clone https://github.com/comfyanonymous/ComfyUI /opt/ComfyUI
   cd /opt/ComfyUI
   pip install -r requirements.txt
   ```
2. Run ComfyUI:
   ```bash
   python3 main.py --listen 0.0.0.0 --port 8188
   ```
3. Edit `config.ini` in **OnMyChat**:
   ```ini
   COMFY_API_URL = http://localhost:8188
   WORKFLOW_PATH = flow.json
   COMFY_OUTPUT_DIR = /opt/ComfyUI/output
   COMFY_INPUT_DIR = /opt/ComfyUI/input
   ```

---

## Advanced: Systemd Services

### **OnMyChat** Service

Create `/etc/systemd/system/onmychat.service`:

```ini
[Unit]
Description=OnMyChat AI Node
After=network.target ollama.service

[Service]
Type=simple
User=onmychat
WorkingDirectory=/opt/onmychat
ExecStart=/opt/onmychat/venv/bin/uvicorn api:app --host 0.0.0.0 --port 8000
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now onmychat
```

### Ollama Service

Ollama installs its own systemd unit automatically. To enable:

```bash
sudo systemctl enable --now ollama
```

---

## Advanced: GPU Acceleration

### NVIDIA CUDA

```bash
# Install NVIDIA drivers
sudo apt install nvidia-driver-550

# Install NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install nvidia-container-toolkit

# Configure Ollama for GPU
ollama stop
sudo systemctl edit ollama.service
# Add:
# [Service]
# Environment="OLLAMA_CUDA=1"
sudo systemctl restart ollama
```

### AMD ROCm

```bash
# Install ROCm
sudo apt install rocm-libs rocm-dev

# Ollama detects ROCm automatically on supported GPUs
ollama stop
sudo systemctl restart ollama
```

---

## Troubleshooting

| Problem | Likely Cause | Solution |
|---------|-------------|----------|
| Connector shows "offline" | NAT / firewall | Check `http://<ip>:8081` is reachable. Enable UPnP or forward ports 9000, 8899. |
| AI node not responding | **OnMyChat** not running | `systemctl status onmychat` and check logs. |
| RAG not finding documents | ChromaDB not populated | Run `/learn <path>` in chat. Check `memory_index/` exists. |
| Slow inference | No GPU / small RAM | Use a smaller model (e.g., `gemma4:4b`) or enable CUDA. |
| Image generation fails | ComfyUI not running | Check `COMFY_API_URL` in `config.ini`. Run ComfyUI manually. |
| `AI_TOKEN` mismatch | Config mismatch | Copy the token from Connector UI → Settings → Integrations → AI Token. |

---

## References

- [Product Specification](PRODUCT_SPECIFICATION.md) — ecosystem overview
- [Search & RAG Architecture](SEARCH_AND_RAG.md) — ChromaDB deep dive
- [User Guide](USER_GUIDE.md) — daily usage
- [Developer's Guide](DEVELOPERS_GUIDE.md) — building from source
- [Ollama](https://ollama.com) — local LLM runner
- [ChromaDB](https://www.trychroma.com) — vector database
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) — image generation

---

**On My Disk** — your data, your AI, your rules.
