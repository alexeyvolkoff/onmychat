# On My Chat — AI Node for On My Disk

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-On%20My%20Disk-0070BB.svg)](https://github.com/alexeyvolkoff/onmydisk)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM%20%26%20Vision-black.svg)](https://ollama.com)
[![Web & PWA](https://img.shields.io/badge/PWA-onmychat.com-7c3aed.svg)](https://onmychat.com/?launch=1)
[![Docs](https://img.shields.io/badge/Docs-onmydisk.org-0ea5e9.svg)](https://onmydisk.org)
[![Discord](https://img.shields.io/badge/Discord-Community-5865F2.svg?logo=discord&logoColor=white)](https://discord.gg/qAQpfmrBPW)

**On My Chat** is the open-source, self-hosted AI node for [**On My Disk**](https://github.com/alexeyvolkoff/onmydisk), powered by **Ollama** (or **RKLLAMA**). It runs on **Linux**, **Windows**, **macOS**, home servers, NAS boxes, and ARM SBCs — giving you private RAG over your local folders, mobile/PWA remote access over peer-to-peer WebRTC without port forwarding, and a built-in Creative Studio with an AI prompt assistant.

---

## Highlights

- **Private Multilingual RAG & Photo Indexing** — Indexes local documents (`PDF`, `DOCX`, `XLSX`, `PPTX`, `MD`, code) and photos directly from your folders using ChromaDB and CPU-friendly `paraphrase-multilingual-MiniLM-L12-v2` embeddings. Bare photos are automatically captioned and OCR-indexed via Ollama vision models (e.g. `gemma4`) without uploading files to third-party clouds.
- **Mobile & PWA Remote Access (Zero Port Forwarding)** — Connect to your home Ollama node from any phone or browser at [**onmychat.com**](https://onmychat.com) (or [**direct.onmydisk.net**](https://direct.onmydisk.net)) over encrypted P2P WebRTC DataChannels. No static IP, router setup, or VPN required.
- **Passwordless Cryptographic Identity (DID)** — Access control is based on W3C-style Decentralized Identifiers (`DID` keypairs with portable `.json` keyfiles) and P2P device pairing (`LinkID` / QR) — **no static API tokens in `config.ini`**.
- **Creative Studio & Prompt Assistant** — Includes an Ollama-powered **AI Art Director** that writes and surgically refines visual prompts, plans multi-scene storyboards, manages character Personas, and drives local **ComfyUI** pipelines (SDXL DMD2, IP-Adapter Plus Face + ReActor FaceSwap, Auto-Inpaint retouch, Krea 2, Anima, video generation, and 1-click CivitAI publishing).
- **Flexible Runtime Modes (`local` · `agentic` · `hybrid`)** — Run everything on a single GPU box (`local`), keep lightweight CPU indexing on an always-on NAS/SBC while tunneling Ollama & ComfyUI over P2P to a desktop GPU (`agentic`), or run on Rockchip RK3588/RK3576 NPUs via `RKLLAMA`.
- **MCP Tools, Voice & Push Notifications** — Native Model Context Protocol (`MCP`) server & client integrations, local neural TTS (`Kokoro-82M`), and Web/mobile push notifications when background tasks or renders complete.

---

![On My Chat Interface](docs/screenshot.png)

---

## Architecture

```text
┌───────────────────────────┐         Signaling & DID Handshake         ┌───────────────────────────┐
│   Browser / Mobile PWA    │ <───────────────────────────────────────> │  direct.onmydisk.net /    │
│      (onmychat.com)       │                                           │       onmychat.com        │
└─────────────┬─────────────┘                                           └─────────────▲─────────────┘
              │                                                                       │
              │            Direct Encrypted P2P Tunnel (WebRTC DataChannel)           │
              └───────────────────────────────────┬───────────────────────────────────┘
                                                  │  (No port forwarding / No cloud storage)
                                                  ▼
                                    ┌───────────────────────────┐
                                    │    On My Disk 3.0 Node    │
                                    │   (Local P2P File Node)   │
                                    └─────────────┬─────────────┘
                                                  │ 127.0.0.1:4000
                                                  ▼
                                    ┌───────────────────────────┐
                                    │     On My Chat Service    │
                                    ├───────────────────────────┤
                                    │ • Ollama (LLM & Vision)   │
                                    │ • ChromaDB (Local RAG)    │
                                    │ • ComfyUI (Studio Render) │
                                    │ • Kokoro TTS & MCP Tools  │
                                    └───────────────────────────┘
```

1. **On My Disk 3.0 Node** (`onmydisk`) shares selected local folders over P2P WebRTC and routes authenticated AI requests to `onmychat` on `localhost` along with the caller's verified `DID` and pairing role (`owner` / `guest`).
2. **`onmychat`** orchestrates chat streaming, background RAG indexing, semantic search, MCP tools, and ComfyUI workflows against your local **Ollama** instance (`http://localhost:11434`) or a P2P-tunneled GPU node.

---

## Quick Start

### 1. Prerequisites

- **OS (All Major Platforms Supported):**
  - **Linux:** Ubuntu, Debian, Armbian, NAS & SBCs (`x86_64` and `arm64`)
  - **Windows:** Windows 10 / 11 (`x64` — per-user install, no Administrator rights needed)
  - **macOS:** Apple Silicon M1–M4 (`arm64` — per-user install, no `sudo` needed)
  - **Clients & Edge:** Any modern browser / PWA, native Android & iOS apps, plus storage/tunnel nodes on OpenWrt routers
- **Python:** 3.10+ *(configured automatically by the one-command installers)*
- **Ollama:** Installed locally for GPU/Metal/CPU inference (`--gpu` / `-Gpu`), or optional in **Agentic P2P** mode (`--agentic` / `-Agentic`) where inference runs over P2P on a remote GPU
- **On My Disk 3.0 Node:** Local P2P WebRTC daemon — installed together with **On My Chat** in one step via the [**⚡ TL;DR Quick Install Guide**](https://onmydisk.org/advanced/#-tldr--quick-install-recommended-agentic-ai-setup)

### 2. One-Command Cross-Platform Install (Recommended)

Follow the [**⚡ TL;DR — Quick Install**](https://onmydisk.org/advanced/#-tldr--quick-install-recommended-agentic-ai-setup) in the official documentation to deploy both the **On My Disk 3.0 Node** and **On My Chat** in a single command on **Linux**, **Windows (`PowerShell`)**, or **macOS (`Terminal`)** — either in lightweight **Agentic P2P** mode (`--agentic` / `-Agentic`) or full local **Ollama GPU** mode (`--gpu` / `-Gpu`).

### 3. Standalone Linux / Rockchip SBC Installer (`install.sh`)

If you already run `onmydisk` on Linux and want to install **On My Chat** directly from this repository into `/opt/onmychat` (with **Ollama** or Rockchip **RKLLAMA** NPU systemd units):

```bash
git clone https://github.com/alexeyvolkoff/onmychat.git
cd onmychat
sudo ./install.sh ollama
```

Pull your preferred Ollama model (for chat + vision photo indexing, `gemma4` is recommended):

```bash
ollama pull gemma4:latest
```

To update an existing `/opt/onmychat` installation later:

```bash
cd /opt/onmychat && sudo ./update.sh
```

### 4. Manual Installation (Cross-Platform Development)

```bash
git clone https://github.com/alexeyvolkoff/onmychat.git
cd onmychat

python3 -m venv venv
source venv/bin/activate   # Windows PowerShell: .\venv\Scripts\Activate.ps1

# Optional on CPU-only / ARM machines to skip CUDA wheels:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
cp config.example.ini config.ini

uvicorn api:app --host 127.0.0.1 --port 4000
```

---

## Configuration (`config.ini`)

No API tokens or shared secrets are needed in `config.ini` — trust and ownership are handled automatically via local `onmydisk-connector` pairing and cryptographic `DID` signatures.

Key settings in [`config.example.ini`](config.example.ini):

```ini
[settings]
GATEWAY_URL = https://direct.onmydisk.net

# Inference mode: local | agentic | hybrid
# AI_PROVIDER = local

# Local Ollama endpoint & models
OLLAMA_URL = http://localhost:11434
DEFAULT_MODEL = gemma4:latest
WORK_MODEL = gemma4:latest
FUN_MODEL = gemma4:latest
VISION_MODEL = gemma4:latest
LLM_NUM_CTX = 32768

# Optional: Local ComfyUI for Creative Studio
COMFY_API_URL = http://localhost:8188
WORKFLOW_PATH = flow.json
COMFY_OUTPUT_DIR = /opt/ComfyUI/output
COMFY_INPUT_DIR = /opt/ComfyUI/input

# Optional: Auto-caption & OCR photos during /rag/index via VISION_MODEL
# AUTO_RECOGNIZE_IMAGES = true

# Optional: Neural Voice (Kokoro-82M)
# TTS_ENABLED = true
```

### Agentic Split Mode (`AI_PROVIDER = agentic`)

If you run `onmychat` on a low-power always-on NAS or mini-PC without a discrete GPU, set `AI_PROVIDER = agentic` (or switch to **Agentic** directly in the web UI under **Settings → AI Node**). Knowledge indexing and ChromaDB stay local on CPU, while Ollama and ComfyUI calls are transparently tunneled over On My Disk P2P to a paired desktop GPU node (`agentic_tunnel.py`).

---

## Pairing & Remote Access

1. Install **On My Disk 3.0 Node** + **On My Chat** via the [**⚡ TL;DR Quick Install**](https://onmydisk.org/advanced/#-tldr--quick-install-recommended-agentic-ai-setup) (Linux, Windows, or macOS).
2. Open [**onmychat.com**](https://onmychat.com/?launch=1) (chat-first PWA) or [**direct.onmydisk.net**](https://direct.onmydisk.net) (workspace + file manager) and create or load your **DID** identity (save your `.json` keyfile for 1-click sign-in on other devices).
3. Pair your machine using its **LinkID**, QR code, or local Claim page (`http://localhost:8081` on Windows/macOS, `http://localhost/` on Linux). Because you paired the node with your cryptographic DID, `onmychat` automatically recognizes you as the **Node Owner** with full private RAG and Creative Studio access — zero manual token copying required.

Learn more about decentralized identity and keyfiles in the [DID Documentation](https://onmydisk.org/did/).

---

## Documentation

- [⚡ TL;DR — Cross-Platform Quick Install (Linux, Windows, macOS)](https://onmydisk.org/advanced/#-tldr--quick-install-recommended-agentic-ai-setup)
- [Personal AI Node Guide](https://onmydisk.org/ainode/)
- [On My Disk & On My Chat Documentation](https://onmydisk.org)
- [Decentralized Identity (DID) Guide](https://onmydisk.org/did/)
- [Personal AI Node Setup Guide (EN)](docs/AI_NODE_SETUP_EN.md) · [RU](docs/AI_NODE_SETUP_RU.md) · [UA](docs/AI_NODE_SETUP_UA.md) · [SI](docs/AI_NODE_SETUP_SI.md)
- [Rockchip NPU (RKLLAMA) Setup Guide (EN)](docs/RKLLAMA_SETUP.md) · [RU](docs/RKLLAMA_SETUP_RU.md)
- [Google Colab Remote GPU Node Notebook](onmydisk_colab_ai_node.ipynb)

---

## License

Licensed under the [Apache License 2.0](LICENSE).
