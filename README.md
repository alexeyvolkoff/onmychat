# OnMyChat — Personal AI Node for OnMyDisk

OnMyChat is a private, local, self-hosted AI orchestrator designed to run as an AI Node on your own hardware. It integrates seamlessly with the **OnMyDisk** ecosystem, allowing you to converse with local LLMs, utilize private RAG (Retrieval-Augmented Generation), search documents semantically, and generate images — all completely offline and within your control.

---

## Key Features

- **Private RAG** — Indexes your local files using ChromaDB and sentence-transformers. No data leaves your home network.
- **Semantic Search** — Finds documents based on meaning and content, rather than simple name matching.
- **Local Inference** — Connects to local LLM engines (Ollama or RKLLAMA for Rockchip NPUs).
- **Tool Protocol (MCP)** — Native Model Context Protocol support to run local integrations.
- **Image Generation** — Seamless integration with ComfyUI running locally on your GPU.
- **Multi-tenant & Group Sharing** — Share your personal AI node safely with friends and family via OnMyDisk group sharing.

---

## Architecture Overview

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   Web Browser    │────>│  OnMyDisk Gateway│────>│   Your Device    │
│  (onmydisk.net)  │     │  (cloud proxy)   │     │   (Connector)    │
└──────────────────┘     └──────────────────┘     └──────────────────┘
                                                            │
                                                            v
                                                     ┌──────────────────┐
                                                     │   OnMyChat AI    │
                                                     │  Files (local)   │
                                                     ├──────────────────┤
                                                     │  Ollama (LLM)    │
                                                     │  ChromaDB (RAG)  │
                                                     │  ComfyUI (Image) │
                                                     └──────────────────┘
```

The **OnMyDisk Connector** (`onmydisk-connector`) exposes your files remotely and routes AI requests to the **OnMyChat** service.
The **OnMyChat** service manages the conversation history, schedules background RAG indexing, executes vector database queries, and interacts with Ollama.

---

## Getting Started

### 1. Prerequisites

- Python 3.10+
- OnMyDisk Connector installed on the same machine
- Ollama or RKLLAMA running locally

### 2. Installation

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/alexeyvolkoff/onmychat.git
cd onmychat

# Set up virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies (CPU-only PyTorch is recommended for light devices/SBCs)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 3. Configuration

Copy the example configuration file and edit it:

```bash
cp config.example.ini config.ini
```

Edit the `config.ini` fields:

```ini
[settings]
GATEWAY_URL = https://onmydisk.net
AI_TOKEN = <generate-your-own-secure-token>

OLLAMA_URL = http://localhost:11434
DEFAULT_MODEL = gemma4:12b

STORAGE_ROOT = /path/to/shared/storage
APP_ROOT_DIR = /path/to/onmychat
```

### 4. Running the Service

Start OnMyChat using Uvicorn:

```bash
source venv/bin/activate
uvicorn api:app --host 0.0.0.0 --port 8000
```

---

## Integrating with OnMyDisk Connector

1. Open your local OnMyDisk Connector Web UI (usually `http://localhost` or `http://<device-ip>`).
2. Go to **Settings → Integrations** and enable **On My Chat**.
3. Set **Base URL** to `http://localhost:8000` and paste the same **AI Token** you configured in `config.ini`.
4. Save the settings. Now you can use the chat and search features through the main OnMyDisk interface at [onmydisk.net](https://onmydisk.net)!

---

## Documentation

For detailed setup guides and specifications, refer to the `docs` folder:
- [Personal AI Node Setup Guide](docs/AI_NODE_SETUP_EN.md) (Russian version: [AI_NODE_SETUP_RU.md](docs/AI_NODE_SETUP_RU.md))
- [RKLLAMA Rockchip NPU Setup Guide](docs/RKLLAMA_SETUP.md) (Russian version: [RKLLAMA_SETUP_RU.md](docs/RKLLAMA_SETUP_RU.md))

---

## License

This project is licensed under the Apache License 2.0. See the LICENSE file for details.
