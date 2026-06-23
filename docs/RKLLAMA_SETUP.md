# Rockchip NPU Setup Instructions (RKLLAMA)

If you are setting up your OnMyChat AI node on a Single Board Computer (SBC) powered by a Rockchip SoC with an NPU (such as RK3588 or RK3576, e.g., Firefly, Orange Pi 5, etc.), standard Ollama is not supported. Use **RKLLAMA** instead to leverage NPU acceleration.

---

## 1. Install and Configure RKLLAMA

Follow these steps to deploy and run the RKLLAMA server:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/NotPunchnox/rkllama.git ~/RKLLAMA
   ```

2. **Install the package:**
   Navigate to the repository and install it in your Python environment (e.g., base environment of Miniconda):
   ```bash
   cd ~/RKLLAMA
   pip install .
   ```

3. **Install RKLLM Models:**
   Place your compiled `.rkllm` model files under `~/RKLLAMA/models/<model-name>/` along with a `Modelfile`.

4. **Create a Systemd Service:**
   Create a systemd unit file `/etc/systemd/system/rkllama.service`:
   ```ini
   [Unit]
   Description=RKLLAMA Server
   After=network.target

   [Service]
   Type=simple
   WorkingDirectory=/home/firefly
   Environment=HOME=/home/firefly
   User=firefly
   ExecStart=/home/firefly/miniconda3/bin/rkllama_server --processor rk3588 --port 8080 --models /home/firefly/RKLLAMA/models
   Restart=on-failure

   [Install]
   WantedBy=multi-user.target
   ```

5. **Start the Service:**
   Reload systemd, enable and start the service:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable rkllama
   sudo systemctl start rkllama
   ```

6. **Verify installation:**
   Verify available models by running:
   ```bash
   curl -s http://localhost:8080/api/tags
   ```

7. **Configure OnMyChat:**
   In your OnMyChat `config.ini`, set:
   ```ini
   OLLAMA_URL = http://localhost:8080
   DEFAULT_MODEL = <your_model_name_from_rkllama_tags>
   ```

---

## 2. OnMyChat Installation on SBC (Avoiding CUDA Downloads)

To prevent downloading gigabytes of unnecessary NVIDIA CUDA/cuDNN packages on the SBC, force a CPU-only PyTorch installation:
```bash
pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

If you encounter an `AttributeError: np.float_ was removed in the NumPy 2.0 release` when starting OnMyChat, downgrade NumPy:
```bash
pip install "numpy<2.0.0"
```

---

## 3. Run OnMyChat as a Systemd Service

Create `/etc/systemd/system/onmychat.service`:
```ini
[Unit]
Description=OnMyChat AI Service
After=network.target rkllama.service

[Service]
Type=simple
User=firefly
WorkingDirectory=/home/firefly/projects/onmychat
Environment="PATH=/home/firefly/projects/onmychat/venv/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=/home/firefly/projects/onmychat/venv/bin/uvicorn api:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable onmychat
sudo systemctl start onmychat
```
