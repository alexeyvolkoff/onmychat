# Инструкции по настройке Rockchip NPU (RKLLAMA)

Если вы настраиваете AI-узел OnMyChat на одноплатном компьютере (SBC) под управлением процессора Rockchip с NPU (например, RK3588 или RK3576, такого как Firefly, Orange Pi 5 и т. д.), стандартная утилита Ollama не поддерживается. Используйте **RKLLAMA** вместо неё для аппаратного ускорения на NPU.

---

> [!TIP]
> **Рекомендуемая автоматическая установка**
> Вы можете настроить OnMyChat и RKLLAMA (включая службы systemd) автоматически с помощью нового скрипта **[install.sh](file:///home/alexey/projects/omd/onmychat/install.sh)**:
> ```bash
> sudo ./install.sh
> ```
> Выберите `rkllama` при запросе.

---

## 1. Установка и настройка RKLLAMA (Вручную)

Выполните следующие шаги для развертывания и запуска сервера RKLLAMA:

1. **Клонируйте репозиторий:**
   ```bash
   git clone https://github.com/NotPunchnox/rkllama.git /opt/rkllama
   ```

2. **Установите пакет:**
   Перейдите в папку репозитория и установите его в ваше Python-окружение (например, в base-окружение Miniconda):
   ```bash
   cd /opt/rkllama
   pip install .
   ```

3. **Установка моделей RKLLM:**
   Разместите ваши скомпилированные файлы моделей `.rkllm` в директории `/opt/rkllama/models/<model-name>/` вместе с файлом описания `Modelfile`.

4. **Создайте службу Systemd:**
   Создайте файл конфигурации службы `/etc/systemd/system/rkllama.service`:
   ```ini
   [Unit]
   Description=RKLLAMA Server
   After=network.target

   [Service]
   Type=simple
   WorkingDirectory=/opt/rkllama
   Environment=HOME=/home/firefly
   User=firefly
   ExecStart=/home/firefly/miniconda3/bin/rkllama_server --processor rk3588 --port 8080 --models /opt/rkllama/models
   Restart=on-failure

   [Install]
   WantedBy=multi-user.target
   ```

5. **Запустите службу:**
   Перезагрузите systemd, включите и запустите службу:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable rkllama
   sudo systemctl start rkllama
   ```

6. **Проверка установки:**
   Проверьте доступные модели с помощью curl:
   ```bash
   curl -s http://localhost:8080/api/tags
   ```

7. **Настройка OnMyChat:**
   В файле `config.ini` для OnMyChat укажите:
   ```ini
   OLLAMA_URL = http://localhost:8080
   DEFAULT_MODEL = <имя_вашей_модели_из_тегов_rkllama>
   ```

---

## 2. Особенности установки OnMyChat на SBC (избегаем загрузки CUDA)

Чтобы не скачивать гигабайты ненужных библиотек NVIDIA CUDA/cuDNN на SBC, устанавливайте зависимости OnMyChat с явным указанием CPU-версии PyTorch:
```bash
pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

If you encounter an `AttributeError: np.float_ was removed in the NumPy 2.0 release` when starting OnMyChat, downgrade NumPy:
```bash
pip install "numpy<2.0.0"
```

---

## 3. Запуск OnMyChat как системного сервиса Systemd

Создайте файл `/etc/systemd/system/onmychat.service`:
```ini
[Unit]
Description=OnMyChat AI Service
After=network.target rkllama.service

[Service]
Type=simple
User=firefly
WorkingDirectory=/opt/onmychat
Environment="PATH=/opt/onmychat/venv/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=/opt/onmychat/venv/bin/uvicorn api:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Включите и запустите службу:
```bash
sudo systemctl daemon-reload
sudo systemctl enable onmychat
sudo systemctl start onmychat
```
