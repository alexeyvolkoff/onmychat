# Настройка персонального AI-узла

> Запусти свою собственную AI-инфраструктуру: приватно, без цензуры, без лимитов.

Это руководство поможет развернуть персональный AI-узел OnMyChat на собственном оборудовании и подключить его к аккаунту OnMyDisk через **OnMyDisk Connector**. После настройки вы получите:

- **Приватный RAG** — ваши файлы индексируются локально, данные не покидают вашу сеть
- **Семантический поиск** — поиск документов по смыслу, а не только по имени
- **Безлимитный чат** — никаких рейт-лимитов и контент-фильтров
- **Генерация изображений** — ComfyUI на вашем GPU
- **Полная приватность** — инференс выполняется на вашем оборудовании

---

## Архитектура

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Браузер        │────>│  OnMyDisk Gateway│────>│  Ваше устройство │
│  (onmydisk.net) │     │  (облачный прокси)│     │  (Connector)     │
└─────────────────┘     └──────────────────┘     └──────────────────┘
                                                         │
                                                         v
                                                  ┌──────────────────┐
                                                  │  OnMyChat AI     │
                                                  │  Файлы (локально)│
                                                  ├──────────────────┤
                                                  │  Ollama (LLM)    │
                                                  │  ChromaDB (RAG)  │
                                                  │  ComfyUI (Image) │
                                                  └──────────────────┘
```

**Connector** (`onmydisk-connector`) предоставляет доступ к вашим файлам через шлюз и анонсирует ваш локальный AI-узел. 
**AI-узел** (`onmychat`) обрабатывает инференс, поиск и RAG — всё за вашим NAT.

---

## Требования к оборудованию

| Компонент | Минимум | Рекомендуется |
|-----------|---------|---------------|
| CPU | 4 ядра | 8+ ядер |
| RAM | 8 ГБ | 32 ГБ |
| Диск | 20 ГБ свободно | 100+ ГБ (SSD) |
| GPU (опционально) | — | NVIDIA GTX 1060 6GB+ |
| ОС | Ubuntu 22.04 / 24.04 | Ubuntu 24.04 LTS |
| Сеть | Доступ в интернет | Установленный OnMyDisk Connector |

---

## Шаг 1: Установка OnMyDisk Connector

Connector — это **продвинутый клиент** On My Disk предназначенный для установки на различные устройства хранения данных. Он работает как системный сервис, открывает доступ к вашим локальным папкам и хостит локальный AI-ассистент.

### Загрузка пакета

Готовые `.deb` пакеты доступны для **amd64** и **arm64**:

| Архитектура | Ссылка |
|-------------|--------|
| x86_64 (amd64) | `https://forge.bineon.team/repo/Ubuntu/focal/onmydisk-connector-amd64.deb` |
| ARM64 | `https://forge.bineon.team/repo/Ubuntu/focal/onmydisk-connector-arm64.deb` |
| ARMHF (armv7) | `https://forge.bineon.team/repo/Ubuntu/focal/onmydisk-connector-armhf.deb` |

Также можно собрать из исходников (см. [Руководство разработчика](DEVELOPERS_GUIDE.md)).

### Установка

```bash
sudo apt update
sudo apt install samba udisks2 libqt5core5a libqt5network5 \
    libqt5websockets5 libqt5widgets5 libqt5sql5 \
    libqt5webchannel5 ffmpeg ruby-full build-essential zlib1g-dev

# Для x86_64 (amd64)
sudo dpkg -i onmydisk-connector-amd64.deb
# Для ARM64 (Raspberry Pi 4/5, Orange Pi, Firefly RK3588 и др. под управлением 64-битной ОС)
# sudo dpkg -i onmydisk-connector-arm64.deb
# Для ARMHF (armv7) (Raspberry Pi 3, Orange Pi и др. под управлением 32-битной ОС)
# sudo dpkg -i onmydisk-connector-armhf.deb


sudo apt --fix-broken install   # если не хватает зависимостей
```
### После установки
Настройте запуск клиента под вашим системным пользователем, чтобы он имел корректный доступ к вашей домашней директории. Для этого отредактируйте `/etc/onmydisk/onmydisk.conf` и добавьте имя вашего пользователя и группу в секцию `[FileNode]`:

```ini
[FileNode]
Port=80
User=firefly  # замените на вашего системного пользователя
Group=firefly # замените на вашу системную группу
```

Сохраните конфигурацию и перезапустите службу:

```bash
sudo systemctl restart onmydisk
```

### Проверка установки

```bash
systemctl status onmydisk
# Должен показать: active (running)
```

Connector слушает **порт 80** по умолчанию. Откройте `http://<ip-или-имя-вашего-устройства>` для доступа к веб-интерфейсу.

---

## Шаг 2: Регистрация и подключение к On My Disk

Изначально ваше устройство будет работать в анонимном режиме, без привязки к учётной записи On My Disk, исключительно в локальной сети. Чтобы получить доступ к вашему устройству через интернет, из мобильного приложения, а также поделиться доступом к файлам и ИИ ассистенту с другими пользователями в вашей группе, выполните следующие действия:

1. Создайте учётную запись On My Disk или войдите с помощью Google в интерфейс шлюза: [onmydisk.net](https://onmydisk.net).
2. Откройте веб-интерфейс Connector: `http://<ip-или-имя-устройства>` и перейдите в настройки профиля **Settings → Profile**.
3. Кликните "Привязать устройство к аккаунту". В появившемся окне выполните вход вашей учётной записью.
4. Проверьте веб-интерфейсе шлюза [onmydisk.net](https://onmydisk.net) - в разделе **My Devices** должно появиться ваше устройство. Теперь вы можете пользоваться вашим устройством удалённо и предоставить доступ к файлам и ИИ ассистенту другим пользователям вашей группы.

---

## Шаг 3: Установка и настройка OnMyChat (AI-узел)

OnMyChat — это дополнительный сервис, который управляет LLM, RAG и поиском.

```bash
# Клонируйте репозиторий
git clone <repository-url> /opt/onmychat
cd /opt/onmychat

# Создайте виртуальное окружение
python3 -m venv venv
source venv/bin/activate

# Установите зависимости
pip install -r requirements.txt

# Опционально: для веб-поиска
# playwright install
```

### Конфигурация

```bash
cp config.example.ini config.ini
```

Отредактируйте `config.ini`:

```ini
[settings]
GATEWAY_URL = https://onmydisk.net
AI_TOKEN = <придумайте-ваш-ai-token>

OLLAMA_URL = http://localhost:11434
DEFAULT_MODEL = gemma4:12b

STORAGE_ROOT = /opt/onmychat/storage
APP_ROOT_DIR = /opt/onmychat
```

После этого откройте веб-интерфейс Connector (`http://<ip-или-имя-устройства>`), перейдите в **Settings → Integrations**, включите **On My Chat**, укажите **Base URL**: `http://localhost:8000` и введите тот же **AI Token** что и в конфиге выше. Connector начнёт проксировать AI-запросы от шлюза OnMyDisk к вашему локальному OnMyChat.

### Запуск

```bash
source venv/bin/activate
python3 api.py
```

AI-узел слушает **порт 8000** по умолчанию.

---

## Шаг 4: Установка и настройка Ollama

Ollama запускает LLM локально на вашем оборудовании.

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Загрузка моделей

```bash
# Основная модель для чата
ollama pull gemma4:12b

# Лёгкая модель для утилитарных задач
ollama pull gemma4:4b

# (Опционально) модель без цензуры
ollama pull igorls/gemma-4-E4B-it-heretic-GGUF:latest
```

### Проверка

```bash
ollama list
curl http://localhost:11434/api/tags
```

> [!IMPORTANT]
> Для GPU-ускорения установите NVIDIA Container Toolkit или ROCm. Если у вас только CPU, используйте модели поменьше (`gemma4:4b` или `qwen2.5:7b`).

### Специальное руководство для плат Rockchip NPU (RK3588 / RK3576, например, Firefly, Orange Pi 5 и др.)

Если вы настраиваете узел на плате с Rockchip NPU, стандартная Ollama не поддерживается. Вместо неё используйте **RKLLAMA**:

1. **Установка и настройка RKLLAMA:**
   - Клонируйте репозиторий: `git clone https://github.com/NotPunchnox/rkllama.git ~/RKLLAMA`
   - Перейдите в папку репозитория и установите пакет в ваше Python-окружение (например, в base-окружение Miniconda):
     ```bash
     cd ~/RKLLAMA
     pip install .
     ```
   - Разместите ваши файлы моделей `.rkllm` в `~/RKLLAMA/models/<model-name>/` вместе с файлом описания `Modelfile`.
   - Создайте службу systemd для автоматического запуска `/etc/systemd/system/rkllama.service`:
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
   - Перезапустите systemd и активируйте службу:
     ```bash
     sudo systemctl daemon-reload && sudo systemctl enable rkllama && sudo systemctl start rkllama
     ```
   - Проверьте доступные модели: `curl -s http://localhost:8080/api/tags`
   - В файле `config.ini` для OnMyChat укажите:
     ```ini
     OLLAMA_URL = http://localhost:8080
     DEFAULT_MODEL = <имя_вашей_модели_из_тегов_rkllama>
     ```

2. **Особенности установки OnMyChat на SBC (избегаем загрузки CUDA):**
   Чтобы не скачивать гигабайты ненужных библиотек NVIDIA CUDA/cuDNN, устанавливайте зависимости OnMyChat с явным указанием CPU-версии PyTorch:
   ```bash
   pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
   ```
   Если при запуске OnMyChat возникает ошибка `AttributeError: np.float_ was removed in the NumPy 2.0 release`, принудительно понизьте версию NumPy:
   ```bash
   pip install "numpy<2.0.0"
   ```

3. **Запуск OnMyChat как системного сервиса:**
   Создайте файл `/etc/systemd/system/onmychat.service`:
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
   Включите и запустите службу:
   ```bash
   sudo systemctl daemon-reload && sudo systemctl enable onmychat && sudo systemctl start onmychat
   ```

---


## RAG и семантический поиск

Встроенная ChromaDB обеспечивает семантический поиск и RAG.

Коллекции знаний, импортированные через `/learn <путь_к_файлу_или_URL> <коллекция>` хранятся в `/opt/onmychat/data/chroma_db/<коллекция>`.
После импорта документа или ссылки проверьте:

```bash
ls /opt/onmychat/memory_index/chroma_db/
# Пример вывод 'omd'
```
Таким образом вы сможете организовать разрозненные документы и внешние источники в логически связанные коллекции знаний. Ссылаясь на коллекцию в запросе (`/explain <коллекция> <ваш_вопрос>`), вы значительно улучшите качество и релевантность ответов ассистента.

Кроме того, система периодически производит автоиндексацию файлов в вашем хранилище. Индекс семантического поиска сохраняется в search_index:

```bash
ls /opt/onmychat/memory_index/search_index/
# Должна быть коллекция 'omd_search'
```

---

## Режимы работы AI-узла

AI-узел, подключенный к шлюзу OnMyDisk, может работать в двух режимах, определяемых аутентификацией пользователя на шлюзе:

- **Публичный режим** — пользователь является гостем. Узел проверяет баланс токенов и применяет лимиты на импорт и генерацию.
- **Приватный режим** — пользователь является владельцем устройства или состоит в списке доступа к устройству. Проверка баланса и лимиты не применяются.

Мы работаем над тем, чтобы владельцы узлов, предоставляемых в публичный доступ, могли зарабатывать за использование их оборудования.
---

## Продвинутое: Генерация изображений (ComfyUI)

1. Установите ComfyUI:
   ```bash
   git clone https://github.com/comfyanonymous/ComfyUI /opt/ComfyUI
   cd /opt/ComfyUI
   pip install -r requirements.txt
   ```
2. Запустите ComfyUI:
   ```bash
   python3 main.py --listen 0.0.0.0 --port 8188
   ```
3. Отредактируйте `config.ini` в OnMyChat:
   ```ini
   COMFY_API_URL = http://localhost:8188
   WORKFLOW_PATH = flow.json
   COMFY_OUTPUT_DIR = /opt/ComfyUI/output
   COMFY_INPUT_DIR = /opt/ComfyUI/input
   ```

---

## Продвинутое: Systemd-сервисы

### OnMyChat Service

Создайте `/etc/systemd/system/onmychat.service`:

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

Ollama устанавливает свой systemd-юнит автоматически. Для включения:

```bash
sudo systemctl enable --now ollama
```

---

## Продвинутое: Ускорение GPU

### NVIDIA CUDA

```bash
# Установка драйверов NVIDIA
sudo apt install nvidia-driver-550

# Установка NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install nvidia-container-toolkit

# Настройка Ollama для GPU
ollama stop
sudo systemctl edit ollama.service
# Добавьте:
# [Service]
# Environment="OLLAMA_CUDA=1"
sudo systemctl restart ollama
```

### AMD ROCm

```bash
# Установка ROCm
sudo apt install rocm-libs rocm-dev

# Ollama автоматически определит ROCm на поддерживаемых GPU
ollama stop
sudo systemctl restart ollama
```

---

## Решение проблем

| Проблема | Вероятная причина | Решение |
|----------|------------------|---------|
| Connector показывает "offline" | NAT / фаервол | Проверьте доступность `http://<ip>:8081`. Включите UPnP или пробросьте порты 9000, 8899. |
| AI-узел не отвечает | OnMyChat не запущен | `systemctl status onmychat`, проверьте логи. |
| RAG не находит документы | ChromaDB не заполнена | Выполните `/learn <путь>` в чате. Проверьте наличие `memory_index/`. |
| Медленный инференс | Нет GPU / мало RAM | Используйте модель поменьше (например, `gemma4:4b`) или включите CUDA. |
| Ошибка генерации изображений | ComfyUI не запущен | Проверьте `COMFY_API_URL` в `config.ini`. Запустите ComfyUI вручную. |
| `AI_TOKEN` не совпадает | Ошибка в конфиге | Скопируйте токен из Connector UI → Settings → Integrations → AI Token. |

---

## Ссылки

- [Спецификация продукта](PRODUCT_SPECIFICATION.md) — обзор экосистемы
- [Архитектура поиска и RAG](SEARCH_AND_RAG.md) — устройство ChromaDB
- [Руководство пользователя](USER_GUIDE.md) — ежедневное использование
- [Руководство разработчика](DEVELOPERS_GUIDE.md) — сборка из исходников
- [Ollama](https://ollama.com) — локальный запуск LLM
- [ChromaDB](https://www.trychroma.com) — векторная база данных
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) — генерация изображений

---

**On My Disk** — ваши данные, ваш AI, ваши правила.
