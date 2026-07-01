# Налаштування персонального AI-вузла

> Запусти власну AI-інфраструктуру: приватно, без цензури, без лімітів.

Цей посібник допоможе розгорнути персональний AI-вузол OnMyChat на власному обладнанні та підключити його до акаунту OnMyDisk через **OnMyDisk Connector**. Після налаштування ви отримаєте:

- **Приватний RAG** — ваші файли індексуються локально, дані не залишають вашу мережу
- **Семантичний пошук** — пошук документів за змістом, а не лише за іменем
- **Безлімітний чат** — жодних рейт-лімітів і контент-фільтрів
- **Генерація зображень** — ComfyUI на вашому GPU
- **Повна приватність** — інференс виконується на вашому обладнанні

---

## Архітектура

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Браузер        │────>│  OnMyDisk Gateway│────>│  Ваш пристрій    │
│  (onmydisk.net) │     │  (хмарний проксі)│     │  (Connector)     │
└─────────────────┘     └──────────────────┘     └──────────────────┘
                                                         │
                                                         v
                                                  ┌──────────────────┐
                                                  │  OnMyChat AI     │
                                                  │  Файли (локально)│
                                                  ├──────────────────┤
                                                  │  Ollama (LLM)    │
                                                  │  ChromaDB (RAG)  │
                                                  │  ComfyUI (Image) │
                                                  └──────────────────┘
```

**Connector** (`onmydisk-connector`) надає доступ до ваших файлів через шлюз та анонсує ваш локальний AI-вузол.
**AI-вузол** (`onmychat`) обробляє інференс, пошук та RAG — все за вашим NAT.

---

## Вимоги до обладнання

| Компонент | Мінімум | Рекомендовано |
|-----------|---------|---------------|
| CPU | 4 ядра | 8+ ядер |
| RAM | 8 ГБ | 32 ГБ |
| Диск | 20 ГБ вільно | 100+ ГБ (SSD) |
| GPU (опціонально) | — | NVIDIA GTX 1060 6GB+ |
| ОС | Ubuntu 22.04 / 24.04 | Ubuntu 24.04 LTS |
| Мережа | Доступ до інтернету | Встановлений OnMyDisk Connector |

---

## Крок 1: Встановлення OnMyDisk Connector

Connector — це **просунутий клієнт** On My Disk, призначений для встановлення на різні пристрої зберігання даних. Він працює як системний сервіс, відкриває доступ до ваших локальних тек та хостить локального AI-асистента.

### Завантаження пакета

Готові `.deb` пакети доступні для **amd64** та **arm64**:

| Архітектура | Посилання |
|-------------|-----------|
| x86_64 (amd64) | `https://forge.bineon.team/repo/Ubuntu/focal/onmydisk-connector-amd64.deb` |
| ARM64 | `https://forge.bineon.team/repo/Ubuntu/focal/onmydisk-connector-arm64.deb` |
| ARMHF (armv7) | `https://forge.bineon.team/repo/Ubuntu/focal/onmydisk-connector-armhf.deb` |

Також можна зібрати з вихідних кодів (див. [Посібник розробника](DEVELOPERS_GUIDE.md)).

### Встановлення

```bash
sudo apt update
sudo apt install samba udisks2 libqt5core5a libqt5network5 \
    libqt5websockets5 libqt5widgets5 libqt5sql5 \
    libqt5webchannel5 ffmpeg ruby-full build-essential zlib1g-dev

# Для x86_64 (amd64)
sudo dpkg -i onmydisk-connector-amd64.deb
# Для ARM64 (Raspberry Pi 4/5, Orange Pi, Firefly RK3588 та ін. під керуванням 64-бітної ОС)
# sudo dpkg -i onmydisk-connector-arm64.deb
# Для ARMHF (armv7) (Raspberry Pi 3, Orange Pi та ін. під керуванням 32-бітної ОС)
# sudo dpkg -i onmydisk-connector-armhf.deb


sudo apt --fix-broken install   # якщо бракує залежностей
```
### Після встановлення
Налаштуйте запуск клієнта під вашим системним користувачем, щоб він мав коректний доступ до вашої домашньої директорії. Для цього відредагуйте `/etc/onmydisk/onmydisk.conf` та додайте ім'я вашого користувача і групу в секцію `[FileNode]`:

```ini
[FileNode]
Port=80
User=firefly  # замініть на вашого системного користувача
Group=firefly # замініть на вашу системну групу
```

Збережіть конфігурацію та перезапустіть службу:

```bash
sudo systemctl restart onmydisk
```

### Перевірка встановлення

```bash
systemctl status onmydisk
# Має показати: active (running)
```

Connector слухає **порт 80** за замовчуванням. Відкрийте `http://<ip-або-ім'я-вашого-пристрою>` для доступу до веб-інтерфейсу.

---

## Крок 2: Реєстрація та підключення до On My Disk

Спочатку ваш пристрій працюватиме в анонімному режимі, без прив'язки до облікового запису On My Disk, виключно в локальній мережі. Щоб отримати доступ до вашого пристрою через інтернет, з мобільного додатку, а також поділитися доступом до файлів та AI-асистента з іншими користувачами, виконайте наступні дії:

1. Створіть обліковий запис On My Disk або увійдіть за допомогою Google в інтерфейс шлюзу: [onmydisk.net](https://onmydisk.net).
2. Відкрийте веб-інтерфейс Connector: `http://<ip-або-ім'я-пристрою>` і перейдіть до налаштувань профілю **Settings → Profile**.
3. Клікніть "Прив'язати пристрій до акаунту". У вікні, що з'явилося, виконайте вхід під своїм обліковим записом.
4. Перевірте у веб-інтерфейсі шлюзу [onmydisk.net](https://onmydisk.net) - у розділі **My Devices** має з'явитися ваш пристрій. Тепер ви можете користуватися вашим пристроєм віддалено та надати доступ до файлів і AI-асистента іншим користувачам вашої групи.

---

## Крок 3: Встановлення та налаштування OnMyChat (AI-вузол)

OnMyChat — це додатковий сервіс, який керує LLM, RAG та пошуком.

```bash
# Клонуйте репозиторій
git clone <repository-url> /opt/onmychat
cd /opt/onmychat

# Створіть віртуальне оточення
python3 -m venv venv
source venv/bin/activate

# Встановіть залежності
pip install -r requirements.txt

# Опціонально: для веб-пошуку
# playwright install
```

### Конфігурація

```bash
cp config.example.ini config.ini
```

Відредагуйте `config.ini`:

```ini
[settings]
GATEWAY_URL = https://onmydisk.net
AI_TOKEN = <придумайте-ваш-ai-token>

OLLAMA_URL = http://localhost:11434
DEFAULT_MODEL = gemma4:12b

APP_ROOT_DIR = /opt/onmychat
```

Після цього відкрийте веб-інтерфейс Connector (`http://<ip-або-ім'я-пристрою>`), перейдіть до **Settings → Integrations**, увімкніть **On My Chat**, вкажіть **Base URL**: `http://localhost:8000` та введіть той самий **AI Token**, що й у конфігу вище. Connector почне проксіювати AI-запити від шлюзу OnMyDisk до вашого локального OnMyChat.

### Запуск

```bash
source venv/bin/activate
python3 api.py
```

AI-вузол слухає **порт 8000** за замовчуванням.

---

## Крок 4: Встановлення та налаштування Ollama

Ollama запускає LLM локально на вашому обладнанні.

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Завантаження моделей

```bash
# Основна модель для чату
ollama pull gemma4:12b

# Легка модель для утилітарних задач
ollama pull gemma4:4b

# (Опціонально) модель без цензури
ollama pull igorls/gemma-4-E4B-it-heretic-GGUF:latest
```

### Перевірка

```bash
ollama list
curl http://localhost:11434/api/tags
```

> [!IMPORTANT]
> Для GPU-прискорення встановіть NVIDIA Container Toolkit або ROCm. Якщо у вас лише CPU, використовуйте моделі менших розмірів (`gemma4:4b` або `qwen2.5:7b`).

### Спеціальний посібник для плат з Rockchip NPU (RK3588 / RK3576, наприклад, Firefly, Orange Pi 5 та ін.)

Якщо ви налаштовуєте вузол на платі з Rockchip NPU, стандартна Ollama не підтримується. Замість неї використовуйте **RKLLAMA**:

1. **Встановлення та налаштування RKLLAMA:**
   - Клонуйте репозиторій: `git clone https://github.com/NotPunchnox/rkllama.git ~/RKLLAMA`
   - Перейдіть до теки репозиторію та встановіть пакет у ваше Python-оточення (наприклад, у base-оточення Miniconda):
     ```bash
     cd ~/RKLLAMA
     pip install .
     ```
   - Розмістіть ваші файли моделей `.rkllm` у `~/RKLLAMA/models/<назва-моделі>/` разом з файлом опису `Modelfile`.
   - Створіть службу systemd для автоматичного запуску `/etc/systemd/system/rkllama.service`:
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
   - Перезавантажте systemd та активуйте службу:
     ```bash
     sudo systemctl daemon-reload && sudo systemctl enable rkllama && sudo systemctl start rkllama
     ```
   - Перевірте доступні моделі: `curl -s http://localhost:8080/api/tags`
   - У файлі `config.ini` для OnMyChat вкажіть:
     ```ini
     OLLAMA_URL = http://localhost:8080
     DEFAULT_MODEL = <назва_вашої_моделі_з_тегів_rkllama>
     ```

2. **Особливості встановлення OnMyChat на SBC (уникаємо завантаження CUDA):**
   Щоб не завантажувати гігабайти непотрібних бібліотек NVIDIA CUDA/cuDNN, встановлюйте залежності OnMyChat з явним зазначенням CPU-версії PyTorch:
   ```bash
   pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
   ```
   Якщо при запуску OnMyChat виникає помилка `AttributeError: np.float_ was removed in the NumPy 2.0 release`, примусово знизьте версію NumPy:
   ```bash
   pip install "numpy<2.0.0"
   ```

3. **Запуск OnMyChat як системного сервісу:**
   Створіть файл `/etc/systemd/system/onmychat.service`:
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
   Увімкніть та запустіть службу:
   ```bash
   sudo systemctl daemon-reload && sudo systemctl enable onmychat && sudo systemctl start onmychat
   ```

---


## RAG та семантичний пошук

Вбудована ChromaDB забезпечує семантичний пошук та RAG.

Колекції знань, імпортовані через `/learn <шлях_до_файлу_або_URL> <колекція>`, зберігаються у `/opt/onmychat/data/chroma_db/<колекція>`.
Після імпорту документа або посилання перевірте:

```bash
ls /opt/onmychat/memory_index/chroma_db/
# Приклад виведення: 'omd'
```
Таким чином ви зможете організувати розрізнені документи та зовнішні джерела в логічно пов'язані колекції знань. Посилаючись на колекцію в запиті (`/explain <колекція> <ваше_питання>`), ви значно покращите якість та релевантність відповідей асистента.

Крім того, система періодично виконує автоіндексацію файлів у вашому сховищі. Індекс семантичного пошуку зберігається в search_index:

```bash
ls /opt/onmychat/memory_index/search_index/
# Має бути колекція 'omd_search'
```

---

## Режими роботи AI-вузла

AI-вузол, підключений до шлюзу OnMyDisk, може працювати у двох режимах, що визначаються автентифікацією користувача на шлюзі:

- **Публічний режим** — користувач є гостем. Вузол перевіряє баланс токенів і застосовує ліміти на імпорт та генерацію.
- **Приватний режим** — користувач є власником пристрою або перебуває у списку доступу до пристрою. Перевірка балансу та ліміти не застосовуються.

Ми працюємо над тим, щоб власники вузлів, наданих у публічний доступ, могли заробляти за використання їхнього обладнання.
---

## Просунуте: Генерація зображень (ComfyUI)

1. Встановіть ComfyUI:
   ```bash
   git clone https://github.com/comfyanonymous/ComfyUI /opt/ComfyUI
   cd /opt/ComfyUI
   pip install -r requirements.txt
   ```
2. Запустіть ComfyUI:
   ```bash
   python3 main.py --listen 0.0.0.0 --port 8188
   ```
3. Відредагуйте `config.ini` у OnMyChat:
   ```ini
   COMFY_API_URL = http://localhost:8188
   WORKFLOW_PATH = flow.json
   COMFY_OUTPUT_DIR = /opt/ComfyUI/output
   COMFY_INPUT_DIR = /opt/ComfyUI/input
   ```

---

## Просунуте: Systemd-сервіси

### OnMyChat Service

Створіть `/etc/systemd/system/onmychat.service`:

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

Ollama встановлює свій systemd-юніт автоматично. Для увімкнення:

```bash
sudo systemctl enable --now ollama
```

---

## Просунуте: Прискорення GPU

### NVIDIA CUDA

```bash
# Встановлення драйверів NVIDIA
sudo apt install nvidia-driver-550

# Встановлення NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install nvidia-container-toolkit

# Налаштування Ollama для GPU
ollama stop
sudo systemctl edit ollama.service
# Додайте:
# [Service]
# Environment="OLLAMA_CUDA=1"
sudo systemctl restart ollama
```

### AMD ROCm

```bash
# Встановлення ROCm
sudo apt install rocm-libs rocm-dev

# Ollama автоматично визначить ROCm на підтримуваних GPU
ollama stop
sudo systemctl restart ollama
```

---

## Вирішення проблем

| Проблема | Ймовірна причина | Рішення |
|----------|------------------|---------| 
| Connector показує "offline" | NAT / файрвол | Перевірте доступність `http://<ip>:8081`. Увімкніть UPnP або прокиньте порти 9000, 8899. |
| AI-вузол не відповідає | OnMyChat не запущений | `systemctl status onmychat`, перевірте логи. |
| RAG не знаходить документи | ChromaDB не заповнена | Виконайте `/learn <шлях>` у чаті. Перевірте наявність `memory_index/`. |
| Повільний інференс | Немає GPU / мало RAM | Використовуйте модель менших розмірів (наприклад, `gemma4:4b`) або увімкніть CUDA. |
| Помилка генерації зображень | ComfyUI не запущений | Перевірте `COMFY_API_URL` у `config.ini`. Запустіть ComfyUI вручну. |
| `AI_TOKEN` не збігається | Помилка в конфігурації | Скопіюйте токен із Connector UI → Settings → Integrations → AI Token. |

---

## Посилання

- [Специфікація продукту](PRODUCT_SPECIFICATION.md) — огляд екосистеми
- [Архітектура пошуку та RAG](SEARCH_AND_RAG.md) — побудова ChromaDB
- [Посібник користувача](USER_GUIDE.md) — щоденне використання
- [Посібник розробника](DEVELOPERS_GUIDE.md) — збірка з вихідних кодів
- [Ollama](https://ollama.com) — локальний запуск LLM
- [ChromaDB](https://www.trychroma.com) — векторна база даних
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) — генерація зображень

---

**On My Disk** — ваші дані, ваш AI, ваші правила.
