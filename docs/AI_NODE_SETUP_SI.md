# Nastavitev osebnega AI vozlišča

> Zaženi svojo lastno AI infrastrukturo: zasebno, brez cenzure, brez omejitev.

Ta vodnik opisuje namestitev osebnega OnMyChat AI vozlišča na vaši lastni opremi in njegovo povezavo z vašim OnMyDisk računom prek **OnMyDisk Connector**. Po nastavitvi boste imeli:

- **Zasebni RAG** — vaše datoteke so indeksirane lokalno, podatki ne zapuščajo vašega omrežja
- **Semantično iskanje** — iščite dokumente po pomenu, ne le po imenu
- **Neomejeno klepetanje** — brez omejitev hitrosti in filtriranja vsebine
- **Generiranje slik** — ComfyUI na vašem GPU
- **Popolna zasebnost** — sklepanje (inference) deluje na vaši opremi

---

## Arhitektura

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Spletni        │────>│  OnMyDisk Gateway│────>│  Vaša naprava    │
│  brskalnik      │     │  (oblačni proxy) │     │  (Connector)     │
│  (onmydisk.net) │     └──────────────────┘     └──────────────────┘
└─────────────────┘                                        │
                                                           v
                                                    ┌──────────────────┐
                                                    │  OnMyChat AI     │
                                                    │  Datoteke (lok.) │
                                                    ├──────────────────┤
                                                    │  Ollama (LLM)    │
                                                    │  ChromaDB (RAG)  │
                                                    │  ComfyUI (Slike) │
                                                    └──────────────────┘
```

**Connector** (`onmydisk-connector`) omogoča dostop do vaših datotek prek prehoda in najavi vaše lokalno AI vozlišče.
**AI vozlišče** (`onmychat`) obdeluje sklepanje, iskanje in RAG — vse za vašim NAT.

---

## Zahteve glede opreme

| Komponenta | Minimalno | Priporočeno |
|------------|-----------|--------------| 
| CPU | 4 jedra | 8+ jeder |
| RAM | 8 GB | 32 GB |
| Disk | 20 GB prosto | 100+ GB (SSD) |
| GPU (izbirno) | — | NVIDIA GTX 1060 6GB+ |
| OS | Ubuntu 22.04 / 24.04 | Ubuntu 24.04 LTS |
| Omrežje | Dostop do interneta | Nameščen OnMyDisk Connector |

---

## 1. korak: Namestitev OnMyDisk Connector

Connector je **napredni odjemalec** On My Disk, namenjen namestitvi na različne naprave za shranjevanje podatkov. Deluje kot sistemska storitev, omogoča dostop do lokalnih map in gosti lokalnega AI asistenta.

### Prenos paketa

Vnaprej zgrajeni `.deb` paketi so na voljo za **amd64** in **arm64**:

| Arhitektura | URL za prenos |
|-------------|---------------|
| x86_64 (amd64) | `https://forge.bineon.team/repo/Ubuntu/focal/onmydisk-connector-amd64.deb` |
| ARM64 | `https://forge.bineon.team/repo/Ubuntu/focal/onmydisk-connector-arm64.deb` |
| ARMHF (armv7) | `https://forge.bineon.team/repo/Ubuntu/focal/onmydisk-connector-armhf.deb` |

Lahko ga tudi zgradite iz izvorne kode (glejte [Razvijalski priročnik](DEVELOPERS_GUIDE.md)).

### Namestitev

```bash
sudo apt update
sudo apt install samba udisks2 libqt5core5a libqt5network5 \
    libqt5websockets5 libqt5widgets5 libqt5sql5 \
    libqt5webchannel5 ffmpeg ruby-full build-essential zlib1g-dev

# Za x86_64 (amd64)
sudo dpkg -i onmydisk-connector-amd64.deb
# Za ARM64 (Raspberry Pi 4/5, Orange Pi, Firefly RK3588 in dr. z 64-bitnim OS)
# sudo dpkg -i onmydisk-connector-arm64.deb
# Za ARMHF (armv7) (Raspberry Pi 3, Orange Pi in dr. z 32-bitnim OS)
# sudo dpkg -i onmydisk-connector-armhf.deb


sudo apt --fix-broken install   # če manjkajo odvisnosti
```
### Po namestitvi
Konfigurirajte odjemalca za zagon pod vašim sistemskim uporabnikom, da zagotovite pravilen dostop do vaše domače mape. Uredite `/etc/onmydisk/onmydisk.conf` in dodajte uporabniško ime ter skupino v razdelku `[FileNode]`:

```ini
[FileNode]
Port=80
User=firefly  # zamenjajte z vašim sistemskim uporabniškim imenom
Group=firefly # zamenjajte z vašo sistemsko skupino
```

Shranite konfiguracijo in ponovno zaženite storitev:

```bash
sudo systemctl restart onmydisk
```

### Preverite namestitev

```bash
systemctl status onmydisk
# Moral bi prikazati: active (running)
```

Connector privzeto posluša na **vratih 80**. Odprite `http://<ip-ali-ime-vaše-naprave>` za dostop do spletnega vmesnika.

---

## 2. korak: Registracija in povezava z On My Disk

Na začetku bo vaša naprava delovala v anonimnem načinu, brez povezave z računom On My Disk, dostopna le v lokalnem omrežju. Za dostop do naprave prek interneta, iz mobilne aplikacije in za deljenje datotek ter AI asistenta z drugimi uporabniki sledite tem korakom:

1. Ustvarite račun On My Disk ali se prijavite z Googlom v vmesnik prehoda: [onmydisk.net](https://onmydisk.net).
2. Odprite spletni vmesnik Connector: `http://<ip-ali-ime-naprave>` in pojdite v nastavitve profila **Settings → Profile**.
3. Kliknite "Poveži napravo z računom". V pojavnem oknu se prijavite s svojim računom.
4. Preverite v spletnem vmesniku prehoda [onmydisk.net](https://onmydisk.net) - vaša naprava bi se morala pojaviti v razdelku **My Devices**. Zdaj lahko svojo napravo uporabljate na daljavo ter delite datoteke in AI asistenta z drugimi uporabniki v vaši skupini.

---

## 3. korak: Namestitev in nastavitev OnMyChat (AI vozlišče)

OnMyChat je dodatna storitev, ki upravlja LLM, RAG in iskanje.

```bash
# Klonirajte repozitorij
git clone <repository-url> /opt/onmychat
cd /opt/onmychat

# Ustvarite virtualno okolje
python3 -m venv venv
source venv/bin/activate

# Namestite odvisnosti
pip install -r requirements.txt

# Izbirno: za spletno iskanje
# playwright install
```

### Nastavitev

```bash
cp config.example.ini config.ini
```

Uredite `config.ini`:

```ini
[settings]
GATEWAY_URL = https://onmydisk.net
AI_TOKEN = <izmislite-si-svoj-ai-token>

OLLAMA_URL = http://localhost:11434
DEFAULT_MODEL = gemma4:12b

STORAGE_ROOT = /opt/onmychat/storage
APP_ROOT_DIR = /opt/onmychat
```

Nato odprite spletni vmesnik Connector (`http://<ip-ali-ime-naprave>`), pojdite na **Settings → Integrations**, omogočite **On My Chat**, navedite **Base URL**: `http://localhost:8000` in vnesite enak **AI Token** kot v zgornji konfiguraciji. Connector bo začel posredovati AI zahteve od prehoda OnMyDisk do vašega lokalnega OnMyChat.

### Zagon

```bash
source venv/bin/activate
python3 api.py
```

AI vozlišče privzeto posluša na **vratih 8000**.

---

## 4. korak: Namestitev in nastavitev Ollame

Ollama poganja LLM lokalno na vaši opremi.

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Nalaganje modelov

```bash
# Glavni model za klepet
ollama pull gemma4:12b

# Lahek model za pomožne naloge
ollama pull gemma4:4b

# (Izbirno) model brez cenzure
ollama pull igorls/gemma-4-E4B-it-heretic-GGUF:latest
```

### Preverjanje

```bash
ollama list
curl http://localhost:11434/api/tags
```

> [!IMPORTANT]
> Za GPU pospešek namestite NVIDIA Container Toolkit ali ROCm. Če imate le CPU, uporabite manjše modele (npr. `gemma4:4b` ali `qwen2.5:7b`).

### Posebna navodila za plošče z Rockchip NPU (RK3588 / RK3576, npr. Firefly, Orange Pi 5 in dr.)

Če nastavljate vozlišče na plošči z Rockchip NPU, standardna Ollama ni podprta. Namesto nje uporabite **RKLLAMA**:

1. **Namestitev in nastavitev RKLLAMA:**
   - Klonirajte repozitorij: `git clone https://github.com/NotPunchnox/rkllama.git ~/RKLLAMA`
   - Pojdite v mapo repozitorija in namestite paket v vaše Python okolje (npr. v osnovno okolje Miniconda):
     ```bash
     cd ~/RKLLAMA
     pip install .
     ```
   - Datoteke modelov `.rkllm` namestite v `~/RKLLAMA/models/<ime-modela>/` skupaj z datoteko opisa `Modelfile`.
   - Ustvarite systemd storitveno datoteko za samodejni zagon `/etc/systemd/system/rkllama.service`:
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
   - Ponovno naložite systemd in aktivirajte storitev:
     ```bash
     sudo systemctl daemon-reload && sudo systemctl enable rkllama && sudo systemctl start rkllama
     ```
   - Preverite razpoložljive modele: `curl -s http://localhost:8080/api/tags`
   - V datoteki `config.ini` za OnMyChat navedite:
     ```ini
     OLLAMA_URL = http://localhost:8080
     DEFAULT_MODEL = <ime_vašega_modela_iz_rkllama_oznak>
     ```

2. **Namestitev OnMyChat na SBC (izogibanje prenosu CUDA):**
   Da se izognete prenosu gigabajtov nepotrebnih knjižnic NVIDIA CUDA/cuDNN, namestite odvisnosti OnMyChat z eksplicitno navedbo CPU različice PyTorch:
   ```bash
   pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
   ```
   Če se ob zagonu OnMyChat pojavi napaka `AttributeError: np.float_ was removed in the NumPy 2.0 release`, prisilno znižajte različico NumPy:
   ```bash
   pip install "numpy<2.0.0"
   ```

3. **Zagon OnMyChat kot sistemska storitev:**
   Ustvarite datoteko `/etc/systemd/system/onmychat.service`:
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
   Omogočite in zaženite storitev:
   ```bash
   sudo systemctl daemon-reload && sudo systemctl enable onmychat && sudo systemctl start onmychat
   ```

---


## RAG in semantično iskanje

Vgrajena ChromaDB zagotavlja semantično iskanje in RAG.

Zbirke znanja, uvožene prek `/learn <pot_do_datoteke_ali_URL> <zbirka>`, se shranjujejo v `/opt/onmychat/data/chroma_db/<zbirka>`.
Po uvozu dokumenta ali povezave preverite:

```bash
ls /opt/onmychat/memory_index/chroma_db/
# Primer izpisa: 'omd'
```
Tako lahko organizirate razpršene dokumente in zunanje vire v logično povezane zbirke znanja. S sklicevanjem na zbirko v poizvedbi (`/explain <zbirka> <vaše_vprašanje>`) boste bistveno izboljšali kakovost in ustreznost odgovorov asistenta.

Poleg tega sistem periodično samodejno indeksira datoteke v vaši shrambi. Indeks semantičnega iskanja se shranjuje v search_index:

```bash
ls /opt/onmychat/memory_index/search_index/
# Vsebovati mora zbirko 'omd_search'
```

---

## Načini delovanja AI vozlišča

AI vozlišče, povezano s prehodom OnMyDisk, lahko deluje v dveh načinih, ki ju določa avtentikacija uporabnika na prehodu:

- **Javni način** — uporabnik je gost. Vozlišče preverja stanje žetonov in uveljavlja omejitve za uvoz in generiranje.
- **Zasebni način** — uporabnik je lastnik naprave ali je na seznamu za dostop do naprave. Preverjanje stanja in omejitve se ne uveljavljajo.

Prizadevamo si, da bi lastniki vozlišč, ki so na voljo javnosti, lahko zaslužili z uporabo njihove opreme.
---

## Napredno: Generiranje slik (ComfyUI)

1. Namestite ComfyUI:
   ```bash
   git clone https://github.com/comfyanonymous/ComfyUI /opt/ComfyUI
   cd /opt/ComfyUI
   pip install -r requirements.txt
   ```
2. Zaženite ComfyUI:
   ```bash
   python3 main.py --listen 0.0.0.0 --port 8188
   ```
3. Uredite `config.ini` v OnMyChat:
   ```ini
   COMFY_API_URL = http://localhost:8188
   WORKFLOW_PATH = flow.json
   COMFY_OUTPUT_DIR = /opt/ComfyUI/output
   COMFY_INPUT_DIR = /opt/ComfyUI/input
   ```

---

## Napredno: Systemd storitve

### OnMyChat storitev

Ustvarite `/etc/systemd/system/onmychat.service`:

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

### Ollama storitev

Ollama samodejno namesti svoj systemd modul. Za omogočitev:

```bash
sudo systemctl enable --now ollama
```

---

## Napredno: GPU pospešek

### NVIDIA CUDA

```bash
# Namestitev gonilnikov NVIDIA
sudo apt install nvidia-driver-550

# Namestitev NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install nvidia-container-toolkit

# Nastavitev Ollame za GPU
ollama stop
sudo systemctl edit ollama.service
# Dodajte:
# [Service]
# Environment="OLLAMA_CUDA=1"
sudo systemctl restart ollama
```

### AMD ROCm

```bash
# Namestitev ROCm
sudo apt install rocm-libs rocm-dev

# Ollama samodejno zazna ROCm na podprtih GPU-jih
ollama stop
sudo systemctl restart ollama
```

---

## Reševanje težav

| Težava | Verjeten vzrok | Rešitev |
|--------|----------------|---------|
| Connector kaže "offline" | NAT / požarni zid | Preverite dostopnost `http://<ip>:8081`. Omogočite UPnP ali posredujte vrata 9000, 8899. |
| AI vozlišče ne odgovarja | OnMyChat ne teče | `systemctl status onmychat`, preverite dnevnike. |
| RAG ne najde dokumentov | ChromaDB ni napolnjena | Zaženite `/learn <pot>` v klepetu. Preverite `memory_index/`. |
| Počasno sklepanje | Ni GPU / premalo RAM | Uporabite manjši model (npr. `gemma4:4b`) ali omogočite CUDA. |
| Napaka pri generiranju slik | ComfyUI ne teče | Preverite `COMFY_API_URL` v `config.ini`. Zaženite ComfyUI ročno. |
| `AI_TOKEN` se ne ujema | Napaka v konfiguraciji | Kopirajte žeton iz Connector UI → Settings → Integrations → AI Token. |

---

## Povezave

- [Specifikacija izdelka](PRODUCT_SPECIFICATION.md) — pregled ekosistema
- [Arhitektura iskanja in RAG](SEARCH_AND_RAG.md) — poglobljen opis ChromaDB
- [Uporabniški priročnik](USER_GUIDE_SI.md) — vsakodnevna uporaba
- [Razvijalski priročnik](DEVELOPERS_GUIDE.md) — gradnja iz izvorne kode
- [Ollama](https://ollama.com) — lokalno poganjanje LLM
- [ChromaDB](https://www.trychroma.com) — vektorska podatkovna baza
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) — generiranje slik

---

**On My Disk** — vaši podatki, vaš AI, vaša pravila.
