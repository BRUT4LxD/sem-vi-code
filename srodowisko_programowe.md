# Środowisko programowe projektu

Dokument zawiera **faktyczne wersje** z folderu `venv` w repozytorium (do cytowania w rozdziale o badaniach) oraz ogólne wskazówki i treść `requirements.txt`.

---

## Faktyczne środowisko badawcze (folder `venv`)

Poniższe wartości pochodzą z odczytu interpretera i pakietów w:

`sem-vi-code\venv\`

### System operacyjny i architektura

| Aspekt | Wartość |
|--------|---------|
| Platforma (z Pythona) | `Windows-10-10.0.26200-SP0` |
| Architektura | `AMD64` |

### Python i narzędzie instalacyjne

| Aspekt | Wartość |
|--------|---------|
| **Wersja Pythona** | **3.11.5** |
| Interpreter w venv | `venv\Scripts\python.exe` |
| Interpreter bazowy (z `venv\pyvenv.cfg`) | `C:\Users\BRUT4LxD\AppData\Local\Programs\Python\Python311\python.exe` |
| `include-system-site-packages` | `false` (środowisko **izolowane**) |
| **pip** | **26.0.1** (dla Pythona 3.11 w tym venv) |

### PyTorch, CUDA i GPU

| Aspekt | Wartość |
|--------|---------|
| **torch** | **2.10.0+cu128** |
| **torchvision** | **0.25.0+cu128** |
| **torchaudio** | **2.10.0+cu128** |
| **CUDA (wbudowana w build PyTorch)** | **12.8** (`torch.version.cuda`) |
| **cuDNN** (numer wersji zwracany przez PyTorch) | **91002** |
| `torch.cuda.is_available()` | `True` (GPU dostępne w momencie odczytu) |

Build `+cu128` oznacza **oficjalny wheel PyTorch ze wsparciem CUDA 12.8** (nie jest to osobna instalacja CUDA Toolkit wymagana do importu `torch` — do treningu na GPU potrzebne są odpowiednie **sterowniki NVIDIA** zgodne z tym stosem).

### Pakiety z `pip freeze` (kompletna lista z tego venv)

```text
absl-py==2.4.0
colorama==0.4.6
contourpy==1.3.3
cycler==0.12.1
filelock==3.20.0
fonttools==4.62.1
fsspec==2025.12.0
grpcio==1.78.0
Jinja2==3.1.6
joblib==1.5.3
kiwisolver==1.5.0
lightning-utilities==0.15.3
Markdown==3.10.2
MarkupSafe==3.0.2
matplotlib==3.10.8
mpmath==1.3.0
networkx==3.6.1
numpy==2.3.5
packaging==26.0
pandas==3.0.1
pillow==12.0.0
protobuf==7.34.1
pyparsing==3.3.2
python-dateutil==2.9.0.post0
scikit-learn==1.8.0
scipy==1.17.1
six==1.17.0
sympy==1.14.0
tensorboard==2.20.0
tensorboard-data-server==0.7.2
threadpoolctl==3.6.0
torch==2.10.0+cu128
torchaudio==2.10.0+cu128
torchmetrics==1.9.0
torchvision==0.25.0+cu128
tqdm==4.67.3
typing_extensions==4.15.0
tzdata==2025.3
Werkzeug==3.1.6
```

### Jak odtworzyć ten opis u siebie (weryfikacja)

Z katalogu `sem-vi-code`, po aktywacji `venv`:

```text
.\venv\Scripts\python.exe --version
.\venv\Scripts\pip.exe freeze
.\venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

---

## Wersja języka Python (kontekst vs `requirements.txt`)

- W **`requirements.txt`** w komentarzach jest mowa o zgodności z **Pythonem 3.8** — to **orientacja dla dolnych granic** pakietów, a **nie** opis faktycznego venv w repozytorium.
- **Faktyczne badania** w tym projekcie (wg folderu `venv`) były prowadzone na **Pythonie 3.11.5**.

W repozytorium **nie ma** pliku `.python-version` ani `pyproject.toml` z przypiętym Pythonem — źródłem prawdy o środowisku badawczym jest niniejszy zapis lub wyeksportowany `pip freeze` z Twojej maszyny w dacie eksperymentu.

---

## Wirtualne środowisko (`venv`)

**Zalecenie:** instalacja wyłącznie w **dedykowanym venv**, aby nie mieszać wersji bibliotek z innymi projektami i z Pythonem systemowym.

### Windows (PowerShell lub CMD)

W tym repozytorium istnieje już folder **`venv`** (nie `.venv`). Nowe środowisko można utworzyć analogicznie:

```text
python -m venv venv
```

Aktywacja (dla folderu `venv`):

- **PowerShell:** `.\venv\Scripts\Activate.ps1`  
  (przy pierwszym uruchomieniu może być konieczna polityka wykonywania skryptów, np. `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`)
- **CMD:** `.\venv\Scripts\activate.bat`

Po aktywacji w wierszu poleceń pojawia się prefiks `(venv)`.

Dalsze kroki (z aktywnym venv):

```text
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Dezaktywacja:** `deactivate`

### Dlaczego `venv`

- Izolacja **wersji** `torch`, `numpy` itd. od innych projektów.
- Możliwość **usunięcia** całego środowiska przez skasowanie folderu `venv` bez wpływu na system.
- Zgodność z typowym workflow na Windows bez dodatkowych menedżerów (conda jest opcjonalna — w projekcie nie jest wymagana).

**Uwaga:** folder `.venv` zwykle dodaje się do **`.gitignore`**, żeby nie commitować binariów i katalogu site-packages.

---

## Instalacja PyTorch (CPU vs GPU / CUDA)

Plik `requirements.txt` używa ogólnych nazw `torch`, `torchvision`, `torchaudio`. Dla **GPU NVIDIA** często korzystniej jest dobrać wheel **z oficjalnego instalatora** (wersja CUDA musi odpowiadać sterownikom), np. instrukcje z [pytorch.org](https://pytorch.org/get-started/locally/), a następnie ewentualnie doinstalować pozostałe pakiety z `requirements.txt` (lub ręcznie dopasować wersje `torch*` do jednego źródła).

- **Sterowniki NVIDIA** oraz (opcjonalnie) **CUDA Toolkit** — zgodnie z wymaganiami wybranego buildu PyTorch.
- W komentarzu w `requirements.txt` jest kontekst **RTX 3070** — typowo trening z **CUDA** po poprawnej instalacji stosu PyTorch+GPU.

---

## Platforma i uruchamianie kodu

- **System:** projekt jest rozwijany i uruchamiany typowo pod **Microsoft Windows**.
- **Katalog roboczy:** skrypty zakładają ścieżki względne (np. `./data/imagenette`, `./models/...`). **Uruchamianie z katalogu głównego repozytorium** (`sem-vi-code`) jako bieżącego katalogu zmniejsza ryzyko błędów ścieżek.
- **Zmienna `PYTHONPATH`:** jeśli importy pakietów projektu (`training`, `data_eng`, …) mają działać z dowolnego katalogu, można ustawić `PYTHONPATH` na katalog główny repozytorium; przy uruchomieniu z katalogu głównego często nie jest to konieczne.

---

## Język i zarządzanie zależnościami

- **Menadżer pakietów:** **pip** (standardowo w parze z venv).
- **Lista zależności:** `requirements.txt` — wersje jako **dolne granice** (`>=`), bez pełnego zamrożenia buildów (brak pliku lock w repozytorium).

---

## Zależności z `requirements.txt` (deklaracja minimalna, nie stan venv)

Konkretne wersje z badawczego `venv` są w sekcji **„Faktyczne środowisko badawcze”** powyżej — mogą być **nowsze** niż dolne granice z pliku.

| Obszar | Pakiety (minimalne wersje) |
|--------|----------------------------|
| Głębokie uczenie | `torch>=2.4.1`, `torchvision>=0.19.1`, `torchaudio>=2.4.1` |
| Obliczenia naukowe | `numpy>=1.24.4`, `scipy>=1.10.1`, `pandas>=2.0.3` |
| Metryki ML | `scikit-learn>=1.3.2`, `torchmetrics>=1.2.0` |
| Wizualizacja | `matplotlib>=3.7.0` |
| Postęp w konsoli | `tqdm>=4.66.1` |
| Logowanie eksperymentów | `tensorboard>=2.14.0` |

Biblioteki pomocnicze (np. **Pillow**) często są **zależnościami pośrednimi** `torchvision`.

---

## Charakter oprogramowania

- **Zestaw modułów i skryptów Pythona** (m.in. `training/`, `data_eng/`, `imagenette_lab/`, `attacks/`, `evaluation/`), bez frameworka webowego w głównych zależnościach.
- **PyTorch**, **torchvision**, **TensorBoard**, **torchmetrics**, **scikit-learn**.

---

## Inne ważne aspekty (dokumentacja i odtwarzalność)

| Aspekt | Opis |
|--------|------|
| **Odtwarzalność eksperymentu** | Same `>=` nie gwarantują identycznych wersji po czasie. Do pracy można dołączyć wynik `pip freeze` z maszyny, na której liczono wyniki końcowe. |
| **Dane i modele** | Zbiory (np. ImageNette) i pliki `.pt` zwykle **nie są** w repozytorium — wymagane lokalne ścieżki zgodne z kodem (`./data/...`, `./models/...`). |
| **Kontrola wersji** | Repozytorium **Git** — folder `venv` zwykle w **`.gitignore`** i nie commituje się razem z kodem. |
| **IDE** | Dowolne (np. VS Code / Cursor / PyCharm); nie jest częścią `requirements.txt`. |
| **Jupyter / testy** | Nie wymienione w `requirements.txt` — instalacja opcjonalna (`pip install jupyter`, `pytest` itd.). |
| **TensorBoard** | Logi w katalogach typu `runs/...` — uruchomienie: `tensorboard --logdir=...` z aktywnym venv. |

---

## Pełna treść `requirements.txt` (stan referencyjny)

```
# Core PyTorch ecosystem - Latest stable versions for RTX 3070 (Python 3.8 compatible)
torch>=2.4.1
torchvision>=0.19.1
torchaudio>=2.4.1

# Scientific computing (Python 3.8 compatible)
numpy>=1.24.4
scipy>=1.10.1
pandas>=2.0.3

# Machine learning metrics
scikit-learn>=1.3.2
torchmetrics>=1.2.0

# Visualization
matplotlib>=3.7.0

# Progress bars
tqdm>=4.66.1

# Logging and monitoring (Python 3.8 compatible)
tensorboard>=2.14.0
```
