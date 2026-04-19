# Środowisko programowe (badania)

Opis środowiska badawczego jest **podzielony na pliki CSV** (łatwy import do arkusza lub LaTeX). Wersje bibliotek odpowiadają odczytowi z folderu `venv` w repozytorium (stan referencyjny).

## Pliki CSV

| Plik | Zawartość |
|------|-----------|
| [srodowisko/srodowisko_systemowe.csv](srodowisko/srodowisko_systemowe.csv) | System operacyjny, CPU, RAM, GPU oraz krótki odczyt platformy z Pythona |
| [srodowisko/srodowisko_biblioteki.csv](srodowisko/srodowisko_biblioteki.csv) | Python, PyTorch / CUDA / cuDNN, interpreter, potem pełna lista pakietów z `pip freeze` |

Nagłówek kolumn: **`Obszar`**, **`Wartość`**.

## Uwagi (nie są w CSV)

- W **`requirements.txt`** komentarze wskazują m.in. **Python 3.8** jako orientację dolnych granic — to **nie** opis faktycznego `venv`; badania w tym projekcie były prowadzone na **Pythonie 3.11.5** (patrz CSV bibliotek).
- Build PyTorch **`+cu128`** — wheel z obsługą CUDA 12.8; do treningu na GPU potrzebne są **sterowniki NVIDIA** zgodne ze stosem.
- **Odtwarzalność:** same `>=` w `requirements.txt` nie zamrażają wersji — do publikacji wart dołączyć aktualny `pip freeze` z maszyny i datą pomiaru.
- **Uruchamianie:** z katalogu głównego repozytorium (`sem-vi-code`), z aktywnym `venv` (PowerShell: `.\venv\Scripts\Activate.ps1`).

### Weryfikacja u siebie

```text
.\venv\Scripts\python.exe --version
.\venv\Scripts\pip.exe freeze
.\venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Po zmianie środowiska zaktualizuj odpowiednio wiersze w plikach CSV w folderze `srodowisko/`.
