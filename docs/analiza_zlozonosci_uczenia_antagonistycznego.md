# Analiza złożoności obliczeniowej uczenia antagonistycznego w eksperymentach ImageNette

## 1. Cel i zakres analizy

Niniejszy dokument opisuje złożoność obliczeniową trzech wariantów uczenia odpornego na przykłady antagonistyczne zaimplementowanych w projekcie:

1. standardowego uczenia antagonistycznego z generowaniem przykładów w locie,
2. progresywnego aktywnego uczenia antagonistycznego,
3. progresywnego pasywnego uczenia antagonistycznego na uprzednio wygenerowanych przykładach.

Analiza jest oparta na aktualnej implementacji w modułach:

- `training/train.py`,
- `imagenette_lab/training/imagenette_adversarial_trainer.py`,
- `imagenette_lab/training/imagenette_adversarial_progressive_trainer.py`,
- `data_eng/adversarial_training_dataset_builder.py`,
- `experiments/imagenette_full_research/runner.py`,
- `experiments/imagenette_full_research/config.yaml`.

Celem analizy nie jest wyłącznie podanie asymptotycznej notacji `O(...)`, ale także rozpisanie składników kosztu w sposób przydatny do opisu eksperymentów w pracy doktorskiej. Z tego powodu oddzielnie traktowane są:

- koszt klasycznego przejścia uczącego przez model,
- koszt walidacji,
- koszt generowania przykładów antagonistycznych,
- koszt wynikający z progresywnego przyrostu zbioru danych,
- koszt przeniesiony między fazą aktywną i pasywną.

## 2. Oznaczenia

W dalszej części dokumentu stosowane są następujące oznaczenia:

| Symbol | Znaczenie |
| --- | --- |
| `M` | liczba architektur/modeli trenowanych w eksperymencie |
| `A` | liczba rozważanych ataków antagonistycznych |
| `N_tr` | liczba czystych obrazów w zbiorze treningowym |
| `N_val` | liczba czystych obrazów w zbiorze walidacyjnym |
| `B` | rozmiar batcha |
| `E_s` | liczba epok w standardowym uczeniu antagonistycznym |
| `E_p` | liczba epok w pasywnym uczeniu antagonistycznym |
| `I` | liczba iteracji w progresywnym aktywnym uczeniu antagonistycznym |
| `E_i` | liczba epok wykonywanych w każdej iteracji progresywnej |
| `T` | liczba przykładów treningowych generowanych na atak w jednej iteracji progresywnej |
| `V` | liczba przykładów walidacyjnych generowanych na atak w jednej iteracji progresywnej |
| `rho` | udział próbek antagonistycznych w batchu w standardowym wariancie on-the-fly |
| `P` | liczba parametrów modelu |
| `C_f` | koszt pojedynczego przejścia w przód dla jednej próbki |
| `C_b` | koszt przejścia wstecznego dla jednej próbki |
| `C_train` | koszt jednego kroku uczącego dla jednej próbki, w przybliżeniu `C_f + C_b` |
| `C_att(a)` | koszt wygenerowania jednej próbki antagonistycznej atakiem `a` |
| `Q_att` | łączna liczba prób wygenerowania przykładów antagonistycznych |

Dla analizowanych eksperymentów ImageNette aktualna konfiguracja nominalna wynosi:

| Parametr | Wartość |
| --- | ---: |
| `M` | 6 |
| `A` | 31 |
| `N_tr` | 9469 |
| `N_val` | 3925 |
| `B` | 32 |
| `I` | 30 |
| `E_i` | 50 |
| `T` | 10 |
| `V` | 2 |
| `E_p` | 200 |
| `rho` | 0.5 |

Wartości `T` i `V` wynikają odpowiednio z pól `images_per_attack_per_iteration` oraz `validation_images_per_attack_per_iteration` w konfiguracji progresywnej.

## 3. Model kosztu

Koszt uczenia sieci neuronowej można rozpatrywać na dwóch poziomach.

Pierwszy poziom to koszt nadzorowanego uczenia klasyfikatora, czyli standardowe przejścia przez model. Dla batcha zawierającego `B` próbek pojedynczy krok uczący obejmuje:

1. przejście w przód,
2. obliczenie funkcji straty,
3. propagację wsteczną,
4. aktualizację parametrów optymalizatorem.

Dla uproszczenia można przyjąć:

```text
C_train ~= C_f + C_b
```

W typowych sieciach konwolucyjnych i transformatorowych koszt propagacji wstecznej jest zwykle większy od kosztu samego przejścia w przód. W przybliżeniach inżynierskich często stosuje się:

```text
C_train ~= 3 * C_f
```

Walidacja nie wykonuje propagacji wstecznej, więc jej koszt jest bliższy:

```text
C_val ~= C_f
```

Drugi poziom to koszt wygenerowania próbki antagonistycznej. Ten składnik zależy od konkretnego ataku. Dla ataków jednoetapowych, takich jak FGSM, koszt może być zbliżony do jednego lub kilku przejść modelu. Dla ataków iteracyjnych, takich jak PGD, BIM, APGD, CW lub DeepFool, koszt rośnie wraz z liczbą kroków optymalizacji wewnętrznej:

```text
C_att(a) ~= k_a * (C_f + C_b)
```

gdzie `k_a` oznacza efektywną liczbę iteracji lub ewaluacji wymaganych przez atak `a`. W praktyce `C_att(a)` może dominować nad kosztem samego douczania klasyfikatora.

Z tego powodu w dokumencie rozdzielane są:

- nadzorowane sample-passes, czyli liczba próbek przetworzonych przez pętlę treningu lub walidacji,
- forward-equivalent passes, czyli orientacyjne przeliczenie kosztu treningu na koszt przejść w przód,
- attack-generation cost, czyli koszt generowania perturbacji.

## 4. Standardowe uczenie antagonistyczne on-the-fly

### 4.1. Charakterystyka implementacji

Standardowy wariant antagonistyczny jest obsługiwany przez `Training.train_imagenette_adversarial(...)` z parametrem `use_preattacked_images=False`. W każdej epoce wykonywana jest funkcja `_train_adversarial_epoch(...)`.

Uwaga metodologiczna: wariant ten jest zaimplementowany w kodzie, ale nie jest uruchamiany w pipeline eksperymentu ImageNette. W `experiments/imagenette_full_research/runner.py` oraz w `config.yaml` (sekcja `run.phases`) nie występuje faza odpowiadająca standardowemu uczeniu on-the-fly; faktycznie wykonywane są jedynie warianty progresywny aktywny (`progressive_active`) oraz progresywny pasywny (`passive`). W konsekwencji parametr `E_s` nie pochodzi z konfiguracji eksperymentu. Wartość `E_s = 200` przyjmowana w dalszej części jest założeniem porównawczym (dla zrównania z `E_p` wariantu pasywnego), a nie wielkością odczytaną z `config.yaml`.

Dla każdego batcha:

1. wybierana jest część próbek o liczności:

```text
num_adv = floor(B * rho)
```

2. dla tej części batcha losowany jest jeden atak z listy `attack_names`,
3. generowane są przykłady antagonistyczne,
4. próbki antagonistyczne są łączone z pozostałymi próbkami czystymi,
5. wykonywany jest klasyczny krok optymalizacji na połączonym batchu.

Istotne jest to, że w tej implementacji nie są wykonywane wszystkie ataki dla każdej próbki. Dla każdego batcha losowany jest jeden atak. Oznacza to, że koszt generowania przykładów antagonistycznych zależy od średniego kosztu wylosowanego ataku, a nie od sumy kosztów wszystkich `A` ataków w każdej epoce.

### 4.2. Złożoność nadzorowanej części treningu

Dla jednego modelu liczba przetworzonych próbek w części treningowej wynosi:

```text
S_train_standard = E_s * N_tr
```

Liczba próbek walidacyjnych:

```text
S_val_standard = E_s * N_val
```

Łącznie:

```text
S_total_standard = E_s * (N_tr + N_val)
```

W przeliczeniu na koszt obliczeniowy:

```text
T_standard_supervised =
    E_s * N_tr  * C_train
  + E_s * N_val * C_val
```

Przy przybliżeniu `C_train ~= 3*C_f` i `C_val ~= C_f`:

```text
T_standard_supervised ~= E_s * (3*N_tr + N_val) * C_f
```

### 4.3. Koszt generowania przykładów antagonistycznych

Dla pełnych batchy liczba próbek antagonistycznych generowanych w jednej epoce jest w przybliżeniu:

```text
N_adv_epoch ~= rho * N_tr
```

Dokładniej, ze względu na `floor(B*rho)`, koszt zależy od liczby batchy i rozmiaru ostatniego batcha:

```text
N_adv_epoch =
    floor(N_tr / B) * floor(B*rho)
  + floor((N_tr mod B) * rho)
```

Dla `N_tr = 9469`, `B = 32`, `rho = 0.5`:

```text
floor(9469 / 32) = 295
9469 mod 32 = 29
N_adv_epoch = 295 * 16 + floor(29 * 0.5) = 4734
```

Dla `E_s = 200`:

```text
N_adv_standard = 4734 * 200 = 946800
```

prób generowania przykładów antagonistycznych na jeden model.

Koszt generowania można zapisać jako:

```text
T_standard_attack =
    E_s * N_adv_epoch * E_a[C_att(a)]
```

gdzie `E_a[C_att(a)]` oznacza oczekiwany koszt ataku losowanego z listy `attack_names`.

Całkowity koszt standardowego wariantu:

```text
T_standard =
    E_s * N_tr  * C_train
  + E_s * N_val * C_val
  + E_s * N_adv_epoch * E_a[C_att(a)]
```

### 4.4. Liczby orientacyjne dla konfiguracji ImageNette

Zakładając `E_s = 200`, `N_tr = 9469`, `N_val = 3925`, `B = 32`, `rho = 0.5`:

| Wielkość | Jeden model | Sześć modeli |
| --- | ---: | ---: |
| treningowe sample-passes | 1 893 800 | 11 362 800 |
| walidacyjne sample-passes | 785 000 | 4 710 000 |
| razem nadzorowane sample-passes | 2 678 800 | 16 072 800 |
| forward-equivalent, bez kosztu ataków | 6 466 400 | 38 798 400 |
| liczba próbek przekazanych do ataków | 946 800 | 5 680 800 |
| batch updates treningowe | 59 200 | 355 200 |
| batch updates walidacyjne | 24 600 | 147 600 |

Najważniejszy wniosek: standardowy wariant ma stosunkowo prostą złożoność względem liczby epok, ale koszt generowania perturbacji jest ponoszony w każdej epoce. Jeżeli lista ataków zawiera metody iteracyjne, koszt `T_standard_attack` może być dominujący mimo niewielkiej liczby nadzorowanych sample-passes.

## 5. Progresywne aktywne uczenie antagonistyczne

### 5.1. Charakterystyka implementacji

Progresywne aktywne uczenie antagonistyczne jest realizowane przez `ImageNetteAdversarialProgressiveTrainer.train_progressive_adversarial_model(...)`. Wariant ten różni się od standardowego uczenia on-the-fly tym, że dane antagonistyczne są generowane w iteracjach i kumulowane.

W każdej iteracji `i`:

1. model generuje nowe przykłady antagonistyczne dla każdego ataku,
2. generowanych jest do `T` przykładów treningowych na atak,
3. generowanych jest do `V` przykładów walidacyjnych na atak,
4. nowe przykłady są dodawane do dotychczasowego zbioru progresywnego,
5. model jest trenowany przez `E_i` epok na zbiorze:

```text
clean_train + progressive_train_adversarial
```

6. model jest walidowany na zbiorze:

```text
clean_val + progressive_val_adversarial
```

7. dodatkowo wykonywana jest walidacja tylko na progresywnym zbiorze antagonistycznym.

Właśnie kumulowanie danych powoduje, że złożoność nie jest liniowa wyłącznie względem liczby iteracji. Rozmiar zbioru treningowego rośnie w każdej iteracji, a model przez wiele epok przetwarza również dane wygenerowane w poprzednich iteracjach.

### 5.2. Rozmiar zbioru w iteracji

Po iteracji `i`, licząc od 1, liczba wygenerowanych przykładów treningowych wynosi nominalnie:

```text
D_adv_train(i) = i * A * T
```

Liczba wygenerowanych przykładów walidacyjnych:

```text
D_adv_val(i) = i * A * V
```

Rozmiar połączonego zbioru treningowego:

```text
D_train(i) = N_tr + i * A * T
```

Rozmiar połączonego zbioru walidacyjnego:

```text
D_val(i) = N_val + i * A * V
```

Rozmiar walidacji antagonistycznej:

```text
D_adv_only(i) = i * A * V
```

### 5.3. Złożoność nadzorowanej części treningu

Liczba treningowych sample-passes dla jednego modelu:

```text
S_train_prog =
    E_i * sum_{i=1}^{I} D_train(i)
```

Po podstawieniu `D_train(i)`:

```text
S_train_prog =
    E_i * sum_{i=1}^{I} (N_tr + i*A*T)
```

```text
S_train_prog =
    E_i * (I*N_tr + A*T * I(I+1)/2)
```

Analogicznie dla walidacji połączonej:

```text
S_val_prog =
    E_i * (I*N_val + A*V * I(I+1)/2)
```

Dla walidacji tylko antagonistycznej:

```text
S_adv_val_prog =
    E_i * (A*V * I(I+1)/2)
```

Łącznie:

```text
S_total_prog =
    S_train_prog + S_val_prog + S_adv_val_prog
```

Koszt obliczeniowy części nadzorowanej:

```text
T_prog_supervised =
    S_train_prog   * C_train
  + S_val_prog     * C_val
  + S_adv_val_prog * C_val
```

Przy przybliżeniu `C_train ~= 3*C_f`:

```text
T_prog_supervised ~=
    (3*S_train_prog + S_val_prog + S_adv_val_prog) * C_f
```

### 5.4. Koszt generowania przykładów antagonistycznych

W każdej iteracji i dla każdego ataku wykonywana jest funkcja zbierająca skuteczne przykłady antagonistyczne. Dla pojedynczego ataku i splitu procedura:

1. wykonuje predykcję na czystym obrazie, aby odrzucić próbki błędnie sklasyfikowane,
2. generuje przykład antagonistyczny,
3. wykonuje predykcję na obrazie antagonistycznym,
4. zapisuje przykład, jeśli atak doprowadził do zmiany decyzji modelu.

Dla nominalnie skutecznej generacji liczba zapisanych przykładów na jeden model wynosi:

```text
N_gen_prog = I * A * (T + V)
```

Jednak rzeczywisty koszt zależy od liczby prób wymaganych do uzyskania skutecznych przykładów. Oznaczmy przez `q_{i,a,tr}` liczbę prób dla ataku `a` w iteracji `i` na zbiorze treningowym, a przez `q_{i,a,val}` analogiczną liczbę dla walidacji. Wtedy:

```text
Q_att_prog =
    sum_{i=1}^{I} sum_{a=1}^{A} (q_{i,a,tr} + q_{i,a,val})
```

Koszt generowania:

```text
T_prog_attack =
    sum_{i=1}^{I} sum_{a=1}^{A}
    [
      q_{i,a,tr}  * (C_f + C_att(a) + C_f)
    + q_{i,a,val} * (C_f + C_att(a) + C_f)
    ]
```

Pierwsze `C_f` odpowiada filtracji poprawnie sklasyfikowanych obrazów czystych, `C_att(a)` generowaniu perturbacji, a drugie `C_f` weryfikacji skuteczności ataku.

W implementacji parametr `max_tries_per_attack` ogranicza liczbę kolejnych nieudanych prób, ale nie jest prostym globalnym ograniczeniem liczby wszystkich prób. Złożoność generowania jest więc zależna od skuteczności ataku i aktualnej odporności modelu w danej iteracji. Im bardziej odporny staje się model, tym trudniejsze może być zebranie wymaganej liczby skutecznych przykładów.

### 5.5. Liczby nominalne dla konfiguracji ImageNette

Dla `A = 31`, `I = 30`, `E_i = 50`, `T = 10`, `V = 2`:

```text
I(I+1)/2 = 465
```

Liczby poniżej są nominalne i stanowią górne ograniczenie. W implementacji działa wczesne zatrzymywanie (`early_stopping_patience = 7`, licznik resetowany na początku każdej iteracji), więc rzeczywista liczba epok w iteracji może być mniejsza niż `E_i = 50`. Faktyczny koszt jest więc nie większy niż podany poniżej.

Liczba wygenerowanych przykładów na jeden model:

```text
N_gen_train = I*A*T = 30*31*10 = 9300
N_gen_val   = I*A*V = 30*31*2  = 1860
N_gen_total = 11160
```

Rozmiary końcowe zbiorów:

```text
D_train(I) = 9469 + 9300 = 18769
D_val(I)   = 3925 + 1860 = 5785
D_adv(I)   = 1860
```

Sample-passes dla jednego modelu:

```text
S_train_prog =
50 * (30*9469 + 31*10*465)
= 21 411 000
```

```text
S_val_prog =
50 * (30*3925 + 31*2*465)
= 7 329 000
```

```text
S_adv_val_prog =
50 * (31*2*465)
= 1 441 500
```

Łącznie:

```text
S_total_prog = 30 181 500 sample-passes / model
```

Dla sześciu modeli:

```text
S_total_prog_all = 181 089 000 sample-passes
```

Forward-equivalent bez kosztu wewnętrznego ataków:

```text
T_prog_supervised ~= 3*S_train_prog + S_val_prog + S_adv_val_prog
```

```text
T_prog_supervised ~= 73 003 500 forward-equivalent passes / model
```

Dla sześciu modeli:

```text
438 021 000 forward-equivalent passes
```

Liczba batchy przy `B = 32`:

| Składnik | Jeden model | Sześć modeli |
| --- | ---: | ---: |
| trening | 669 850 | 4 019 100 |
| walidacja połączona | 229 800 | 1 378 800 |
| walidacja tylko antagonistyczna | 45 750 | 274 500 |
| razem | 945 400 | 5 672 400 |

Najważniejszy wniosek: progresywny aktywny wariant jest kosztowny z dwóch powodów. Po pierwsze, wielokrotnie przetwarza coraz większy skumulowany zbiór danych, co daje składnik trójkątny `I(I+1)/2`. Po drugie, w każdej iteracji uruchamia pełny mechanizm generowania przykładów antagonistycznych dla wielu ataków.

## 6. Progresywne pasywne uczenie antagonistyczne

### 6.1. Charakterystyka implementacji

Wariant pasywny jest realizowany przez `ImageNetteAdversarialTrainer.train_adversarial_model(...)` z parametrem `use_preattacked_images=True`. W eksperymencie `runner.py` modele pasywne są trenowane na obrazach zapisanych wcześniej przez wariant progresywny aktywny:

```text
final_research/data/attacks/progressive/active
```

Dla każdej architektury używany jest odpowiadający jej folder, np.:

```text
resnet18_progressive_adv
swin_t_progressive_adv
efficientnet_b0_progressive_adv
```

Wariant pasywny nie generuje przykładów antagonistycznych w trakcie treningu. Zamiast tego:

1. ładuje wcześniej zapisane obrazy attacked,
2. ładuje czyste obrazy ImageNette,
3. buduje zbalansowany zbiór clean+attacked,
4. wykonuje klasyczne uczenie nadzorowane na tym zbiorze.

Przy aktualnej konfiguracji efektywne wartości to:

```text
clean_to_attacked_ratio = 1.0   # wartość domyślna; nie występuje jawnie w sekcji passive_adversarial
augment_clean_to_match_attacked = true   # ustawione jawnie w config.yaml
```

Pole `clean_to_attacked_ratio` nie jest podane w bloku `passive_adversarial` w `config.yaml`; runner nie przekazuje go do `train_adversarial_model(...)`, więc używana jest wartość domyślna `1.0` z `build_imagenette_adversarial_training_loaders(...)`. Oznacza to, że liczba czystych próbek jest dopasowywana do liczby próbek attacked. W rezultacie rozmiar zbioru treningowego jest w przybliżeniu:

```text
D_passive_train = 2 * N_att_train
```

a rozmiar zbioru walidacyjnego:

```text
D_passive_val = 2 * N_att_val_used
```

gdzie `N_att_val_used` może zostać ograniczone przez `train_test_split`.

### 6.2. Złożoność treningu pasywnego

Dla jednego modelu:

```text
S_train_passive = E_p * D_passive_train
```

```text
S_val_passive = E_p * D_passive_val
```

```text
S_total_passive = E_p * (D_passive_train + D_passive_val)
```

Koszt:

```text
T_passive =
    S_train_passive * C_train
  + S_val_passive   * C_val
```

Przy przybliżeniu `C_train ~= 3*C_f`:

```text
T_passive ~= (3*S_train_passive + S_val_passive) * C_f
```

Nie występuje tu składnik `C_att(a)` w trakcie treningu. Jest to podstawowa różnica względem standardowego uczenia on-the-fly i progresywnego aktywnego wariantu. Koszt ataków został poniesiony wcześniej, w fazie generowania danych przez wariant aktywny.

Podobnie jak w wariancie progresywnym, `E_p = 200` jest górnym ograniczeniem. W konfiguracji pasywnej działa wczesne zatrzymywanie (`early_stopping_patience = 20`), więc rzeczywista liczba epok może być mniejsza, a podane liczby należy traktować jako maksymalne.

### 6.3. Nominalny koszt pasywny przy pełnej generacji progresywnej

Jeżeli przyjmiemy nominalne wartości z wariantu progresywnego:

```text
N_att_train = I*A*T = 9300
N_att_val   = I*A*V = 1860
```

to:

```text
D_passive_train = 2 * 9300 = 18600
D_passive_val   = 2 * 1860 = 3720
```

Dla `E_p = 200`:

```text
S_train_passive = 200 * 18600 = 3 720 000
```

```text
S_val_passive = 200 * 3720 = 744 000
```

```text
S_total_passive = 4 464 000 sample-passes / model
```

Dla sześciu modeli:

```text
S_total_passive_all = 26 784 000 sample-passes
```

Forward-equivalent:

```text
T_passive ~= 3*3 720 000 + 744 000
= 11 904 000 forward-equivalent passes / model
```

Dla sześciu modeli:

```text
71 424 000 forward-equivalent passes
```

### 6.4. Koszt pasywny dla aktualnie zapisanych danych

W aktualnym katalogu `final_research/data/attacks/progressive/active` liczba zapisanych obrazów różni się od wartości nominalnej. Wynika to z rzeczywistej skuteczności ataków, powtórzeń uruchomień oraz ograniczeń zbierania przykładów. Dla aktualnie zapisanych danych koszt pasywnego treningu wynosi:

| Model | attacked train | attacked val użyte | train size clean+attacked | val size clean+attacked | sample-passes train | sample-passes val |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| densenet121 | 7691 | 1538 | 15382 | 3076 | 3 076 400 | 615 200 |
| efficientnet_b0 | 7520 | 1504 | 15040 | 3008 | 3 008 000 | 601 600 |
| inception_v3 | 7780 | 1556 | 15560 | 3112 | 3 112 000 | 622 400 |
| mobilenet_v2 | 7627 | 1525 | 15254 | 3050 | 3 050 800 | 610 000 |
| resnet18 | 14675 | 2935 | 29350 | 5870 | 5 870 000 | 1 174 000 |
| swin_t | 8673 | 1734 | 17346 | 3468 | 3 469 200 | 693 600 |
| razem | 53966 | 10792 | 107932 | 21584 | 21 586 400 | 4 316 800 |

Łącznie dla sześciu modeli:

```text
S_total_passive_actual = 25 903 200 sample-passes
```

Jest to koszt samego pasywnego treningu i walidacji. Nie obejmuje on kosztu wcześniejszego wygenerowania obrazów przez wariant aktywny.

## 7. Porównanie wariantów

### 7.1. Porównanie strukturalne

| Wariant | Generowanie przykładów | Charakter danych treningowych | Główny składnik kosztu |
| --- | --- | --- | --- |
| standardowy on-the-fly | w każdej epoce, dla części batcha | rozmiar datasetu stały | koszt ataków powtarzany przez wszystkie epoki |
| progresywny aktywny | na początku każdej iteracji, dla każdego ataku | dataset rośnie iteracyjnie | koszt ataków + trójkątny koszt ponownego uczenia na rosnącym zbiorze |
| progresywny pasywny | brak w trakcie treningu | stały zbiór clean+attacked z dysku | klasyczne uczenie nadzorowane na gotowych przykładach |

### 7.2. Porównanie nominalnych sample-passes

Poniższe zestawienie porównuje część nadzorowaną, bez wewnętrznego kosztu ataków.

| Wariant | Jeden model | Sześć modeli | Uwagi |
| --- | ---: | ---: | --- |
| standardowy on-the-fly, `E_s=200` | 2 678 800 | 16 072 800 | nie obejmuje kosztu generowania 946 800 próbek attacked na model |
| progresywny aktywny | 30 181 500 | 181 089 000 | obejmuje trening i dwie walidacje w każdej iteracji; nie obejmuje kosztu wewnętrznego ataków |
| progresywny pasywny, nominalnie | 4 464 000 | 26 784 000 | nie obejmuje kosztu wcześniejszej generacji danych |
| progresywny pasywny, aktualne dane z dysku | zależnie od modelu | 25 903 200 | koszt policzony dla aktualnie zapisanych obrazów |

### 7.3. Porównanie forward-equivalent

Przy założeniu:

```text
trening jednej próbki ~= 3 forward passes
walidacja jednej próbki ~= 1 forward pass
```

otrzymujemy:

| Wariant | Jeden model | Sześć modeli | Bez czego? |
| --- | ---: | ---: | --- |
| standardowy on-the-fly, `E_s=200` | 6 466 400 | 38 798 400 | bez kosztu ataków on-the-fly |
| progresywny aktywny | 73 003 500 | 438 021 000 | bez kosztu wewnętrznego ataków progresywnych |
| progresywny pasywny, nominalnie | 11 904 000 | 71 424 000 | bez kosztu wcześniejszej generacji danych |

### 7.4. Relacja kosztów

W części nadzorowanej progresywny aktywny wariant jest około:

```text
30 181 500 / 2 678 800 ~= 11.27
```

razy większy niż standardowy wariant on-the-fly przy `E_s = 200`, jeśli pominiemy koszt samych ataków.

Jest też około:

```text
30 181 500 / 4 464 000 ~= 6.76
```

razy większy niż nominalny pasywny wariant progresywny, również w części nadzorowanej.

Porównania te nie oznaczają jednak, że standardowy wariant jest zawsze tańszy w czasie rzeczywistym. Standardowy wariant generuje przykłady antagonistyczne w każdej epoce, więc jego rzeczywisty koszt może szybko rosnąć dla ataków iteracyjnych. Progresywny pasywny wariant jest tani w fazie treningu, ponieważ nie wykonuje ataków, ale jego koszt całkowity powinien być liczony razem z kosztem fazy aktywnej, która wytworzyła dane.

## 8. Złożoność pamięciowa

### 8.1. Standardowe uczenie on-the-fly

W standardowym wariancie on-the-fly pamięć GPU jest zdominowana przez:

- parametry modelu `P`,
- stany optymalizatora, w przypadku Adam zwykle około `2P` dodatkowych tensorów,
- aktywacje dla batcha,
- tensory potrzebne do wygenerowania przykładów antagonistycznych.

Asymptotycznie:

```text
O(P + B * S_img + memory_attack)
```

gdzie `S_img` oznacza rozmiar reprezentacji obrazu i aktywacji pośrednich. Ataki iteracyjne mogą zwiększać pamięć tymczasową, ale zwykle nie wymagają trzymania całego datasetu antagonistycznego w pamięci GPU.

### 8.2. Progresywny aktywny

Wariant aktywny przechowuje skumulowane przykłady antagonistyczne w strukturach Pythona jako lista tensorów CPU:

```text
progressive_train_dataset
progressive_test_dataset
```

Pamięć RAM rośnie liniowo z liczbą wygenerowanych obrazów:

```text
O(I * A * (T + V) * S_img)
```

Przy ImageNette i obrazach `3 x 224 x 224` jeden obraz float32 zajmuje:

```text
3 * 224 * 224 * 4 B = 602112 B ~= 0.57 MiB
```

Dla nominalnych `11160` obrazów na model:

```text
11160 * 0.57 MiB ~= 6.4 GiB
```

Jest to orientacyjny koszt pamięci CPU dla tensorów float32, jeżeli wszystkie przykłady są przechowywane jednocześnie jako tensory. Dodatkowo, przy `save_generated_images=true`, obrazy są zapisywane na dysku jako PNG, co przenosi część kosztu na przestrzeń dyskową.

Pamięć GPU podczas treningu jest podobna do klasycznego treningu dla batcha `B`, natomiast podczas generowania ataków zależy od konkretnego ataku.

### 8.3. Progresywny pasywny

Wariant pasywny ładuje obrazy z dysku przez `Dataset`/`DataLoader`, dlatego nie musi utrzymywać całego zbioru w GPU. Koszt pamięci GPU jest zbliżony do zwykłego uczenia nadzorowanego:

```text
O(P + B * S_img)
```

Koszt dyskowy wynika z liczby zapisanych obrazów antagonistycznych:

```text
O(N_att_train + N_att_val)
```

W tym wariancie głównym narzutem systemowym może być I/O z dysku, szczególnie jeśli obrazy PNG są dekodowane w czasie treningu bez większej liczby workerów DataLoadera.

## 9. Interpretacja metodologiczna

### 9.1. Standardowe uczenie antagonistyczne

Standardowy wariant on-the-fly jest metodologicznie prosty: model w każdej epoce widzi świeżo wygenerowane przykłady antagonistyczne. Zaletą jest adaptacyjność przykładów do aktualnych parametrów modelu. Wadą jest powtarzany koszt ataków. Każda epoka wymaga ponownego wygenerowania perturbacji, nawet dla obrazów, które były już wcześniej przetwarzane.

W kontekście pracy doktorskiej standardowy wariant można traktować jako punkt odniesienia dla pytania: jaki jest koszt uzyskania odporności, jeżeli przykłady antagonistyczne generujemy bezpośrednio w pętli uczenia.

### 9.2. Progresywny aktywny wariant

Progresywny aktywny wariant ma większy koszt obliczeniowy, ale realizuje inną strategię badawczą. Model jest cyklicznie konfrontowany z nowymi przykładami wygenerowanymi względem jego aktualnego stanu, a zbiór przykładów antagonistycznych narasta. Dzięki temu można analizować odporność modelu w warunkach stopniowego poszerzania przestrzeni zakłóceń.

Cena tej strategii jest wysoka:

- każde `T` i `V` zwiększa zbiór we wszystkich kolejnych iteracjach,
- koszt uczenia ma składnik `I(I+1)/2`,
- generacja jest wykonywana dla wielu ataków,
- walidacja jest wykonywana zarówno na zbiorze mieszanym, jak i na zbiorze antagonistycznym.

Złożoność aktywnego wariantu można więc interpretować jako koszt adaptacyjnego, wieloatakowego curriculum odpornościowego.

### 9.3. Progresywny pasywny wariant

Wariant pasywny jest obliczeniowo najtańszy w samej pętli treningu, ponieważ korzysta z już istniejącego zbioru przykładów antagonistycznych. Jest to praktycznie klasyczne uczenie nadzorowane na zbiorze mieszanym clean+attacked.

Jego ograniczeniem jest brak adaptacyjności w trakcie treningu. Przykłady antagonistyczne zostały wygenerowane wcześniej, dla innego modelu lub wcześniejszego stanu procesu. Jeżeli model pasywny nauczy się odporności na ten konkretny rozkład zakłóceń, nie oznacza to automatycznie odporności na świeżo generowane ataki względem aktualnych parametrów.

Metodologicznie wariant pasywny jest dobrym sposobem na sprawdzenie, ile odporności można uzyskać, traktując przykłady antagonistyczne jako statyczną augmentację danych.

## 10. Podsumowanie

Złożoność badanych wariantów można streścić następująco:

```text
standard on-the-fly:
O(E_s * (N_tr*C_train + N_val*C_val + rho*N_tr*E_a[C_att]))
```

```text
progressive active:
O(E_i * (I*N_tr + A*T*I(I+1)/2) * C_train)
+ O(E_i * (I*N_val + 2*A*V*I(I+1)/2) * C_val)
+ O(Q_att_prog * C_att)
```

```text
progressive passive:
O(E_p * (D_passive_train*C_train + D_passive_val*C_val))
```

Najważniejsze wnioski:

1. Standardowe uczenie antagonistyczne ma liniową złożoność względem liczby epok, ale koszt ataków jest ponoszony w każdej epoce.
2. Progresywne aktywne uczenie ma składnik trójkątny `I(I+1)/2`, ponieważ dane antagonistyczne są kumulowane i wielokrotnie używane w kolejnych iteracjach.
3. Progresywne aktywne uczenie jest najbardziej kosztowne obliczeniowo, szczególnie po uwzględnieniu kosztu ataków iteracyjnych.
4. Progresywne pasywne uczenie jest najtańsze w pętli treningu, ale nie jest samodzielnie darmowe metodologicznie, ponieważ wymaga wcześniejszego wytworzenia zbioru attacked.
5. W analizie eksperymentalnej należy oddzielać koszt treningu klasyfikatora od kosztu generowania przykładów antagonistycznych, ponieważ te dwa składniki skalują się inaczej i odpowiadają na inne pytania badawcze.

