# Progresywne uczenie antagonistyczne dla ImageNette

## Cel i motywacja

Progresywne uczenie antagonistyczne jest wariantem treningu odpornościowego, w którym model nie otrzymuje jednorazowo stałego zbioru przykładów antagonistycznych. Zamiast tego przykłady są generowane iteracyjnie na podstawie aktualnego stanu modelu, a następnie dokładane do rosnącego zbioru treningowego. Dzięki temu model jest regularnie wystawiany na nowe warianty perturbacji, które pozostają skuteczne wobec bieżących wag sieci.

W klasycznym treningu antagonistycznym istnieje ryzyko, że model szybko dopasuje się do ograniczonego, wcześniej wygenerowanego zbioru ataków. Podejście progresywne ogranicza ten problem, ponieważ po każdej rundzie fine-tuningu ataki są generowane ponownie. Zbiór antagonistyczny staje się historią słabości modelu z kolejnych etapów treningu, a nie pojedynczą migawką podatności sprzed uczenia.

Jednym z głównych celów tej metody jest uzyskanie niewrażliwości modelu na subtelne, celowo zaprojektowane zmiany w obrazie. Ataki antagonistyczne często nie polegają na semantycznej zmianie treści obrazu, lecz na wprowadzeniu niewielkich perturbacji, które są słabo zauważalne dla człowieka, ale znacząco zmieniają odpowiedź modelu. Progresywne uczenie antagonistyczne ma stopniowo zmniejszać wpływ takich wyinżynierowanych modyfikacji, ponieważ model jest wielokrotnie trenowany na przykładach, które wcześniej wykorzystywały jego aktualne słabości.

Dodatkową motywacją jest zwiększenie kosztu adaptacji po stronie atakującego. Jeżeli atakujący wie, że model był trenowany antagonistycznie, może próbować odtworzyć podobny trening i przygotować atak przeciwko modelowi o zbliżonej odporności. W przypadku pojedynczego etapu treningu antagonistycznego taka strategia jest relatywnie prostsza do przewidzenia. W przypadku treningu progresywnego atakujący musiałby jednak odtworzyć nie tylko samą metodę ataku, lecz także sekwencję kolejnych iteracji, losowy dobór próbek, skuteczne przykłady zaakceptowane w każdej rundzie oraz zmiany wag modelu po każdej fazie douczania.

Można interpretować to jako problem liczby „refleksji” lub poziomów adaptacji. Model po jednej iteracji jest odporny na pewien zbiór ataków, więc atakujący może dostosować się do tej wersji. Po wielu iteracjach model uwzględnia jednak historię kolejnych prób obejścia odporności. Im dalsza iteracja procesu, tym trudniej założyć, że atakujący odtworzy dokładnie tę samą ścieżkę treningową i przygotuje perturbację skuteczną względem końcowego modelu. Z tego powodu ataki projektowane względem wcześniejszych lub mniej zaadaptowanych wersji modelu powinny stawać się coraz mniej skuteczne.

Metoda może również ograniczać przenaszalność ataków między modelami. Jeżeli model staje się mniej wrażliwy na subtelne, specjalnie zaprojektowane zmiany konkretnych pikseli, to perturbacje wygenerowane dla jednej architektury lub jednej wersji modelu powinny rzadziej zachowywać skuteczność po przeniesieniu na inny model. Innymi słowy, progresywne uczenie antagonistyczne nie tylko wzmacnia odporność wobec ataków generowanych bezpośrednio na dany model, ale może także zmniejszać użyteczność perturbacji jako uniwersalnych lub transferowalnych wzorców błędu.

Implementacja znajduje się w `imagenette_lab/training/imagenette_adversarial_progressive_trainer.py`, a etap eksperymentu uruchamiany jest przez główny program badawczy `experiments/imagenette_full_research/runner.py`.

## Wkład metody

Wprowadzona metoda stanowi aktywne rozszerzenie klasycznego uczenia antagonistycznego. Jej główny wkład polega na tym, że zbiór przykładów antagonistycznych nie jest traktowany jako statyczny zasób przygotowany przed treningiem, lecz jako dynamicznie rozwijana pamięć przypadków, które były skuteczne wobec kolejnych wersji modelu.

Najważniejsze elementy metody:

- **Generowanie ataków względem aktualnego modelu**: w każdej iteracji przykłady antagonistyczne są tworzone na podstawie bieżących wag sieci, a nie względem modelu sprzed rozpoczęcia treningu odpornościowego.
- **Kumulowanie historycznych słabości modelu**: skuteczne przykłady z poprzednich iteracji nie są usuwane. Trafiają do rosnącego zbioru, który reprezentuje historię podatności modelu.
- **Aktywny dobór trudnych przykładów**: do zbioru antagonistycznego trafiają wyłącznie przykłady, które faktycznie zmieniły decyzję modelu. Metoda nie zakłada, że każda perturbacja jest równie wartościowa treningowo.
- **Kontrola kosztu przez cierpliwość Fmax**: metoda ogranicza liczbę kolejnych nieskutecznych prób, dzięki czemu ataki, które przestają znajdować błędy modelu, nie dominują czasu obliczeniowego eksperymentu.
- **Jednoczesne monitorowanie danych czystych i antagonistycznych**: trening odbywa się na zbiorze mieszanym, a walidacja obejmuje zarówno połączony zbiór walidacyjny, jak i skumulowaną część antagonistyczną.
- **Zwiększenie kosztu adaptacji atakującego**: końcowy model jest wynikiem wielu iteracji generowania, selekcji i douczania, więc jego odtworzenie przez atakującego wymagałoby rekonstrukcji całej ścieżki treningowej, a nie tylko znajomości architektury i pojedynczej procedury ataku.
- **Potencjalne ograniczenie przenaszalności ataków**: model uczony na wielu iteracjach subtelnych perturbacji powinien słabiej reagować na specyficzne wzorce pikselowe, co może zmniejszać skuteczność ataków przenoszonych z innych modeli.

Tak zdefiniowany proces można traktować jako formę aktywnego curriculum learning, w którym poziom trudności danych rośnie wraz z modelem. Model sam, poprzez swoje aktualne błędy, współdecyduje o tym, jakie przykłady zostaną dołączone do kolejnych etapów treningu.

## Różnica względem klasycznego uczenia antagonistycznego

Klasyczne uczenie antagonistyczne najczęściej przyjmuje jedną z dwóch form. Pierwsza polega na wcześniejszym wygenerowaniu zbioru przykładów antagonistycznych i późniejszym trenowaniu modelu na tym stałym zbiorze. Druga generuje przykłady on-the-fly w trakcie epok treningowych, ale zwykle nie zachowuje długoterminowej pamięci wygenerowanych wcześniej przykładów.

Proponowane podejście różni się od obu wariantów:

- W przeciwieństwie do treningu na statycznym zbiorze, przykłady są generowane ponownie w każdej iteracji, już po zmianie wag modelu.
- W przeciwieństwie do prostego generowania on-the-fly, skuteczne przykłady są kumulowane i pozostają dostępne w kolejnych iteracjach.
- Zbiór treningowy rośnie progresywnie, zamiast być stałą mieszaniną danych czystych i antagonistycznych.
- Ataki są filtrowane przez skuteczność: do pamięci trafiają tylko te przykłady, które faktycznie spowodowały błędną predykcję.
- Metoda naturalnie tworzy sekwencję coraz trudniejszych etapów, ponieważ ataki są generowane względem coraz bardziej dostrojonego modelu.

W efekcie klasyczne uczenie antagonistyczne odpowiada na pytanie: „jak nauczyć model odporności na znany zbiór perturbacji?”, natomiast progresywne uczenie antagonistyczne odpowiada na pytanie: „jak iteracyjnie identyfikować i zapamiętywać nowe słabości modelu podczas wzmacniania jego odporności?”.

## Hipoteza badawcza

Hipoteza badawcza stojąca za metodą jest następująca:

> Model trenowany na skumulowanym zbiorze przykładów antagonistycznych, generowanych iteracyjnie względem kolejnych stanów własnych wag, osiągnie wyższą odporność na szerokie spektrum ataków niż model trenowany wyłącznie na danych czystych lub model trenowany na statycznym zbiorze przykładów antagonistycznych.

Uzasadnienie tej hipotezy wynika z adaptacyjnego charakteru procedury. Jeżeli model po danej iteracji przestaje być podatny na część wcześniejszych perturbacji, kolejne ataki są generowane już względem jego nowszej wersji. Proces powinien więc przesuwać uwagę treningu z łatwych, wcześniej opanowanych przypadków na te przykłady, które nadal odsłaniają aktualne słabości modelu.

Dodatkowo kumulowanie wcześniejszych skutecznych przykładów zmniejsza ryzyko zapominania odporności uzyskanej w poprzednich iteracjach. Model nie trenuje wyłącznie na najnowszym zestawie ataków, lecz na pełnej historii znalezionych przykładów antagonistycznych.

Drugim elementem hipotezy jest założenie, że skuteczność ataków projektowanych względem wcześniejszych etapów treningu będzie spadać wraz z odległością od końcowej iteracji modelu. Model po wielu rundach powinien być mniej wrażliwy na małe, celowo zaprojektowane zmiany w obrazie, ponieważ takie zmiany były wielokrotnie odkrywane, filtrowane i włączane do zbioru uczącego. Dotyczy to również silnego scenariusza ataku białej skrzynki: nawet jeśli atakujący zna architekturę i aktualne wagi modelu, końcowy model powinien mieć mniejszą podatność na subtelne perturbacje, ponieważ jego proces treningowy był systematycznie wzmacniany przykładami tego typu.

W tym sensie progresywność pełni także funkcję utrudnienia dla atakującego. Im więcej iteracji aktywnego uczenia antagonistycznego, tym więcej etapów adaptacji należałoby odtworzyć, aby przygotować atak odpowiadający końcowej wersji modelu. Prawdopodobieństwo, że atakujący dokładnie powtórzy wieloetapową ścieżkę generowania przykładów, selekcji skutecznych perturbacji i douczania modelu, jest niższe niż w przypadku jednorazowego treningu antagonistycznego.

Trzecim elementem hipotezy jest spadek przenaszalności ataków między modelami. Jeżeli końcowy model jest mniej podatny na lokalne, wyinżynierowane zmiany pikseli, to przykłady antagonistyczne wygenerowane na innym modelu powinny mieć mniejszą szansę wywołania błędnej decyzji. Taki efekt byłby szczególnie istotny w scenariuszach, w których atakujący nie atakuje bezpośrednio końcowego modelu, lecz korzysta z modelu zastępczego i próbuje przenieść perturbację na właściwy system.

## Założenia metody

Metoda opiera się na kilku założeniach dotyczących modelu, danych, ataków oraz kosztu obliczeniowego eksperymentu.

- **Model bazowy powinien mieć sensowną jakość na danych czystych**. Aktywne generowanie przykładów antagonistycznych filtruje próbki błędnie sklasyfikowane przed atakiem. Oznacza to, że metoda zakłada istnienie modelu, który poprawnie rozpoznaje istotną część czystych obrazów i dopiero na tej podstawie można szukać perturbacji zmieniających jego decyzję.

- **Skuteczny przykład antagonistyczny jest wartościowy treningowo**. Jeżeli perturbacja zmienia decyzję modelu dla obrazu poprawnie sklasyfikowanego przed atakiem, to taki przykład ujawnia realną słabość aktualnego modelu. Z tego powodu metoda zapisuje i kumuluje tylko skuteczne przypadki.

- **Słabości modelu zmieniają się po douczaniu**. Po każdej iteracji aktualizowane są wagi modelu, więc wcześniejsze perturbacje mogą przestać być wystarczająco skuteczne. Dlatego nowe ataki są generowane względem bieżącej wersji modelu, a nie wyłącznie raz przed rozpoczęciem treningu.

- **Starsze skuteczne ataki nadal mają wartość**. Nawet jeżeli model po kilku iteracjach staje się odporniejszy, wcześniejsze przykłady antagonistyczne reprezentują historię podatności, której model nie powinien zapomnieć. Z tego powodu zbiór antagonistyczny jest kumulowany, a nie nadpisywany w każdej iteracji.

- **Mieszanina danych czystych i antagonistycznych jest konieczna dla zachowania jakości klasyfikacji**. Celem metody nie jest wyłącznie obniżenie skuteczności ataków, ale także utrzymanie kompetencji modelu na obrazach niezmodyfikowanych. Dlatego trening odbywa się na połączonym zbiorze czystym i antagonistycznym.

- **Subtelne perturbacje można częściowo neutralizować przez ekspozycję iteracyjną**. Metoda zakłada, że wielokrotne uczenie na przykładach zawierających małe, celowo zaprojektowane zmiany pikseli zmniejsza wrażliwość modelu na podobne modyfikacje w przyszłości.

- **Wiele iteracji zwiększa koszt adaptacji atakującego**. Końcowy model jest wynikiem całej ścieżki generowania ataków, selekcji skutecznych przykładów i douczania. Atakujący, który chciałby przygotować równie dopasowany atak, musiałby odtworzyć nie tylko architekturę i wagi, ale także wieloetapowy proces adaptacji.

- **Cierpliwość Fmax jest heurystyką kosztu obliczeniowego**. Długa seria kolejnych nieskutecznych prób sugeruje, że dany atak w aktualnej iteracji przestaje efektywnie znajdować nowe słabości modelu. Cierpliwość Fmax pozwala zakończyć takie generowanie wcześniej i przenieść zasoby obliczeniowe na kolejne ataki lub kolejne etapy.

- **Aktywnie wygenerowany zbiór może mieć wartość pasywną**. Zapisane przykłady antagonistyczne mogą być później użyte do uczenia innych modeli, bez ponownego uruchamiania aktywnego procesu generowania ataków.

## Uzasadnienie nazwy metody

Opisywaną w tym dokumencie fazę można określić jako **aktywne progresywne uczenie antagonistyczne**.

Słowo **aktywne** oznacza, że trudne przykłady są generowane w trakcie eksperymentu przez sam proces treningowy. Model nie korzysta wyłącznie z pasywnego, przygotowanego wcześniej zbioru danych. Zamiast tego jego bieżące decyzje są używane do wyboru obrazów, które zostaną zaatakowane, oraz do oceny, czy wygenerowany przykład jest wartościowy.

Słowo **progresywne** odnosi się do iteracyjnego wzrostu zbioru antagonistycznego. Po każdej rundzie generowania i treningu nowe skuteczne przykłady są dokładane do istniejącego zbioru. Kolejne iteracje nie zaczynają od pustej pamięci, lecz rozwijają poprzednie etapy.

Słowo **antagonistyczne** wskazuje, że przykłady treningowe są tworzone przez mechanizmy działające przeciwko modelowi. Ich celem jest znalezienie takich perturbacji obrazu, które prowadzą do błędnej klasyfikacji, a następnie wykorzystanie tych przypadków do wzmocnienia modelu.

Pełna nazwa oddaje więc trzy kluczowe aspekty: aktywne wyszukiwanie błędów, progresywne gromadzenie trudnych przykładów oraz antagonistyczny charakter generowanych perturbacji.

## Aktywna i pasywna odmiana metody

Progresywne uczenie antagonistyczne można podzielić na dwie powiązane odmiany: **aktywną** oraz **pasywną**. Obie korzystają z tej samej idei wzmacniania odporności modelu przez przykłady antagonistyczne, ale różnią się momentem generowania ataków oraz sposobem wykorzystania powstałego zbioru danych.

### Aktywne progresywne uczenie antagonistyczne

Aktywna odmiana jest głównym przedmiotem tego dokumentu i odpowiada fazie `progressive_active`. W tym wariancie generowanie przykładów antagonistycznych oraz douczanie modelu zachodzą w jednej pętli treningowej.

W każdej iteracji:

1. Aktualna wersja modelu jest używana do wygenerowania nowych skutecznych ataków.
2. Przykłady antagonistyczne są filtrowane przez skuteczność, czyli do zbioru trafiają tylko te, które zmieniły predykcję modelu.
3. Nowe przykłady są dopisywane do skumulowanego zbioru antagonistycznego.
4. Ten sam model jest dalej trenowany na danych czystych oraz na powiększonym zbiorze przykładów antagonistycznych.

Kluczową cechą wariantu aktywnego jest sprzężenie zwrotne: model generuje dane treningowe względem własnego aktualnego stanu, a następnie natychmiast uczy się na przykładach, które ujawniły jego bieżące słabości.

### Pasywne progresywne uczenie antagonistyczne

Pasywna odmiana wykorzystuje zbiór danych zapisany podczas aktywnego progresywnego uczenia antagonistycznego, ale nie generuje nowych ataków w trakcie własnego treningu. W tym wariancie aktywny etap pełni rolę generatora danych, a pasywny etap pełni rolę osobnego treningu na gotowym zbiorze.

Proces wygląda następująco:

1. Najpierw uruchamiany jest wariant aktywny, który generuje i zapisuje skuteczne obrazy antagonistyczne na dysku.
2. Następnie tworzony jest nowy model, który nie był wcześniej trenowany progresywnie.
3. Ten nowy model jest uczony na mieszaninie obrazów czystych oraz zapisanych obrazów antagonistycznych pochodzących z aktywnego etapu.
4. Podczas pasywnego treningu nie zachodzi już aktywne wyszukiwanie nowych słabości bieżącego modelu.

Różnica jest więc zasadnicza: w wariancie aktywnym model sam współtworzy dane, na których będzie się dalej uczył, natomiast w wariancie pasywnym model korzysta z gotowego zbioru przykładów antagonistycznych wygenerowanych wcześniej przez inny proces treningowy.

Pasywna odmiana pozwala sprawdzić, czy zbiór antagonistyczny wygenerowany w aktywnym procesie ma wartość transferowalną jako materiał treningowy dla nowego modelu. Innymi słowy, bada ona, czy historia słabości jednego modelu lub jednej procedury aktywnej może poprawić odporność modelu trenowanego później bez aktywnego generowania ataków.

Dodatkową zaletą wariantu pasywnego jest możliwość ponownego użycia oraz agregacji zapisanych obrazów antagonistycznych bez konieczności każdorazowego uruchamiania pełnego aktywnego procesu douczania. Jeżeli zbiór zaatakowanych obrazów został już dostarczony lub wcześniej wygenerowany, można wykorzystać go jako gotowy zasób treningowy dla innego modelu albo włączyć go do szerszego zbioru danych. Pozwala to oddzielić kosztowny etap aktywnego generowania ataków od późniejszego etapu uczenia modeli na danych czystych i zaatakowanych.

Ma to znaczenie praktyczne w sytuacjach, w których aktywne douczanie całego modelu uwzględniającego dodatkowe dane byłoby zbyt kosztowne obliczeniowo lub czasowo. Wariant pasywny umożliwia wtedy wykorzystanie efektów wcześniejszego aktywnego eksperymentu oraz tego z drugiej porcji danych.

## Zakres eksperymentu

Aktualna konfiguracja eksperymentu jest zdefiniowana w `experiments/imagenette_full_research/config.yaml`.

### Faza pipeline'u

Konfiguracja pozwala uruchamiać wybrane etapy procesu badawczego. Aktualnie w pliku konfiguracyjnym aktywnie uruchamiany jest etap aktywnego progresywnego uczenia antagonistycznego:

```yaml
run:
  phases:
    - progressive_active
```

Dla badań nad progresywnym uczeniem antagonistycznym pełny protokół obejmuje: trening modeli bazowych, walidację modeli bazowych, bezpośrednie ataki na modele bazowe, aktywne progresywne uczenie antagonistyczne, walidację modeli aktywnie progresywnych, bezpośrednie ataki na modele aktywnie progresywne, pasywne uczenie antagonistyczne, walidację modeli pasywnych, bezpośrednie ataki na modele pasywne oraz badanie przenaszalności ataków.

### Modele

Eksperyment jest wykonywany dla architektur:

```yaml
training:
  config_name: advanced
  full_finetune: true
  architectures:
    - resnet18
    - densenet121
    - efficientnet_b0
    - mobilenet_v2
    - swin_t
    - inception_v3
```

Ważne: faza aktywnego uczenia progresywnego nie tworzy świeżych modeli ImageNet od zera. Dla każdej architektury ładowany jest wcześniej wytrenowany model normalny, zgodnie ze wzorcem:

```text
final_research/models/normal/<arch>_advanced.pt
```

Dopiero tak załadowany model jest dalej trenowany progresywnie na mieszaninie danych czystych i antagonistycznych.

### Ataki

Lista ataków jest zdefiniowana w konfiguracji eksperymentu:

```yaml
attacks:
  names:
    - APGD
    - APGDT
    - BIM
    - CW
    - DeepFool
    - DIFGSM
    - EADEN
    - EADL1
    - EOTPGD
    - FAB
    - FFGSM
    - FGSM
    - GN
    - Jitter
    - MIFGSM
    - NIFGSM
    - PGD
    - PGDL2
    - PGDRS
    - PGDRSL2
    - RFGSM
    - SINIFGSM
    - SPSA
    - TIFGSM
    - TPGD
    - UPGD
    - VMIFGSM
    - VNIFGSM
    - OnePixel
    - Pixle
    - Square
```

Dla każdego ataku trainer próbuje wygenerować określoną liczbę skutecznych przykładów antagonistycznych w każdej iteracji.

### Parametry progresywnego treningu

Sekcja `progressive` steruje główną dynamiką eksperymentu:

```yaml
progressive:
  learning_rate: 0.0001
  iterations: 30
  epochs_per_iteration: 50
  batch_size: 32
  images_per_attack_per_iteration: 10
  validation_images_per_attack_per_iteration: 2
  max_tries_per_attack: 50
  early_stopping_patience: 7
  scheduler_type: step
  weight_decay: 0.0001
  save_generated_images: true
```

Znaczenie parametrów:

- `learning_rate`: początkowy współczynnik uczenia używany w każdej iteracji progresywnej. W aktualnej konfiguracji wynosi `0.0001`.
- `iterations`: liczba rund progresywnego generowania i trenowania.
- `epochs_per_iteration`: maksymalna liczba epok fine-tuningu wykonywana po wygenerowaniu nowych przykładów w danej iteracji.
- `batch_size`: rozmiar porcji danych dla treningu i walidacji na połączonym zbiorze.
- `images_per_attack_per_iteration`: docelowa liczba skutecznych przykładów antagonistycznych generowanych dla każdego ataku na część treningową w jednej iteracji.
- `validation_images_per_attack_per_iteration`: docelowa liczba skutecznych przykładów antagonistycznych generowanych dla każdego ataku na część walidacyjną w jednej iteracji.
- Cierpliwość Fmax (`max_tries_per_attack`): limit kolejnych nieudanych prób dla danego ataku. Jeżeli w ostatnich `X` próbach nie uda się wygenerować skutecznego przykładu, generowanie dla tego ataku jest przerywane.
- `early_stopping_patience`: liczba epok bez poprawy, po której trening w bieżącej iteracji może zostać zatrzymany.
- `scheduler_type`: harmonogram współczynnika uczenia, w tym przypadku `step`.
- `weight_decay`: regularyzacja L2 w optymalizatorze.
- `save_generated_images`: zapisuje wygenerowane obrazy antagonistyczne do katalogu eksperymentu.

Parametry uczenia aktywnego są więc ustawione konserwatywnie: model jest dalej dostrajany z niskim początkowym współczynnikiem uczenia `0.0001` oraz regularyzacją wag `0.0001`. Ma to ograniczać ryzyko gwałtownego zniszczenia reprezentacji wyuczonych podczas standardowego treningu na czystych danych.

Harmonogram typu `step` zmniejsza współczynnik uczenia w trakcie epok danej iteracji. Domyślnie co 5 epok wartość jest mnożona przez `0.8`. Przy `epochs_per_iteration: 50` oznacza to, że w obrębie jednej iteracji trening zaczyna się od `0.0001`, po 5 epokach przechodzi do `0.00008`, po kolejnych 5 epokach do `0.000064` itd.

Ważne jest to, że optymalizator i harmonogram są tworzone od nowa na początku każdej iteracji progresywnej. Wagi modelu nie są resetowane, ale współczynnik uczenia w kolejnej iteracji ponownie startuje od wartości `0.0001` i dopiero w trakcie tej iteracji maleje według tego samego harmonogramu. Dzięki temu każda nowa runda, po dodaniu kolejnych przykładów antagonistycznych, rozpoczyna dostrajanie z tą samą początkową intensywnością uczenia.

## Metoda działania

### 1. Inicjalizacja modeli

Runner dla każdej architektury:

1. Buduje ścieżkę do normalnie wytrenowanego modelu: `final_research/models/normal/<arch>_advanced.pt`.
2. Ładuje model przez `load_model_imagenette(...)`.
3. Przekazuje załadowany model do `ImageNetteAdversarialProgressiveTrainer`.
4. Przygotowuje ścieżkę zapisu wyniku progresywnego: `final_research/models/progressive/active/<arch>_progressive_adv.pt`.

W praktyce oznacza to, że progresywny etap jest kontynuacją normalnego treningu, a nie treningiem od bazowych wag ImageNet.

### 2. Generowanie przykładów antagonistycznych

Na początku każdej iteracji procedura korzysta z dwóch rodzajów odczytu danych ImageNette:

- losowego odczytu pojedynczych obrazów do generowania nowych ataków,
- stabilnego odczytu czystych danych do budowania części treningowej i walidacyjnej.

Oznacza to, że wybór oraz kolejność czystych obrazów używanych do ataku są losowane przez mechanizm odczytu danych. Procedura nie atakuje deterministycznie pierwszych `N` obrazów z katalogu. Dla każdego ataku przechodzi po czystych obrazach w losowej kolejności i zbiera skuteczne przykłady do momentu osiągnięcia założonego limitu albo przerwania przez cierpliwość Fmax. W praktyce dwa uruchomienia tego samego eksperymentu mogą wygenerować inny zestaw przykładów antagonistycznych, jeżeli nie ustawiono jawnie ziaren losowości dla bibliotek używanych przez trening, odczyt danych i same ataki.

Losowość dotyczy przede wszystkim doboru kandydatów do ataku i kolejności ich przetwarzania. Dodatkowo część algorytmów ataku posiada własny komponent stochastyczny, więc nawet dla tego samego obrazu wynik perturbacji może zależeć od stanu generatorów losowych.

Dla każdego ataku wykonywana jest następująca procedura:

1. Dla bieżącego ataku tworzona jest instancja przez `AttackFactory.get_attack(attack_name, model)`.
2. Loader generacyjny zwraca czysty obraz i etykietę.
3. Aktualny model klasyfikuje czysty obraz.
4. Jeżeli model myli się na czystym obrazie, próbka jest odrzucana i nie jest używana do generowania przykładu antagonistycznego.
5. Dla poprawnie sklasyfikowanego obrazu generowany jest przykład antagonistyczny.
6. Wynik ataku jest przycinany do zakresu `[0, 1]` przez `normalize_adversarial_image(...)`.
7. Model klasyfikuje wygenerowany obraz antagonistyczny.
8. Przykład jest uznany za skuteczny, jeśli predykcja modelu po ataku różni się od etykiety.
9. Skuteczne przykłady są zapisywane w pamięci jako `(adv_image, label)`.
10. Dla skutecznych przykładów zapisywany jest także `AttackResult`, który służy do obliczania średnich metryk odległości perturbacji.
11. Jeżeli `save_generated_images: true`, obraz jest także zapisywany na dysku jako PNG.

Mechanizm cierpliwości Fmax działa jako limit kolejnych porażek. Każdy skuteczny przykład zeruje licznik nieudanych prób, a każda nieskuteczna próba zwiększa licznik `failed_streak`. Jeżeli licznik osiągnie wartość z konfiguracji, generowanie dla danego ataku zostaje zakończone.

Warto podkreślić, że cierpliwość Fmax nie jest limitem całkowitej liczby prób. Jest to limit kolejnych nieudanych prób. Jeżeli atak regularnie znajduje skuteczne przykłady, licznik jest resetowany po każdym sukcesie i proces może trwać dłużej. Jeżeli przez dłuższą serię kandydatów nie udaje się znaleźć skutecznego przykładu, generowanie dla tego ataku kończy się wcześniej.

### 3. Kumulowanie danych

Trainer utrzymuje dwa rosnące zbiory:

- `progressive_train_dataset`: skuteczne przykłady antagonistyczne dla treningu,
- `progressive_test_dataset`: skuteczne przykłady antagonistyczne dla walidacji.

Po każdej iteracji nowe przykłady są dokładane do istniejących list. Następnie tworzony jest połączony zbiór:

```text
combined_train_dataset = clean_train_dataset + progressive_train_dataset
combined_val_dataset   = clean_val_dataset   + progressive_test_dataset
```

W efekcie model w kolejnych iteracjach trenuje na coraz większym zbiorze, który zawiera zarówno czyste dane, jak i wszystkie skuteczne przykłady antagonistyczne wygenerowane w poprzednich rundach.

Nowo wygenerowane przykłady nie zastępują starszych przykładów antagonistycznych. Są do nich dokładane. Oznacza to, że model w iteracji `k` widzi czyste dane oraz sumę skutecznych ataków wygenerowanych w iteracjach `1..k`. Zbiór antagonistyczny pełni więc rolę pamięci historycznych podatności modelu.

Czysty zbiór danych jest dołączany w całości, natomiast część antagonistyczna jest skumulowaną listą obrazów wygenerowanych aktywnie podczas treningu. Przy każdej iteracji tworzony jest nowy połączony zbiór danych. Odczyt treningowy dla tego zbioru jest losowany, więc kolejność porcji danych w trakcie dostrajania również nie jest stała.

### 4. Trening w iteracji

Po przygotowaniu zbioru danych wywoływana jest metoda `Training.train_imagenette_adversarial_progressive(...)`. Dla każdej iteracji:

1. Tworzony jest nowy optymalizator `Adam`.
2. Tworzony jest nowy harmonogram współczynnika uczenia zgodny z konfiguracją.
3. Model jest trenowany przez maksymalnie `epochs_per_iteration` epok.
4. Walidacja odbywa się na połączonym zbiorze czystym i antagonistycznym.
5. Osobno monitorowana jest walidacja na skumulowanym zbiorze antagonistycznym.
6. Najlepszy model danej iteracji jest zapisywany z sufiksem iteracji.

Wagi modelu nie są resetowane pomiędzy iteracjami. Każda kolejna iteracja kontynuuje dostrajanie modelu po poprzedniej iteracji. Resetowany jest natomiast optymalizator i harmonogram współczynnika uczenia, dlatego każda iteracja rozpoczyna uczenie od początkowej wartości `learning_rate`, a następnie zmniejsza ją w trakcie epok zgodnie z harmonogramem `step`.

### 5. Zapisywane artefakty

W typowym uruchomieniu program zapisuje:

- progresywne modele: `final_research/models/progressive/active/`,
- wygenerowane obrazy antagonistyczne: `final_research/data/attacks/progressive/active/`,
- logi TensorBoard pod wspólnym katalogiem `final_research/runs/`.

Jeżeli `save_generated_images` jest włączone, obrazy są zapisywane w strukturze:

```text
<attacked_images_folder>/<train|test>/<model_progressive_adv>/<attack>/<label>/progressive_iter<it>_<timestamp>.png
```

## Protokół ewaluacji

Protokół ewaluacji obejmuje pełną sekwencję etapów badawczych, nawet jeżeli część z nich jest w danym uruchomieniu zakomentowana w konfiguracji. Zakomentowanie etapu oznacza jedynie, że nie jest wykonywany w bieżącym przebiegu programu. Nie oznacza natomiast, że etap nie należy do protokołu porównawczego.

Pełny protokół dla badań nad progresywnym uczeniem antagonistycznym obejmuje:

1. Trening modeli bazowych na czystych danych (`train_baseline`).
2. Walidację modeli bazowych na czystych danych (`validate_normal`).
3. Ewaluację modeli bazowych przez bezpośrednie ataki (`direct_normal`).
4. Aktywne progresywne uczenie antagonistyczne (`progressive_active`).
5. Walidację modeli aktywnie progresywnych na czystych danych (`validate_progressive_active`).
6. Ewaluację modeli aktywnie progresywnych przez bezpośrednie ataki (`direct_progressive_active`).
7. Pasywne uczenie antagonistyczne na zbiorze wygenerowanym aktywnie (`passive`).
8. Walidację modeli pasywnie progresywnych na czystych danych (`validate_passive`).
9. Ewaluację modeli pasywnie progresywnych przez bezpośrednie ataki (`direct_passive`).
10. Badanie przenaszalności ataków dla wszystkich grup modeli (`transferability`).

Nazwy w nawiasach są roboczymi identyfikatorami etapów w konfiguracji i kodzie. W opisie naukowym należy posługiwać się nazwami merytorycznymi: model bazowy, model aktywnie progresywny, model pasywnie progresywny, walidacja na danych czystych, bezpośrednia ewaluacja antagonistyczna oraz badanie przenaszalności.

### 1. Punkt odniesienia: modele normalne

Pierwszym punktem odniesienia są modele trenowane standardowo na czystym zbiorze ImageNette. Etap treningu bazowego (`train_baseline`) zapisuje wytrenowane modele do:

```text
final_research/models/normal/
```

Nazwy zapisanych modeli mają format zależny od architektury i wariantu treningu:

```text
<architektura>_<wariant_treningu>.pt
```

W obecnej konfiguracji wariant treningu ma wartość `advanced`, a lista architektur obejmuje `resnet18`, `densenet121`, `efficientnet_b0`, `mobilenet_v2`, `swin_t` oraz `inception_v3`. Faza aktywna nie tworzy nowych modeli od zera, lecz wczytuje właśnie te normalnie wytrenowane modele. Oznacza to, że ewaluacja progresywna jest porównaniem modelu przed i po aktywnym douczaniu antagonistycznym.

### 2. Walidacja na czystych danych

Walidacja czysta mierzy, czy wzmacnianie odporności nie niszczy podstawowej jakości klasyfikacji obrazów bez perturbacji.

Dla modeli normalnych etap walidacji (`validate_normal`):

1. Wczytuje wszystkie zapisane modele z `final_research/models/normal/`.
2. Rozpoznaje architekturę na podstawie nazwy pliku.
3. Waliduje modele na czystym zbiorze walidacyjnym ImageNette.
4. Zapisuje zbiorcze wyniki do:

```text
final_research/results/normal/clean_validation_summary.csv
```

5. Dodatkowo zapisuje wyniki per klasa dla każdego modelu.

Dla modeli po aktywnym progresywnym uczeniu antagonistycznym etap walidacji progresywnej (`validate_progressive_active`) wykonuje analogiczną ocenę modeli z:

```text
final_research/models/progressive/active/
```

Wyniki trafiają do:

```text
final_research/results/progressive/active/clean_validation_summary.csv
```

Dla modeli trenowanych pasywnie etap walidacji pasywnej (`validate_passive`) ocenia modele z:

```text
final_research/models/progressive/passive/
```

Wyniki są zapisywane jako:

```text
final_research/results/progressive/passive/clean_validation_passive_summary.csv
```

Porównanie tych trzech plików odpowiada na pytanie, jak zmienia się jakość klasyfikacji czystych obrazów pomiędzy modelem normalnym, aktywnie douczonym modelem progresywnym oraz modelem pasywnym trenowanym na zapisanych przykładach antagonistycznych.

### 3. Ewaluacja odporności przez bezpośrednie ataki

Odporność modeli jest oceniana przez ponowne wykonanie tego samego zestawu ataków na zbiorze testowym ImageNette. Lista ataków jest wspólna dla wszystkich porównywanych wariantów, co pozwala zestawiać wyniki między modelami bez zmiany protokołu ataku.

Dla modeli normalnych etap bezpośrednich ataków (`direct_normal`):

1. Ładuje testową część zbioru ImageNette.
2. Dla każdej architektury wskazuje wytrenowany model z `final_research/models/normal/`.
3. Uruchamia bezpośrednie ataki antagonistyczne.
4. Zapisuje wyniki CSV do:

```text
final_research/results/attacks/normal/
```

Dla modeli aktywnie progresywnych etap bezpośrednich ataków (`direct_progressive_active`):

1. Ładuje całą testową część zbioru ImageNette.
2. Wczytuje modele z `final_research/models/progressive/active/` w formacie `<architektura>_progressive_adv.pt`.
3. Uruchamia ten sam zestaw ataków.
4. Zapisuje wyniki do:

```text
final_research/results/attacks/progressive/active/
```

Dla modeli pasywnych etap bezpośrednich ataków (`direct_passive`):

1. Ładuje całą testową część zbioru ImageNette.
2. Wczytuje modele z `final_research/models/progressive/passive/` w formacie `<architektura>_adv_passive.pt`.
3. Uruchamia ten sam zestaw ataków.
4. Zapisuje wyniki do:

```text
final_research/results/attacks/progressive/passive/
```

Najważniejsze porównanie odporności polega na zestawieniu wyników z katalogów `normal`, `progressive/active` oraz `progressive/passive` dla tej samej architektury i tego samego ataku. W szczególności należy analizować spadek skuteczności ataków, zmianę `AD` i `RAD`, metryki jakości po ataku oraz odległości perturbacji `L0_pixels`, `L1`, `L2` i `Linf`.

### 4. Ewaluacja aktywnego treningu progresywnego

Faza aktywnego progresywnego uczenia antagonistycznego (`progressive_active`) jest jednocześnie etapem treningu i źródłem części danych ewaluacyjnych. Dla każdej architektury program:

1. Wczytuje normalnie wytrenowany model z `final_research/models/normal/`.
2. Generuje skuteczne przykłady antagonistyczne względem aktualnego modelu.
3. Zapisuje wygenerowane obrazy, jeżeli `save_generated_images` jest włączone.
4. Trenuje model na mieszaninie danych czystych i skumulowanych danych antagonistycznych.
5. Zapisuje końcowy model do:

```text
final_research/models/progressive/active/
```

Jeżeli zapisywanie obrazów jest aktywne, przykłady antagonistyczne powstające podczas treningu są zapisywane w:

```text
final_research/data/attacks/progressive/active/
```

Ten katalog pełni podwójną rolę. Po pierwsze dokumentuje, jakie przykłady zostały uznane za skuteczne w aktywnym procesie. Po drugie stanowi wejście dla pasywnego uczenia antagonistycznego.

### 5. Ewaluacja pasywnej odmiany metody

Faza pasywnego uczenia antagonistycznego (`passive`) sprawdza, czy zbiór wygenerowany aktywnie ma wartość treningową dla nowego modelu, który nie uczestniczył w aktywnym procesie generowania ataków. Program tworzy nowy model danej architektury, konfiguruje go do pełnego dostrajania i trenuje na obrazach czystych oraz wcześniej zaatakowanych obrazach z:

```text
final_research/data/attacks/progressive/active/
```

Modele pasywne są zapisywane do:

```text
final_research/models/progressive/passive/
```

Ten etap pozwala oddzielić dwie hipotezy:

1. Czy aktywne douczanie poprawia odporność konkretnego modelu, który sam generował swoje trudne przykłady.
2. Czy zbiór przykładów wygenerowany aktywnie jest użyteczny również jako gotowy materiał treningowy dla innego modelu.

### 6. Ewaluacja przenaszalności ataków

Badanie przenaszalności ataków (`transferability`) sprawdza, czy zapisane przykłady antagonistyczne zachowują skuteczność po przeniesieniu na inne modele. W pełnym protokole ten etap powinien obejmować wszystkie trzy grupy modeli:

1. Modele bazowe trenowane wyłącznie na danych czystych.
2. Modele po aktywnym progresywnym uczeniu antagonistycznym.
3. Modele po pasywnym uczeniu antagonistycznym.

Oznacza to, że przenaszalność należy analizować zarówno dla ataków wygenerowanych względem modeli normalnych, jak i względem modeli aktywnie oraz pasywnie progresywnych. Każda grupa pełni wtedy rolę potencjalnego źródła perturbacji, a pozostałe modele mogą pełnić rolę modeli docelowych.

Ten etap jest szczególnie istotny dla hipotezy o ograniczeniu przenaszalności ataków. Jeżeli progresywne uczenie antagonistyczne zmniejsza wrażliwość modeli na subtelne, wyinżynierowane perturbacje, to skuteczność przykładów wygenerowanych na modelu źródłowym powinna spadać po przeniesieniu ich na modele docelowe. Najważniejsze jest porównanie, czy ataki skuteczne wobec modeli bazowych tracą skuteczność wobec modeli aktywnie lub pasywnie progresywnych oraz czy ataki wygenerowane wobec modeli progresywnych są mniej uniwersalne między architekturami.

### 7. Zalecany sposób porównywania wyników

Minimalny protokół porównawczy powinien obejmować trzy grupy modeli:

1. **Modele normalne**: modele trenowane standardowo na danych czystych.
2. **Modele aktywnie progresywne**: te same architektury po aktywnym progresywnym uczeniu antagonistycznym.
3. **Modele pasywnie progresywne**: nowe modele trenowane pasywnie na danych czystych i przykładach wygenerowanych przez aktywną fazę.

Dla każdej architektury należy porównać:

1. Wyniki walidacji na czystych danych, aby ocenić koszt odporności na obrazach bez perturbacji.
2. Wyniki bezpośrednich ataków, aby ocenić odporność na ten sam zestaw metod antagonistycznych.
3. Wyniki przenaszalności, aby sprawdzić, czy perturbacje zachowują skuteczność między modelami.

Interpretacja powinna uwzględniać kompromis między odpornością i jakością klasyfikacji czystych obrazów. Najsilniejszy wynik metody występuje wtedy, gdy model progresywny obniża skuteczność ataków oraz przenaszalność perturbacji, a jednocześnie utrzymuje akceptowalną dokładność na czystym zbiorze walidacyjnym.

## Pseudokod algorytmu

### Aktywne progresywne uczenie antagonistyczne

Poniższy pseudokod przedstawia aktywną odmianę progresywnego uczenia antagonistycznego, czyli wariant używany w fazie `progressive_active`.

```text
Wejście:
    A              = lista architektur modeli
    T              = lista ataków antagonistycznych
    I              = liczba iteracji progresywnych
    E              = liczba epok treningu w jednej iteracji
    K_train        = liczba skutecznych obrazów antagonistycznych
                     generowanych na atak dla zbioru treningowego
    K_val          = liczba skutecznych obrazów antagonistycznych
                     generowanych na atak dla zbioru walidacyjnego
    Fmax           = maksymalna liczba kolejnych nieudanych prób
    D_train_clean  = czysty zbiór treningowy
    D_val_clean    = czysty zbiór walidacyjny

Dla każdej architektury a w A:
    model <- wczytaj model a wytrenowany wcześniej na danych czystych

    D_train_adv <- pusty zbiór przykładów antagonistycznych
    D_val_adv   <- pusty zbiór przykładów antagonistycznych

    Dla iteracji i = 1..I:
        D_train_new <- pusty zbiór nowych przykładów treningowych
        D_val_new   <- pusty zbiór nowych przykładów walidacyjnych

        Dla każdego ataku t w T:
            attack <- utwórz atak t dla aktualnego modelu

            train_successes <- 0
            train_failed_streak <- 0

            Dopóki train_successes < K_train
                  oraz train_failed_streak < Fmax:

                (x, y) <- pobierz kolejny losowy czysty obraz z D_train_clean

                Jeżeli model(x) != y:
                    pomiń x
                    kontynuuj

                x_adv <- attack(x, y)
                x_adv <- przytnij x_adv do zakresu [0, 1]

                Jeżeli model(x_adv) != y:
                    dodaj (x_adv, y) do D_train_new
                    train_successes <- train_successes + 1
                    train_failed_streak <- 0
                W przeciwnym razie:
                    train_failed_streak <- train_failed_streak + 1

            val_successes <- 0
            val_failed_streak <- 0

            Dopóki val_successes < K_val
                  oraz val_failed_streak < Fmax:

                (x, y) <- pobierz kolejny losowy czysty obraz z D_val_clean

                Jeżeli model(x) != y:
                    pomiń x
                    kontynuuj

                x_adv <- attack(x, y)
                x_adv <- przytnij x_adv do zakresu [0, 1]

                Jeżeli model(x_adv) != y:
                    dodaj (x_adv, y) do D_val_new
                    val_successes <- val_successes + 1
                    val_failed_streak <- 0
                W przeciwnym razie:
                    val_failed_streak <- val_failed_streak + 1

        D_train_adv <- D_train_adv ∪ D_train_new
        D_val_adv   <- D_val_adv ∪ D_val_new

        D_train_combined <- D_train_clean ∪ D_train_adv
        D_val_combined   <- D_val_clean ∪ D_val_adv

        Trenuj model przez maksymalnie E epok na D_train_combined
        Waliduj model na D_val_combined oraz D_val_adv
        Zapisz najlepszy model z bieżącej iteracji

    Zapisz wynik końcowy modelu

Wyjście:
    wytrenowane modele progresywne,
    skumulowane zbiory przykładów antagonistycznych,
    metryki treningowe i walidacyjne.
```

Najważniejszą własnością algorytmu jest to, że `D_train_adv` i `D_val_adv` nie są resetowane między iteracjami. Są one rozszerzane o nowe skuteczne przykłady, dzięki czemu model trenuje na historii przypadków, które w kolejnych etapach okazały się dla niego problematyczne.

### Pasywne progresywne uczenie antagonistyczne

Pasywna odmiana korzysta z obrazów antagonistycznych zapisanych wcześniej przez aktywne progresywne uczenie antagonistyczne. W tym wariancie nie powstają nowe ataki względem aktualnie trenowanego modelu. Model uczy się na gotowej mieszaninie danych czystych oraz zaatakowanych, a koszt aktywnego generowania przykładów jest ponoszony wcześniej, w osobnej fazie.

```text
Wejście:
    A                  = lista architektur modeli
    D_train_clean      = czysty zbiór treningowy
    D_val_clean        = czysty zbiór walidacyjny
    D_train_adv_saved  = zapisany treningowy zbiór przykładów antagonistycznych
                         wygenerowany w aktywnej fazie progresywnej
    D_val_adv_saved    = zapisany walidacyjny zbiór przykładów antagonistycznych
                         wygenerowany w aktywnej fazie progresywnej
    E                  = liczba epok treningu pasywnego
    B                  = rozmiar batcha
    P                  = parametry treningu pasywnego
                         learning_rate, scheduler, weight_decay,
                         early_stopping_patience, gradient_clip_norm

Dla każdej architektury a w A:
    model <- utwórz nowy model a
    model <- wczytaj wagi startowe dla standardowego transfer learningu

    D_train_passive <- D_train_clean ∪ D_train_adv_saved
    D_val_passive   <- D_val_clean ∪ D_val_adv_saved

    Jeżeli konfiguracja wymaga zbalansowania udziału danych czystych:
        D_train_passive <- zwiększ udział danych czystych względem
                           liczby przykładów antagonistycznych

    train_loader <- utwórz odczyt treningowy z D_train_passive
                    z porcją danych o rozmiarze B
    val_loader   <- utwórz odczyt walidacyjny z D_val_passive
                    z porcją danych o rozmiarze B

    Trenuj model przez maksymalnie E epok na danych treningowych
    Po każdej epoce waliduj model na danych walidacyjnych

    Jeżeli metryka walidacyjna poprawia się:
        zapisz najlepszy model

    Jeżeli przez określoną liczbę epok nie ma poprawy:
        zatrzymaj trening wcześniej

    Zapisz końcowe metryki treningu pasywnego

Wyjście:
    modele wytrenowane pasywnie na danych czystych i zapisanych
    przykładach antagonistycznych,
    metryki walidacyjne,
    zapisane modele pasywne.
```

Najważniejszą różnicą względem wariantu aktywnego jest brak sprzężenia zwrotnego pomiędzy trenowanym modelem i generatorem ataków. Pasywny model nie wpływa na to, jakie przykłady antagonistyczne znajdą się w zbiorze. Korzysta z gotowej pamięci przykładów wygenerowanych wcześniej, dzięki czemu można badać, czy aktywnie zebrany zbiór ma wartość treningową również dla nowych modeli.

## Diagram przepływu

```mermaid
flowchart TD
    A[Start fazy progressive_active] --> B[Wczytaj config.yaml]
    B --> C[Odczytaj architektury, ataki i parametry progressive]
    C --> D[Załaduj modele wytrenowane normalnie]
    D --> E[Utwórz ImageNetteAdversarialProgressiveTrainer]
    E --> F{Dla każdego modelu}

    F --> G[Iteracja progresywna i = 1..N]
    G --> H[Utwórz odczyt danych czystych i generacyjnych]
    H --> I[Losowo iteruj po czystych obrazach]
    I --> J{Dla każdego ataku}

    J --> K[Odrzuć obrazy błędne na czysto]
    K --> L[Wygeneruj obrazy antagonistyczne]
    L --> M{Czy atak zmienił predykcję?}
    M -- Tak --> N[Dodaj przykład do progressive dataset]
    N --> O[Wyzeruj failed_streak]
    M -- Nie --> P[Zwiększ failed_streak]
    P --> Q{failed_streak >= Fmax?}
    Q -- Tak --> R[Zakończ generowanie dla tego ataku]
    Q -- Nie --> I
    O --> S{Osiągnięto images_per_attack_per_iteration?}
    S -- Nie --> I
    S -- Tak --> R

    R --> T[Połącz clean dataset z progressive dataset]
    T --> U[Trenuj model przez epochs_per_iteration]
    U --> V[Waliduj na combined val oraz zbiorze antagonistycznym]
    V --> W[Zapisz najlepszy model iteracji]
    W --> X{Czy są kolejne iteracje?}
    X -- Tak --> G
    X -- Nie --> Y[Zapisz końcowy wynik modelu]
    Y --> Z{Czy są kolejne modele?}
    Z -- Tak --> F
    Z -- Nie --> AA[Koniec fazy progressive_active]
```

## Interpretacja metody

Najważniejszą cechą tej procedury jest sprzężenie zwrotne pomiędzy modelem i generatorem ataków. Model po każdej iteracji zmienia swoje granice decyzyjne, więc kolejne ataki są generowane względem nowszej, potencjalnie odporniejszej wersji modelu. Jeżeli atak nadal znajduje skuteczne perturbacje, przykłady trafiają do zbioru treningowego. Jeżeli przez dłuższą serię prób atak nie znajduje skutecznego przykładu, mechanizm cierpliwości Fmax ogranicza koszt obliczeniowy i przechodzi dalej.

Metoda działa więc jak aktywne wzmacnianie odporności: model jest uczony na czystych danych oraz na stale rozszerzanej pamięci przykładów, które w przeszłości były dla niego trudne. Dzięki temu końcowy model powinien zachować kompetencję klasyfikacji czystych obrazów, a jednocześnie poprawiać odporność na szeroką rodzinę ataków antagonistycznych.

## Ograniczenia i założenia

- Generowane są wyłącznie przykłady z obrazów poprawnie sklasyfikowanych przed atakiem.
- Obrazy kandydackie do ataku są pobierane w losowej kolejności, więc bez kontrolowanych ziaren losowości dobór i kolejność próbek nie są deterministyczne.
- Zbiór antagonistyczny rośnie w pamięci procesu, dlatego koszt pamięci zwiększa się wraz z liczbą iteracji i liczbą ataków.
- Cierpliwość Fmax jest heurystyką kosztu obliczeniowego: mniejsza wartość przyspiesza eksperyment, ale może zmniejszyć liczbę znalezionych skutecznych przykładów.
- Skuteczność treningu zależy od różnorodności ataków wybranych w konfiguracji; zbyt wąski zestaw ataków może prowadzić do odporności wyspecjalizowanej tylko pod konkretne metody.
- Każda iteracja tworzy nowy optymalizator i scheduler, ale kontynuuje trening tych samych wag modelu.
