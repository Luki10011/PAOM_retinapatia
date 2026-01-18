# Analiza dostępnych wyników konkursu APTOS 2019 Blindness Detection

## 1. Wstęp: kontekst i cel konkursu

APTOS 2019 Blindness Detection to konkurs zorganizowany na platformie Kaggle. Celem tego konkursu było stworzenie modelu AI zdolnego do wykrywania i oceny stopnia zaawansowania retinopatii cukrzycowej na podstawie zdjęć dna oka. Retinopatia jest powikłaniem cukrzycy i jedną z głównych przyczyn ślepoty w wieku produkcyjnym.

### Zadanie
Uczestnicy musieli stworzyć model, który na podstawie zdjęcia dna oka przypisze pacjentowi jedną z pięciu ocen w skali porządkowej:
* `0` - Brak retinopatii (No DR)
* `1` - Łagodna (Mild)
* `2` - Umiarkowana (Moderate)
* `3` - Ciężka (Severe)
* `4` - Proliferacyjna (Proliferative DR)

---

## 2. Metryka oceny

### Metryka oceny: Quadratic Weighted Kappa (QWK)
Modele były oceniane przy użyciu Kwadratowej Ważonej Kappy (QWK).
* Metryka ta mierzy zgodność między dwoma ocenami (lekarza i modelu).
    * Wartość 1 oznacza pełną zgodność.
    * Wartość 0 oznacza zgodność losową.
* Przez zastosowanie kwadratu w metryce duże pomyłki są karane nieproporcjonalnie surowiej.
    * Pomylenie klasy 0 z 1 to mały błąd.
    * Pomylenie klasy 0 z 4 to ogromny błąd, który drastycznie obniża wynik.
* To wymusiło na uczestnikach traktowanie problemu nie jako prostej klasyfikacji (gdzie każdy błąd waży tyle samo), ale jako problemu regresji lub klasyfikacji porządkowej .

---

## 3. Analiza rozwiązań: architektura i modele

W konkursie zauważalna jest dominacja rodziny modeli EfficientNet, choć czołowe miejsca często korzystały z zespołów (ensembling) różnych architektur.

* EfficientNet (B3-B7): Używany przez większość topowych zespołów (2, 4, 5, 8, 9, 12, 13 miejsce). Modele te oferowały świetny stosunek wydajności do liczby parametrów.
* SE-ResNeXt50/101: Bardzo popularny model wspomagający, często łączony z EfficientNetami w celu zwiększenia różnorodności predykcji (1, 7, 9, 11, 13 miejsce).
* InceptionV4 / InceptionResNetV2: Wykorzystane przez zwycięzcę oraz 7 miejsce.

---

## 4. Strategia dotycząca danych

Zbiór danych z 2019 roku był relatywnie mały, co wymusiło na uczestnikach kreatywne podejście.

### Wykorzystanie zewnętrznych danych (2015 Dataset)
Prawie każde czołowe rozwiązanie korzystało ze starego zbioru danych z podobnego konkursu Kaggle z 2015 roku.
* Pre-training: Trenowanie modelu najpierw na danych z 2015 r., a potem fine-tuning na danych z 2019 r. (2, 4, 7, 12 miejsca).
* Łączenie zbiorów: Niektórzy (1, 5, 8 miejsce) po prostu połączyli dane z 2015 i 2019 roku w jeden duży zbiór treningowy, aby zwiększyć generalizację modelu.

### Pseudo-Labeling
Wielu uczestników z czołówki (1, 2, 5, 7, 8, 11, 13 miejsce) użyło techniki Pseudo-Labeling.
1.  Model trenowano na dostępnych danych treningowych.
2.  Używano go do przewidzenia etykiet dla zbioru testowego.
3.  Najpewniejsze predykcje dodawano do zbioru treningowego jako "nowe" dane.
4.  Model trenowano ponownie na powiększonym zbiorze.
Pozwoliło to modelom "dostroić się" do specyfiki zdjęć testowych, które mogły różnić się od treningowych.

---

## 5. Przetwarzanie wstępne zdjęć:

Wyróżniły się dwa główne podejścia do obróbki zdjęć:

### "Ben's Preprocessing"
Metoda spopularyzowana przez Benjamina Grahama. Stosowana przez 4, 9, 11 i 13 miejsce. Ta metoda polegała na:
1.  Wykadrowaniu zdjęcia tak, aby usunąć czarne tło..
2.  Przeskalowaniu obrazu, aby oko zajmowało większość kadru (często "Circular Crop").
3.  Korekcji kolorów (rozmycie Gaussa dodane do obrazu w celu ujednolicenia oświetlenia i wydobycia detali naczyń krwionośnych.).  

### Minimalizm
Niektórzy (w tym zwycięzca) uznali, że nowoczesne sieci neuronowe poradzą sobie bez skomplikowanego preprocessingu. Ograniczyli się do zwykłego zmiany rozmiaru i prostego przycięcia. Zwycięzca argumentował, że "jakość obrazów jest wystarczająca dla głębokich sieci".

---

## 6. Trening i funkcje straty

### Regresja zamiast Klasyfikacji
Mimo że zadanie było klasyfikacją (5 klas), większość czołowych rozwiązań traktowała to jako problem regresji.
* Model przewidywał liczbę ciągłą (np. 2.7).
* Zastosowano thresholding, aby zamienić wynik na klasę. Zwycięsca zastosował ręczną optymalizację progów.
* Loss Function: Używano MSE lub SmoothL1Loss, które naturalnie karzą większe odchylenia mocniej niż małe.

### Augmentacja i Pooling
* Augmentacja: Stosowano bardzo silną augmentację danych, aby zapobiec przeuczeniu na małym zbiorze. Biblioteka Albumentations była standardem. Wykorzystane techniki to: Obracanie, odbicia lustrzane, zmiana jasności/kontrastu, rozmycie, Zoom (ważne, bo dane z 2019 mają różne przybliżenia).
* Pooling: 1 miejsce zastosowało Generalized Mean Pooling (GeM) zamiast standardowego Average Pooling. Wzór na GeM pozwala sieci dynamicznie decydować, czy skupiać się na średniej jasności cech, czy na ich maksimach (co jest przydatne przy wykrywaniu małych zmian patologicznych).

---

## 7. Wyniki i wnioski:

| Miejsce | Strategia / Model | Kluczowy element |
| :--- | :--- | :--- |
| **1** | Inception + ResNeXt | GeM Pooling + Optymalizacja progów |
| **2** | EfficientNet B3-B5 | Iteracyjne Pseudo-labeling |
| **4** | EfficientNet B2-B7 | Różne rozmiary modeli dla różnych rozdzielczości obrazu |
| **7** | SeResNext + Inception | Pre-training na 2015 + Bardzo silna augmentacja |

### Podsumowanie Strategiczne
1.  Zrozumienie metryki: Zastosowanie regresji zamiast klasyfikacji było kluczowe dla maksymalizacji wyniku QWK.
2.  Zarządzanie danymi: Wygrali ci, którzy efektywnie wykorzystali stare dane (2015) i pseudo-labeling, zamiast skupiać się tylko na architekturze sieci.
3.  Uczenie zespołowe Uśrednianie wyników z różnych architektur i różnych "seedów" było niezbędne do zmniejszenia błędów i osiągnięcia wysokiego QWK.