import jiwer

# 1. ZŁOTY STANDARD 
ground_truth = """
Dzień dobry. Witam wszystkich ponownie. Teraz na tych laboratoriach opowiemy sobie trochę o klasyfikacji. Klasyfikacji, tylko części. Dzisiaj się skupimy na doborze hiperparametrów w walidacji krzyżowej, badaniu jakości modelu i dalszych etapach. Omówimy sobie elementy, które gdzieś tam powinny się znaleźć. Dzisiaj i jutro na konwersatoriach, jak też sobie opowiemy o konkretnych modelach. Konkretnych modelach, które oprócz tego naszego benchmarkowego, czyli regresji logistycznej mogą być używane, czyli najbardziej popularnych, najbardziej użytecznych, czyli wszystkich opartych na Drzewach decyzyjnych, Random Forest, wszystkich rebustowych, algorytmy rebustowych, XGBoost i wszystkich związanych z Ensemble Learning, opowiemy sobie właśnie o SVM, Support Vector Machines i tak dalej. I powiemy, jakie są tam hiperparametry, i w jaki sposób je tutaj dobierać. Dobra, i tak. Ok, czyli ostatnim razem zakończyliśmy na tym, że właśnie opowiedzieliśmy sobie o czyszczeniu z brakujących wartości. Opowiedzieliśmy sobie dokładnie jakie są techniki brakujących wartości, co powinniśmy z nimi robić, kiedy dropujemy, kiedy imputujemy, jakie są metody imputacji, czyli tam imputacja średnią, medianą, imputacja stratyfikowana, imputacja za pomocą modeli, czyli na przykład k-najbliższych sąsiadów do tego, żebyśmy imputowali. Opowiedzieliśmy sobie o różnych rodzajach wartości brakujących, czyli o Missing Not at Random, Missing at Random, Missing Completely at Random, czyli wszystkie te typy i co z każdym typem możemy zrobić, z czego on wynika. Później opowiedzieliśmy sobie o skalowaniu i normalizacji cech. Tak, mówiliśmy sobie o, kiedy skalujemy, kiedy normalizujemy, czym się od siebie różnią MinMaxScaler, standard normalization i dzięki temu właśnie opowiedzieliśmy sobie, kiedy jest to istotne, kiedy jest istotne dla jakich modeli. Mówiliśmy sobie, że modele tam, gdzie bazujemy na odległościach euklidesowych, na przykład k-najbliższych sąsiadów, Support Vector Machines, regresja logistyczna będzie wymagała skalowania i normalizacji cech. Mówiliśmy sobie, że niektóre z tych modeli nie będą tego wymagały, żebyśmy skalowali cechy, więc zawsze ten pipeline preprocessingowy powinien być dostosowany też do modelu. Jeżeli skalujemy dane to,  i damy to do modelu, który nie wymaga skalowania, to też się nic nie stanie ale. Tutaj zawsze jest po tym, jak sobie dobierzemy model, zawsze możemy stworzyć jakiś pipeline, który będzie dedykowany dla tego modelu. No i oczywiście mówiliśmy sobie o tym, że te wszystkie elementy zawsze powinny być tworzone w, po podziale na zbiór treningowy i testowy tak? Gdzie na przykład, jak tworzymy imputację, osobno imputujemy dla zbioru testowego, osobno imputujemy dla zbioru treningowego. Jeżeli tworzymy skalowanie, osobno skalujemy dla zbioru testowego i osobno dla zbioru treningowego. Na przykład mówimy, tam jest fit_transform, i na przykład później tylko transform. Czyli wykorzystujemy w odpowiedni sposób zbiór testowy i treningowy, cały preprocessing robimy w ten sposób, żeby informacje i"""

# 2. WYNIK Z WHISPERA (Surowy tekst z błędami)
whisper_output = """Dzień dobry, witam wszystkich ponownie. I teraz... ellesze na kt違室, jak pow eyeball, ndFlrkflrע. Klasifikacji, tylko część dzisiaj się skupimy na... koniektyta i kości modelu i talszych. i dzisiaj wyczuje i zachujcie do mówimy, by się creeping gdzieś tam powinny się znaleźć. i dzisiaj i jutro na konwersyteturie, jak to już sobie opowiem o konkretnych modelach. aplikacja Będzbękowego 3-Liu egysji logistycznej mogą być używane, czyli tych najbardziej populariznej, niebezpiecznej. użytecknych czyli wszystkie opaity, to trzeba decyzjny random forest. przez kilkubów keżdy i ucząc i wszystkich związanych z N-sab. deals Aye 긴工 Atw brak I powiemy, jakie są tam hipyporypory metrii. w jaki sposób je tutaj dopiegać. Dobry, tak. Ok, czyli ostatnim razem z... Zakończyliśmy, że właśnie opowiedzieliśmy. o czyszczeniu, o zupełnie niebeekujących wartości. Opowicieliśmy sobie o czyszczeniu, o zupełnie niebeekujących wartości. Dokładnie jakie są techniki. Wydwijmy się z nimi robiczki, gdzie dopujemę, kiedy imputujemy, i są metodynputacji. i tam imputacja średnio medianą imputacę stratifikowa. imputacja za pomocą modeli, czyli na przykład na najbliższych sędziardów do tego Myśmy imputowali. Obawiliśmy sobie już dynkuły za jak wajtności. b airculusbr少 ale a me sculpt do выдajere fantasia po comprehensity id latency, recommendation onotic kind of chody enterprises committee velotycznej co to możemy sobie zakończyć. Później opowiedzieliśmy sobie o skalowanie. i minimalizacji cek. Mówiliśmy sobie o kiedy skalu... workshopsриłem je kiedy normalizujemy czy mieliśmy comme deg Bitega oszczulem m 2009 warestant Where normal z normalization, ... i... i dim, quedarowoStudional. P взять ten temat, hmm... Kiedy kiedy jest to, kiedy jest to modely? Wiedzimy sobie, że modelę tam gdzie bazujemy na odległość, jak euklitezowych. independencey quien nie dzęło by się und ecological inspired i będzie wymagała skalodania i normalizacji cek. Mówiliśmy sobie, że niektóre z tych modeli nie będą. Oῖ team tego supermer one lasonden, to cie Pop & Pикeline Unifone i powinien być to stosowany też do modelu. Daj wskalujemy dane. i tam jest modelu, który nie wymaga. skalowania coś się nic nie staje, ale tutaj zawsze w pot... Jak sobie dobierzemy model, zawsze możemy stworzyć jakieś pipeline, tu je będzie dydykowany. do tego modelu. No i oczywiście mówiliśmy sobie o... O tym, że te wszystkie elementy zawsze powinny być. Narazę na reiterate. popoziale na test. i wyltreni go wy i testowę. Znaczy, że tworzymy imputację o st... i odszedł osobny i putujemy dla spieryt testowego, osobny i putujemy dla spieryt cheningowego. Jeżeli tworzymy skalowanie, osobno skalujemy dla z tego. To jest to wego idzie dla osób, no dla zbyt chłyninkowej. Tak, np. mówimy... tam jest fit transform i np. później tylko transform. czyli wykorzystujemy odpowiedni sposób odpowiedni sposób z próbutestowy i trajningowi cały preces ingrobiłem w ten sposób, żeby informacja."""

# 3. Normalizacja tekstu przed porównaniem
transformation = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveWhiteSpace(replace_by_space=True),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords()  # <--- TA LINIJKA ROZWIĄZUJE PROBLEM (Tnie zdania na listę słów)
])

print("Uruchamianie kalkulatora Word Error Rate (WER)...")

print("Uruchamianie kalkulatora Word Error Rate (WER)...")

# 4. Obliczenie wartości WER 
wer = jiwer.wer(
    ground_truth, 
    whisper_output, 
    reference_transform=transformation,  
    hypothesis_transform=transformation
)

print(f"\n" + "="*50)
print(f" 🎯 WSKAŹNIK BŁĘDÓW (WER): {wer * 100:.2f}%")
print("="*50)

# 5. Wizualizacja błędów
print("\nSzczegółowa analiza dopasowania (Alignment):")
alignment = jiwer.process_words(
    ground_truth, 
    whisper_output, 
    reference_transform=transformation,   
    hypothesis_transform=transformation
)
# Wypisanie wizualizacji do konsoli
print(jiwer.visualize_alignment(alignment))