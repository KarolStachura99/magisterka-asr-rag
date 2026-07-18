# System transkrypcji ASR ze wsparciem kontekstowym RAG + korekta SLM

Praca magisterska: w pełni lokalny system transkrypcji polskojęzycznych
wykładów akademickich nasyconych żargonem technicznym, działający na konsumenckim
sprzęcie (i5-6300HQ, 12 GB RAM, GTX 960M 4 GB).

**Problem:** błędy Out-of-Vocabulary („ensom legning", „kapytwektorim maszyn")
czynią surową transkrypcję bezużyteczną, a rozwiązania chmurowe są wykluczone
ze względu na prywatność nagrań i koszty.

**Architektura wynikowa:** materiały źródłowe (PDF/PPTX) → hybrydowy ekstraktor
terminologii (spaCy POS + KeyBERT) → słownik kontekstowy → ASR (faster-whisper
base INT8 + VAD) → korekta SLM (Ollama, llama3.2) → transkrypcja finalna.

## Kluczowe wyniki

| Konfiguracja | WER | RTF |
|---|---|---|
| Whisper base (baseline) | 36.73% | 0.1794 |
| Naiwny streaming (sztywne paczki 5 s) | 66.1% | >1 |
| faster-whisper INT8 + VAD (wsadowo) | 36.97% | 0.17 |
| + korekta SLM llama3.2 ze słownikiem (3B; wariant 0.5B odrzucony) | **36.49%** | 0.67 |

Wyniki dotyczą trybu wsadowego na fragmencie testowym (300 s audio, ground
truth 422 słowa). RTF wiersza SLM obejmuje pełny potok (transkrypcja +
korekta względem długości audio) - koszt najlepszej jakości, który uzasadnia
asynchroniczną korektę w aplikacji live. Metryki trybu na żywo zbiera
`app/metrics_live.csv`.

Wnioski negatywne (udokumentowane): biasing dekodera (initial_prompt/hotwords)
destabilizuje mały model; SLM 0.5B niewystarczający do korekty.

## Aplikacja transkrypcji na żywo (`app/`)

Finał wdrożeniowy projektu: telefon jako bezprzewodowy mikrofon, komputer jako
serwer, transkrypcja na żywo na obu ekranach, korekta SLM dochodząca
asynchronicznie. Całość w sieci lokalnej, zero chmury.

<img width="952" height="535" alt="image" src="https://github.com/user-attachments/assets/361a2ad1-2eac-4107-8dc7-874839fb327c" />
<img width="299" height="654" alt="image" src="https://github.com/user-attachments/assets/6bd0efd9-5d4e-42df-ac90-2fcc231dfaf6" />


Przepływ: telefon (getUserMedia → AudioWorklet → PCM int16 16 kHz)
→ WebSocket → serwer FastAPI → bufor + webrtcvad (segmenty cięte ciszą,
nie zegarem - wniosek z Eksperymentu 2) → faster-whisper (kolejka, CPU)
→ broadcast na wszystkie podglądy → asynchroniczna kolejka SLM → Ollama
→ podmiana segmentu na ekranie. Kontekst sesji ładowany z PDF
(`rag_context6.py` jako podproces) albo wpisywany tekstem.

Zabezpieczenia jakości w trybie live:
- filtr halucynacji: odrzucanie segmentów o `compression_ratio > 2.4`
  + tłumik serii powtórzonych słów,
- strażnik korekt SLM: odpowiedź modelu jest odrzucana, gdy zmienia długość
  tekstu o >40%, zawiera wyciek promptu/słownika lub nowe linie
  („korekta nigdy nie pogarsza"),
- awaria Ollamy nie zatrzymuje transkrypcji (korekta wyłącza się sama).

Metryki każdej sesji trafiają do `app/metrics_live.csv` (latencja od końca
wypowiedzi do tekstu, RTF narastający, długości kolejek).

### Uruchomienie

```powershell
# 1. Środowisko (Windows, Python 3.11, aktywny venv)
python -m pip install -r requirements.txt
python -m pip install fastapi "uvicorn[standard]" webrtcvad-wheels python-multipart

# 2. Certyfikat self-signed (jednorazowo; wymagany przez getUserMedia na mobile)
mkdir app\certs
openssl req -x509 -newkey rsa:2048 -keyout app\certs\key.pem -out app\certs\cert.pem -days 365 -nodes -subj "/CN=magisterka-live"

# 3. Ollama z modelem korekcyjnym (opcjonalnie; bez niej działa samo ASR)
ollama pull llama3.2

# 4. Start
python app\server.py
```

Komputer: `https://localhost:8443/` (podgląd + ładowanie kontekstu).
Telefon w tej samej sieci Wi-Fi: `https://<IP-komputera>:8443/mic`
(ostrzeżenie o certyfikacie → „Zaawansowane → Przejdź"). Wymaga reguły
zapory dla portu 8443 (TCP, przychodzące).

## Struktura repozytorium

```
app/                    # aplikacja live (serwer FastAPI + frontend)
exp1_...  – exp6_...    # eksperymenty (zamrożone jako dokumentacja badań)
rag_context6.py         # hybrydowa ekstrakcja terminologii (finalna wersja)
extracted_dictionaries/ # wygenerowane słowniki kontekstowe
metrics_results/        # metryki eksperymentów (źródło danych dashboardu)
alignments/             # alignmenty błędów S/D/I z ewaluacji
dashboard.py            # dashboard Streamlit z metrykami eksperymentów
evaluate_pipeline.py    # ewaluacja WER/RTF z alignmentem błędów
coverage_check.py       # pokrycie błędów przez słownik kontekstowy
ground_truth/           # transkrypcje referencyjne
```

Pełna dokumentacja eksperymentów (hipotezy, wyniki, wnioski) prowadzona
w Notion; praca magisterska w przygotowaniu.
