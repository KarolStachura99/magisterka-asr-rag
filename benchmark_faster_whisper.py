"""
Benchmark wydajnościowy silnika faster-whisper (CTranslate2).
Ewaluacja kwantyzacji INT8 w środowisku z ograniczoną pamięcią VRAM (4GB).
"""

import time
from faster_whisper import WhisperModel

def run_faster_whisper_benchmark(audio_path: str, model_size: str = "base"):
    """
    Uruchamia transkrypcję i oblicza metryki wydajnościowe (Latencja, RTF).
    """
    print(f"Inicjalizacja silnika faster-whisper (Model: {model_size})...")
    print("Ładowanie na GPU z kwantyzacją INT8...")
    
    # Inicjalizacja modelu compute_type="int8" to kluczowa optymalizacja dla GTX 960M
    try:
        model = WhisperModel("base", device="cpu", compute_type="int8")
    except Exception as e:
        print(f"Błąd inicjalizacji CUDA. Sprawdź sterowniki. Szczegóły: {e}")
        return

    print(f"Rozpoczynam transkrypcję pliku: {audio_path}")
    start_time = time.time()

    # Wywołanie inferencji. Zwraca iterator (segments) oraz metadane (info)
    # beam_size=5 to standard zapewniający balans między szybkością a dokładnością
    # segments, info = model.transcribe(audio_path, beam_size=5, language="pl")
    segments, info = model.transcribe(audio_path, beam_size=5, vad_filter=True)

    print(f"Parametry audio: Czas trwania = {info.duration:.2f}s, Język = {info.language} ({info.language_probability:.2f})")
    print("-" * 50)

    full_text = ""
    # Pętla zwraca transkrypcje paczek w locie
    for segment in segments:
        full_text += segment.text + " "
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text.strip()}")

    end_time = time.time()
    latency = end_time - start_time
    
    # Obliczenie współczynnika RTF (Real-Time Factor). 
    # RTF = czas_przetwarzania / czas_trwania_audio. Wynik < 1.0 oznacza działanie w czasie rzeczywistym.
    rtf = latency / info.duration if info.duration > 0 else 0

    print("-" * 50)
    print("WYNIKI BENCHMARKU:")
    print(f"Czas operacji całkowity: {latency:.2f} sekund")
    print(f"Wskaźnik RTF: {rtf:.4f}")
    
    return full_text

if __name__ == "__main__":

    sciezka_do_audio = "C:/Users/Karol Stachura/Desktop/Projekt_Magisterski/audio_files/test_poczatek.mp3"
    
    try:
        run_faster_whisper_benchmark(sciezka_do_audio)
    except FileNotFoundError:
        print("Nie znaleziono pliku audio. Sprawdź ścieżkę.")