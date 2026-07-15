"""
Benchmark wydajnościowy silnika faster-whisper (CTranslate2).
Ewaluacja inferencji CPU z kwantyzacją INT8 + filtr Silero VAD
na sprzęcie bez wsparcia CUDA 12 (GTX 960M).
"""

import time
import os
import psutil
from faster_whisper import WhisperModel

def run_faster_whisper_benchmark(audio_path: str, model_size: str = "base"):
    process = psutil.Process(os.getpid())
    mem_start = process.memory_info().rss / 1024**2

    print(f"Inicjalizacja silnika faster-whisper (Model: {model_size})...")
    print("Ładowanie na CPU z kwantyzacją INT8...")

    # compute_type="int8" - kluczowa optymalizacja pamięciowa dla starszego sprzętu
    try:
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
    except Exception as e:
        print(f"Błąd inicjalizacji modelu. Szczegóły: {e}")
        return
    mem_after_load = process.memory_info().rss / 1024**2

    print(f"Rozpoczynam transkrypcję pliku: {audio_path}")
    start_time = time.time()

    # beam_size=5: domyślny standard faster-whisper (balans szybkość/dokładność)
    # language="pl": wymuszone dla spójności z baseline (Eksperyment 1)
    # vad_filter=True: Silero VAD odrzuca ciszę przed dekoderem
    segments, info = model.transcribe(
        audio_path, beam_size=5, language="pl", vad_filter=True
    )

    print(f"Parametry audio: Czas trwania = {info.duration:.2f}s, "
          f"Język = {info.language} ({info.language_probability:.2f})")
    print("-" * 50)

    full_text = ""
    # UWAGA: segments to leniwy iterator - inferencja dzieje się w tej pętli, pomiar czasu musi objąć całą pętlę
    for segment in segments:
        full_text += segment.text + " "
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text.strip()}")

    latency = time.time() - start_time
    mem_peak = process.memory_info().rss / 1024**2

    rtf = latency / info.duration if info.duration > 0 else 0

    # Zapis transkrypcji 
    output_dir = "output_transcriptions_txt"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, "faster_whisper_vad.txt")
    with open(file_path, "w", encoding="utf-8-sig") as f:
        f.write(full_text.strip())

    print("-" * 50)
    print("WYNIKI BENCHMARKU (faster-whisper base, CPU, INT8, VAD):")
    print(f"Długość audio:          {info.duration:.2f} s")
    print(f"Czas operacji całkowity: {latency:.2f} s")
    print(f"Wskaźnik RTF:           {rtf:.4f}")
    print(f"RAM przed załadowaniem: {mem_start:.0f} MB")
    print(f"RAM po załadowaniu:     {mem_after_load:.0f} MB (model: +{mem_after_load-mem_start:.0f} MB)")
    print(f"RAM po transkrypcji:    {mem_peak:.0f} MB")
    print(f"\n[SUKCES] Zapisano do: {file_path}")

    return full_text

if __name__ == "__main__":
    sciezka_do_audio = "audio_files/test_poczatek.mp3"
    try:
        run_faster_whisper_benchmark(sciezka_do_audio)
    except FileNotFoundError:
        print("Nie znaleziono pliku audio. Sprawdź ścieżkę.")