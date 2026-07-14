import whisper
import time
import torch
import psutil, os

process = psutil.Process(os.getpid())

device = "cuda" if torch.cuda.is_available() else "cpu"
print("--- Diagnostyka ---")
print(f"Używane urządzenie do obliczeń: {device.upper()}")

mem_start = process.memory_info().rss / 1024**2

print("\nŁadowanie modelu Whisper (wersja 'base')...")
model = whisper.load_model("base", device=device)
mem_after_load = process.memory_info().rss / 1024**2

audio_file = "audio_files/test_poczatek.mp3"

print(f"\nRozpoczynam transkrypcję pliku: {audio_file}...")
start_time = time.time()

try:
    # fp16=False - jawna deklaracja precyzji (CPU nie wspiera FP16),
    # eliminuje warning i gwarantuje powtarzalność pomiarów
    result = model.transcribe(audio_file, language="pl", fp16=False, verbose=False)
    processing_time = time.time() - start_time
    mem_peak = process.memory_info().rss / 1024**2

    # RTF - spójna metryka z eksperymentem faster-whisper
    audio_duration = result["segments"][-1]["end"] if result["segments"] else 0
    rtf = processing_time / audio_duration if audio_duration else float("nan")

    print("\n--- SUROWA TRANSKRYPCJA (Baseline) ---")
    print(result["text"])
    print("--------------------------------------")
    print("\n--- METRYKI (Baseline: openai-whisper, CPU, FP32) ---")
    print(f"Długość audio:          {audio_duration:.2f} s")
    print(f"Czas przetwarzania:     {processing_time:.2f} s")
    print(f"RTF (Real-Time Factor): {rtf:.4f}")
    print(f"RAM przed załadowaniem: {mem_start:.0f} MB")
    print(f"RAM po załadowaniu:     {mem_after_load:.0f} MB (model: +{mem_after_load-mem_start:.0f} MB)")
    print(f"RAM po transkrypcji:    {mem_peak:.0f} MB")

    output_dir = "output_transcriptions_txt"
    os.makedirs(output_dir, exist_ok=True)
    
    file_path = os.path.join(output_dir, "baseline_whisper_base.txt")

    with open(file_path, "w", encoding="utf-8-sig") as f:
        f.write(result["text"])
    print(f"\n[SUKCES] Zapisano do: {file_path}")

except Exception as e:
    print(f"\nWystąpił błąd: {e}")