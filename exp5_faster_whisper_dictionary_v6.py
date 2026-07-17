"""
Eksperyment: Faster-whisper + słownik kontekstowy (initial_prompt).
A/B względem exp3_faster_whisper_vad: identyczna konfiguracja,
jedyną zmienną jest initial_prompt (słownik z rag_context 5).
"""
import time
import os
import psutil
from faster_whisper import WhisperModel

PROMPT_FILE = "extracted_dictionaries/extracted_dictionary_v6_Lab2 - Wstęp do klasyfikacji.txt"
MODE = "hotwords"   # przełącznik: "prompt" albo "hotwords"
AUDIO_PATH = "audio_files/test_poczatek.mp3"
OUTPUT_PATH = f"output_transcriptions_txt/faster_whisper_vad_{MODE}6.txt"

def run_benchmark(model_size: str = "base"):
    process = psutil.Process(os.getpid())
    mem_start = process.memory_info().rss / 1024**2

    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        domain_prompt = f.read().strip()
    print(f"Załadowany initial_prompt ({len(domain_prompt.split())} słów):")
    print(domain_prompt, "\n")

    print(f"Inicjalizacja faster-whisper (Model: {model_size}, CPU, INT8)...")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    mem_after_load = process.memory_info().rss / 1024**2

    start_time = time.time()
    kwargs = {"initial_prompt": domain_prompt} if MODE == "prompt" else {"hotwords": domain_prompt}
    segments, info = model.transcribe(
        AUDIO_PATH, beam_size=5, language="pl",
        vad_filter=True, **kwargs
    )

    full_text = ""
    for segment in segments:
        full_text += segment.text + " "
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text.strip()}")

    latency = time.time() - start_time
    mem_peak = process.memory_info().rss / 1024**2
    rtf = latency / info.duration if info.duration > 0 else 0

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8-sig") as f:
        f.write(full_text.strip())

    print("-" * 50)
    print("WYNIKI (faster-whisper base, CPU, INT8, VAD + initial_prompt):")
    print(f"Długość audio:           {info.duration:.2f} s")
    print(f"Czas operacji całkowity: {latency:.2f} s")
    print(f"Wskaźnik RTF:            {rtf:.4f}")
    print(f"RAM: {mem_start:.0f} -> {mem_after_load:.0f} (+{mem_after_load-mem_start:.0f} model) -> {mem_peak:.0f} MB")
    print(f"\n[SUKCES] Zapisano do: {OUTPUT_PATH}")

if __name__ == "__main__":
    run_benchmark()