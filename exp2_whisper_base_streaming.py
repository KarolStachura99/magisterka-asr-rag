import whisper
import time
import torch
import os
import psutil
from pydub import AudioSegment

audio_file = "audio_files/test_poczatek.mp3"
chunk_length_ms = 5000
temp_file = "temp_chunk.wav"
output_filename = "output_transcriptions_txt/streaming_whisper_base_5s.txt"

process = psutil.Process(os.getpid())
device = "cuda" if torch.cuda.is_available() else "cpu"
print("--- Diagnostyka ---")
print(f"Używane urządzenie: {device.upper()}")

print("\nŁadowanie modelu Whisper (wersja 'base')...")
model = whisper.load_model("base", device=device)
mem_after_load = process.memory_info().rss / 1024**2

print(f"Wczytywanie pliku {audio_file}...")
audio = AudioSegment.from_mp3(audio_file)
total_length_ms = len(audio)

print("\n--- SYMULACJA STRUMIENIOWANIA (LIVE CHUNKING, 5 s) ---")
print("Format: [Przedział paczki] Latencja | Tekst")
print("-" * 70)

latencies = []
full_text_parts = []
total_start = time.time()

for i in range(0, total_length_ms, chunk_length_ms):
    chunk = audio[i:i+chunk_length_ms]
    chunk.export(temp_file, format="wav")

    start_time = time.time()
    result = model.transcribe(temp_file, language="pl", fp16=False)
    latency = time.time() - start_time

    latencies.append(latency)
    full_text_parts.append(result["text"].strip())

    print(f"[{i/1000:05.1f}s - {(i+chunk_length_ms)/1000:05.1f}s] "
          f"Latencja: {latency:.2f}s | {result['text']}")

total_time = time.time() - total_start
mem_end = process.memory_info().rss / 1024**2

# Zapis transkrypcji do ewaluacji WER
full_text = " ".join(full_text_parts)
with open(output_filename, "w", encoding="utf-8-sig") as f:
    f.write(full_text)

if os.path.exists(temp_file):
    os.remove(temp_file)

audio_duration = total_length_ms / 1000.0
avg_lat = sum(latencies) / len(latencies)
over_budget = sum(1 for l in latencies if l > chunk_length_ms / 1000.0)

print("\n--- METRYKI (Streaming naiwny: openai-whisper base, CPU, FP32) ---")
print(f"Długość audio:            {audio_duration:.2f} s")
print(f"Liczba paczek:            {len(latencies)} (po {chunk_length_ms/1000:.0f} s)")
print(f"Czas całkowity:           {total_time:.2f} s")
print(f"RTF:                      {total_time/audio_duration:.4f}")
print(f"Latencja paczki śr./min/maks: {avg_lat:.2f} / {min(latencies):.2f} / {max(latencies):.2f} s")
print(f"Paczek przekraczających budżet 5 s: {over_budget}/{len(latencies)}"
      f" ({'system NIE nadąża za żywo' if over_budget else 'system nadąża za żywo'})")
print(f"RAM po załadowaniu / na końcu: {mem_after_load:.0f} / {mem_end:.0f} MB")
print(f"\n[SUKCES] Transkrypcja zapisana do: {output_filename}")