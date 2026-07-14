import whisper
import time
import torch
import os
from pydub import AudioSegment


audio_file = "audio_files/test_poczatek.mp3"
chunk_length_ms = 5000 
temp_file = "temp_chunk.wav" 

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- Diagnostyka ---")
print(f"Używane urządzenie: {device.upper()}")

print("\nŁadowanie modelu Whisper (wersja 'base')...")
model = whisper.load_model("base", device=device)

print(f"Wczytywanie pliku {audio_file} do pamięci podręcznej...")
audio = AudioSegment.from_mp3(audio_file)
total_length_ms = len(audio)

print("\n--- ROZPOCZYNAM SYMULACJĘ STRUMIENIOWANIA (LIVE CHUNKING) ---")
print("Format: [Przedział czasu paczki] Opóźnienie (TTFT) | Tekst")
print("-" * 70)

# Pętla przesuwająca się co 5 sekund po całym nagraniu
for i in range(0, total_length_ms, chunk_length_ms):
    # Wycięcie małej paczki dźwięku
    chunk = audio[i:i+chunk_length_ms]
    
    chunk.export(temp_file, format="wav")
    
    start_time = time.time()
    
    result = model.transcribe(temp_file, language="pl")
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    chunk_start_sec = i / 1000.0
    chunk_end_sec = (i + chunk_length_ms) / 1000.0
    
    print(f"[{chunk_start_sec:05.1f}s - {chunk_end_sec:05.1f}s] TTFT: {processing_time:.2f}s | {result['text']}")


if os.path.exists(temp_file):
    os.remove(temp_file)
    
print("\n--- ZAKOŃCZONO STRUMIENIOWANIE ---")