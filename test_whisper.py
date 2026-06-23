import whisper
import time
import torch

# 1. Sprawdzenie CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- Diagnostyka ---")
print(f"Używane urządzenie do obliczeń: {device.upper()}")

# 2. Ładowanie modelu (na start bierzemy najmniejszy, żeby sprawdzić czy działa)
print("\nŁadowanie modelu Whisper (wersja 'base')...")
model = whisper.load_model("base", device=device)

# audio_file = "Uczenie maszynowe i biblioteki programistyczne AI-20260314_103649-Nagrywanie spotkania.mp3" 
audio_file = "audio_files/test_poczatek.mp3"  # Plik testowy z wyciętym fragmentem 

print(f"\nRozpoczynam transkrypcję pliku: {audio_file}...")
start_time = time.time()

try:
    # 3. Właściwa transkrypcja nasłuchująca żargonu
    result = model.transcribe(audio_file, language="pl") # Wymuszamy język polski
    end_time = time.time()

    processing_time = end_time - start_time
    
    print("\n--- SUROWA TRANSKRYPCJA (Baseline) ---")
    print(result["text"])
    print("--------------------------------------")
    print(f"\nCzas przetwarzania (na karcie GTX 960M): {processing_time:.2f} sekund")
    
    # ---------------------------------------------------------
    # 4. ZAPIS DO PLIKU ( wymuszenie UTF-8)
    # ---------------------------------------------------------
    output_filename = "baseline_whisper_base.txt"
    with open(output_filename, "w", encoding="utf-8-sig") as f:
        f.write(result["text"])
        
    print(f"\n[SUKCES] Transkrypcja została bezbłędnie zapisana z polskimi znakami do pliku: {output_filename}")
    
except Exception as e:
    print(f"\nWystąpił błąd! Sprawdź czy plik {audio_file} istnieje. Treść błędu: {e}")