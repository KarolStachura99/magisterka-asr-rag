import re
import os

plik_wejsiowy = "output_transcriptions_txt/faster_whisper_vad.txt"

if not os.path.exists(plik_wejsiowy):
    print(f"BŁĄD: System nie widzi pliku {plik_wejsiowy}.")
    print("Sprawdź w VS Code po lewej stronie, czy plik dokładnie tak się nazywa.")
else:
    # 1. Tworzenie nowej nazwy pliku (dodanie _clean)
    baza, ext = os.path.splitext(plik_wejsiowy)
    plik_wyjsciowy = f"{baza}_clean{ext}"

    # 2. Odczyt oryginalnego tekstu
    with open(plik_wejsiowy, 'r', encoding='utf-8') as f:
        text = f.read()

    # 3. Usuwanie znaczników czasu (np. [25.10s -> 33.10s])
    clean_text = re.sub(r'\[.*?\]\s*', '', text)

    # 4. Zapis do NOWEGO pliku (plik_wyjsciowy)
    with open(plik_wyjsciowy, 'w', encoding='utf-8') as f:
        f.write(clean_text)

    print("Sukces! Oryginalny plik pozostał bez zmian.")
    print(f"Oczyszczony tekst zapisano jako: {plik_wyjsciowy}")