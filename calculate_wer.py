import jiwer
import os

GROUND_TRUTH_FILE = "ground_truth/ground_truth_Uczenie maszynowe i biblioteki programistyczne AI-20260314_103649-Nagrywanie spotkania.txt"

HYPOTHESES_FILES = {
    "1. Whisper Base (Baseline)": "output_transcriptions_txt/baseline_whisper_base.txt",
    #"2. Faster-Whisper + VAD": "output_transcriptions_txt/faster_whisper_vad_clean.txt"
}

# Transformacja ujednolicająca (GŁÓWNA METRYKA)
normalize_transform = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(), 
])

# Transformacja surowa (Case Sensitive)
raw_transform = jiwer.Compose([
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(), 
])

def load_text(filepath):
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def main():
    print("--- ROZPOCZYNAM EWALUACJĘ WER ---")
    ground_truth = load_text(GROUND_TRUTH_FILE)
    if not ground_truth:
        print("BŁĄD: Nie znaleziono Złotego Standardu!")
        return

    for nazwa, sciezka in HYPOTHESES_FILES.items():
        hypothesis = load_text(sciezka)
        if not hypothesis:
            print(f"[{nazwa}] -> Brak pliku ({sciezka}) - pomijam.")
            continue

        # Pełna analiza błędów (WER + S/D/I) na transformacji znormalizowanej
        output = jiwer.process_words(
            ground_truth, hypothesis,
            reference_transform=normalize_transform,
            hypothesis_transform=normalize_transform
        )

        # Dodatkowy WER surowy (case-sensitive, z interpunkcją)
        wer_raw = jiwer.wer(
            ground_truth, hypothesis,
            reference_transform=raw_transform,
            hypothesis_transform=raw_transform
        )

        # CER - błąd na poziomie znaków
        cer = jiwer.cer(ground_truth, hypothesis)

        print(f"\n[{nazwa}]")
        print(f"  -> WER (znormalizowany - główny): {output.wer * 100:.2f}%")
        print(f"  -> WER (surowy - case sensitive): {wer_raw * 100:.2f}%")
        print(f"  -> CER:            {cer * 100:.2f}%")
        print(f"  -> MER:            {output.mer * 100:.2f}%")
        print(f"  -> WIL:            {output.wil * 100:.2f}%")
        print(f"  -> Substitutions:  {output.substitutions}")
        print(f"  -> Deletions:      {output.deletions}")
        print(f"  -> Insertions:     {output.insertions}")
        print(f"  -> Słów w referencji: {output.hits + output.substitutions + output.deletions}")

        # Opcjonalnie: zapis dopasowania do pliku (źródło przykładów błędów OOV)
        with open(f"alignment_{nazwa.split('.')[0].strip()}.txt", "w", encoding="utf-8") as f:
            f.write(jiwer.visualize_alignment(output))

if __name__ == "__main__":
    main()