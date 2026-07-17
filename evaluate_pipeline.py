"""
evaluate_pipeline.py - jednolita ewaluacja wszystkich eksperymentów.
Uruchomienie: python evaluate_pipeline.py
Wyniki: alignments/alignment_<eksperyment>.txt + results/metrics_log.csv
"""
import jiwer
import os
import csv
from datetime import datetime
import re

GROUND_TRUTH_FILE = "ground_truth/ground_truth_Uczenie maszynowe i biblioteki programistyczne AI-20260314_103649-Nagrywanie spotkania.txt"

# Rejestr eksperymentów: nazwa -> plik transkrypcji
EXPERIMENTS = {
    "whisper_base_baseline":  "output_transcriptions_txt/baseline_whisper_base.txt",
    "whisper_base_streaming": "output_transcriptions_txt/streaming_whisper_base_5s.txt",
    "faster_whisper_vad":     "output_transcriptions_txt/faster_whisper_vad.txt",
    "faster_whisper_vad_prompt": "output_transcriptions_txt/faster_whisper_vad_prompt.txt",
    "faster_whisper_vad_prompt6":   "output_transcriptions_txt/faster_whisper_vad_prompt6.txt",
    "faster_whisper_vad_hotwords6": "output_transcriptions_txt/faster_whisper_vad_hotwords6.txt",
    "faster_whisper_vad_slm": "output_transcriptions_txt/faster_whisper_vad_slm.txt",
    "faster_whisper_vad_slm_qwen":  "output_transcriptions_txt/faster_whisper_vad_slm.txt",
    "faster_whisper_vad_slm_llama": "output_transcriptions_txt/faster_whisper_vad_slm_llama.txt",
}

ALIGN_DIR = "alignments"
RESULTS_CSV = "metrics_results/metrics_log.csv"

normalize_transform = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])
raw_transform = jiwer.Compose([
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])


def clean_text(text: str) -> str:
    text = re.sub(r'\[.*?\]\s*', '', text)   # znaczniki czasu [xx.x s - yy.y s] (z txt_cleaner.py)
    text = text.replace("\ufffd", " ")        # znaki zastępcze �
    text = " ".join(text.split())             # wielokrotne spacje / nowe linie
    return text

def load_text(filepath):
    if not os.path.exists(filepath):
        return None
    with open(filepath, "r", encoding="utf-8-sig") as f:
        return f.read()

def main():
    os.makedirs(ALIGN_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)

    ground_truth = load_text(GROUND_TRUTH_FILE)
    if not ground_truth:
        print("BŁĄD: Nie znaleziono ground truth!")
        return
    ground_truth = clean_text(ground_truth)

    write_header = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8-sig") as csvfile:
        writer = csv.writer(csvfile, delimiter=";")
        if write_header:
            writer.writerow(["data", "eksperyment", "wer_norm", "wer_raw",
                             "cer", "mer", "wil", "S", "D", "I", "H", "slowa_ref"])

        for name, path in EXPERIMENTS.items():
            hypothesis = load_text(path)
            if not hypothesis:
                print(f"[{name}] brak pliku ({path}) - pomijam.")
                continue
            hypothesis = clean_text(hypothesis)

            out = jiwer.process_words(
                ground_truth, hypothesis,
                reference_transform=normalize_transform,
                hypothesis_transform=normalize_transform)
            wer_raw = jiwer.wer(ground_truth, hypothesis,
                                reference_transform=raw_transform,
                                hypothesis_transform=raw_transform)
            cer = jiwer.cer(ground_truth, hypothesis)
            n_ref = out.hits + out.substitutions + out.deletions

            # Alignment z nazwą eksperymentu
            align_path = os.path.join(ALIGN_DIR, f"alignment_{name}.txt")
            with open(align_path, "w", encoding="utf-8") as f:
                f.write(jiwer.visualize_alignment(out))

            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M"), name,
                f"{out.wer*100:.2f}", f"{wer_raw*100:.2f}", f"{cer*100:.2f}",
                f"{out.mer*100:.2f}", f"{out.wil*100:.2f}",
                out.substitutions, out.deletions, out.insertions, out.hits, n_ref])

            print(f"\n[{name}]")
            print(f"  WER {out.wer*100:.2f}% | WER raw {wer_raw*100:.2f}% | CER {cer*100:.2f}%")
            print(f"  S={out.substitutions} D={out.deletions} I={out.insertions} H={out.hits} / {n_ref} słów")
            print(f"  alignment -> {align_path}")

    print(f"\nMetryki dopisane do: {RESULTS_CSV}")

if __name__ == "__main__":
    main()