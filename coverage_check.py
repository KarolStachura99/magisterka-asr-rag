"""
coverage_check.py - diagnostyka pokrycia słownika kontekstowego.
Mierzy, jaki odsetek błędnie transkrybowanych słów referencji (S+D)
pokrywa lista initial_prompt. UWAGA: analiza wyłącznie post-hoc -
wyniki NIE mogą służyć do rozszerzania listy (przeciek danych testowych).
"""
import jiwer
import spacy
import re
import os

GROUND_TRUTH_FILE = "ground_truth/ground_truth_Uczenie maszynowe i biblioteki programistyczne AI-20260314_103649-Nagrywanie spotkania.txt"
HYPOTHESIS_FILE = "output_transcriptions_txt/faster_whisper_vad.txt"
PROMPT_FILE = "extracted_dictionaries/extracted_dictionary_v6_Lab2 - Wstęp do klasyfikacji.txt"

normalize_transform = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])

def clean_text(text):
    text = re.sub(r'\[.*?\]\s*', '', text)
    text = text.replace("\ufffd", " ")
    return " ".join(text.split())

def load(p):
    with open(p, "r", encoding="utf-8-sig") as f:
        return f.read()

print("Ładowanie spaCy...")
nlp = spacy.load("pl_core_news_md")

def lemmas(words):
    """Mapa: słowo -> lemat (lowercase)."""
    doc = nlp(" ".join(words))
    return {t.text.lower(): t.lemma_.lower() for t in doc}

gt = clean_text(load(GROUND_TRUTH_FILE))
hyp = clean_text(load(HYPOTHESIS_FILE))
prompt_terms = [t.strip().lower() for t in load(PROMPT_FILE).split(",") if t.strip()]

# Zbiór form i lematów z promptu (frazy rozbijamy też na słowa składowe)
prompt_words = set()
for term in prompt_terms:
    for w in term.split():
        prompt_words.add(w)
prompt_lemmas = set(lemmas(list(prompt_words)).values()) | prompt_words

# Alignment: wyciągamy słowa referencji z operacji S i D
out = jiwer.process_words(gt, hyp,
                          reference_transform=normalize_transform,
                          hypothesis_transform=normalize_transform)
ref_words = out.references[0]
error_words = []
for chunk in out.alignments[0]:
    if chunk.type in ("substitute", "delete"):
        error_words.extend(ref_words[chunk.ref_start_idx:chunk.ref_end_idx])

# Pokrycie: czy lemat błędnego słowa jest na liście promptu
err_lemmas = lemmas(error_words)
covered, uncovered = [], []
for w in error_words:
    (covered if err_lemmas.get(w, w) in prompt_lemmas or w in prompt_lemmas
     else uncovered).append(w)

n = len(error_words)
print(f"\n--- POKRYCIE SŁOWNIKA KONTEKSTOWEGO (S+D = {n} wystąpień) ---")
print(f"Pokryte przez listę:   {len(covered)} ({len(covered)/n*100:.1f}%)")
print(f"Niepokryte:            {len(uncovered)} ({len(uncovered)/n*100:.1f}%)")

from collections import Counter
print("\nPokryte (top):", Counter(covered).most_common(15))
print("\nNiepokryte (top):", Counter(uncovered).most_common(25))