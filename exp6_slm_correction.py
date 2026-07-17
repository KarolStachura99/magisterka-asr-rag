"""
Eksperyment 6: Warstwa korekcyjna SLM (Ollama qwen2.5:0.5b).
Wejście: transkrypcja wariantu A (exp3) + słownik v6 (rag_context 6).
Wyjście: skorygowana transkrypcja do ewaluacji WER.
Konserwatywna korekta: tylko zniekształcona terminologia, bez parafraz.
"""
import time
import os
import re
import requests

TRANSCRIPT_FILE = "output_transcriptions_txt/faster_whisper_vad.txt"
DICTIONARY_FILE = "extracted_dictionaries/extracted_dictionary_v6_Lab2 - Wstęp do klasyfikacji.txt"
OUTPUT_FILE = "output_transcriptions_txt/faster_whisper_vad_slm.txt"

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2:latest"
OUTPUT_FILE = "output_transcriptions_txt/faster_whisper_vad_slm_llama.txt"
OPTIONS = {"num_ctx": 4096, "temperature": 0.1, "num_predict": 256}

SYSTEM_PROMPT = (
    "Jesteś korektorem transkrypcji wykładu o uczeniu maszynowym. "
    "Otrzymujesz fragment transkrypcji z błędami fonetycznymi oraz słownik "
    "poprawnej terminologii. Twoje zadanie: popraw WYŁĄCZNIE zniekształcone "
    "terminy specjalistyczne na ich poprawne formy. NIE parafrazuj, NIE zmieniaj "
    "szyku zdań, NIE dodawaj ani NIE usuwaj słów poza korektą terminów, NIE "
    "komentuj. Zakaz znaczników Markdown. Zwróć tylko poprawiony tekst."
)

def split_chunks(text, max_words=60):
    """Dzieli tekst na fragmenty po zdaniach, ~60 słów (limit num_predict=150)."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current = [], []
    for s in sentences:
        current.append(s)
        if sum(len(x.split()) for x in current) >= max_words:
            chunks.append(" ".join(current)); current = []
    if current:
        chunks.append(" ".join(current))
    return chunks

def correct_chunk(chunk, dictionary):
    user_msg = (f"Słownik poprawnej terminologii: {dictionary}\n\n"
                f"Fragment transkrypcji do korekty:\n{chunk}\n\n"
                f"Poprawiony tekst:")
    r = requests.post(OLLAMA_URL, json={
        "model": MODEL, "stream": False, "options": OPTIONS,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": user_msg}]})
    r.raise_for_status()
    return r.json()["message"]["content"].strip()

if __name__ == "__main__":
    with open(TRANSCRIPT_FILE, encoding="utf-8-sig") as f:
        transcript = f.read().strip()
    with open(DICTIONARY_FILE, encoding="utf-8") as f:
        dictionary = f.read().strip()

    chunks = split_chunks(transcript)
    print(f"Transkrypcja: {len(transcript.split())} słów -> {len(chunks)} fragmentów")
    print(f"Model: {MODEL} | Opcje: {OPTIONS}\n" + "-" * 60)

    corrected, latencies = [], []
    t_total = time.time()
    for i, ch in enumerate(chunks, 1):
        t0 = time.time()
        try:
            out = correct_chunk(ch, dictionary)
        except Exception as e:
            print(f"[{i}] BŁĄD ({e}) - zachowuję oryginał")
            out = ch
        lat = time.time() - t0
        latencies.append(lat)
        corrected.append(out)
        print(f"[{i}/{len(chunks)}] {lat:.2f}s")
        print(f"  PRZED: {ch[:90]}...")
        print(f"  PO:    {out[:90]}...")

    total = time.time() - t_total
    with open(OUTPUT_FILE, "w", encoding="utf-8-sig") as f:
        f.write(" ".join(corrected))

    print("-" * 60)
    print(f"WYNIKI (korekta SLM: {MODEL}):")
    print(f"Fragmentów: {len(chunks)} | Czas łączny: {total:.2f} s "
          f"| Latencja śr./maks.: {sum(latencies)/len(latencies):.2f} / {max(latencies):.2f} s")
    print(f"[SUKCES] Zapisano do: {OUTPUT_FILE}")