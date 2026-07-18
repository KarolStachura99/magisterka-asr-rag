"""
Etap 1: serwer transkrypcji na żywo.
Telefon -> /ws/audio (PCM int16 16 kHz) -> bufor + VAD -> faster-whisper -> /ws/view (JSON).

Uruchomienie (z katalogu projektu, w aktywnym venv):
    python app\server.py
Serwer nasłuchuje na https://0.0.0.0:8443 (certyfikat: app/certs/, patrz README poniżej pliku).
"""

import asyncio
import json
import time
from pathlib import Path

import numpy as np
import webrtcvad
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel

# ----------------------------------------------------------------------------
# Konfiguracja — wartości zgodne z ustaleniami z eksperymentów (exp1-exp6)
# ----------------------------------------------------------------------------
SAMPLE_RATE = 16000          # Hz; telefon przepróbkowuje do tej wartości
FRAME_MS = 30                # rozmiar ramki VAD (webrtcvad: 10/20/30 ms)
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000        # 480 próbek
FRAME_BYTES = FRAME_SAMPLES * 2                        # int16 = 2 bajty
VAD_AGGRESSIVENESS = 2       # 0-3; 2 = rozsądny kompromis dla mowy z telefonu
SILENCE_END_MS = 600         # tyle ciszy po mowie kończy segment
SILENCE_END_FRAMES = SILENCE_END_MS // FRAME_MS
PREROLL_MS = 300             # tyle audio sprzed początku mowy doklejamy do segmentu
PREROLL_FRAMES = PREROLL_MS // FRAME_MS
MIN_SEGMENT_S = 0.5          # krótsze segmenty odrzucamy (trzaski, pojedyncze ramki)
MAX_SEGMENT_S = 15.0         # awaryjne cięcie długiego monologu

MODEL_NAME = "base"
WHISPER_KWARGS = dict(
    language="pl",
    beam_size=5,
    vad_filter=True,         # druga linia obrony wewnątrz Whispera
    temperature=0.0,         # deterministycznie, bez fallbacków
)

app = FastAPI()

# ----------------------------------------------------------------------------
# Stan globalny
# ----------------------------------------------------------------------------
model: WhisperModel | None = None
viewers: set[WebSocket] = set()          # klienci /ws/view (komputer + telefon)
segment_queue: asyncio.Queue = asyncio.Queue()
seg_counter = 0

# Metryki narastające (do rozdziału "wdrożenie na żywo")
metrics = {
    "audio_s_total": 0.0,        # ile sekund mowy przetworzono
    "proc_s_total": 0.0,         # ile sekund liczył je Whisper
    "segments": 0,
}

METRICS_CSV = Path(__file__).parent / "metrics_live.csv"
SESSION_START = time.strftime("%Y-%m-%d %H:%M:%S")


def log_metrics_csv(seg_id, audio_s, proc_s, latency_s, rtf_total, queue, text_len):
    new_file = not METRICS_CSV.exists()
    with METRICS_CSV.open("a", encoding="utf-8") as f:
        if new_file:
            f.write("session,seg_id,audio_s,proc_s,latency_s,rtf_total,queue,text_len\n")
        f.write(f"{SESSION_START},{seg_id},{audio_s:.2f},{proc_s:.2f},"
                f"{latency_s:.2f},{rtf_total:.3f},{queue},{text_len}\n")


@app.on_event("startup")
async def startup() -> None:
    global model
    print(f"[startup] Ładuję faster-whisper '{MODEL_NAME}' (cpu, int8)...")
    model = WhisperModel(MODEL_NAME, device="cpu", compute_type="int8")
    print("[startup] Model gotowy.")
    asyncio.create_task(transcription_worker())


# ----------------------------------------------------------------------------
# Broadcast do wszystkich podglądów
# ----------------------------------------------------------------------------
async def broadcast(payload: dict) -> None:
    dead = []
    for ws in viewers:
        try:
            await ws.send_text(json.dumps(payload, ensure_ascii=False))
        except Exception:
            dead.append(ws)
    for ws in dead:
        viewers.discard(ws)


# ----------------------------------------------------------------------------
# Worker: sekwencyjna transkrypcja segmentów z kolejki (CPU = 1 naraz)
# ----------------------------------------------------------------------------
MAX_WORD_REPEATS = 3         # dłuższe serie tego samego słowa = pętla halucynacji
COMPRESSION_RATIO_MAX = 2.4  # próg z literatury Whispera; wyżej = tekst-pętla


def _collapse_repeats(text: str) -> str:
    """Skraca serie identycznych słów ('razy razy razy...' -> 'razy razy razy')."""
    words = text.split()
    out, run = [], 0
    for i, w in enumerate(words):
        run = run + 1 if i > 0 and w.lower().strip(".,!?") == words[i - 1].lower().strip(".,!?") else 1
        if run <= MAX_WORD_REPEATS:
            out.append(w)
    return " ".join(out)


def _transcribe_sync(audio: np.ndarray) -> str:
    segments, _info = model.transcribe(audio, **WHISPER_KWARGS)
    parts = []
    for s in segments:
        # Halucynacje mają charakterystycznie wysoką kompresowalność (powtórki).
        if s.compression_ratio is not None and s.compression_ratio > COMPRESSION_RATIO_MAX:
            print(f"  [filtr] odrzucam segment (compression_ratio="
                  f"{s.compression_ratio:.1f}): {s.text[:60]!r}...")
            continue
        parts.append(s.text.strip())
    return _collapse_repeats(" ".join(parts).strip())


async def transcription_worker() -> None:
    while True:
        seg_id, audio, seg_end_wallclock = await segment_queue.get()
        audio_s = len(audio) / SAMPLE_RATE
        t0 = time.perf_counter()
        text = await asyncio.to_thread(_transcribe_sync, audio)
        proc_s = time.perf_counter() - t0

        metrics["audio_s_total"] += audio_s
        metrics["proc_s_total"] += proc_s
        metrics["segments"] += 1
        # Latencja end-to-end (dolne ograniczenie): od końca mowy do gotowego tekstu.
        latency_s = time.time() - seg_end_wallclock
        rtf_total = metrics["proc_s_total"] / max(metrics["audio_s_total"], 1e-6)

        print(f"[seg {seg_id}] {audio_s:.1f}s audio, {proc_s:.1f}s proc, "
              f"latencja {latency_s:.1f}s, RTF narast. {rtf_total:.2f} | {text!r}")

        # Metryki do CSV — surowe dane do rozdziału o wdrożeniu na żywo.
        log_metrics_csv(seg_id, audio_s, proc_s, latency_s, rtf_total,
                        segment_queue.qsize(), len(text))

        if not text:
            continue   # cisza/trzask: liczymy w metrykach, nie wyświetlamy

        await broadcast({
            "type": "segment",
            "id": seg_id,
            "text": text if text else "(brak rozpoznanej mowy)",
            "audio_s": round(audio_s, 2),
            "proc_s": round(proc_s, 2),
            "latency_s": round(latency_s, 2),
            "rtf_total": round(rtf_total, 3),
            "queue": segment_queue.qsize(),
            "corrected": False,   # pole pod etap 3 (SLM)
        })


# ----------------------------------------------------------------------------
# /ws/audio — strumień PCM z telefonu + maszyna stanów VAD
# ----------------------------------------------------------------------------
@app.websocket("/ws/audio")
async def ws_audio(ws: WebSocket) -> None:
    global seg_counter
    await ws.accept()
    print("[audio] Telefon podłączony.")
    await broadcast({"type": "status", "text": "Mikrofon podłączony"})

    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    pending = bytearray()            # bajty, które nie złożyły się jeszcze w ramkę
    preroll: list[bytes] = []        # ostatnie ramki ciszy przed mową
    segment = bytearray()            # aktualnie zbierany segment mowy
    in_speech = False
    silence_run = 0

    def flush_segment():
        """Zamknij segment i wrzuć do kolejki transkrypcji."""
        global seg_counter
        nonlocal segment
        audio = np.frombuffer(bytes(segment), dtype=np.int16).astype(np.float32) / 32768.0
        segment = bytearray()
        if len(audio) / SAMPLE_RATE < MIN_SEGMENT_S:
            return
        seg_counter += 1
        segment_queue.put_nowait((seg_counter, audio, time.time()))

    try:
        while True:
            data = await ws.receive_bytes()
            pending.extend(data)
            while len(pending) >= FRAME_BYTES:
                frame = bytes(pending[:FRAME_BYTES])
                del pending[:FRAME_BYTES]
                is_speech = vad.is_speech(frame, SAMPLE_RATE)

                if in_speech:
                    segment.extend(frame)
                    if is_speech:
                        silence_run = 0
                    else:
                        silence_run += 1
                        if silence_run >= SILENCE_END_FRAMES:
                            in_speech = False
                            silence_run = 0
                            flush_segment()
                    if len(segment) / 2 / SAMPLE_RATE >= MAX_SEGMENT_S:
                        flush_segment()          # awaryjne cięcie, mowa trwa dalej
                else:
                    if is_speech:
                        in_speech = True
                        segment.extend(b"".join(preroll))   # dokleja nagłos
                        segment.extend(frame)
                        preroll.clear()
                    else:
                        preroll.append(frame)
                        if len(preroll) > PREROLL_FRAMES:
                            preroll.pop(0)
    except WebSocketDisconnect:
        if in_speech:
            flush_segment()          # nie gub ostatniej wypowiedzi
        print("[audio] Telefon rozłączony.")
        await broadcast({"type": "status", "text": "Mikrofon rozłączony"})


# ----------------------------------------------------------------------------
# /ws/view — podgląd transkrypcji (komputer i telefon)
# ----------------------------------------------------------------------------
@app.websocket("/ws/view")
async def ws_view(ws: WebSocket) -> None:
    await ws.accept()
    viewers.add(ws)
    try:
        while True:
            await ws.receive_text()   # nic nie oczekujemy, trzymamy połączenie
    except WebSocketDisconnect:
        viewers.discard(ws)


# ----------------------------------------------------------------------------
# Statyczny frontend
# ----------------------------------------------------------------------------
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "viewer.html")


@app.get("/mic")
async def mic() -> FileResponse:
    return FileResponse(STATIC_DIR / "phone.html")


if __name__ == "__main__":
    import uvicorn
    certs = Path(__file__).parent / "certs"
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8443,
        ssl_certfile=str(certs / "cert.pem"),
        ssl_keyfile=str(certs / "key.pem"),
    )