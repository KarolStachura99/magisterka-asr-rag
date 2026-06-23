from pydub import AudioSegment
print("Wczytywanie długiego pliku...")

audio = AudioSegment.from_mp3("Uczenie maszynowe i biblioteki programistyczne AI-20260314_103649-Nagrywanie spotkania.mp3")

start_min = 15
end_min = 17

# pydub operuje na milisekundach
start_ms = start_min * 60 * 1000
end_ms = end_min * 60 * 1000

print(f"Wycinam fragment od {start_min}:00 do {end_min}:00...")
# Składnia Pythona do wycinania "od-do" (slicing)
short_audio = audio[start_ms:end_ms]

short_audio.export("audio_files/test_srodek.mp3", format="mp3")
print("Zapisano test_srodek.mp3! Możesz testować.")