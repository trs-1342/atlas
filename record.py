import sounddevice as sd
from scipy.io.wavfile import write
import os
import numpy as np

DEVICE_ID = 3
duration = 2

info = sd.query_devices(DEVICE_ID, "input")
fs = int(info["default_samplerate"])

print("Mikrofon:", info["name"])
print("Sample rate:", fs)

label = input("kelime (hey_atlas/ac/kapat): ").strip()
path = f"data/{label}"
os.makedirs(path, exist_ok=True)

i = len([f for f in os.listdir(path) if f.endswith(".wav")])

while True:
    cmd = input("enter = kayıt | -1 = çık: ")
    if cmd == "-1":
        break

    print("kayıt...")
    audio = sd.rec(
        int(duration * fs),
        samplerate=fs,
        channels=1,
        dtype="int16",
        device=DEVICE_ID
    )
    sd.wait()

    filename = f"{path}/{label}_{i}.wav"
    write(filename, fs, audio)

    print("kaydedildi:", filename)
    i += 1
