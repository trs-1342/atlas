import sounddevice as sd
import numpy as np

devices = sd.query_devices()

for i, d in enumerate(devices):
    if d['max_input_channels'] > 0:
        print(i, d['name'])

DEVICE_ID = int(input("Device seç: "))

def callback(indata, frames, time, status):
    volume = np.linalg.norm(indata) * 10
    print(int(volume))

with sd.InputStream(device=DEVICE_ID, channels=1, samplerate=48000, callback=callback):
    while True:
        pass
