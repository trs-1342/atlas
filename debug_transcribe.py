import sounddevice as sd
import queue
import json
import subprocess
import threading
import time
from vosk import Model, KaldiRecognizer

DEVICE_ID = 3
MODEL_PATH = "vosk-model-small-tr-0.3"

q = queue.Queue()
last_mic_state = None


def get_default_source():
    result = subprocess.run(
        ["pactl", "get-default-source"],
        capture_output=True,
        text=True
    )
    return result.stdout.strip()


def get_source_mute(source):
    result = subprocess.run(
        ["pactl", "get-source-mute", source],
        capture_output=True,
        text=True
    )
    out = result.stdout.strip().lower()
    if "yes" in out:
        return True
    if "no" in out:
        return False
    return None


def mic_state_monitor():
    global last_mic_state
    source = get_default_source()

    while True:
        state = get_source_mute(source)
        if state is not None and state != last_mic_state:
            last_mic_state = state
            if state:
                print("Mikrofon kapatıldı")
            else:
                print("Mikrofon açıldı")
        time.sleep(0.7)


def callback(indata, frames, time_info, status):
    if status:
        return
    q.put(bytes(indata))


device_info = sd.query_devices(DEVICE_ID, "input")
samplerate = int(device_info["default_samplerate"])

print(f"Mikrofon: {device_info['name']}")
print(f"Sample rate: {samplerate}")

model = Model(MODEL_PATH)
rec = KaldiRecognizer(model, samplerate)
rec.SetWords(False)

threading.Thread(target=mic_state_monitor, daemon=True).start()

print("Konuş. Çıkmak için Ctrl+C")

with sd.RawInputStream(
    device=DEVICE_ID,
    samplerate=samplerate,
    blocksize=8000,
    dtype="int16",
    channels=1,
    callback=callback
):
    while True:
        data = q.get()

        if rec.AcceptWaveform(data):
            text = json.loads(rec.Result()).get("text", "").strip()
            if text:
                print("FINAL:", text)
        else:
            partial = json.loads(rec.PartialResult()).get("partial", "").strip()
            if partial:
                print("PARTIAL:", partial)
