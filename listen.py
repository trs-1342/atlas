import os
import time
import joblib
import numpy as np
import sounddevice as sd
from datetime import datetime
from scipy.signal import resample_poly

MODEL_FILE = "atlas_model.joblib"
DEVICE_ID = 3

RECORD_SEC = 2
BLOCK_SIZE = 1024
MIN_CONFIDENCE = 0.82
COOLDOWN = 1.5

TARGET_SR = 16000
N_FFT = 512
WIN = 400
HOP = 160
N_MELS = 40
FEAT_T = 60

os.makedirs("logs", exist_ok=True)


def log(msg):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line)

    with open("logs/atlas.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)


def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)


def build_mel_filterbank():
    bins = N_FFT // 2 + 1
    mel_points = np.linspace(hz_to_mel(80), hz_to_mel(8000), N_MELS + 2)
    hz_points = mel_to_hz(mel_points)
    freqs = np.linspace(0, TARGET_SR / 2, bins)

    fb = np.zeros((N_MELS, bins), dtype=np.float32)

    for m in range(1, N_MELS + 1):
        left, center, right = hz_points[m - 1], hz_points[m], hz_points[m + 1]

        left_mask = (freqs >= left) & (freqs <= center)
        right_mask = (freqs >= center) & (freqs <= right)

        fb[m - 1, left_mask] = (freqs[left_mask] - left) / (center - left)
        fb[m - 1, right_mask] = (right - freqs[right_mask]) / (right - center)

    return fb


MEL_FB = build_mel_filterbank()
HANN = np.hanning(WIN).astype(np.float32)


def trim_silence(x):
    energy = np.abs(x)
    threshold = max(0.015, energy.max() * 0.18)
    idx = np.where(energy > threshold)[0]

    if len(idx) == 0:
        return x

    pad = int(0.05 * TARGET_SR)
    start = max(0, idx[0] - pad)
    end = min(len(x), idx[-1] + pad)

    return x[start:end]


def log_mel_features(x):
    frames = []
    pos = 0

    while pos + WIN <= len(x):
        frame = x[pos:pos + WIN] * HANN
        frame = np.pad(frame, (0, N_FFT - WIN))
        spectrum = np.abs(np.fft.rfft(frame, N_FFT)).astype(np.float32)
        frames.append(spectrum)
        pos += HOP

    if not frames:
        return np.zeros(N_MELS * FEAT_T, dtype=np.float32)

    spec = np.stack(frames)
    mel = spec @ MEL_FB.T
    mel = np.log1p(mel)

    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    mel = mel.T

    old_t = np.linspace(0, 1, mel.shape[1])
    new_t = np.linspace(0, 1, FEAT_T)

    resized = np.zeros((N_MELS, FEAT_T), dtype=np.float32)

    for i in range(N_MELS):
        resized[i] = np.interp(new_t, old_t, mel[i])

    return resized.flatten()


def extract_features(x, sr):
    x = x.astype(np.float32) / 32768.0

    if sr != TARGET_SR:
        x = resample_poly(x, TARGET_SR, sr).astype(np.float32)

    x = trim_silence(x)

    max_len = TARGET_SR * RECORD_SEC

    if len(x) > max_len:
        x = x[:max_len]
    else:
        x = np.pad(x, (0, max_len - len(x)))

    return log_mel_features(x)


def calibrate_noise(sr):
    log("Gürültü ölçülüyor, 1 saniye sessiz kal")
    audio = sd.rec(
        int(sr * 1),
        samplerate=sr,
        channels=1,
        dtype="int16",
        device=DEVICE_ID
    )
    sd.wait()

    x = audio.reshape(-1).astype(np.float32)
    rms = int(np.sqrt(np.mean(x ** 2)))

    threshold = max(800, rms * 2.5)

    log(f"Gürültü RMS: {rms}")
    log(f"Ses eşiği: {int(threshold)}")

    return threshold


bundle = joblib.load(MODEL_FILE)
model = bundle["model"]

info = sd.query_devices(DEVICE_ID, "input")
sr = int(info["default_samplerate"])

log(f"Mikrofon: {info['name']}")
log(f"Sample rate: {sr}")

threshold = calibrate_noise(sr)
last_detect = 0

log("Hazır. Konuşmanı bekliyorum...")

with sd.InputStream(
    device=DEVICE_ID,
    samplerate=sr,
    channels=1,
    dtype="int16",
    blocksize=BLOCK_SIZE
) as stream:

    while True:
        data, _ = stream.read(BLOCK_SIZE)
        x = data.reshape(-1).astype(np.float32)

        rms = int(np.sqrt(np.mean(x ** 2)))

        if rms < threshold:
            continue

        chunks = [x.astype(np.int16)]
        target = int(sr * RECORD_SEC)

        while sum(len(c) for c in chunks) < target:
            data, _ = stream.read(BLOCK_SIZE)
            chunks.append(data.reshape(-1).astype(np.int16))

        audio = np.concatenate(chunks)

        feat = extract_features(audio, sr).reshape(1, -1)

        pred = model.predict(feat)[0]
        prob = model.predict_proba(feat)[0].max()

        now = time.time()

        if pred == "negatif":
            log(f"NEGATİF RED -> {pred} | güven: {prob:.2f} | rms: {rms}")
            continue

        if prob >= MIN_CONFIDENCE and now - last_detect > COOLDOWN:
            last_detect = now
            log(f"ALGILANDI -> {pred} | güven: {prob:.2f} | rms: {rms}")
        else:
            log(f"RED -> {pred} | güven: {prob:.2f} | rms: {rms}")
