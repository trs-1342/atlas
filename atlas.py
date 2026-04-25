print("ATLAS_VERSION=action_v1")

import os
import time
import joblib
import signal
import subprocess
import numpy as np
import sounddevice as sd

from datetime import datetime
from scipy.signal import resample_poly


# =========================
# CONFIG
# =========================

MODEL_FILE = "atlas_model.joblib"
LOG_FILE = "logs/atlas.log"

DEVICE_ID = 3

APP_NAME = "KWrite"
APP_CMD = ["kwrite"]
APP_PROCESS = "kwrite"

RECORD_SEC = 2
BLOCK_SIZE = 1024

MIN_CONFIDENCE = 0.82
MIN_MARGIN = 0.12

COOLDOWN = 2.0

NEGATIVE_LABELS = {"negatif", "negative", "noise", "other"}
OPEN_LABELS = {"ac", "aç"}
CLOSE_LABELS = {"kapat"}
WAKE_LABELS = {"hey_atlas", "hey atlas"}

N_FFT = 512
WIN = 400
HOP = 160
N_MELS = 40
FEAT_T = 60


# =========================
# LOG
# =========================

os.makedirs("logs", exist_ok=True)


def log(message):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line)

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def notify(title, message):
    log(f"Bildirim -> {title}: {message}")

    try:
        subprocess.run(
            ["notify-send", title, message],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except Exception:
        pass


# =========================
# APP CONTROL
# =========================

def is_app_running():
    result = subprocess.run(
        ["pgrep", "-x", APP_PROCESS],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    return result.returncode == 0


def open_app():
    if is_app_running():
        notify("Atlas", f"{APP_NAME} zaten açık.")
        return

    subprocess.Popen(
        APP_CMD,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True
    )

    notify("Atlas", f"{APP_NAME} açıldı.")


def close_app():
    if not is_app_running():
        notify("Atlas", f"{APP_NAME} zaten kapalı.")
        return

    subprocess.run(
        ["pkill", "-TERM", "-x", APP_PROCESS],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    time.sleep(0.5)

    if is_app_running():
        subprocess.run(
            ["pkill", "-KILL", "-x", APP_PROCESS],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        notify("Atlas", f"{APP_NAME} zorla kapatıldı.")
    else:
        notify("Atlas", f"{APP_NAME} kapatıldı.")


def handle_command(label, confidence, margin, rms):
    label = label.strip()

    if label in NEGATIVE_LABELS:
        log(f"NEGATİF RED -> {label} | güven: {confidence:.2f} | margin: {margin:.2f} | rms: {rms}")
        return

    if confidence < MIN_CONFIDENCE:
        log(f"RED düşük güven -> {label} | güven: {confidence:.2f} | margin: {margin:.2f} | rms: {rms}")
        return

    if margin < MIN_MARGIN:
        log(f"RED kararsız -> {label} | güven: {confidence:.2f} | margin: {margin:.2f} | rms: {rms}")
        return

    if label in WAKE_LABELS:
        notify("Atlas", "Dinliyorum.")
        return

    if label in OPEN_LABELS:
        log(f"KOMUT -> AÇ | güven: {confidence:.2f} | margin: {margin:.2f} | rms: {rms}")
        open_app()
        return

    if label in CLOSE_LABELS:
        log(f"KOMUT -> KAPAT | güven: {confidence:.2f} | margin: {margin:.2f} | rms: {rms}")
        close_app()
        return

    log(f"BİLİNMEYEN ETİKET -> {label} | güven: {confidence:.2f}")


# =========================
# MODEL LOAD
# =========================

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(f"Model bulunamadı: {MODEL_FILE}")

bundle = joblib.load(MODEL_FILE)

if isinstance(bundle, dict):
    model = bundle["model"]
    TARGET_SR = int(bundle.get("target_sr", 16000))
else:
    model = bundle
    TARGET_SR = 16000

classes = list(model.classes_)

log(f"Model yüklendi: {MODEL_FILE}")
log(f"Etiketler: {classes}")


# =========================
# FEATURE EXTRACTION
# =========================

def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)


def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)


def build_mel_filterbank():
    bins = N_FFT // 2 + 1

    mel_points = np.linspace(
        hz_to_mel(80),
        hz_to_mel(8000),
        N_MELS + 2
    )

    hz_points = mel_to_hz(mel_points)
    freqs = np.linspace(0, TARGET_SR / 2, bins)

    fb = np.zeros((N_MELS, bins), dtype=np.float32)

    for m in range(1, N_MELS + 1):
        left = hz_points[m - 1]
        center = hz_points[m]
        right = hz_points[m + 1]

        left_mask = (freqs >= left) & (freqs <= center)
        right_mask = (freqs >= center) & (freqs <= right)

        fb[m - 1, left_mask] = (freqs[left_mask] - left) / (center - left)
        fb[m - 1, right_mask] = (right - freqs[right_mask]) / (right - center)

    return fb


MEL_FB = build_mel_filterbank()
HANN = np.hanning(WIN).astype(np.float32)


def trim_silence(x):
    energy = np.abs(x)

    if energy.max() <= 0:
        return x

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


def extract_features(audio, source_sr):
    x = audio.astype(np.float32)

    if np.max(np.abs(x)) > 1:
        x = x / 32768.0

    if source_sr != TARGET_SR:
        x = resample_poly(x, TARGET_SR, source_sr).astype(np.float32)

    x = trim_silence(x)

    max_len = TARGET_SR * RECORD_SEC

    if len(x) > max_len:
        x = x[:max_len]
    else:
        x = np.pad(x, (0, max_len - len(x)))

    return log_mel_features(x)


# =========================
# AUDIO
# =========================

def calibrate_noise(sr):
    log("Gürültü ölçülüyor, 1 saniye sessiz kal.")

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

    threshold = int(max(800, rms * 2.5))

    log(f"Gürültü RMS: {rms}")
    log(f"Ses eşiği: {threshold}")

    return threshold


def predict_audio(audio, sr):
    feat = extract_features(audio, sr).reshape(1, -1)

    probs = model.predict_proba(feat)[0]

    order = np.argsort(probs)[::-1]

    best_idx = order[0]
    second_idx = order[1] if len(order) > 1 else order[0]

    label = classes[best_idx]
    confidence = float(probs[best_idx])
    second_confidence = float(probs[second_idx])
    margin = confidence - second_confidence

    return label, confidence, margin


# =========================
# MAIN LOOP
# =========================

def main():
    info = sd.query_devices(DEVICE_ID, "input")
    sr = int(info["default_samplerate"])

    log(f"Mikrofon: {info['name']}")
    log(f"Sample rate: {sr}")

    threshold = calibrate_noise(sr)

    last_action_time = 0

    log("Atlas hazır. Komut bekleniyor: hey_atlas / ac / kapat")

    with sd.InputStream(
        device=DEVICE_ID,
        samplerate=sr,
        channels=1,
        dtype="int16",
        blocksize=BLOCK_SIZE
    ) as stream:

        while True:
            data, _ = stream.read(BLOCK_SIZE)
            chunk = data.reshape(-1).astype(np.float32)

            rms = int(np.sqrt(np.mean(chunk ** 2)))

            if rms < threshold:
                continue

            chunks = [chunk.astype(np.int16)]
            target_len = int(sr * RECORD_SEC)

            while sum(len(c) for c in chunks) < target_len:
                data, _ = stream.read(BLOCK_SIZE)
                chunks.append(data.reshape(-1).astype(np.int16))

            audio = np.concatenate(chunks)

            label, confidence, margin = predict_audio(audio, sr)

            now = time.time()

            if now - last_action_time < COOLDOWN:
                log(f"COOLDOWN -> {label} | güven: {confidence:.2f} | margin: {margin:.2f} | rms: {rms}")
                continue

            last_action_time = now

            handle_command(label, confidence, margin, rms)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Atlas durduruldu.")
    except Exception as e:
        log(f"HATA: {type(e).__name__}: {e}")
        raise
