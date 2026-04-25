import os
import joblib
import numpy as np
from scipy.io.wavfile import read
from scipy.signal import resample_poly
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold

DATA_DIR = "data"
MODEL_FILE = "atlas_model.joblib"

TARGET_SR = 16000
MAX_SEC = 2
N_FFT = 512
WIN = 400
HOP = 160
N_MELS = 40
FEAT_T = 60


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


def resample_audio(x, sr):
    if sr == TARGET_SR:
        return x
    return resample_poly(x, TARGET_SR, sr).astype(np.float32)


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


def extract_features(path):
    sr, x = read(path)

    if x.ndim > 1:
        x = x[:, 0]

    x = x.astype(np.float32)

    if np.max(np.abs(x)) > 1:
        x = x / 32768.0

    x = resample_audio(x, sr)
    x = trim_silence(x)

    max_len = TARGET_SR * MAX_SEC

    if len(x) > max_len:
        x = x[:max_len]
    else:
        x = np.pad(x, (0, max_len - len(x)))

    return log_mel_features(x)


X = []
y = []

labels = sorted([
    d for d in os.listdir(DATA_DIR)
    if os.path.isdir(os.path.join(DATA_DIR, d))
])

for label in labels:
    folder = os.path.join(DATA_DIR, label)
    files = [f for f in os.listdir(folder) if f.endswith(".wav")]

    print(f"{label}: {len(files)} kayıt")

    for file in files:
        path = os.path.join(folder, file)
        X.append(extract_features(path))
        y.append(label)

X = np.array(X)
y = np.array(y)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(
        kernel="rbf",
        probability=True,
        class_weight="balanced",
        C=3.0,
        gamma="scale"
    ))
])

model.fit(X, y)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv)

print("Accuracy:", round(scores.mean() * 100, 2), "%")
print("Toplam kayıt:", len(y))
print("Etiketler:", labels)

joblib.dump({
    "model": model,
    "labels": labels,
    "target_sr": TARGET_SR
}, MODEL_FILE)

print("Kaydedildi:", MODEL_FILE)
