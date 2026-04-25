print("ATLAS_VERSION=ml_v3")

import sounddevice as sd
import subprocess, argparse, time, sys
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneOut, cross_val_score

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE_ID  = 3
STATE_FILE = "atlas_templates.npz"
TARGET_SR  = 16000

POS_COUNT  = 10
NEG_GROUPS = [
    ("atla",  6), ("hatla",  4), ("batla",  4),
    ("ata",   4), ("altı",   4), ("hatlı",  4),
    ("adlas", 4), ("test",   3),
]

FRAME_SIZE    = 1024
RECORD_SEC    = 0.75   # onset sonrası sabit kayıt süresi
COOLDOWN      = 2.0
MIN_SEC       = 0.12
MAX_SEC       = 0.60
REARM_SILENCE = 12     # onset sonrası kaç frame sessizlik gerekir (re-arm)

# Mel
N_MELS = 40;  N_FFT = 512;  HOP = 160;  WIN = 400
F_MIN = 80.0; F_MAX = 8000.0
FEAT_T = 40

SVM_THR = 0.65
SIB_LOW = 3500;  SIB_HIGH = 8000
SIB_WIN_MS = 180

last_trigger = 0.0
_noise_amp   = 0.01   # global — kalibrasyon sonrası güncellenir

# ── Mel filterbank ────────────────────────────────────────────────────────────
def _build_fb():
    hz2mel = lambda h: 2595 * np.log10(1 + h / 700)
    mel2hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    bins  = N_FFT // 2 + 1
    mels  = np.linspace(hz2mel(F_MIN), hz2mel(F_MAX), N_MELS + 2)
    hzs   = mel2hz(mels)
    freqs = np.linspace(0, TARGET_SR / 2, bins)
    fb    = np.zeros((N_MELS, bins), dtype=np.float32)
    for m in range(1, N_MELS + 1):
        l, c, r = hzs[m-1], hzs[m], hzs[m+1]
        lm = (freqs >= l) & (freqs <= c)
        rm = (freqs >= c) & (freqs <= r)
        if c > l: fb[m-1, lm] = (freqs[lm] - l) / (c - l)
        if r > c: fb[m-1, rm] = (r - freqs[rm]) / (r - c)
    return fb

MEL_FB = _build_fb()
_HANN  = np.hanning(WIN).astype(np.float32)

# ── Feature extraction ────────────────────────────────────────────────────────
def _resample(x, sr):
    return resample_poly(x, TARGET_SR, sr).astype(np.float32) if sr != TARGET_SR else x

def _trim(x):
    """
    Gürültüye dayanıklı trim: eşik noise_amp * 2.5 ile env.max() * 0.25'in
    büyüğü — yüksek taban gürültüsünde de konuşmayı ayırt eder.
    """
    if not x.size: return x
    w   = max(1, int(0.01 * TARGET_SR))
    env = np.convolve(np.abs(x), np.ones(w) / w, "same")
    thr = max(_noise_amp * 2.5, env.max() * 0.25, 0.015)
    idx = np.where(env > thr)[0]
    if not idx.size: return np.array([], np.float32)
    pad = int(0.020 * TARGET_SR)
    return x[max(0, idx[0]-pad) : min(len(x), idx[-1]+pad)]

def log_mel(x):
    frames, pos = [], 0
    while pos + WIN <= len(x):
        f = x[pos:pos+WIN] * _HANN
        if WIN < N_FFT:
            f = np.concatenate([f, np.zeros(N_FFT - WIN, np.float32)])
        frames.append(np.abs(np.fft.rfft(f, N_FFT)).astype(np.float32))
        pos += HOP
    if not frames:
        return np.zeros((N_MELS, 1), np.float32)
    raw = np.stack(frames) @ MEL_FB.T
    mel = np.log1p(raw + 1e-6).astype(np.float32)
    mu  = mel.mean(); std = mel.std() + 1e-6
    return ((mel - mu) / std).T

def extract(x, input_sr):
    """
    (mel, vs, cr, trimmed_16k) — trimmed trailing-silence YOK.
    sibilance ve SVM buradan hesaplanır.
    """
    x  = _resample(x, input_sr)
    cr = float(np.mean(np.abs(x) >= 0.999))
    x  = _trim(x)
    vs = len(x) / TARGET_SR if len(x) else 0.0
    if x.size < int(MIN_SEC * TARGET_SR):
        return None, vs, cr, None
    mx = float(np.max(np.abs(x)))
    if mx > 0: x = x / mx
    if len(x) > int(MAX_SEC * TARGET_SR):
        x  = x[:int(MAX_SEC * TARGET_SR)]
        vs = len(x) / TARGET_SR
    return log_mel(x), vs, cr, x

# ── SVM ───────────────────────────────────────────────────────────────────────
def _resize_mel(mel, T=FEAT_T):
    n, t = mel.shape
    if t == T: return mel
    old = np.linspace(0, 1, t); new = np.linspace(0, 1, T)
    out = np.empty((n, T), np.float32)
    for i in range(n): out[i] = np.interp(new, old, mel[i])
    return out

def mel_to_feat(mel):
    return _resize_mel(mel).flatten()

def train_svm(pos_bank, neg_bank):
    X = np.array([mel_to_feat(m) for m in pos_bank] +
                 [mel_to_feat(m) for m in neg_bank])
    y = np.array([1]*len(pos_bank) + [0]*len(neg_bank))
    clf = Pipeline([
        ('sc',  StandardScaler()),
        ('svm', SVC(kernel='rbf', probability=True,
                    class_weight='balanced', C=2.0, gamma='scale')),
    ])
    clf.fit(X, y)
    try:
        cv = cross_val_score(clf, X, y, cv=LeaveOneOut(), scoring='accuracy')
        tp = int(cv[y==1].sum()); tn = int(cv[y==0].sum())
        print(f"  LOO  pos={tp}/{len(pos_bank)}  neg={tn}/{len(neg_bank)}  "
              f"genel={cv.mean():.1%}")
        bad = [i for i,(s,yi) in enumerate(zip(cv,y)) if yi==1 and s==0]
        if bad: print(f"  Dikkat: pos[{bad}] sınıflandırılamadı")
    except Exception: pass
    return clf

# ── Sibilance ─────────────────────────────────────────────────────────────────
def sibilance(trimmed):
    n   = int(TARGET_SR * SIB_WIN_MS / 1000)
    seg = trimmed[-n:] if len(trimmed) >= n else trimmed
    if not seg.size: return 0.0
    sz  = max(N_FFT, len(seg))
    pw  = np.abs(np.fft.rfft(seg, sz)) ** 2
    fr  = np.linspace(0, TARGET_SR / 2, len(pw))
    tot = pw.sum() + 1e-12
    return float(pw[(fr >= SIB_LOW) & (fr <= SIB_HIGH)].sum() / tot)

# ── Karar ─────────────────────────────────────────────────────────────────────
def decide(mel, trimmed, clf, vs, sib_thr):
    if not (MIN_SEC <= vs <= MAX_SEC):
        print(f"  atlandı süre={vs:.2f}s")
        return False
    prob = float(clf.predict_proba([mel_to_feat(mel)])[0][1])
    sib  = "–"
    ok   = prob >= SVM_THR
    if ok:
        sib = sibilance(trimmed)
        ok  = sib >= sib_thr
    s = sib if isinstance(sib, str) else f"{sib:.3f}"
    print(f"  prob={prob:.3f} sib={s} vs={vs:.2f} → {'★ KABUL' if ok else 'red'}")
    return ok

# ── Gürültü kalibrasyon ───────────────────────────────────────────────────────
def calibrate(sr):
    """
    Gürültü tabanını ölç. İki değer döndürür:
      start_rms  — konuşma başlangıcı eşiği (int16 ölçeği)
      sib_thr    — gürültüye göre ayarlı sibilance eşiği
    Yüksek gürültülü ortamlarda sibilance devre dışı bırakılır.
    """
    global _noise_amp
    print("Gürültü ölçülüyor (1.0s sessiz ol)...", end=" ", flush=True)
    n    = int(sr * 1.0)
    data = sd.rec(n, samplerate=sr, channels=1, dtype="int16", device=DEVICE_ID)
    sd.wait()
    rms = int(np.sqrt(np.mean(data.astype(np.float64)**2))) + 1
    print(f"taban={rms}")

    # Gürültü amplitude'ü (0-1 ölçeği)
    _noise_amp = rms / 32768.0

    # START_RMS: gürültünün 2.5 katı, min 800, max 18000
    start_rms = int(np.clip(rms * 2.5, 800, 18000))

    # Sibilance eşiği: gürültü yüksekse (SNR kötüyse) sib devre dışı
    # Gürültü > 5000 RMS → yeterli SNR yok → sib kontrolü atla
    if rms > 5000:
        sib_thr = 0.0   # devre dışı
        print(f"  Yüksek gürültü (RMS={rms}) — sibilance devre dışı")
        print(f"  Öneri: sistem sesini/mikrofon kazancını düşür")
    else:
        sib_thr = 0.15
        print(f"  Düşük gürültü — sibilance aktif (thr={sib_thr})")

    print(f"  START_RMS={start_rms}  SIB_THR={sib_thr}")
    return start_rms, sib_thr

# ── Sabit-süreli kayıt (onset-triggered) ─────────────────────────────────────
def _record_onset(sr, start_rms):
    """
    Onset (RMS >= start_rms) bekle, sonra RECORD_SEC kadar kayıt al.
    END_RMS yok — sabit süre. Trailing silence _trim ile temizlenir.
    Bu yaklaşım yüksek gürültülü ortamlarda çok daha güvenilir.
    """
    target = int(sr * RECORD_SEC)
    armed  = True
    rearm_cnt = 0

    with sd.InputStream(device=DEVICE_ID, samplerate=sr,
                        blocksize=FRAME_SIZE, channels=1, dtype="int16") as st:
        while True:
            data, _ = st.read(FRAME_SIZE)
            x   = np.asarray(data).reshape(-1).astype(np.int16)
            rms = int(np.sqrt(np.mean(x.astype(np.float64)**2)))

            if not armed:
                rearm_cnt = rearm_cnt + 1 if rms < start_rms else 0
                if rearm_cnt >= REARM_SILENCE: armed = True; rearm_cnt = 0
                continue

            if rms >= start_rms:
                # Onset! İlk chunk + geri kalan sabit süreyi kaydet
                chunks = [x.copy()]
                collected = FRAME_SIZE
                while collected < target:
                    d2, _ = st.read(FRAME_SIZE)
                    chunks.append(np.asarray(d2).reshape(-1).astype(np.int16))
                    collected += FRAME_SIZE
                return np.concatenate(chunks).astype(np.float32) / 32768, rms

# ── Template I/O ──────────────────────────────────────────────────────────────
def save_templates(pos, neg):
    pa = np.empty(len(pos), object); na = np.empty(len(neg), object)
    for i, m in enumerate(pos): pa[i] = m.astype(np.float32)
    for i, m in enumerate(neg): na[i] = m.astype(np.float32)
    np.savez_compressed(STATE_FILE, pos_mels=pa, neg_mels=na)
    print(f"Kaydedildi ({len(pos)} pos, {len(neg)} neg)")

def load_and_train():
    p = Path(STATE_FILE)
    if not p.exists():
        print("Template yok — önce: python atlas.py --enroll"); return None
    d = np.load(p, allow_pickle=True)
    if "pos_mels" not in d:
        print("Eski format — yeniden enroll:  python atlas.py --enroll"); return None
    pos_bank = d["pos_mels"]; neg_bank = d["neg_mels"]
    print(f"Template: {len(pos_bank)} pos  {len(neg_bank)} neg")
    return train_svm(pos_bank, neg_bank)

# ── Enroll ────────────────────────────────────────────────────────────────────
def collect(label, count, sr, start_rms):
    out = []
    for i in range(count):
        print(f"  [{label}] {i+1}/{count} → söyle")
        raw_f, onset_rms = _record_onset(sr, start_rms)
        mel, vs, cr, _ = extract(raw_f, sr)
        if np.max(np.abs(raw_f)) >= 0.999 or cr > 0.05:
            print("  ✗ clip"); continue
        if mel is None:
            print(f"  ✗ çok kısa ({vs:.2f}s)"); continue
        if vs > MAX_SEC:
            print(f"  ✗ çok uzun ({vs:.2f}s)"); continue
        out.append(mel)
        print(f"  ✓ {vs:.2f}s  T={mel.shape[1]}  onset_rms={onset_rms}")
    return out

def enroll():
    info = sd.query_devices(DEVICE_ID, "input")
    sr   = int(info["default_samplerate"])
    print(f"Mikrofon: {info['name']}  SR={sr}\n")
    start_rms, _ = calibrate(sr)
    print()

    print("[POZ] 'atlas' — net söyle, /s/ ile bitir")
    pos = collect("atlas", POS_COUNT, sr, start_rms)

    neg = []
    print("\n[NEG]")
    for word, cnt in NEG_GROUPS:
        neg.extend(collect(word, cnt, sr, start_rms))

    if len(pos) < 6 or len(neg) < 10:
        print("Yetersiz kayıt"); return

    print(f"\nSVM eğitiliyor ({len(pos)} pos, {len(neg)} neg)...")
    train_svm(pos, neg)
    save_templates(pos, neg)
    print("Kullanım: python atlas.py")

# ── Run ───────────────────────────────────────────────────────────────────────
def run():
    global last_trigger
    clf = load_and_train()
    if clf is None: return

    info = sd.query_devices(DEVICE_ID, "input")
    sr   = int(info["default_samplerate"])
    print(f"Mikrofon: {info['name']}  SR={sr}")
    start_rms, sib_thr = calibrate(sr)
    print(f"\nSVM_THR={SVM_THR}  SIB_THR={sib_thr}")
    print("Hazır — istediğin an söyle\n")

    while True:
        raw_f, onset_rms = _record_onset(sr, start_rms)
        print(f"[ses onset_rms={onset_rms}]")

        mel, vs, cr, tri = extract(raw_f, sr)

        if mel is None or cr > 0.05:
            print(f"  atlandı (vs={vs:.2f} cr={cr:.2f})")
            continue

        ok  = decide(mel, tri, clf, vs, sib_thr)
        now = time.time()
        if ok and (now - last_trigger) >= COOLDOWN:
            last_trigger = now
            print("★★★ Atlas algılandı! ★★★")
            subprocess.run(["notify-send", "Atlas", "Dinliyorum..."],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--enroll", action="store_true")
    args = p.parse_args()
    try:
        enroll() if args.enroll else run()
    except KeyboardInterrupt:
        print("\nDurduruluyor...")
        sys.exit(0)
