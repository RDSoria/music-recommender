#!/usr/bin/env python3
"""
OpenL3-based audio feature extractor (v2):
- Stereo-aware loading (keeps spatial/timbre cues for embeddings)
- Configurable hop size for OpenL3 frames
- Optional full-track analysis ( --seconds 0 )
- Per-file timing + real-time factor (RTF)
- Optional stable 128-D reduction via persisted Scaler+PCA

Example:
  python extractor_openl3_v2.py \
    --music-dir "/path/to/folder" \
    --out-dir "./features_json" \
    --seconds 180 \
    --hop-size 0.25 \
    --embedding-size 512 \
    --input-repr mel256 \
    --content-type music \
    --reduce-to-128
"""

import argparse, sys, json, time, traceback
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import librosa

try:
    import pyloudnorm as pyln
    HAVE_PYLN = True
except Exception:
    HAVE_PYLN = False

try:
    import openl3
except ImportError as e:
    sys.stderr.write(
        "ERROR: openl3 is not installed.\n"
        "Install one of:\n"
        "  pip install openl3 'tensorflow-cpu<2.16'\n"
        "  # or on Apple Silicon:\n"
        "  pip install openl3 tensorflow-macos\n"
    )
    raise

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from joblib import dump, load

SUPPORTED_EXTS = {".mp3", ".flac", ".wav", ".m4a", ".aac", ".ogg"}
TARGET_SR = 48000  # OpenL3 commonly uses 48 kHz
PCA_PATH = "openl3_pca.pkl"
SCALER_PATH = "openl3_scaler.pkl"

# Key profiles (Krumhansl–Schmuckler, simplified)
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=np.float32)
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=np.float32)
KEY_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

def find_audio_files(root: str) -> List[Path]:
    return sorted([p for p in Path(root).rglob("*")
                   if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS])

def load_audio(path: Path, seconds: int) -> Tuple[np.ndarray, int]:
    """
    Returns (y, sr) where y is (n_samples, n_channels) float32 in [-1,1].
    - Preserves stereo if present (librosa returns (n_channels, n_samples) when mono=False).
    - Normalizes per channel to peak=1.
    """
    y, sr = librosa.load(str(path), sr=TARGET_SR, mono=False,
                         duration=seconds if seconds > 0 else None)
    if y.ndim == 1:
        # mono -> (n_samples, 1)
        y = y[np.newaxis, :]
    # y shape: (n_channels, n_samples) -> transpose to (n_samples, n_channels)
    y = y.T
    # normalize per channel
    peak = np.max(np.abs(y), axis=0, keepdims=True)
    peak[peak == 0] = 1.0
    y = (y / peak).astype(np.float32)
    return y, sr

def to_mono(y: np.ndarray) -> np.ndarray:
    """Simple average to mono from (n_samples, n_channels)."""
    if y.ndim == 2 and y.shape[1] > 1:
        return np.mean(y, axis=1)
    if y.ndim == 2 and y.shape[1] == 1:
        return y[:, 0]
    return y  # already mono 1-D

def tempo_and_confidence(y_mono: np.ndarray, sr: int) -> Tuple[float, float]:
    oenv = librosa.onset.onset_strength(y=y_mono, sr=sr)
    ac = librosa.autocorrelate(oenv, max_size=oenv.shape[0] // 2)
    ac = ac[1:]
    if ac.size == 0 or np.max(ac) <= 0:
        tempo, _ = librosa.beat.beat_track(y=y_mono, sr=sr, trim=False)
        tempo = float(np.atleast_1d(tempo)[0])
        return float(np.clip(tempo, 60, 200)), 0.0
    conf = float(np.max(ac) / (np.sum(ac) + 1e-9))
    tempo, _ = librosa.beat.beat_track(y=y_mono, sr=sr, onset_envelope=oenv, trim=False)
    tempo = float(np.atleast_1d(tempo)[0])
    if tempo < 60: tempo *= 2
    if tempo > 200: tempo /= 2
    return float(np.clip(tempo, 60, 200)), conf

def loudness_lufs_or_rms(y_mono: np.ndarray, sr: int) -> float:
    if HAVE_PYLN:
        try:
            meter = pyln.Meter(sr)
            return float(meter.integrated_loudness(y_mono.astype(np.float64)))
        except Exception:
            pass
    rms = np.sqrt(np.mean(y_mono**2)) + 1e-12
    return float(20.0 * np.log10(rms))

def detect_key_and_confidence(y_mono: np.ndarray, sr: int) -> Tuple[str, float]:
    chroma = librosa.feature.chroma_stft(y=y_mono, sr=sr)
    cmean = np.mean(chroma, axis=1)
    best_score, best_i, best_mode = -999.0, 0, "maj"
    for i in range(12):
        maj = np.corrcoef(np.roll(MAJOR_PROFILE, i), cmean)[0, 1]
        min_ = np.corrcoef(np.roll(MINOR_PROFILE, i), cmean)[0, 1]
        maj = -1 if np.isnan(maj) else maj
        min_ = -1 if np.isnan(min_) else min_
        if maj >= min_ and maj > best_score:
            best_score, best_i, best_mode = maj, i, "maj"
        if min_ > maj and min_ > best_score:
            best_score, best_i, best_mode = min_, i, "min"
    key = f"{KEY_NAMES[best_i]}{'m' if best_mode=='min' else ''}"
    conf = float(max(0.0, min(1.0, (best_score + 1.0) / 2.0)))
    return key, conf

def get_openl3_embedding(y: np.ndarray, sr: int, *, content_type: str,
                         input_repr: str, embedding_size: int, hop_size: float) -> np.ndarray:
    """
    y: (n_samples, n_channels) or (n_samples,) float32 waveform
    Returns mean-pooled OpenL3 embedding (1D).
    """
    emb, ts = openl3.get_audio_embedding(
        y, sr,
        content_type=content_type,
        input_repr=input_repr,
        embedding_size=embedding_size,
        center=True,
        hop_size=hop_size
    )
    if emb.ndim == 1:
        pooled = emb
    else:
        pooled = np.mean(emb, axis=0)
    return pooled.astype(np.float32)

def fit_or_load_scaler_pca(X: np.ndarray, target_dim: int = 128) -> Tuple[StandardScaler, PCA]:
    if Path(SCALER_PATH).exists() and Path(PCA_PATH).exists():
        scaler = load(SCALER_PATH)
        pca = load(PCA_PATH)
        return scaler, pca
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X)
    n_samples, n_features = Xs.shape
    n_components = min(target_dim, n_features, max(2, n_samples - 1))
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(Xs)
    dump(scaler, SCALER_PATH)
    dump(pca, PCA_PATH)
    return scaler, pca

def transform_to_128(v: np.ndarray, scaler: StandardScaler, pca: PCA) -> np.ndarray:
    v2 = scaler.transform(v.reshape(1, -1))
    e = pca.transform(v2)[0].astype(np.float32)
    out = np.zeros(128, dtype=np.float32)
    n = min(128, e.shape[0])
    out[:n] = e[:n]
    return out

def process_all(files: List[Path], seconds: int, content_type: str,
                input_repr: str, embedding_size: int, hop_size: float) -> List[Dict]:
    records = []
    for p in files:
        t0 = time.time()
        try:
            y, sr = load_audio(p, seconds)         # (n_samples, n_channels)
            y_mono = to_mono(y)                    # 1-D for tempo/key/loudness
            bpm, tconf = tempo_and_confidence(y_mono, sr)
            loud = loudness_lufs_or_rms(y_mono, sr)
            key, kconf = detect_key_and_confidence(y_mono, sr)
            o3 = get_openl3_embedding(y, sr,
                                      content_type=content_type,
                                      input_repr=input_repr,
                                      embedding_size=embedding_size,
                                      hop_size=hop_size)
            elapsed = time.time() - t0
            analyzed_sec = y.shape[0] / sr  # n_samples / sr
            rtf = analyzed_sec / elapsed if elapsed > 0 else 0.0
            print(f"[OK] {p}  time={elapsed:.2f}s  analyzed={analyzed_sec:.1f}s  RTF={rtf:.2f}")

            records.append({
                "path": str(p),
                "bpm": bpm,
                "tempo_confidence": tconf,
                "loudness": loud,
                "musical_key": key,
                "key_confidence": kconf,
                "openl3": o3,   # native size (512 or 6144)
            })
        except Exception:
            print(f"[ERR] {p}\n{traceback.format_exc()}", file=sys.stderr)
    if not records:
        raise RuntimeError("No audio processed successfully.")
    return records

def write_jsons(records: List[Dict], out_dir: Path, reduce_to_128: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    if reduce_to_128:
        X = np.vstack([r["openl3"] for r in records])
        scaler, pca = fit_or_load_scaler_pca(X, target_dim=128)
    else:
        scaler, pca = None, None

    for r in records:
        emb = transform_to_128(r["openl3"], scaler, pca) if reduce_to_128 else r["openl3"]
        data = {
            "bpm": round(float(r["bpm"]), 2),
            "tempo_confidence": round(float(r["tempo_confidence"]), 3),
            "musical_key": r["musical_key"],
            "key_confidence": round(float(r["key_confidence"]), 3),
            "loudness": round(float(r["loudness"]), 2),
            "embedding": [float(x) for x in emb.tolist()],
        }
        out_path = Path(out_dir) / (Path(r["path"]).name.replace(Path(r["path"]).suffix, ".json"))
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, separators=(",", ":"), indent=2)
        print(f"[WROTE] {out_path}")

def main():
    ap = argparse.ArgumentParser(description="OpenL3 music extractor (v2) -> per-file JSONs")
    ap.add_argument("--music-dir", required=True)
    ap.add_argument("--out-dir", default="./features_json")
    ap.add_argument("--seconds", type=int, default=90, help="Analyze first N seconds (0 = full track)")
    ap.add_argument("--content-type", default="music", choices=["music", "env"])
    ap.add_argument("--input-repr", default="mel256", choices=["mel256", "mel128", "linear"])
    ap.add_argument("--embedding-size", type=int, default=512, choices=[512, 6144])
    ap.add_argument("--hop-size", type=float, default=0.25, help="Seconds between OpenL3 frames (smaller = more detail)")
    ap.add_argument("--reduce-to-128", action="store_true", help="Project to stable 128-D via persisted scaler+PCA")
    args = ap.parse_args()

    files = find_audio_files(args.music_dir)
    if not files:
        print("No audio files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(files)} audio files. Processing…")
    records = process_all(files, args.seconds, args.content_type, args.input_repr,
                          args.embedding_size, args.hop_size)
    write_jsons(records, Path(args.out_dir), reduce_to_128=args.reduce_to_128)
    print("Done.")

if __name__ == "__main__":
    main()