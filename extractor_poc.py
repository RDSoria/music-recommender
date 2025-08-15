#!/usr/bin/env python3
"""
PoC audio feature extractor:
- Scans a folder for audio files (mp3, flac, wav, m4a, ogg, aac)
- Computes BPM, musical key, loudness (LUFS or RMS dB), and a 128-D embedding
- Fits PCA on this batch (components limited by samples/features), then pads to 128 dims
- Writes one JSON per input file in --out-dir

Usage:
  python extractor_poc.py --music-dir "/path/to/folder" --out-dir "./features_json" --seconds 75
"""

import argparse, os, sys, json, traceback
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import librosa
try:
    import pyloudnorm as pyln
    HAVE_PYLN = True
except Exception:
    HAVE_PYLN = False

from sklearn.decomposition import PCA

SUPPORTED_EXTS = {".mp3", ".flac", ".wav", ".m4a", ".aac", ".ogg"}
TARGET_SR = 22050  # analysis sample rate

# Krumhansl–Schmuckler key profiles (simplified)
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=np.float32)
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=np.float32)
KEY_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

def find_audio_files(root: str) -> List[Path]:
    root_p = Path(root)
    files = []
    for p in root_p.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            files.append(p)
    return sorted(files)

def load_audio(path: Path, seconds: int) -> Tuple[np.ndarray, int]:
    y, sr = librosa.load(str(path), sr=TARGET_SR, mono=True, duration=seconds if seconds > 0 else None)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    return y.astype(np.float32), sr

def tempo_bpm(y: np.ndarray, sr: int) -> float:
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr, trim=False)
    tempo = float(np.atleast_1d(tempo)[0])  # safe scalar for numpy 1.25+
    # Clamp + common octave correction
    if tempo < 60: tempo *= 2
    if tempo > 200: tempo /= 2
    return float(np.clip(tempo, 60, 200))

def loudness_lufs_or_rms(y: np.ndarray, sr: int) -> float:
    # Prefer LUFS; fallback to RMS dBFS-like if pyloudnorm unavailable/fails
    if HAVE_PYLN:
        try:
            meter = pyln.Meter(sr)  # ITU-R BS.1770
            return float(meter.integrated_loudness(y.astype(np.float64)))
        except Exception:
            pass
    rms = np.sqrt(np.mean(y**2)) + 1e-12
    return float(20.0 * np.log10(rms))

def detect_key(y: np.ndarray, sr: int) -> str:
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
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
    return f"{KEY_NAMES[best_i]}{'m' if best_mode=='min' else ''}"

def feature_vector(y: np.ndarray, sr: int) -> np.ndarray:
    # Expressive yet compact vector (~120–140 dims): stats over spectral, MFCC, chroma, contrast, etc.
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    spec_centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    spec_bw = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    chroma = librosa.feature.chroma_stft(S=S, sr=sr)

    def stats(x: np.ndarray) -> np.ndarray:
        return np.array([np.mean(x), np.std(x)], dtype=np.float32)

    parts = [
        stats(spec_centroid),
        stats(spec_bw),
        stats(rolloff),
        stats(zcr),
        np.r_[np.mean(contrast, axis=1), np.std(contrast, axis=1)].astype(np.float32),
        np.r_[np.mean(mfcc, axis=1), np.std(mfcc, axis=1)].astype(np.float32),
        np.r_[np.mean(chroma, axis=1), np.std(chroma, axis=1)].astype(np.float32),
    ]
    v = np.concatenate(parts).astype(np.float32)
    return v

def pad_or_trim(vec: np.ndarray, dim: int) -> np.ndarray:
    out = np.zeros(dim, dtype=np.float32)
    n = min(dim, vec.shape[0])
    out[:n] = vec[:n]
    return out

def process_all(files: List[Path], seconds: int) -> Tuple[List[Dict], PCA, int]:
    """
    Returns:
      - records: list of dicts with metrics and raw features per track
      - fitted PCA
      - raw feature dimensionality
    """
    records = []
    for p in files:
        try:
            y, sr = load_audio(p, seconds)
            bpm = tempo_bpm(y, sr)
            loud = loudness_lufs_or_rms(y, sr)
            key = detect_key(y, sr)
            raw = feature_vector(y, sr)
            records.append({
                "path": str(p),
                "bpm": bpm,
                "loudness": loud,
                "musical_key": key,
                "raw": raw,  # numpy array
            })
            print(f"[OK] {p}")
        except Exception:
            print(f"[ERR] {p}\n{traceback.format_exc()}", file=sys.stderr)

    if not records:
        raise RuntimeError("No audio processed successfully.")

    raw_mat = np.vstack([r["raw"] for r in records])
    raw_dim = raw_mat.shape[1]
    n_samples = raw_mat.shape[0]

    # PCA components must be <= min(n_samples, raw_dim); keep at least 2
    target_dim = min(128, raw_dim, max(2, n_samples - 1))
    pca = PCA(n_components=target_dim, random_state=42)
    pca.fit(raw_mat)
    return records, pca, raw_dim

def write_jsons(records: List[Dict], pca: PCA, raw_dim: int, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in records:
        emb = pca.transform(r["raw"].reshape(1, -1))[0].astype(np.float32)
        emb128 = pad_or_trim(emb, 128)  # ensure 128-D output
        data = {
            "bpm": round(float(r["bpm"]), 2),
            "musical_key": r["musical_key"],
            "loudness": round(float(r["loudness"]), 2),  # LUFS if available, else RMS dBFS-like
            "embedding": [float(x) for x in emb128.tolist()],
        }
        base = Path(r["path"]).name
        out_path = out_dir / Path(base).with_suffix(".json").name
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, separators=(",", ":"), indent=2)
        print(f"[WROTE] {out_path}")

def main():
    ap = argparse.ArgumentParser(description="Audio feature extractor PoC -> per-file JSONs")
    ap.add_argument("--music-dir", required=True, help="Folder with audio files")
    ap.add_argument("--out-dir", default="./features_json", help="Where to write JSON files")
    ap.add_argument("--seconds", type=int, default=75, help="Analyze first N seconds (0 = full file)")
    args = ap.parse_args()

    files = find_audio_files(args.music_dir)
    if not files:
        print("No audio files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(files)} audio files. Processing…")
    records, pca, raw_dim = process_all(files, args.seconds)
    write_jsons(records, pca, raw_dim, Path(args.out_dir))
    print("Done.")

if __name__ == "__main__":
    main()