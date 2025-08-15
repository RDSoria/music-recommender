#!/usr/bin/env python3
"""
Recommend the next song from precomputed JSON features.

Inputs:
  --features-dir   Path to folder containing *.json from extractor_poc.py
  --song           Seed song name (matches JSON basename, case-insensitive; partial OK)
  --topk           How many recommendations to show (default 1)

Scoring (0..1 higher is better):
  score = 0.70 * cosine(embedding)
        + 0.15 * harmonic_compatibility(key)
        + 0.10 * bpm_closeness
        + 0.05 * loudness_closeness
"""

import argparse, json
from pathlib import Path
import numpy as np
import math
from typing import List, Dict, Tuple

KEY_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
KEY_INDEX = {k:i for i,k in enumerate(KEY_NAMES)}

def load_library(features_dir: Path) -> List[Dict]:
    items = []
    for p in sorted(features_dir.glob("*.json")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            vec = np.array(data.get("embedding", []), dtype=np.float32)
            name = p.stem  # basename without .json
            items.append({
                "name": name,
                "path": str(p),
                "bpm": float(data.get("bpm", 0)),
                "loudness": float(data.get("loudness", 0)),
                "key": str(data.get("musical_key", "C")).strip(),
                "emb": vec,
            })
        except Exception as e:
            print(f"[WARN] skip {p}: {e}")
    if not items:
        raise RuntimeError("No feature JSON files found.")
    return items

def find_seed(items: List[Dict], needle: str) -> Dict:
    n = needle.lower().strip()
    # exact (case-insensitive) match to basename
    for it in items:
        if it["name"].lower() == n:
            return it
    # fallback: partial substring match
    for it in items:
        if n in it["name"].lower():
            return it
    raise ValueError(f"Song '{needle}' not found in features.")

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    da = float(np.linalg.norm(a))
    db = float(np.linalg.norm(b))
    if da == 0 or db == 0:
        return 0.0
    return float(np.dot(a, b) / (da * db))

def parse_key(k: str) -> Tuple[int, str]:
    """Return (pitch_class 0..11, mode 'maj'|'min'). Accepts like 'C', 'C#m', 'Am'."""
    k = k.strip().upper().replace("M", "")  # tolerate 'Cm', 'C#m', 'CM'
    mode = "min" if k.endswith("M") else ("min" if k.endswith("MIN") else "maj")  # defensive
    # Fix common forms: 'AM' (ambiguous), we'll detect minor if endswith 'M' lower-case in source
    # Simpler approach: if ends with 'M' in original (lowercase), consider minor.
    # But extractor emitted 'm' for minor; keep a robust parse:
    if k.endswith("M") and not any(k[:-1] == x for x in KEY_NAMES):
        # If not a valid base when removing 'M', treat as major
        mode = "maj"
    base = k.replace("M", "").replace("MIN", "")
    # Map enharmonic 'DB' -> 'C#', 'EB' -> 'D#', etc. Minimal map:
    base = base.replace("DB","C#").replace("EB","D#").replace("GB","F#").replace("AB","G#").replace("BB","A#")
    if base not in KEY_INDEX:
        # last resort: default C major
        return 0, "maj"
    return KEY_INDEX[base], mode

def circle_distance(a: int, b: int) -> int:
    d = abs(a - b) % 12
    return min(d, 12 - d)

def harmonic_compatibility(k1: str, k2: str) -> float:
    """
    Very simple harmonic rule-of-thumb:
      1.0 same key & mode
      0.95 same tonic, different mode (e.g., C vs Cm)
      0.95 relative major/minor (Am <-> C)
      else decay by circle-of-fifths/semitone distance
    """
    pc1, m1 = parse_key(k1)
    pc2, m2 = parse_key(k2)
    if pc1 == pc2 and m1 == m2:
        return 1.0
    # same tonic different mode
    if pc1 == pc2 and m1 != m2:
        return 0.95
    # relative major/minor: minor +3 semitones => relative major (approx)
    if m1 != m2 and circle_distance((pc1 + (3 if m1 == "min" else -3)) % 12, pc2) == 0:
        return 0.95
    # general decay by semitone distance (0..6)
    d = circle_distance(pc1, pc2)  # 0..6
    # map 0->1.0, 1->0.9, 2->0.8, ..., 6->0.4
    return max(0.4, 1.0 - 0.1 * d)

def closeness_gaussian(delta: float, sigma: float) -> float:
    """Return 0..1, 1 means identical; sigma controls tolerance."""
    return float(math.exp(-0.5 * (delta / sigma) ** 2))

def score_pair(a: Dict, b: Dict) -> float:
    cos = cosine(a["emb"], b["emb"])  # 0..1
    harm = harmonic_compatibility(a["key"], b["key"])  # 0..1
    bpm_close = closeness_gaussian(a["bpm"] - b["bpm"], sigma=8.0)  # ~within ±8 BPM
    loud_close = closeness_gaussian(a["loudness"] - b["loudness"], sigma=3.0)  # within ~3 dB
    return 0.70 * cos + 0.15 * harm + 0.10 * bpm_close + 0.05 * loud_close

def recommend(seed: Dict, items: List[Dict], topk: int = 1) -> List[Tuple[Dict, float]]:
    scored = []
    for it in items:
        if it["name"] == seed["name"]:
            continue
        s = score_pair(seed, it)
        scored.append((it, s))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:topk]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", required=True, help="Path to features_json")
    ap.add_argument("--song", required=True, help="Seed song name (basename, case-insensitive; partial OK)")
    ap.add_argument("--topk", type=int, default=1, help="How many recommendations to print")
    args = ap.parse_args()

    items = load_library(Path(args.features_dir))
    seed = find_seed(items, args.song)

    recs = recommend(seed, items, topk=args.topk)
    if not recs:
        print("No candidates.")
        return

    print(f"\nSeed: {seed['name']}  |  BPM {seed['bpm']:.1f}  Key {seed['key']}  Loud {seed['loudness']:.1f} dB")
    print("-" * 70)
    for i, (it, s) in enumerate(recs, 1):
        # For transparency, show components
        cos = cosine(seed["emb"], it["emb"])
        harm = harmonic_compatibility(seed["key"], it["key"])
        bpm_close = closeness_gaussian(seed["bpm"] - it["bpm"], sigma=8.0)
        loud_close = closeness_gaussian(seed["loudness"] - it["loudness"], sigma=3.0)
        print(f"{i}. {it['name']}")
        print(f"   Score: {s:.4f}  (cos {cos:.3f}, harm {harm:.3f}, bpm {bpm_close:.3f}, loud {loud_close:.3f})")
        print(f"   BPM {it['bpm']:.1f}  Key {it['key']}  Loud {it['loudness']:.1f} dB")
    print()

if __name__ == "__main__":
    main()