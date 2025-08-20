#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import math

# ---------- existing helpers (trimmed) ----------

KEY_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
KEY_INDEX = {k:i for i,k in enumerate(KEY_NAMES)}

def load_library(features_dir: Path):
    items = []
    for p in sorted(features_dir.glob("*.json")):
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        items.append({
            "name": p.stem,
            "path": str(p),
            "bpm": float(data.get("bpm", 0)),
            "loudness": float(data.get("loudness", 0)),
            "key": str(data.get("musical_key", "C")),
            "emb": np.array(data.get("embedding", []), dtype=np.float32),
            "key_conf": float(data.get("key_confidence", 1.0)),
            "tempo_conf": float(data.get("tempo_confidence", 1.0)),
        })
    if not items:
        raise RuntimeError("No features found.")
    return items

def load_clusters(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        cj = json.load(f)
    return cj  # {k, clusters:{name: {cluster, dist}}, centroids: [...]}

def find_seed(items, needle: str):
    n = needle.lower().strip()
    for it in items:
        if it["name"].lower() == n: return it
    for it in items:
        if n in it["name"].lower(): return it
    raise ValueError(f"Song '{needle}' not found")

def cosine(a,b):
    da, db = np.linalg.norm(a), np.linalg.norm(b)
    if da==0 or db==0: return 0.0
    return float(np.dot(a,b)/(da*db))

def parse_key(k: str):
    k = k.strip()
    minor = k.endswith("m")
    base = k[:-1] if minor else k
    base = base.upper().replace("DB","C#").replace("EB","D#").replace("GB","F#").replace("AB","G#").replace("BB","A#")
    pc = KEY_INDEX.get(base, 0)
    return pc, ("min" if minor else "maj")

def circle_distance(a,b):
    d = abs(a-b)%12
    return min(d,12-d)

def harmonic_compatibility(k1,k2):
    pc1,m1 = parse_key(k1); pc2,m2 = parse_key(k2)
    if pc1==pc2 and m1==m2: return 1.0
    if pc1==pc2 and m1!=m2: return 0.95
    if m1!=m2 and circle_distance((pc1 + (3 if m1=="min" else -3))%12, pc2)==0:
        return 0.95
    d = circle_distance(pc1, pc2)
    return max(0.4, 1.0 - 0.1*d)

def closeness_gaussian(delta, sigma):
    return float(math.exp(-0.5*(delta/sigma)**2))

def score_pair(a,b):
    cos = cosine(a["emb"], b["emb"])
    harm = harmonic_compatibility(a["key"], b["key"]) * min(a["key_conf"], b.get("key_conf",1.0))
    bpm_close = closeness_gaussian(a["bpm"] - b["bpm"], sigma=8.0) * min(a["tempo_conf"], b.get("tempo_conf",1.0))
    loud_close = closeness_gaussian(a["loudness"] - b["loudness"], sigma=3.0)
    return 0.70*cos + 0.15*harm + 0.10*bpm_close + 0.05*loud_close, (cos, harm, bpm_close, loud_close)

# ---------- cluster-aware recommend ----------

def recommend_clustered(seed, items, clusters_json, topk=3, backfill=0.25):
    # map song -> cluster
    song2cl = clusters_json["clusters"]
    k = clusters_json["k"]

    seed_cl = song2cl.get(seed["name"], {"cluster": None})["cluster"]
    if seed_cl is None:
        # no cluster info: fall back to plain ranking
        pool = [it for it in items if it["name"] != seed["name"]]
    else:
        same = [it for it in items if it["name"] != seed["name"] and song2cl.get(it["name"],{}).get("cluster")==seed_cl]
        if len(same) >= max(3, topk):
            pool = same
        else:
            # backfill with nearest-cluster candidates by centroid distance
            # compute centroid distances
            centers = np.array(clusters_json["centroids"], dtype=np.float32)  # shape (k, d) in unit space
            # We don't have each song's normalized vector here; cosine fallback via labels:
            # Take all non-seed-cluster songs but downweight scores later.
            others = [it for it in items if it["name"] != seed["name"] and it not in same]
            # allow backfill fraction of candidates from others
            keep = int(max(1, backfill * topk))
            pool = same + others[:keep]

    scored = []
    for it in pool:
        s, parts = score_pair(seed, it)
        # small boost if same cluster; small penalty if not
        same_cluster = (song2cl.get(it["name"],{}).get("cluster") == seed_cl)
        s += (0.05 if same_cluster else -0.05)
        scored.append((it, s, parts, same_cluster))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:topk]

def main():
    ap = argparse.ArgumentParser(description="Recommend next song using cluster-aware filtering")
    ap.add_argument("--features-dir", required=True)
    ap.add_argument("--clusters", required=True, help="clusters.json produced by build_clusters.py")
    ap.add_argument("--song", required=True)
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()

    items = load_library(Path(args.features_dir))
    seed = find_seed(items, args.song)
    clusters_json = load_clusters(Path(args.clusters))

    recs = recommend_clustered(seed, items, clusters_json, topk=args.topk)

    print(f"\nSeed: {seed['name']}  |  BPM {seed['bpm']:.1f}  Key {seed['key']}  Loud {seed['loudness']:.1f} dB")
    print("-"*70)
    for i,(it,score,(cos,harm,bpm,loud),same_cluster) in enumerate(recs,1):
        tag = " (same cluster)" if same_cluster else " (backfill)"
        print(f"{i}. {it['name']}{tag}")
        print(f"   Score: {score:.4f}  (cos {cos:.3f}, harm {harm:.3f}, bpm {bpm:.3f}, loud {loud:.3f})")
        print(f"   BPM {it['bpm']:.1f}  Key {it['key']}  Loud {it['loudness']:.1f} dB")
    print()

if __name__ == "__main__":
    main()