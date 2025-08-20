#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def load_embeddings(dir_path: Path):
    items = []
    for p in sorted(dir_path.glob("*.json")):
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        emb = np.asarray(data["embedding"], dtype=np.float32)
        items.append({"name": p.stem, "path": str(p), "emb": emb})
    if not items:
        raise RuntimeError("No JSON feature files found.")
    # Normalize to unit length -> cosine distance becomes Euclidean
    X = np.stack([it["emb"] for it in items], axis=0)
    X = normalize(X)  # L2
    return items, X

def choose_k(X, k_min=2, k_max=10):
    # For very small libraries, clamp
    n = X.shape[0]
    if n <= 3: return 2
    k_max = min(k_max, n-1)
    best_k, best_s = None, -1
    for k in range(k_min, max(k_min, k_max)+1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        if len(set(labels)) == 1:
            continue
        s = silhouette_score(X, labels, metric="euclidean")
        if s > best_s:
            best_k, best_s = k, s
    return best_k or 2

def main():
    ap = argparse.ArgumentParser(description="Cluster songs by embedding (unsupervised genre/style)")
    ap.add_argument("--features-dir", required=True)
    ap.add_argument("--out", default="clusters.json")
    ap.add_argument("--k", type=int, default=0, help="If 0, auto-select via silhouette")
    args = ap.parse_args()

    items, X = load_embeddings(Path(args.features_dir))
    k = args.k or choose_k(X, 2, min(12, max(3, X.shape[0]-1)))
    km = KMeans(n_clusters=k, n_init=25, random_state=42)
    labels = km.fit_predict(X)
    centers = km.cluster_centers_  # unit-length approx

    # distances to own centroid (smaller -> more central)
    dists = np.linalg.norm(X - centers[labels], axis=1)

    out = {
        "k": int(k),
        "clusters": {it["name"]: {"cluster": int(lbl), "dist": float(dist)} 
                     for it, lbl, dist in zip(items, labels, dists)},
        "centroids": centers.tolist()
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Wrote {args.out}  (k={k}, songs={len(items)})")

if __name__ == "__main__":
    main()