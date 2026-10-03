import json
import pickle
import os
import zipfile
import argparse
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import hdbscan


#Config
USAGES_EMB_DIR  = "embeddings/SV/usages"
OUTPUT_DIR      = "outputs/SV"
LANG            = "SV"
MIN_K           = 2
MAX_K           = 8
RANDOM_STATE    = 42


def load_embeddings(word):
    path = os.path.join(USAGES_EMB_DIR, f"{word}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def get_all_words():
    return [
        Path(f).stem
        for f in os.listdir(USAGES_EMB_DIR)
        if f.endswith(".pkl")
    ]


def find_best_k(embeddings):
    best_k     = MIN_K
    best_score = -1

    for k in range(MIN_K, min(MAX_K + 1, len(embeddings))):
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        if len(set(labels)) < 2:
            continue

        score = silhouette_score(embeddings, labels)

        if score > best_score:
            best_score = score
            best_k     = k

    return best_k, best_score


#Clusterin Methods

def cluster_kmeans(word_data):
    embeddings         = np.array([e["embedding"] for e in word_data])
    best_k, best_score = find_best_k(embeddings)
    print(f"  Best k={best_k} (silhouette={best_score:.4f})")

    labels = KMeans(
        n_clusters=best_k, random_state=RANDOM_STATE, n_init=10
    ).fit_predict(embeddings)

    return labels


def cluster_agglomerative(word_data):
    embeddings         = np.array([e["embedding"] for e in word_data])
    best_k, best_score = find_best_k(embeddings)
    print(f"  Best k={best_k} (silhouette={best_score:.4f})")

    labels = AgglomerativeClustering(
        n_clusters=best_k
    ).fit_predict(embeddings)

    return labels


def cluster_hdbscan(word_data):
    embeddings = np.array([e["embedding"] for e in word_data])

    clusterer = hdbscan.HDBSCAN(min_cluster_size=3)
    labels    = clusterer.fit_predict(embeddings)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = list(labels).count(-1)
    print(f"  Clusters found={n_clusters}, noise points={n_noise}")

    # Fallback: if everything is noise, assign all to cluster 0
    if n_clusters == 0:
        print("  Warning: all points classified as noise, assigning to cluster 0")
        labels = np.zeros(len(embeddings), dtype=int)
        return labels

    # Assign each noise point to nearest cluster centroid
    if n_noise > 0:
        # Compute centroid for each cluster
        unique_clusters = [c for c in set(labels) if c != -1]
        centroids = np.array([
            embeddings[labels == c].mean(axis=0)
            for c in unique_clusters
        ])

        # For each noise point find nearest centroid
        noise_indices = np.where(labels == -1)[0]
        for idx in noise_indices:
            noise_vec  = embeddings[idx]

            # Cosine similarity between noise point and each centroid
            similarities = np.dot(centroids, noise_vec) / (
                np.linalg.norm(centroids, axis=1) * np.linalg.norm(noise_vec) + 1e-10
            )

            nearest_cluster   = unique_clusters[np.argmax(similarities)]
            labels[idx]       = nearest_cluster

        print(f"  Reassigned {n_noise} noise points to nearest centroid")

    return labels


#Main

METHODS = {
    "kmeans"       : cluster_kmeans,
    "agglomerative": cluster_agglomerative,
    "hdbscan"      : cluster_hdbscan
}


def main():
    parser = argparse.ArgumentParser(description="Subtask 1a Clustering")
    parser.add_argument(
        "--method",
        type=str,
        choices=["kmeans", "agglomerative", "hdbscan"],
        default="kmeans",
        help="Clustering method to use (default: kmeans)"
    )
    args = parser.parse_args()

    cluster_fn = METHODS[args.method]
    print(f"Method: {args.method}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    words = get_all_words()
    print(f"Found {len(words)} words: {words}")

    results = []

    for word in words:
        print(f"\nClustering: {word}")

        word_data = load_embeddings(word)
        labels    = cluster_fn(word_data)

        for entry, label in zip(word_data, labels):
            results.append({
                "word"         : entry["word"],
                "period_label" : entry["period_label"],
                "sentence_id"  : entry["sentence_id"],
                "label"        : [int(label)]
            })

        print(f"  {len(word_data)} usages → {len(set(labels))} clusters")

    # Save JSONL with correct naming convention
    jsonl_filename = f"{LANG}_subtask1.jsonl"
    jsonl_path     = os.path.join(OUTPUT_DIR, jsonl_filename)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for entry in results:
            f.write(json.dumps(entry) + "\n")

    print(f"\nSaved → {jsonl_path}")

    # Zip the output
    zip_path = os.path.join(OUTPUT_DIR, "submission.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(jsonl_path, arcname=jsonl_filename)

    print(f"Zipped → {zip_path}")
    print(f"\nDone.")


if __name__ == "__main__":
    main()