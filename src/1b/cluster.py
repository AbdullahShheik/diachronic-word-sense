import json
import pickle
import os
import zipfile
import argparse
from pathlib import Path

import numpy as np
from sklearn.cluster import AgglomerativeClustering


#Config
LANG            = "SV"
USAGES_EMB_DIR  = f"embeddings/{LANG}/usages"
OUTPUT_DIR      = f"outputs/{LANG}/1b"

# Subtask 1b is scored on how each word's sense distribution shifts between
# consecutive periods (JSD), ranked across ALL words at once (Spearman).
# So we want:
#   - clusters shared across periods (cluster all usages of a word together)
#   - the same granularity rule for every word (one absolute distance threshold),
#     so JSD values are comparable between words
# Embeddings are tightly packed (median pairwise cosine distance ~0.03), so the
# threshold is small.
# Defaults chosen from a grid sweep on SV dev (threshold x min-size x margin):
# 0.08 / 1 / 0.01 gave Spearman 0.671, F-BCubed 0.729, and was stable across
# margins. Merging small clusters (min-size > 1) generally hurt 1b, and with
# min-size >= 3 most words collapse into a single cluster at thresholds >= 0.06.
DEFAULT_THRESHOLD = 0.08   # cosine distance at which to stop merging clusters
DEFAULT_MIN_SIZE  = 1      # clusters smaller than this are merged into nearest one (1 = off)
DEFAULT_MARGIN    = 0.01   # >0 enables a second label for ambiguous usages


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


def normalize(vectors):
    return vectors / (np.linalg.norm(vectors, axis=-1, keepdims=True) + 1e-10)


def compute_centroids(embeddings, labels):
    clusters  = sorted(set(labels))
    centroids = np.array([embeddings[labels == c].mean(axis=0) for c in clusters])
    return clusters, normalize(centroids)


#Pipeline Steps

def cluster_threshold(embeddings, threshold):
    # Single usage: nothing to cluster
    if len(embeddings) < 2:
        return np.zeros(len(embeddings), dtype=int)

    clusterer = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric="cosine",
        linkage="average"
    )
    return clusterer.fit_predict(embeddings)


def merge_small_clusters(embeddings, labels, min_size):
    labels = labels.copy()

    # Repeatedly fold the smallest undersized cluster into its nearest neighbour
    while True:
        clusters, sizes = np.unique(labels, return_counts=True)
        if len(clusters) < 2 or sizes.min() >= min_size:
            break

        smallest = clusters[np.argmin(sizes)]
        _, centroids = compute_centroids(embeddings, labels)

        own_idx      = list(clusters).index(smallest)
        similarities = centroids @ centroids[own_idx]
        similarities[own_idx] = -np.inf

        nearest = clusters[np.argmax(similarities)]
        labels[labels == smallest] = nearest

    # Relabel to 0..n-1
    _, labels = np.unique(labels, return_inverse=True)
    return labels


def assign_labels(embeddings, labels, margin):
    # Hard labels only
    if margin <= 0:
        return [[int(l)] for l in labels]

    clusters, centroids = compute_centroids(embeddings, labels)
    if len(clusters) < 2:
        return [[int(l)] for l in labels]

    similarities = embeddings @ centroids.T     # shape: (n_usages, n_clusters)

    results = []
    for i, own in enumerate(labels):
        own_sim = similarities[i, own]
        others  = similarities[i].copy()
        others[own] = -np.inf
        second  = int(np.argmax(others))

        # Ambiguous usage: almost as close to another sense → give both labels
        if own_sim - others[second] < margin:
            results.append([int(own), second])
        else:
            results.append([int(own)])

    return results


#Main

def main():
    parser = argparse.ArgumentParser(description="Subtask 1b Clustering")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Cosine distance threshold for agglomerative clustering (default: {DEFAULT_THRESHOLD})"
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=DEFAULT_MIN_SIZE,
        help=f"Minimum cluster size; smaller clusters are merged (default: {DEFAULT_MIN_SIZE})"
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=DEFAULT_MARGIN,
        help=f"Similarity margin for adding a second label; 0 disables (default: {DEFAULT_MARGIN})"
    )
    args = parser.parse_args()

    print(f"threshold={args.threshold}, min_size={args.min_size}, margin={args.margin}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    words = get_all_words()
    print(f"Found {len(words)} words: {words}")

    results = []

    for word in words:
        print(f"\nClustering: {word}")

        word_data  = load_embeddings(word)
        embeddings = normalize(np.array([e["embedding"] for e in word_data]))

        labels = cluster_threshold(embeddings, args.threshold)
        print(f"  Threshold clustering → {len(set(labels))} clusters")

        labels = merge_small_clusters(embeddings, labels, args.min_size)
        print(f"  After merging small clusters → {len(set(labels))} clusters")

        label_lists = assign_labels(embeddings, labels, args.margin)
        n_multi     = sum(len(l) > 1 for l in label_lists)
        if n_multi:
            print(f"  {n_multi} usages given two labels")

        for entry, label in zip(word_data, label_lists):
            results.append({
                "word"         : entry["word"],
                "period_label" : entry["period_label"],
                "sentence_id"  : entry["sentence_id"],
                "label"        : label
            })

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
