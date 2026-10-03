import json
import pickle
import os
import zipfile
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


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


def find_best_k(embeddings, min_k=MIN_K, max_k=MAX_K):
    best_k     = min_k
    best_score = -1

    for k in range(min_k, min(max_k + 1, len(embeddings))):
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        if len(set(labels)) < 2:
            continue

        #Silhouette score
        score = silhouette_score(embeddings, labels)

        if score > best_score:
            best_score = score
            best_k     = k

    return best_k, best_score


def cluster_word(word_data):
    embeddings = np.array([entry["embedding"] for entry in word_data])

    best_k, best_score = find_best_k(embeddings)
    print(f"  Best k={best_k} (silhouette={best_score:.4f})")

    kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    return labels


def get_all_words():
    return [
        Path(f).stem
        for f in os.listdir(USAGES_EMB_DIR)
        if f.endswith(".pkl")
    ]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    words = get_all_words()
    print(f"Found {len(words)} words: {words}")

    results = []

    for word in words:
        print(f"\nClustering: {word}")

        word_data = load_embeddings(word)
        labels    = cluster_word(word_data)

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