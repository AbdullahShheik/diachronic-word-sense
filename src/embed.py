import json
import pickle
import os
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


#Config
USAGES_PATH      = "dev/SV/usages.jsonl"
DEFINITIONS_PATH = "dev/SV/definitions.jsonl"
USAGES_OUT_DIR   = "embeddings/SV/usages"
DEFINITIONS_OUT  = "embeddings/SV/definitions.pkl"
MODEL_NAME       = "pierluigic/xl-lexeme"


def load_model():
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model


def get_target_embedding(sentence, start, end, tokenizer, model):
    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=512
    )

    offset_mapping = inputs.pop("offset_mapping")[0]

    with torch.no_grad():
        outputs = model(**inputs)

    hidden_states = outputs.last_hidden_state[0]  # shape: (seq_len, 768)

    # Find which tokens correspond to the target word using character offsets
    target_indices = [
        idx for idx, (tok_start, tok_end) in enumerate(offset_mapping)
        if tok_start >= start and tok_end <= end and tok_end > tok_start
    ]

    if target_indices:
        # Average embeddings of all target word tokens
        target_embedding = hidden_states[target_indices].mean(dim=0)
    else:
        # Fallback: average all non-special tokens
        print(f"  Warning: no tokens found for offsets [{start}:{end}] in sentence, using fallback.")
        target_embedding = hidden_states[1:-1].mean(dim=0)

    return target_embedding.cpu().numpy()


def embed_usages(tokenizer, model):
    print("\nEmbedding usages...")

    # Group all usages by word first
    word_usages = defaultdict(list)
    with open(USAGES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            word_usages[entry["word"]].append(entry)

    os.makedirs(USAGES_OUT_DIR, exist_ok=True)

    for word, usages in word_usages.items():
        print(f"\nProcessing word: {word} ({len(usages)} usages)")
        word_data = []

        for usage in tqdm(usages, desc=f"  {word}"):
            embedding = get_target_embedding(
                sentence  = usage["text"],
                start     = usage["start"],
                end       = usage["end"],
                tokenizer = tokenizer,
                model     = model
            )

            word_data.append({
                "word"         : usage["word"],
                "sentence_id"  : usage["sentence_id"],
                "period_label" : usage["period_label"],
                "year"         : usage["year"],
                "embedding"    : embedding       # numpy array shape (768,)
            })

        out_path = os.path.join(USAGES_OUT_DIR, f"{word}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(word_data, f)
        print(f"  Saved {len(word_data)} embeddings → {out_path}")


def embed_definitions(tokenizer, model):
    print("\nEmbedding definitions...")

    definitions_data = {}

    with open(DEFINITIONS_PATH, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f]

    for entry in tqdm(entries, desc="  definitions"):
        word       = entry["word"]
        definition = entry["definition"]

        # For definitions we embed the entire definition text as a sentence
        embedding = get_target_embedding(
            sentence  = definition,
            start     = 0,
            end       = len(definition),
            tokenizer = tokenizer,
            model     = model
        )

        definitions_data[word] = {
            "word"       : word,
            "definition" : definition,
            "embedding"  : embedding    # numpy array shape (768,)
        }

    os.makedirs(os.path.dirname(DEFINITIONS_OUT), exist_ok=True)
    with open(DEFINITIONS_OUT, "wb") as f:
        pickle.dump(definitions_data, f)
    print(f"  Saved {len(definitions_data)} definition embeddings → {DEFINITIONS_OUT}")


def main():
    tokenizer, model = load_model()
    embed_usages(tokenizer, model)
    embed_definitions(tokenizer, model)
    print("\nDone. All embeddings saved.")


if __name__ == "__main__":
    main()