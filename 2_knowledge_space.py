#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import normalize as sk_normalize

# -------------------------------
INPUT_FILE = r"D:/DataforPractice/ContentNovelty/1_df_filtered.csv"
MODEL_PATH = r"D:/LLM/specter"
OUTPUT_DIR = r"D:/DataforPractice/ContentNovelty/out"
BATCH_SIZE = 16
MAX_LENGTH = 512
NORMALIZE = True   # True면 L2 정규화 수행
# -------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_model(model_path: str):
    use_cuda = torch.cuda.is_available()

    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    # model = AutoModel.from_pretrained(model_path, torch_dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,          # 호환성 우선
        attn_implementation="eager"         # flash-attn/xformers 비활성화
    )
    
    device = torch.device('cuda' if use_cuda else 'cpu')
    model.to(device)
    model.eval()
    return tokenizer, model, device

@torch.no_grad()
def encode_batch(texts, tokenizer, model, device, max_length=512):
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length
    ).to(device)

    use_amp = (device.type == "cuda")
    if use_amp:
        with torch.cuda.amp.autocast():
            outputs = model(**encoded)
    else:
        outputs = model(**encoded)

    cls = outputs.last_hidden_state[:, 0, :]
    return cls.detach().cpu().numpy()

def main():
    # Load data
    df = pd.read_csv(INPUT_FILE)
    required_cols = ["itemtitle", "abstract", "EU_NUTS_ID", "period", "subject"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input CSV: {missing}")

    # Drop empty abstracts
    df = df.dropna(subset=["abstract"])
    df = df[df["abstract"].str.strip() != ""]
    df = df.reset_index(drop=True)

    # Combine text
    df["itemtitle"] = df["itemtitle"].fillna("")
    df["text"] = (df["itemtitle"] + ". " + df["abstract"]).str.strip()

    # Load model
    tokenizer, model, device = load_model(MODEL_PATH)

    # Embed in batches
    texts = df["text"].tolist()
    all_embs = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding documents"):
        batch_texts = texts[i:i+BATCH_SIZE]
        embs = encode_batch(batch_texts, tokenizer, model, device, max_length=MAX_LENGTH)
        all_embs.append(embs)
    embeddings = np.vstack(all_embs) if len(all_embs) > 0 else np.zeros((0, 768), dtype=np.float32)

    if NORMALIZE and embeddings.shape[0] > 0:
        embeddings = sk_normalize(embeddings, norm="l2", axis=1)

    # Save baseline outputs
    np.save(os.path.join(OUTPUT_DIR, "embeddings.npy"), embeddings)
    df[["EU_NUTS_ID", "period", "subject", "itemtitle"]].to_csv(os.path.join(OUTPUT_DIR, "row_index.csv"), index=False)

    # Build knowledge spaces
    knowledge_spaces = defaultdict(list)
    for idx, row in df.iterrows():
        key = (row["EU_NUTS_ID"], row["period"], row["subject"])
        knowledge_spaces[key].append(embeddings[idx])

    # Save knowledge spaces as NPZ
    npz_dict = {}
    for (nuts, period, subject), vecs in knowledge_spaces.items():
        k = f"{nuts}||{period}||{subject}"
        npz_dict[k] = np.vstack(vecs) if len(vecs) > 0 else np.zeros((0, embeddings.shape[1]), dtype=embeddings.dtype)
    np.savez_compressed(os.path.join(OUTPUT_DIR, "knowledge_spaces.npz"), **npz_dict)

    # 1) 2_knowledge_spaces.csv  (per-row vectors as JSON strings)
    ks_rows = []
    for (nuts, period, subject), vecs in knowledge_spaces.items():
        for vec in vecs:
            ks_rows.append({
                "EU_NUTS_ID": nuts,
                "period": period,
                "subject": subject,
                "embedding": json.dumps(vec.tolist(), ensure_ascii=False)
            })
    pd.DataFrame(ks_rows, columns=["EU_NUTS_ID", "period", "subject", "embedding"]).to_csv(
        os.path.join(OUTPUT_DIR, "2_knowledge_spaces.csv"), index=False
    )

    # 2) centroids.csv (as before) + 2_centroids.csv (same content, different filename)
    rows = []
    for (nuts, period, subject), vecs in knowledge_spaces.items():
        if len(vecs) == 0:
            continue
        arr = np.vstack(vecs)
        centroid = arr.mean(axis=0)
        rows.append({
            "EU_NUTS_ID": nuts,
            "period": period,
            "subject": subject,
            "centroid": json.dumps(centroid.tolist(), ensure_ascii=False)
        })
    centroids_df = pd.DataFrame(rows, columns=["EU_NUTS_ID", "period", "subject", "centroid"])
    centroids_df.to_csv(os.path.join(OUTPUT_DIR, "centroids.csv"), index=False)
    centroids_df.to_csv(os.path.join(OUTPUT_DIR, "2_centroids.csv"), index=False)

    print(f"Done. Saved to: {OUTPUT_DIR}")
    print("Files: embeddings.npy, row_index.csv, knowledge_spaces.npz, 2_knowledge_spaces.csv, centroids.csv, 2_centroids.csv")

if __name__ == "__main__":
    main()
