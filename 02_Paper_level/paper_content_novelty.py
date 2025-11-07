import os
os.environ["XFORMERS_DISABLED"] = "1"  # xFormers FMHA deactivate
import pandas as pd
import numpy as np
import torch
try:
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
except Exception:
    pass
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import gc 

### Settings
SAVE_PATH = r"F:/LLM/specter2_base"
DATA_DIR  = r"E:/DataforPractice/ContentNovelty/"

# Parquet and Feather output paths
OUT_KNOW_PARQUET = os.path.join(DATA_DIR, "Paper/2_knowledge_spaces.parquet")
OUT_CENT_PARQUET = os.path.join(DATA_DIR, "Paper/2_centroids.parquet")
OUT_EXT_PARQUET  = os.path.join(DATA_DIR, "Paper/3_external_with_novelty.parquet")
OUT_INT_NOV_PARQUET = os.path.join(DATA_DIR, "Paper/3_internal_with_novelty.parquet")
OUT_INT_NOV_FEATHER = os.path.join(DATA_DIR, "Paper/3_internal_with_novelty.feather")

OUT_KNOW_FEATHER = os.path.join(DATA_DIR, "Paper/2_knowledge_spaces.feather")
OUT_CENT_FEATHER = os.path.join(DATA_DIR, "Paper/2_centroids.feather")
OUT_EXT_FEATHER  = os.path.join(DATA_DIR, "Paper/3_external_with_novelty.feather")

# CSV output paths
OUT_INT_NOV_CSV = os.path.join(DATA_DIR, "Paper/3_internal_with_novelty.csv")
OUT_EXT_CSV = os.path.join(DATA_DIR, "Paper/3_external_with_novelty.csv")

BATCH_SIZE  = 32
MAX_LENGTH  = 256
NORMALIZE   = True
SAVE_KNOW   = True

### Dependencies Check
try:
    import pyarrow  # noqa: F401
except Exception as e:
    raise RuntimeError(
        "This script saves tables in Parquet/Feather. Please install pyarrow:\n"
        "    pip install pyarrow"
    ) from e

### Model Load
tokenizer = AutoTokenizer.from_pretrained(SAVE_PATH)
model = AutoModel.from_pretrained(SAVE_PATH)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
tqdm.write(f"Device: {device} | Model on: {next(model.parameters()).device}")

### Functions
def l2_normalize(mat: np.ndarray, axis=1, eps=1e-12) -> np.ndarray:
    """Row-wise L2 normalize."""
    denom = np.sqrt((mat * mat).sum(axis=axis, keepdims=True)) + eps
    return mat / denom

def encode_texts(texts, batch_size=BATCH_SIZE, max_length=MAX_LENGTH, normalize=NORMALIZE) -> np.ndarray:
    """Return np.ndarray [N, D] of CLS embeddings (optionally L2-normalized)."""
    vecs = []
    for s in tqdm(range(0, len(texts), batch_size), desc="    Embedding batches", leave=False):
        batch = texts[s:s+batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", truncation=True,
            padding=True, max_length=max_length
        )
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            cls = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy().astype(np.float32)
        vecs.append(cls)
    if not vecs:
        return np.empty((0, model.config.hidden_size), dtype=np.float32)
    mat = np.vstack(vecs).astype(np.float32)
    return l2_normalize(mat) if normalize else mat

def vec_to_str(v: np.ndarray) -> str:
    return " ".join(f"{x:.6f}" for x in v.tolist())

def cosine_distance_from_normalized(v: np.ndarray, c: np.ndarray) -> float:
    """v, c are L2-normalized: cosine distance = 1 - dot."""
    return float(1.0 - np.dot(v, c))

def save_tables(df, parquet_path, feather_path, label=""):
    if df is None or df.empty:
        tqdm.write(f"[SAVE] {label}: empty → skipped")
        return
    # Ensure folder exists before saving
    os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
    df.to_parquet(parquet_path, index=False)
    df.reset_index(drop=True).to_feather(feather_path)
    tqdm.write(f"[SAVE] {label}: {len(df):,} rows\n       - {parquet_path}\n       - {feather_path}")

### Data Load & Deduplication
affinst = pd.read_csv(os.path.join(DATA_DIR, "affinst_ed.csv"))
publication = pd.read_csv(os.path.join(DATA_DIR, "publication_ed.csv"))

tqdm.write("--- Deduplicating affinst table (ignoring ID) ---")
key_cols = ['pubid', 'period', 'subject', 'type']
original_rows = len(affinst)

affinst_dedup = affinst.drop_duplicates(subset=key_cols, keep='first').copy()
new_rows = len(affinst_dedup)
tqdm.write(f"Affinst rows reduced from {original_rows:,} to {new_rows:,} (duplicates removed)")

#########################################
# Pass 1: Internal Embedding & Global Centroid
#########################################
tqdm.write("--- Pass 1: Processing ALL Internal Docs & Building Global Centroids ---")

df_int_all = (
    affinst_dedup.query('type == "Internal"') 
    .merge(publication, on="pubid", how="inner")
    .copy()
)

before = len(df_int_all)
df_int_all = df_int_all[df_int_all["abstract"].notna() & (df_int_all["abstract"].str.strip() != "")]
if df_int_all.empty:
    raise RuntimeError("No valid 'Internal' documents found to build centroids.")

df_int_all["input_text"] = df_int_all["abstract"].fillna("")
emb_mat_int = encode_texts(df_int_all["input_text"].tolist())

finite_mask_int = np.isfinite(emb_mat_int).all(axis=1)
kept = int(finite_mask_int.sum())
if kept == 0:
    raise RuntimeError("All 'Internal' embeddings are invalid.")

df_int_all = df_int_all.loc[finite_mask_int].reset_index(drop=True)
emb_mat_int = emb_mat_int[finite_mask_int]
df_int_all["__emb_vec"] = list(emb_mat_int)

tqdm.write(f"Internal docs for centroids: {kept}/{before}")

rows_cent = []
centroids_global = {}  
grp = df_int_all.groupby(["period", "subject"], dropna=False)["__emb_vec"].apply(list)

pbar_cent = tqdm(grp.items(), desc="  Building global centroids", unit="group")
for (per, subj), vec_list in pbar_cent:
    if len(vec_list) == 0: continue
    mat = np.vstack(vec_list).astype(np.float32)
    centroid = mat.mean(axis=0)
    centroid = l2_normalize(centroid.reshape(1, -1))[0]
    
    centroids_global[(per, subj)] = centroid
    
    rows_cent.append({
        "period": per, "subject": subj,
        "centroid": vec_to_str(centroid), "n_docs": mat.shape[0],
    })

df_cent = pd.DataFrame(rows_cent) if rows_cent else pd.DataFrame()
save_tables(df_cent, OUT_CENT_PARQUET, OUT_CENT_FEATHER, label="centroids")


#########################################
# Pass 2: Calculate Internal Novelty
#########################################
tqdm.write("--- Pass 2: Calculating Internal Novelty (against global centroids) ---")

def compute_novelty_global(row):
    key = (row['period'], row['subject'])
    centroid = centroids_global.get(key)
    if centroid is None: return np.nan
    v = row["__emb_vec"]
    return cosine_distance_from_normalized(v, centroid)

df_int_all["content_novelty"] = df_int_all.apply(compute_novelty_global, axis=1)

df_int_nov = df_int_all[["ID", "period", "subject", "pubid", "content_novelty"]].copy()
save_tables(df_int_nov, OUT_INT_NOV_PARQUET, OUT_INT_NOV_FEATHER, label="internal_novelty")

if SAVE_KNOW:
    rows_know = [{
        "ID": r["ID"], "period": r["period"],
        "subject": r["subject"], "pubid": r["pubid"],
        "embedding": vec_to_str(r["__emb_vec"]),
    } for _, r in df_int_all.iterrows()]
    df_know = pd.DataFrame(rows_know) if rows_know else pd.DataFrame()
    save_tables(df_know, OUT_KNOW_PARQUET, OUT_KNOW_FEATHER, label="knowledge_spaces")

del df_int_all, emb_mat_int, finite_mask_int, df_know, df_int_nov, rows_know
gc.collect()


#########################################
# Pass 3: External Embedding & Novelty
#########################################
tqdm.write("--- Pass 3: Processing ALL External Docs & Calculating Novelty (against global centroids) ---")

if not centroids_global:
    tqdm.write("[WARN] No global centroids were built. Skipping external novelty calculation.")
    df_ext_out = pd.DataFrame()
else:
    df_ext_all = (
        affinst_dedup.query('type == "External"') 
        .merge(publication, on="pubid", how="inner")
        .copy()
    )

    before = len(df_ext_all)
    df_ext_all = df_ext_all[df_ext_all["abstract"].notna() & (df_ext_all["abstract"].str.strip() != "")]

    if df_ext_all.empty:
        tqdm.write("No valid 'External' documents found.")
        df_ext_out = pd.DataFrame()
    else:
        df_ext_all["input_text"] = df_ext_all["abstract"].fillna("")
        emb_mat_ext = encode_texts(df_ext_all["input_text"].tolist())

        finite_mask_ext = np.isfinite(emb_mat_ext).all(axis=1)
        kept = int(finite_mask_ext.sum())
        
        if kept == 0:
            tqdm.write("All 'External' embeddings are invalid.")
            df_ext_out = pd.DataFrame()
        else:
            df_ext_all = df_ext_all.loc[finite_mask_ext].reset_index(drop=True)
            emb_mat_ext = emb_mat_ext[finite_mask_ext]
            df_ext_all["__emb_vec"] = list(emb_mat_ext)
            tqdm.write(f"External docs for novelty: {kept}/{before}")

            df_ext_all["content_novelty"] = df_ext_all.apply(compute_novelty_global, axis=1)

            n_nan = int(df_ext_all["content_novelty"].isna().sum())
            n_val = len(df_ext_all) - n_nan
            tqdm.write(f"External novelty calculated: ok={n_val}, nan={n_nan} (no centroid match)")

            df_ext_out = df_ext_all[[
                "ID", "period", "subject", "pubid", "content_novelty"
            ]].copy()
            
            del df_ext_all, emb_mat_ext, finite_mask_ext
            gc.collect()

save_tables(df_ext_out, OUT_EXT_PARQUET, OUT_EXT_FEATHER, label="external_novelty")


##########################
# Final CSV Save
##########################
if 'df_int_nov' not in locals(): 
    try:
        df_int_nov = pd.read_parquet(OUT_INT_NOV_PARQUET)
    except Exception:
        df_int_nov = pd.DataFrame()
        
if not df_int_nov.empty:
    df_int_nov.to_csv(OUT_INT_NOV_CSV, index=False, encoding='utf-8-sig')
    tqdm.write(f"[SAVE] internal_novelty_csv: {len(df_int_nov):,} rows\n       - {OUT_INT_NOV_CSV}")

if not df_ext_out.empty:
    df_ext_out.to_csv(OUT_EXT_CSV, index=False, encoding='utf-8-sig')
    tqdm.write(f"[SAVE] external_novelty_csv: {len(df_ext_out):,} rows\n       - {OUT_EXT_CSV}")

tqdm.write("--- All processing finished. ---")