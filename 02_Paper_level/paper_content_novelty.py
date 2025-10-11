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

### Settings
SAVE_PATH = r"F:/LLM/specter2_base"
DATA_DIR  = r"F:/DataforPractice/ContentNovelty/"

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
        "     pip install pyarrow"
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
    for s in tqdm(range(0, len(texts), batch_size), desc="   Embedding batches", leave=False):
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

### Data Load
affinst = pd.read_csv(os.path.join(DATA_DIR, "affinst_ed.csv"))
publication = pd.read_csv(os.path.join(DATA_DIR, "publication_ed.csv"))

ID_list = affinst['ID'].dropna().unique().tolist()
tqdm.write(f"Total unique IDs: {len(ID_list):,}")

##########################
# Internal Embedding & Centroid
##########################
rows_know = []
rows_cent = []
rows_int_novelty = []
centroids_by_id = {}

pbar_ids_p1 = tqdm(ID_list, desc="Pass1 INTERNAL (per ID)", unit="ID")
for the_id in pbar_ids_p1:
    df_int = (
        affinst.loc[affinst["ID"] == the_id]
        .query('type == "Internal"')
        .merge(publication, on="pubid", how="inner")
        .copy()
    )

    before = len(df_int)
    
    df_int = df_int[df_int["abstract"].notna() & (df_int["abstract"].str.strip() != "")]
    if df_int.empty:
        pbar_ids_p1.set_postfix(skipped="no-internal")
        continue

    df_int["input_text"] = df_int["abstract"].fillna("")
    emb_mat = encode_texts(df_int["input_text"].tolist())

    finite_mask = np.isfinite(emb_mat).all(axis=1)
    kept = int(finite_mask.sum())
    if kept == 0:
        pbar_ids_p1.set_postfix(skipped="invalid-emb")
        continue
    df_int = df_int.loc[finite_mask].reset_index(drop=True)
    emb_mat = emb_mat[finite_mask]
    df_int["__emb_vec"] = list(emb_mat)

    pbar_ids_p1.set_postfix(rows=f"{kept}/{before}")

    centroids_by_id[the_id] = {}
    grp = df_int.groupby(["period", "subject"], dropna=False)["__emb_vec"].apply(list)
    
    for (per, subj), vec_list in tqdm(grp.items(), desc="   Build centroids", leave=False):
        if len(vec_list) == 0: continue
        mat = np.vstack(vec_list).astype(np.float32)
        centroid = mat.mean(axis=0)
        centroid = l2_normalize(centroid.reshape(1, -1))[0]
        
        centroids_by_id[the_id][(per, subj)] = centroid
        
        rows_cent.append({
            "ID": the_id, "period": per, "subject": subj,
            "centroid": vec_to_str(centroid), "n_docs": mat.shape[0],
        })

    id_centroids = centroids_by_id.get(the_id, {})
    if id_centroids:
        def compute_internal_novelty(row):
            key = (row['period'], row['subject'])
            centroid = id_centroids.get(key)
            if centroid is None: return np.nan
            v = row["__emb_vec"]
            return cosine_distance_from_normalized(v, centroid)
        df_int["content_novelty"] = df_int.apply(compute_internal_novelty, axis=1)
    else:
        df_int["content_novelty"] = np.nan

    if SAVE_KNOW:
        rows_know.extend([{
            "ID": the_id, "period": r["period"],
            "subject": r["subject"], "pubid": r["pubid"],
            "embedding": vec_to_str(r["__emb_vec"]),
        } for _, r in df_int.iterrows()])
    
    # EU_NUTS_ID가 없는 최종 결과 저장
    rows_int_novelty.extend(
        df_int[["ID", "period", "subject", "pubid", "content_novelty"]]
        .to_dict("records")
    )

df_know = pd.DataFrame(rows_know) if rows_know else pd.DataFrame()
df_cent  = pd.DataFrame(rows_cent) if rows_cent else pd.DataFrame()
df_int_nov = pd.DataFrame(rows_int_novelty) if rows_int_novelty else pd.DataFrame()

if SAVE_KNOW and not df_know.empty:
    save_tables(df_know, OUT_KNOW_PARQUET, OUT_KNOW_FEATHER, label="knowledge_spaces")
save_tables(df_cent, OUT_CENT_PARQUET, OUT_CENT_FEATHER, label="centroids")
save_tables(df_int_nov, OUT_INT_NOV_PARQUET, OUT_INT_NOV_FEATHER, label="internal_novelty")

##########################
# External Embedding & Novelty
##########################
rows_ext = []
pbar_ids_p2 = tqdm(ID_list, desc="Pass2 EXTERNAL (per ID)", unit="ID")

for the_id in pbar_ids_p2:
    df_ext = (
        affinst.loc[affinst["ID"] == the_id]
        .query('type == "External"')
        .merge(publication, on="pubid", how="inner")
        .copy()
    )

    before = len(df_ext)

    df_ext = df_ext[df_ext["abstract"].notna() & (df_ext["abstract"].str.strip() != "")]
    if df_ext.empty:
        pbar_ids_p2.set_postfix(skipped="no-external")
        continue

    df_ext["input_text"] = df_ext["abstract"].fillna("")
    emb_ext = encode_texts(df_ext["input_text"].tolist())

    finite_mask = np.isfinite(emb_ext).all(axis=1)
    kept = int(finite_mask.sum())
    if kept == 0:
        pbar_ids_p2.set_postfix(skipped="invalid-emb")
        continue
    df_ext = df_ext.loc[finite_mask].reset_index(drop=True)
    emb_ext = emb_ext[finite_mask]
    df_ext["__emb_vec"] = list(emb_ext)

    id_centroids = centroids_by_id.get(the_id, {})
    if not id_centroids:
        pbar_ids_p2.set_postfix(skipped="no-centroid")
        continue

    def compute_novelty(row):
        key = (row['period'], row['subject'])
        centroid = id_centroids.get(key)
        if centroid is None:
            return np.nan
        v = row["__emb_vec"]
        return cosine_distance_from_normalized(v, centroid)

    df_ext["content_novelty"] = df_ext.apply(compute_novelty, axis=1)

    n_nan = int(df_ext["content_novelty"].isna().sum())
    n_val = len(df_ext) - n_nan
    pbar_ids_p2.set_postfix(rows=f"{kept}/{before}", novelty_ok=n_val, novelty_nan=n_nan)
    
    rows_ext.extend(df_ext[[
        "ID","period","subject","pubid","content_novelty"
    ]].to_dict("records"))

df_ext_out = pd.DataFrame(rows_ext) if rows_ext else pd.DataFrame()
save_tables(df_ext_out, OUT_EXT_PARQUET, OUT_EXT_FEATHER, label="external_novelty")

if not df_int_nov.empty:
    df_int_nov.to_csv(OUT_INT_NOV_CSV, index=False, encoding='utf-8-sig')
    tqdm.write(f"[SAVE] internal_novelty_csv: {len(df_int_nov):,} rows\n       - {OUT_INT_NOV_CSV}")

if not df_ext_out.empty:
    df_ext_out.to_csv(OUT_EXT_CSV, index=False, encoding='utf-8-sig')
    tqdm.write(f"[SAVE] external_novelty_csv: {len(df_ext_out):,} rows\n       - {OUT_EXT_CSV}")