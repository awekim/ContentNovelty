import os
os.environ["XFORMERS_DISABLED"] = "1"   # xFormers FMHA 비활성화
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

# =========================
# Config & Paths
# =========================
SAVE_PATH = r"D:/LLM/specter"
DATA_DIR  = r"D:/DataforPractice/ContentNovelty/"

OUT_KNOW_PARQUET = os.path.join(DATA_DIR, "2_knowledge_spaces.parquet")
OUT_CENT_PARQUET = os.path.join(DATA_DIR, "2_centroids.parquet")
OUT_EXT_PARQUET  = os.path.join(DATA_DIR, "3_external_with_novelty.parquet")

OUT_KNOW_FEATHER = os.path.join(DATA_DIR, "2_knowledge_spaces.feather")
OUT_CENT_FEATHER = os.path.join(DATA_DIR, "2_centroids.feather")
OUT_EXT_FEATHER  = os.path.join(DATA_DIR, "3_external_with_novelty.feather")

BATCH_SIZE  = 32
MAX_LENGTH  = 256   # 요청대로 256
NORMALIZE   = True  # 일관 정규화
SAVE_KNOW   = True  # knowledge_spaces 저장 여부(용량 크면 False 권장)

# =========================
# Dependencies Check
# =========================
try:
    import pyarrow  # noqa: F401
except Exception as e:
    raise RuntimeError(
        "This script saves tables in Parquet/Feather. Please install pyarrow:\n"
        "    pip install pyarrow"
    ) from e

# =========================
# Model
# =========================
tokenizer = AutoTokenizer.from_pretrained(SAVE_PATH)
model = AutoModel.from_pretrained(SAVE_PATH)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
tqdm.write(f"Device: {device} | Model on: {next(model.parameters()).device}")

# =========================
# Utils
# =========================
def l2_normalize(mat: np.ndarray, axis=1, eps=1e-12) -> np.ndarray:
    """Row-wise L2 normalize."""
    denom = np.sqrt((mat * mat).sum(axis=axis, keepdims=True)) + eps
    return mat / denom

def encode_texts(texts, batch_size=BATCH_SIZE, max_length=MAX_LENGTH, normalize=NORMALIZE) -> np.ndarray:
    """Return np.ndarray [N, D] of CLS embeddings (optionally L2-normalized)."""
    vecs = []
    for s in tqdm(range(0, len(texts), batch_size), desc="  Embedding batches", leave=False):
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
    df.to_parquet(parquet_path, index=False)
    df.reset_index(drop=True).to_feather(feather_path)
    tqdm.write(f"[SAVE] {label}: {len(df):,} rows\n       - {parquet_path}\n       - {feather_path}")

# =========================
# Data
# =========================
affinst = pd.read_csv(os.path.join(DATA_DIR, "affinst_ed.csv"))
publication = pd.read_csv(os.path.join(DATA_DIR, "publication_ed.csv"))

ID_list = affinst['ID'].dropna().unique().tolist()
tqdm.write(f"Total unique IDs: {len(ID_list):,}")

# =========================
# Pass 1: INTERNAL 임베딩 & (ID별) 센트로이드
# =========================
rows_know = []    # (선택) 개별 임베딩 저장
rows_cent = []    # ID별 (EU_NUTS_ID, period, subject) 센트로이드
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
    # 1) 초록 유효성 필터 (앞당김)
    df_int = df_int[df_int["abstract"].notna() & (df_int["abstract"].str.strip() != "")]
    filtered = len(df_int)
    if df_int.empty:
        pbar_ids_p1.set_postfix(skipped="no-internal")
        continue

    # 2) 입력 텍스트 구성
    df_int["input_text"] = df_int["abstract"].fillna("")

    # 3) 임베딩
    emb_mat = encode_texts(df_int["input_text"].tolist())

    # 4) NaN/Inf 필터링 (임베딩 직후)
    finite_mask = np.isfinite(emb_mat).all(axis=1)
    kept = int(finite_mask.sum())
    if kept == 0:
        pbar_ids_p1.set_postfix(skipped="invalid-emb")
        continue
    df_int = df_int.loc[finite_mask].reset_index(drop=True)
    emb_mat = emb_mat[finite_mask]
    df_int["__emb_vec"] = list(emb_mat)

    pbar_ids_p1.set_postfix(rows=f"{kept}/{before}")

    # 5) (선택) knowledge_spaces 저장용 행 추가
    if SAVE_KNOW:
        rows_know.extend([{
            "ID": the_id,
            "EU_NUTS_ID": r["EU_NUTS_ID"],
            "period": r["period"],
            "subject": r["subject"],
            "pubid": r["pubid"],
            "embedding": vec_to_str(r["__emb_vec"]),
        } for _, r in df_int.iterrows()])

    # 6) 그룹별 센트로이드(문서 임베딩은 이미 L2 정규화됨 → 평균 후 다시 정규화)
    centroids_by_id[the_id] = {}
    grp = df_int.groupby(["EU_NUTS_ID", "period", "subject"], dropna=False)["__emb_vec"].apply(list)

    for (nuts, per, subj), vec_list in tqdm(grp.items(), desc="  Build centroids", leave=False):
        if len(vec_list) == 0:
            continue
        mat = np.vstack(vec_list).astype(np.float32)
        centroid = mat.mean(axis=0)
        centroid = l2_normalize(centroid.reshape(1, -1))[0]  # 재정규화
        centroids_by_id[the_id][(nuts, per, subj)] = centroid

        rows_cent.append({
            "ID": the_id,
            "EU_NUTS_ID": nuts,
            "period": per,
            "subject": subj,
            "centroid": vec_to_str(centroid),
            "n_docs": mat.shape[0],
        })

# 저장 (Parquet + Feather)
df_know = pd.DataFrame(rows_know) if rows_know else pd.DataFrame()
df_cent  = pd.DataFrame(rows_cent) if rows_cent else pd.DataFrame()

if SAVE_KNOW and not df_know.empty:
    save_tables(df_know, OUT_KNOW_PARQUET, OUT_KNOW_FEATHER, label="knowledge_spaces")
save_tables(df_cent, OUT_CENT_PARQUET, OUT_CENT_FEATHER, label="centroids")

# =========================
# Pass 2: EXTERNAL 임베딩 & (ID별 센트로이드 대비) 참신성
# =========================
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
    # 1) 초록 유효성 필터 (앞당김)
    df_ext = df_ext[df_ext["abstract"].notna() & (df_ext["abstract"].str.strip() != "")]
    filtered = len(df_ext)
    if df_ext.empty:
        pbar_ids_p2.set_postfix(skipped="no-external")
        continue

    # 2) 입력 텍스트
    df_ext["input_text"] = df_ext["abstract"].fillna("")

    # 3) 임베딩
    emb_ext = encode_texts(df_ext["input_text"].tolist())

    # 4) NaN/Inf 필터링 (임베딩 직후)
    finite_mask = np.isfinite(emb_ext).all(axis=1)
    kept = int(finite_mask.sum())
    if kept == 0:
        pbar_ids_p2.set_postfix(skipped="invalid-emb")
        continue
    df_ext = df_ext.loc[finite_mask].reset_index(drop=True)
    emb_ext = emb_ext[finite_mask]
    df_ext["__emb_vec"] = list(emb_ext)

    # 5) 같은 ID의 센트로이드만 사용
    id_centroids = centroids_by_id.get(the_id, {})
    if not id_centroids:
        pbar_ids_p2.set_postfix(skipped="no-centroid")
        continue

    # 6) 참신성 계산: 코사인 거리 (1 - dot), 모두 정규화된 벡터 기준
    def compute_novelty(row):
        key = (row['EU_NUTS_ID'], row['period'], row['subject'])
        centroid = id_centroids.get(key)
        if centroid is None:
            return np.nan
        v = row["__emb_vec"]
        return cosine_distance_from_normalized(v, centroid)

    df_ext["content_novelty"] = df_ext.apply(compute_novelty, axis=1)

    # 진행중 통계 표시
    n_nan = int(df_ext["content_novelty"].isna().sum())
    n_val = len(df_ext) - n_nan
    pbar_ids_p2.set_postfix(rows=f"{kept}/{before}", novelty_ok=n_val, novelty_nan=n_nan)

    rows_ext.extend(df_ext[[
        "ID","EU_NUTS_ID","period","subject","pubid","content_novelty"
    ]].to_dict("records"))

df_ext_out = pd.DataFrame(rows_ext) if rows_ext else pd.DataFrame()
save_tables(df_ext_out, OUT_EXT_PARQUET, OUT_EXT_FEATHER, label="external_novelty")
