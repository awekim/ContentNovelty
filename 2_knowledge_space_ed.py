import os
os.environ["XFORMERS_DISABLED"] = "1"   # xFormers FMHA 비활성화
# os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 필요시 주석 해제: 강제로 CPU만 사용
import torch
try:
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
except Exception:
    pass
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_distances

### path settings
SAVE_PATH = r"D:/LLM/specter"                          
DATA_DIR = r"D:/DataforPractice/ContentNovelty/"       
OUT_KNOW = os.path.join(DATA_DIR, "2_knowledge_spaces.csv")
OUT_CENT = os.path.join(DATA_DIR, "2_centroids.csv")
OUT_EXT  = os.path.join(DATA_DIR, "3_external_with_novelty.csv")

### Load LLM
tokenizer = AutoTokenizer.from_pretrained(SAVE_PATH)
model = AutoModel.from_pretrained(SAVE_PATH)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Device:", device)
print("Model on:", next(model.parameters()).device)

# ========= Utils =========
def encode_texts(texts, batch_size=16, max_length=256):
    """SPECTER CLS 임베딩을 배치로 반환 (np.ndarray, shape=[N, D])."""
    all_vecs = []
    for s in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[s:s+batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", truncation=True,
            padding=True, max_length=max_length
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            cls = outputs.last_hidden_state[:, 0, :]        # [CLS]
        all_vecs.append(cls.detach().cpu().numpy())
    return np.vstack(all_vecs) if all_vecs else np.empty((0, model.config.hidden_size))

def vec_to_str(v: np.ndarray) -> str:
    """CSV 저장용: 공백으로 join (읽을 때는 str.split 후 float 변환)."""
    return " ".join(f"{x:.6f}" for x in v.tolist())

def str_to_vec(s: str) -> np.ndarray:
    return np.array([float(x) for x in s.split(" ")])

# ========= Data =========
affinst = pd.read_csv(os.path.join(DATA_DIR, "affinst_ed.csv"))
publication = pd.read_csv(os.path.join(DATA_DIR, "publication_ed.csv"))

ID_list = affinst['ID'].unique().tolist()

# ========= Pass 1: INTERNAL 임베딩 & 센트로이드 =========
rows_know = []   # 세부 관측치(knowledge spaces)
rows_cent = []   # (EU_NUTS_ID, period, subject)별 센트로이드

for the_id in ID_list:
    print("\n=== INTERNAL | ID:", the_id, "===")
    df_int = (
        affinst.loc[affinst["ID"] == the_id]
        .query('type == "Internal"')
        .merge(publication, on="pubid", how="inner")
        .copy()
    )
    # 제목/초록 정리
    df_int = df_int[df_int["abstract"].notna() & (df_int["abstract"].str.strip() != "")]
    if df_int.empty:
        print("  -> No Internal rows after filtering. Skipping.")
        continue

    df_int["input_text"] = df_int["abstract"].fillna("")
    # 임베딩
    vecs = encode_texts(df_int["input_text"].tolist(), batch_size=16)
    df_int["__emb_vec"] = list(vecs)

    # knowledge_spaces 저장용
    for _, r in df_int.iterrows():
        rows_know.append({
            "ID": the_id,
            "EU_NUTS_ID": r["EU_NUTS_ID"],
            "period": r["period"],
            "subject": r["subject"],
            "pubid": r["pubid"],
            "embedding": vec_to_str(r["__emb_vec"]),
        })

    # 그룹별 centroid
    grp = df_int.groupby(["EU_NUTS_ID", "period", "subject"], dropna=False)["__emb_vec"].apply(list)
    for (nuts, per, subj), vec_list in grp.items():
        mat = np.vstack(vec_list) if len(vec_list) else np.empty((0, model.config.hidden_size))
        if mat.size == 0:
            continue
        mat = mat[~np.isnan(mat).any(axis=1)]
        if mat.shape[0] == 0:
            continue
        centroid = mat.mean(axis=0)
        rows_cent.append({
            "ID": the_id,
            "EU_NUTS_ID": nuts,
            "period": per,
            "subject": subj,
            "centroid": vec_to_str(centroid),
            "n_docs": mat.shape[0],
        })

df_know = pd.DataFrame(rows_know)
df_cent = pd.DataFrame(rows_cent)
if not df_know.empty:
    df_know.to_csv(OUT_KNOW, index=False)
if not df_cent.empty:
    df_cent.to_csv(OUT_CENT, index=False)
print(f"\nSaved:\n - {OUT_KNOW} ({len(df_know)} rows)\n - {OUT_CENT} ({len(df_cent)} rows)")

# ========= Pass 2: EXTERNAL 임베딩 & novelty =========
centroids = {
    (row["EU_NUTS_ID"], row["period"], row["subject"]): str_to_vec(row["centroid"])
    for _, row in df_cent.iterrows()
}

rows_ext = []
for the_id in ID_list:
    print("\n=== EXTERNAL | ID:", the_id, "===")
    df_ext = (
        affinst.loc[affinst["ID"] == the_id]
        .query('type == "External"')
        .merge(publication, on="pubid", how="inner")
        .copy()
    )
    df_ext = df_ext[df_ext["abstract"].notna() & (df_ext["abstract"].str.strip() != "")]
    if df_ext.empty:
        print("  -> No External rows after filtering. Skipping.")
        continue

    df_ext["input_text"] = df_ext["abstract"].fillna("")
    vecs_ext = encode_texts(df_ext["input_text"].tolist(), batch_size=16)
    df_ext["__emb_vec"] = list(vecs_ext)

    def compute_novelty(row):
        key = (row['EU_NUTS_ID'], row['period'], row['subject'])
        centroid = centroids.get(key)
        if centroid is None:
            return np.nan
        v = row["__emb_vec"]
        if np.isnan(v).any():
            return np.nan
        return float(cosine_distances(v.reshape(1, -1), centroid.reshape(1, -1))[0][0])

    df_ext["content_novelty"] = df_ext.apply(compute_novelty, axis=1)

    rows_ext.extend(df_ext[[
        "ID","EU_NUTS_ID","period","subject","pubid","content_novelty"
    ]].to_dict("records"))

df_ext_out = pd.DataFrame(rows_ext)
if not df_ext_out.empty:
    df_ext_out.to_csv(OUT_EXT, index=False)
    print(f"\nSaved:\n - {OUT_EXT} ({len(df_ext_out)} rows)")
else:
    print("\nNo External novelty rows to save.")
