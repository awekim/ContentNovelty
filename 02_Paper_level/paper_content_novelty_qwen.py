import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import gc

# pip install flash-attn --no-build-isolation

# ==========================================
# 1. 설정 (Settings)
# ==========================================
# 로컬 모델 경로 (이미 다운로드 완료된 경로)
LOCAL_MODEL_PATH = r"F:/LLM/qwen3_emb_0.6b"
DATA_DIR  = r"E:/Data_for_Practice/ContentNovelty/"

# RTX 5090 최적화 설정
# 8B 모델 기준 128~256 사이가 적당합니다. (VRAM 32GB 활용)
BATCH_SIZE  = 128 
MAX_LENGTH  = 512   
NORMALIZE   = True  

# 출력 경로 (Qwen 모델임을 명시하기 위해 파일명 수정)
OUT_CENT_PARQUET = os.path.join(DATA_DIR, "Paper/2_centroids_qwen.parquet")
OUT_INT_NOV_CSV = os.path.join(DATA_DIR, "Paper/3_internal_with_novelty_qwen.csv")
OUT_EXT_CSV = os.path.join(DATA_DIR, "Paper/3_external_with_novelty_qwen.csv")

# ==========================================
# 2. 모델 로드 (로컬 경로 참조 및 5090 최적화)
# ==========================================
tqdm.write(f"--- Loading Qwen3-8B from Local: {LOCAL_MODEL_PATH} ---")

tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, trust_remote_code=True)

# flash-attn 대신 PyTorch 내장 가속(sdpa)을 사용합니다.
model = AutoModel.from_pretrained(
    LOCAL_MODEL_PATH,
    torch_dtype=torch.bfloat16,    # dtype 대신 torch_dtype 사용 (경고 방지)
    trust_remote_code=True,
    attn_implementation="sdpa"     # 최적화된 내장 엔진 사용
    # attn_implementation="flash_attention_2"  # Flash Attention 2 활성화
).to("cuda")

model.eval()
device = torch.device("cuda")

# ==========================================
# 3. 유틸리티 함수
# ==========================================
def encode_texts(texts, batch_size=BATCH_SIZE, max_length=MAX_LENGTH):
    instruction = "Represent this scientific abstract for academic novelty analysis: "
    vecs = []
    
    for s in tqdm(range(0, len(texts), batch_size), desc="    Embedding Progress", leave=False):
        batch = texts[s:s+batch_size]
        input_texts = [instruction + str(t) for t in batch]
        
        inputs = tokenizer(
            input_texts, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=max_length
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # [CLS] 토큰 추출
            embeddings = outputs.last_hidden_state[:, 0, :]
            
            if NORMALIZE:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            # 핵심 수정: numpy는 bfloat16을 지원하지 않으므로 
            # 반드시 .float()를 호출하여 float32로 변환 후 cpu로 넘겨야 합니다.
            vecs.append(embeddings.float().cpu().numpy())
            
    # 840만 건의 경우 vstack 시 메모리 부족이 올 수 있으므로 주의가 필요합니다.
    return np.vstack(vecs)

def cosine_distance(v, c):
    return float(1.0 - np.dot(v, c))

# ==========================================
# 4. 데이터 로드 및 분석 시작
# ==========================================
tqdm.write("--- Loading & Cleaning Data ---")
affinst = pd.read_csv(os.path.join(DATA_DIR, "affinst_ed.csv")).drop_duplicates(subset=['pubid', 'period', 'subject', 'type'])
publication = pd.read_csv(os.path.join(DATA_DIR, "publication_ed.csv"))

# --- Pass 1: Internal Docs & Centroids ---
df_int = affinst.query('type == "Internal"').merge(publication, on="pubid", how="inner")
df_int = df_int[df_int["abstract"].notna()].reset_index(drop=True)

tqdm.write(f"Encoding {len(df_int):,} Internal documents...")
emb_int = encode_texts(df_int["abstract"].tolist())
df_int["__emb_vec"] = list(emb_int)

# 중심점(Centroid) 계산 및 저장
centroids_global = {}
cent_rows = []
for (per, subj), group in df_int.groupby(["period", "subject"]):
    vecs = np.vstack(group["__emb_vec"].values)
    avg_vec = vecs.mean(axis=0)
    avg_vec = avg_vec / (np.linalg.norm(avg_vec) + 1e-12)
    centroids_global[(per, subj)] = avg_vec
    cent_rows.append({"period": per, "subject": subj, "n_docs": len(group)})

pd.DataFrame(cent_rows).to_parquet(OUT_CENT_PARQUET)

# --- Pass 2: Internal Novelty ---
def get_novelty(row):
    key = (row['period'], row['subject'])
    centroid = centroids_global.get(key)
    if centroid is None: return np.nan
    return cosine_distance(row["__emb_vec"], centroid)

df_int["content_novelty"] = df_int.apply(get_novelty, axis=1)
df_int[["ID", "period", "subject", "pubid", "content_novelty"]].to_csv(OUT_INT_NOV_CSV, index=False)

# --- Pass 3: External Novelty ---
df_ext = affinst.query('type == "External"').merge(publication, on="pubid", how="inner")
df_ext = df_ext[df_ext["abstract"].notna()].reset_index(drop=True)

if not df_ext.empty:
    tqdm.write(f"Encoding {len(df_ext):,} External documents...")
    emb_ext = encode_texts(df_ext["abstract"].tolist())
    df_ext["__emb_vec"] = list(emb_ext)
    df_ext["content_novelty"] = df_ext.apply(get_novelty, axis=1)
    df_ext[["ID", "period", "subject", "pubid", "content_novelty"]].to_csv(OUT_EXT_CSV, index=False)

tqdm.write("--- All tasks completed successfully ---")