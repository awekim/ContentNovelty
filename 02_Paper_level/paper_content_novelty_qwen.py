import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

LOCAL_MODEL_PATH = r"F:/LLM/qwen3_emb_8b"
DATA_DIR = r"E:/Data_for_Practice/ContentNovelty/"
EMB_TEMP_PATH = os.path.join(DATA_DIR, "temp_embeddings.dat") 

BATCH_SIZE = 256  
MAX_LENGTH = 512
EMB_DIM = 4096    # Qwen-8B의 일반적인 hidden_size 

OUT_CENT_PARQUET = os.path.join(DATA_DIR, "Paper/2_centroids_qwen.parquet")
OUT_INT_NOV_CSV = os.path.join(DATA_DIR, "Paper/3_internal_with_novelty_qwen.csv")
OUT_EXT_CSV = os.path.join(DATA_DIR, "Paper/3_external_with_novelty_qwen.csv")

device = torch.device("cuda")

### Load model
tqdm.write("--- Loading Model ---")
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(
    LOCAL_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="sdpa"
).to(device).eval()

### Load data
tqdm.write("--- Loading Data ---")
affinst = pd.read_csv(os.path.join(DATA_DIR, "affinst_ed.csv"), 
                      usecols=['pubid', 'period', 'subject', 'type', 'ID'])
affinst = affinst.drop_duplicates(subset=['pubid', 'period', 'subject', 'type'])

publication = pd.read_csv(os.path.join(DATA_DIR, "publication_ed.csv"), 
                          usecols=['pubid', 'abstract'])

df_all = affinst.merge(publication, on="pubid", how="inner")
del affinst, publication # 원본 데이터 삭제

df_all = df_all[df_all["abstract"].notna()].reset_index(drop=True)
total_docs = len(df_all)

### Create Memmap embedding
tqdm.write(f"--- Encoding {total_docs:,} documents to Memmap ---")

# 디스크에 거대 행렬 공간 생성 (RAM 대신 HDD/SSD 공간 사용)
fp = np.memmap(EMB_TEMP_PATH, dtype='float32', mode='w+', shape=(total_docs, EMB_DIM))

instruction = "Represent this scientific abstract for academic novelty analysis: "

with torch.no_grad():
    for i in tqdm(range(0, total_docs, BATCH_SIZE), desc="Embedding"):
        end = min(i + BATCH_SIZE, total_docs)
        batch_texts = [instruction + str(t) for t in df_all["abstract"].iloc[i:end]]
        
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, 
                           padding=True, max_length=MAX_LENGTH).to(device)
        
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :] #Depending on embeding feature: [0, 0, :] or [:, -1, :] 
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        fp[i:end, :] = embeddings.float().cpu().numpy()
        
        if i % (BATCH_SIZE * 10) == 0:
            fp.flush() 

### Measure centroid and novelty
tqdm.write("--- Calculating Centroids and Novelty ---")

centroids = {}
df_all["content_novelty"] = 0.0

for (per, subj), group in tqdm(df_all.groupby(["period", "subject"]), desc="Centroids"):
    indices = group.index.values
    group_vecs = fp[indices, :]
    
    avg_vec = group_vecs.mean(axis=0)
    avg_vec /= (np.linalg.norm(avg_vec) + 1e-12)
    centroids[(per, subj)] = avg_vec
    
    similarities = np.dot(group_vecs, avg_vec)
    df_all.loc[indices, "content_novelty"] = 1.0 - similarities

### Save results
tqdm.write("--- Saving Results ---")

df_all[df_all["type"] == "Internal"][["ID", "period", "subject", "pubid", "content_novelty"]].to_csv(OUT_INT_NOV_CSV, index=False)
df_all[df_all["type"] == "External"][["ID", "period", "subject", "pubid", "content_novelty"]].to_csv(OUT_EXT_CSV, index=False)

cent_data = [{"period": k[0], "subject": k[1]} for k in centroids.keys()]
pd.DataFrame(cent_data).to_parquet(OUT_CENT_PARQUET)

# 임시 파일 삭제 여부 결정 (필요시 보관)
# fp._mmap.close() 
# os.remove(EMB_TEMP_PATH)

tqdm.write("--- All tasks completed successfully ---")