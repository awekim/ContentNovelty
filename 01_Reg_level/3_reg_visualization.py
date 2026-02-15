import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import itertools 

### Settings #################################################
TARGET_COMBINATIONS = [
    # (Region, Subject, Period)
    ("UKM23", "Astronomy & Astrophysics", 6),
    ("UKM23", "Physics", 6),
    ("ITF33", "Cell Biology", 6),
    ("ITH44", "Chemistry", 3)
]

MODEL_PATH = r"E:/LLM/specter2_base" 
DATA_DIR = r"E:/DataforPractice/ContentNovelty/"
KNOW_SUBDIR = os.path.join(DATA_DIR, "Region/KnowledgeSpaces_ByID_ALL_Periods")
OUTPUT_IMAGE_DIR = os.path.join(DATA_DIR, "Visualization_Outputs_Specific")
OUTPUT_IMAGE_FORMAT = "png"
OUTPUT_DPI = 150 

### CUDA/Torch Settings ###########################
# os.environ["XFORMERS_DISABLED"] = "1" 

def l2_normalize(mat: np.ndarray, axis=1, eps=1e-12) -> np.ndarray:
    denom = np.sqrt((mat * mat).sum(axis=axis, keepdims=True)) + eps
    return mat / denom

def vec_to_str(v: np.ndarray) -> str:
    return " ".join(f"{x:.6f}" for x in v.tolist())

def encode_texts(texts, tokenizer, model, device, batch_size=32, max_length=256, normalize=True) -> np.ndarray:
    vecs = []
    for s in range(0, len(texts), batch_size):
        batch = texts[s:s+batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", truncation=True, padding=True, max_length=max_length
        ).to(device) 
        with torch.no_grad():
            outputs = model(**inputs)
            cls = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy().astype(np.float32)
        vecs.append(cls)
    
    if not vecs:
        return np.empty((0, model.config.hidden_size), dtype=np.float32)
    
    mat = np.vstack(vecs).astype(np.float32)
    return l2_normalize(mat) if normalize else mat

if __name__ == "__main__":
    if not TARGET_COMBINATIONS:
        raise ValueError("TARGET_COMBINATIONS list missing.")

    print(f"Initial setup... Analyzing {len(TARGET_COMBINATIONS)} specific combinations.")
    os.makedirs(KNOW_SUBDIR, exist_ok=True)
    os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModel.from_pretrained(MODEL_PATH)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("Warning: CUDA not available. Running on CPU.")
    
    print(f"Model loaded on device: {device}")
    model.to(device)

    affinst = pd.read_csv(os.path.join(DATA_DIR, "affinst_ed.csv"))
    publication = pd.read_csv(os.path.join(DATA_DIR, "publication_ed.csv"))
    
    unique_regions_to_process = sorted(list(set([combo[0] for combo in TARGET_COMBINATIONS])))
    
    print("\n" + "="*50)
    print(f"Starting to build regional PCA maps for {len(unique_regions_to_process)} regions...")
    print("="*50)

    for current_region in tqdm(unique_regions_to_process, desc="Processing Regions", unit="region"):
        
        tqdm.write(f"\n--- Processing Region: {current_region} ---")
        tqdm.write("Step 1: Loading ALL Internal papers for this region to build PCA map...")
        
        df_region_int_ALL = (
            affinst[
                (affinst["EU_NUTS_ID"] == current_region) & 
                (affinst["type"] == "Internal")
            ]
            .merge(publication, on="pubid", how="inner")
            .dropna(subset=['abstract'])
            .copy()
        )
        df_region_int_ALL = df_region_int_ALL[df_region_int_ALL["abstract"].str.strip() != ""]

        if df_region_int_ALL.empty:
            tqdm.write(f"[Warning] No 'Internal' papers found for {current_region} at all. Skipping Region.")
            continue

        internal_embeddings_768d_ALL = encode_texts(
            df_region_int_ALL["abstract"].tolist(), tokenizer, model, device
        )
        
        df_know_region = df_region_int_ALL[['ID', 'EU_NUTS_ID', 'period', 'subject', 'pubid']].copy()
        df_know_region['embedding'] = [vec_to_str(vec) for vec in internal_embeddings_768d_ALL]
        output_feather = os.path.join(KNOW_SUBDIR, f"know_{current_region}_ALL_PERIODS.feather")
        df_know_region.to_feather(output_feather)
        
        if len(internal_embeddings_768d_ALL) < 3:
             tqdm.write(f"[Warning] Not enough data points ({len(internal_embeddings_768d_ALL)}) for PCA. Skipping Region.")
             continue
        
        pca_region = PCA(n_components=3, random_state=42)
        internal_embeddings_3d_ALL = pca_region.fit_transform(internal_embeddings_768d_ALL)
        
        df_region_int_ALL[['x', 'y', 'z']] = internal_embeddings_3d_ALL
        
        tqdm.write("Step 2: Loading and projecting ALL External papers for this region...")
        df_region_ext_ALL = (
            affinst[
                (affinst["EU_NUTS_ID"] == current_region) & 
                (affinst["type"] == "External")
            ]
            .merge(publication, on="pubid", how="inner")
            .dropna(subset=['abstract'])
            .copy()
        )
        df_region_ext_ALL = df_region_ext_ALL[df_region_ext_ALL["abstract"].str.strip() != ""]

        external_embeddings_3d_ALL = np.empty((0, 3)) 
        
        if not df_region_ext_ALL.empty:
            external_texts = df_region_ext_ALL["abstract"].tolist()
            external_embeddings_768d_ALL = encode_texts(
                external_texts, tokenizer, model, device
            )
            external_embeddings_3d_ALL = pca_region.transform(external_embeddings_768d_ALL)
            df_region_ext_ALL[['x', 'y', 'z']] = external_embeddings_3d_ALL
        
        tqdm.write(f"Step 3: Plotting requested combinations for {current_region}...")

        region_combos_to_plot = [combo for combo in TARGET_COMBINATIONS if combo[0] == current_region]

        for (reg, current_subject, current_period) in region_combos_to_plot:
            
            # Internal Centroid 
            int_mask = (df_region_int_ALL['subject'] == current_subject) & (df_region_int_ALL['period'] == current_period)
            int_points_3d = df_region_int_ALL.loc[int_mask, ['x', 'y', 'z']].values
            
            if int_points_3d.shape[0] == 0:
                tqdm.write(f"  [Info] Skipping plot for ({reg}, {current_subject}, {current_period}): No matching 'Internal' data.")
                continue
            
            internal_centroid_3d = np.mean(int_points_3d, axis=0)

            # External Centroid 
            ext_mask = (df_region_ext_ALL['subject'] == current_subject) & (df_region_ext_ALL['period'] == current_period)
            ext_points_3d = df_region_ext_ALL.loc[ext_mask, ['x', 'y', 'z']].values
            
            external_centroid_3d = np.array([]) 
            if ext_points_3d.shape[0] > 0:
                external_centroid_3d = np.mean(ext_points_3d, axis=0)
            
            fig = plt.figure(figsize=(16, 12))
            ax = fig.add_subplot(111, projection='3d')

            # Layer 1: Internal 
            ax.scatter(
                internal_embeddings_3d_ALL[:, 0], internal_embeddings_3d_ALL[:, 1], internal_embeddings_3d_ALL[:, 2],
                color='cornflowerblue', s=10, 
                alpha=0.05, 
                label=f"Internal Space (All Periods, Region: {current_region})",
                zorder=1,
                depthshade=False
            )
            
            # Layer 2: Internal Centroid 
            ax.scatter(
                internal_centroid_3d[0], internal_centroid_3d[1], internal_centroid_3d[2],
                color='blue',
                marker='*',
                s=500,
                edgecolors='black', 
                linewidth=1.5, 
                label=f"Internal Centroid (Subj: {current_subject}, Per: {current_period})",
                zorder=10,
                alpha=1.0
            )

            # Layer 3: External Centroid 
            if external_centroid_3d.shape[0] > 0:
                ax.scatter(
                    external_centroid_3d[0], external_centroid_3d[1], external_centroid_3d[2],
                    color='red',
                    marker='*',
                    s=500,
                    edgecolors='black', 
                    linewidth=1.5, 
                    label=f"External Centroid (Subj: {current_subject}, Per: {current_period})",
                    zorder=10,
                    alpha=1.0
                )
            else:
                 tqdm.write(f"  [Info] No 'External' data for ({reg}, {current_subject}, {current_period}).")

            ax.set_title(f"Centroid Comparison (Region: {current_region}, Subj: {current_subject}, Per: {current_period})", fontsize=18, pad=20)
            ax.set_xlabel("PCA Dimension 1", fontsize=12)
            ax.set_ylabel("PCA Dimension 2", fontsize=12)
            ax.set_zlabel("PCA Dimension 3", fontsize=12)
            ax.view_init(elev=20, azim=-60) 
            ax.legend(loc='upper left', bbox_to_anchor=(0.01, 0.95))
            
            safe_subject = str(current_subject).replace('/', '_').replace('\\', '_').replace(' ', '')
            
            output_path = os.path.join(OUTPUT_IMAGE_DIR, f"plot_{current_region}_s_{safe_subject}_p{current_period}.{OUTPUT_IMAGE_FORMAT}")
            fig.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches='tight')
            plt.close(fig) 
        
        tqdm.write(f"--- Region {current_region} finished. ---")
        
    print("\n" + "="*50)
    print("All regions processed. Script finished.")
    print(f"Outputs saved in: {OUTPUT_IMAGE_DIR}")
    print(f"Region-specific data saved in: {KNOW_SUBDIR}")
    print("="*50)