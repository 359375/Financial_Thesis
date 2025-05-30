# -*- coding: utf-8 -*-


import os, random, sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer, util
from scipy.stats import ttest_rel

# ---------------------------------------------------------------------------
# 0. reproducibility & device
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device =", DEVICE)

# ---------------------------------------------------------------------------
# 1. company list (hard‑coded paths)
# ---------------------------------------------------------------------------
BASE = r"C:\Users\99235\Desktop\thesisB\财报\DataClean\GUI\screenshots"
COMPANIES = [
    {
        "name": "BHP",
        "gpt_csv": os.path.join(BASE, "BHPAnnualReport2024", "QA_Temp07", "bhp_qa_output.csv"),
        "gem_csv": os.path.join(BASE, "BHPAnnualReport2024", "QA_Temp07_gemini", "bhp_qa_output.csv"),
    },
    {
        "name": "WoolworthsGroup",
        "gpt_csv": os.path.join(BASE, "WoolworthsGroupAnnualReport2024", "QA_Temp07", "wws_qa_output.csv"),
        "gem_csv": os.path.join(BASE, "WoolworthsGroupAnnualReport2024", "QA_Temp07_gemini", "wws_qa_output.csv"),
    },
    {
        "name": "CMW",
        "gpt_csv": os.path.join(BASE, "CommonWealthAnualReport2024", "QA_Temp07", "cmw_qa_output.csv"),
        "gem_csv": os.path.join(BASE, "CommonWealthAnualReport2024", "QA_Temp07_gemini", "cmw_qa_output.csv"),
    },
]

ROOT_OUT = r"C:\Users\99235\Desktop\thesisB\财报\DataClean\GUI\Bias_Analysis\model_analysis\mpnetV2"
os.makedirs(ROOT_OUT, exist_ok=True)

# ---------------------------------------------------------------------------
# 2. settings
# ---------------------------------------------------------------------------
USE_Q_ONLY = False
BATCH_SIZE = 128
EMB_MODEL  = "sentence-transformers/all-mpnet-base-v2"

# embedding_model_id  = "sentence-transformers/all-mpnet-base-v2"
# embedding_model_id = "BAAI/bge-large-en-v1.5"
# embedding_model_id = "intfloat/e5-large-v2"



# ---------------------------------------------------------------------------
# 3. helper functions
# ---------------------------------------------------------------------------

def load_series(csv_path, q_only=False):
    df = pd.read_csv(csv_path).dropna().reset_index(drop=True)
    return df["Question"].astype(str) if q_only else df["Question"].astype(str) + " " + df["Answer"].astype(str)


def encode(model, texts):
    emb = model.encode(texts.tolist(), convert_to_tensor=True, batch_size=BATCH_SIZE, show_progress_bar=False)
    return torch.nn.functional.normalize(emb, p=2, dim=1)


def gem_to_gpt_similarity(gem_emb, gpt_emb):
    sims = []
    for vec in gem_emb:
        cos = util.cos_sim(vec, gpt_emb)[0]
        sims.append(float(torch.max(cos)))
    return np.array(sims)

# ---------------------------------------------------------------------------
# 4. load model once
# ---------------------------------------------------------------------------
model = SentenceTransformer(EMB_MODEL, device=DEVICE)

all_scores = []  # for combined KDE

for comp in COMPANIES:
    name   = comp["name"]
    gpt_csv    = comp["gpt_csv"]
    gem_csv    = comp["gem_csv"]
    out_dir    = os.path.join(ROOT_OUT, name.lower())
    os.makedirs(out_dir, exist_ok=True)

    # encode
    gpt_txt = load_series(gpt_csv, USE_Q_ONLY)
    gem_txt = load_series(gem_csv, USE_Q_ONLY)
    if len(gpt_txt) != len(gem_txt):
        print(f"❌ Row mismatch for {name}, skip!"); continue

    emb_gpt = encode(model, gpt_txt)
    emb_gem = encode(model, gem_txt)

    sims = gem_to_gpt_similarity(emb_gem, emb_gpt)
    all_scores.append((name, sims))

    # save csv
    pd.DataFrame({"cosine_similarity": sims}).to_csv(os.path.join(out_dir, f"{name}_gem2gpt_mpnetV2.csv"), index=False)

    # summary
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"=== {name}  Gemini → GPT (mpnetV2) ===\n")
        fh.write(f"mean={sims.mean():.4f}, std={sims.std():.4f}, >0.9={(sims>0.9).mean():.1%}\n")
        fh.write(f"paired t-test vs 1.0  p={ttest_rel(sims, np.ones_like(sims)).pvalue:.4e}\n")

    # histogram
    plt.figure(figsize=(6,4))
    plt.hist(sims, bins=20, color="#87CEEB", edgecolor="black")
    plt.xlabel("Cosine Similarity"); plt.ylabel("Freq")
    plt.title(f"{name}: Gem→GPT Similarity (mpnetV2)")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{name}_hist.png"), dpi=300); plt.close()

# ---------------------------------------------------------------------------
# 5. combined KDE plot
# ---------------------------------------------------------------------------
plt.figure(figsize=(8,5))
for name, sims in all_scores:
    sns.kdeplot(sims, label=name, fill=True, alpha=.3, clip=(0,1))
plt.xlabel("Cosine Similarity"); plt.ylabel("Density")
plt.title("Gemini → GPT Similarity Distribution (mpnetV2, 3 Companies)")
plt.xlim(0,1); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(ROOT_OUT, "combined_kde.png"), dpi=300); plt.close()
print("Combined KDE saved →", os.path.join(ROOT_OUT, "combined_kde.png"))
