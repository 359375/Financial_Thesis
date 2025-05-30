import os, random, sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer, util
from scipy.stats import ttest_rel

# =============================================================================
# 0. reproducibility & device
# =============================================================================
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device =", DEVICE)

# =============================================================================
# 1. file paths (GPT-only temperature analysis)
# =============================================================================
BASE_DIR = r"C:\Users\99235\Desktop\thesisB\财报\DataClean\GUI"
T07_CSV = os.path.join(BASE_DIR, r"screenshots\BHPAnnualReport2024\QA_Temp07\bhp_qa_output.csv")
T03_CSV = os.path.join(BASE_DIR, r"screenshots\BHPAnnualReport2024\QA_Temp03\bhp_qa_output.csv")
T00_CSV = os.path.join(BASE_DIR, r"screenshots\BHPAnnualReport2024\QA_Temp0\bhp_qa_output.csv")

OUT_DIR = os.path.join(BASE_DIR, r"Bias_Analysis\temp_analysis\gpt_temp\e5\bhp")
os.makedirs(OUT_DIR, exist_ok=True)
CSV_PATH     = os.path.join(OUT_DIR, "pairwise_similarity_results.csv")
CSV_DIFF     = os.path.join(OUT_DIR, "incremental_drift.csv")
FIG_BOX      = os.path.join(OUT_DIR, "temp_similarity_boxplot.png")
FIG_KDE      = os.path.join(OUT_DIR, "temp_similarity_kde.png")
FIG_HEATMAP  = os.path.join(OUT_DIR, "similarity_heatmap.png")
SUMMARY_TXT  = os.path.join(OUT_DIR, "result_summary.txt")

# =============================================================================
# 2. settings
# =============================================================================
USE_Q_ONLY   = False
BATCH_SIZE   = 128
EMB_MODEL_ID = "intfloat/e5-large-v2"

# =============================================================================
# 3. helpers
# =============================================================================

def load_texts(csv_path: str, q_only: bool = False) -> pd.Series:
    df = pd.read_csv(csv_path).dropna().reset_index(drop=True)
    return df["Question"].astype(str) if q_only else df["Question"].astype(str) + " " + df["Answer"].astype(str)


def encode(model, texts):
    vec = model.encode(texts.tolist(), convert_to_tensor=True, batch_size=BATCH_SIZE, show_progress_bar=True)
    return torch.nn.functional.normalize(vec, p=2, dim=1)


def best_match(src_emb, tgt_emb):
    """return array of best cosine similarity (src→tgt)"""
    sims = []
    for v in src_emb:
        sims.append(float(torch.max(util.cos_sim(v, tgt_emb))))
    return np.array(sims)


def stats_line(name, arr):
    return f"{name:<8} mean={arr.mean():.4f} std={arr.std():.4f} >0.9={(arr>0.9).mean():.1%}"

# =============================================================================
# 4. load data
# =============================================================================
TXT_00 = load_texts(T00_CSV, USE_Q_ONLY)
TXT_03 = load_texts(T03_CSV, USE_Q_ONLY)
TXT_07 = load_texts(T07_CSV, USE_Q_ONLY)

if len({len(TXT_00), len(TXT_03), len(TXT_07)}) != 1:
    print("✖ CSV 行数不一致"); sys.exit(1)
print("Loaded", len(TXT_00), "samples per temperature")

# =============================================================================
# 5. build model & embeddings
# =============================================================================
model = SentenceTransformer(EMB_MODEL_ID).to(DEVICE)
EMB_00, EMB_03, EMB_07 = map(lambda t: encode(model, t), [TXT_00, TXT_03, TXT_07])

# =============================================================================
# 6. compute all 3×3 similarities (single-direction)
# =============================================================================
SIM = {}
labels = ["T0", "T0.3", "T0.7"]
emb_map = {"T0": EMB_00, "T0.3": EMB_03, "T0.7": EMB_07}
for src in labels:
    for tgt in labels:
        if src == tgt:
            SIM[(src, tgt)] = np.ones(len(TXT_00))
        else:
            SIM[(src, tgt)] = best_match(emb_map[src], emb_map[tgt])

# convenience aliases
sim_00_03 = SIM[("T0", "T0.3")]
sim_00_07 = SIM[("T0", "T0.7")]
sim_03_07 = SIM[("T0.3", "T0.7")]

# =============================================================================
# 7. incremental drift (0→0.7) minus (0→0.3)
# =============================================================================
inc_drift = sim_00_07 - sim_00_03  # element-wise
pd.DataFrame({"incremental_drift": inc_drift}).to_csv(CSV_DIFF, index=False)

# =============================================================================
# 8. save raw similarities (mean values)
# =============================================================================
mean_matrix = np.zeros((3,3))
for i, src in enumerate(labels):
    for j, tgt in enumerate(labels):
        mean_matrix[i,j] = SIM[(src, tgt)].mean()

pd.DataFrame(mean_matrix, index=labels, columns=labels).to_csv(CSV_PATH, index_label="source→target")

# =============================================================================
# 9. stats txt
# =============================================================================
with open(SUMMARY_TXT, "w", encoding="utf-8") as fh:
    fh.write("=== Mean similarities ===\n")
    fh.write(pd.DataFrame(mean_matrix, index=labels, columns=labels).to_string())
    fh.write("\n\n=== Distribution summaries ===\n")
    for n,a in zip(["0→0.3","0→0.7","0.3→0.7"],[sim_00_03,sim_00_07,sim_03_07]):
        fh.write(stats_line(n,a)+"\n")
print("summary saved →", SUMMARY_TXT)

# =============================================================================
# 10. plots
# =============================================================================
# boxplot
plt.figure(figsize=(8,3))
plt.boxplot([sim_00_03, sim_00_07, sim_03_07], labels=["0→0.3","0→0.7","0.3→0.7"], patch_artist=True, boxprops=dict(facecolor="#87CEEB"))
plt.title("Semantic Similarity across Temperatures (GPT)")
plt.ylabel("Cosine Similarity"); plt.grid(alpha=.3, axis="y")
plt.tight_layout(); plt.savefig(FIG_BOX, dpi=300); plt.close()

# kde
plt.figure(figsize=(8,4))
for arr,lbl in zip([sim_00_03,sim_00_07,sim_03_07],["0→0.3","0→0.7","0.3→0.7"]):
    sns.kdeplot(arr, label=lbl, fill=True, clip=(0,1), alpha=.4, linewidth=2)
plt.title("KDE of Cosine Similarities (Semantic Alignment)")
plt.xlabel("Cosine Similarity"); plt.ylabel("Density"); plt.xlim(0,1); plt.legend(); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig(FIG_KDE, dpi=300); plt.close()

# heatmap of mean matrix
plt.figure(figsize=(5,4))
ax = sns.heatmap(mean_matrix, annot=True, vmin=0, vmax=1, xticklabels=labels, yticklabels=labels, cmap="Blues")
ax.set_title("Mean Similarity Heatmap (src→tgt)")
plt.tight_layout(); plt.savefig(FIG_HEATMAP, dpi=300); plt.close()
