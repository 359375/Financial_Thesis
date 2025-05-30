import os, json, sys, random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer, util
from scipy.stats import ttest_rel


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# ---------------------------------------------------------------------------
# 1.  File paths  (KEEPING YOUR ORIGINAL ABSOLUTE PATHS)
# ---------------------------------------------------------------------------
temp07_csv = r"C:\Users\99235\Desktop\thesisB\财报\DataClean\GUI\screenshots\BHPAnnualReport2024\QA_Temp07\bhp_qa_output.csv"
temp03_csv = r"C:\Users\99235\Desktop\thesisB\财报\DataClean\GUI\screenshots\BHPAnnualReport2024\QA_Temp03\bhp_qa_output.csv"
temp00_csv = r"C:\Users\99235\Desktop\thesisB\财报\DataClean\GUI\screenshots\BHPAnnualReport2024\QA_Temp0\bhp_qa_output.csv"

out_csv     = r"C:\Users\99235\Desktop\thesisB\财报\DataClean\GUI\Bias_Analysis\temp_analysis\gpt_temp\mpnetV2\bhp\pairwise_similarity_results.csv"
fig_box     = r"C:\Users\99235\Desktop\thesisB\财报\DataClean\GUI\Bias_Analysis\temp_analysis\gpt_temp\mpnetV2\bhp\temp_similarity_boxplot.png"
fig_kde     = r"C:\Users\99235\Desktop\thesisB\财报\DataClean\GUI\Bias_Analysis\temp_analysis\gpt_temp\mpnetV2\bhp\temp_similarity_kde.png"
summary_txt = r"C:\Users\99235\Desktop\thesisB\财报\DataClean\GUI\Bias_Analysis\temp_analysis\gpt_temp\mpnetV2\bhp\result_summary.txt"

# ---------------------------------------------------------------------------
# 2.  Settings
# ---------------------------------------------------------------------------
use_question_only   = False
batch_size          = 128

embedding_model_id  = "sentence-transformers/all-mpnet-base-v2"
# embedding_model_id = "BAAI/bge-large-en-v1.5"
# embedding_model_id = "intfloat/e5-large-v2"

# ---------------------------------------------------------------------------
# 3.  Helper
# ---------------------------------------------------------------------------
def load_texts(csv_path: str, use_q_only: bool) -> pd.Series:
    df = pd.read_csv(csv_path).dropna().reset_index(drop=True)
    if use_q_only:
        return df["Question"].astype(str)
    return df["Question"].astype(str) + " " + df["Answer"].astype(str)

# ---------------------------------------------------------------------------
# 4.  Read all three temperature files
# ---------------------------------------------------------------------------
texts_00 = load_texts(temp00_csv, use_question_only)
texts_03 = load_texts(temp03_csv, use_question_only)
texts_07 = load_texts(temp07_csv, use_question_only)

assert (len(texts_00) == len(texts_03) == len(texts_07)), "Sample counts are not aligned!"
print(f"Loaded {len(texts_00)} aligned samples for each temperature.")

# ---------------------------------------------------------------------------
# 5.  Load model
# ---------------------------------------------------------------------------
model = SentenceTransformer(embedding_model_id, device=device)

# ---------------------------------------------------------------------------
# 6.  Generate embeddings  (unit-normalised)
# ---------------------------------------------------------------------------
def encode(texts):
    emb = model.encode(texts.tolist(),
                       convert_to_tensor=True,
                       device=device,
                       batch_size=batch_size,
                       show_progress_bar=True)
    return torch.nn.functional.normalize(emb, p=2, dim=1)  # unit-norm

emb_00, emb_03, emb_07 = map(encode, [texts_00, texts_03, texts_07])

# ---------------------------------------------------------------------------
# 7.  Cosine similarity  (values now guaranteed in [0,1])
# ---------------------------------------------------------------------------
sim_00_03 = util.cos_sim(emb_00, emb_03).diagonal().cpu().numpy()
sim_00_07 = util.cos_sim(emb_00, emb_07).diagonal().cpu().numpy()
sim_03_07 = util.cos_sim(emb_03, emb_07).diagonal().cpu().numpy()

# ---------------------------------------------------------------------------
# 8.  Descriptive statistics & paired t-tests
# ---------------------------------------------------------------------------
def describe(tag, arr):
    return f"{tag:12s} → mean={arr.mean():.4f}, std={arr.std():.4f}, " \
           f">0.9={np.mean(arr>0.9):.2%}, max={arr.max():.4f}, min={arr.min():.4f}"

stats_lines = [
    describe("0.0 vs 0.3", sim_00_03),
    describe("0.0 vs 0.7", sim_00_07),
    describe("0.3 vs 0.7", sim_03_07)
]

pvals = {
    "0.0–0.3 vs 0.3–0.7": ttest_rel(sim_00_03, sim_03_07).pvalue,
    "0.0–0.3 vs 0.0–0.7": ttest_rel(sim_00_03, sim_00_07).pvalue,
    "0.3–0.7 vs 0.0–0.7": ttest_rel(sim_03_07, sim_00_07).pvalue
}

# write summary txt ----------------------------------------------------------
with open(summary_txt, "w", encoding="utf-8") as fh:
    fh.write("=== Descriptive Statistics ===\n")
    for line in stats_lines:
        fh.write(line + "\n")
    fh.write("\n=== Paired t-test p-values ===\n")
    for k, v in pvals.items():
        fh.write(f"{k} : p = {v:.5e}\n")
print(f"Summary written → {summary_txt}")

# ---------------------------------------------------------------------------
# 9.  Save raw similarities CSV
# ---------------------------------------------------------------------------
pd.DataFrame({
    "sim_00_03": sim_00_03,
    "sim_00_07": sim_00_07,
    "sim_03_07": sim_03_07
}).to_csv(out_csv, index=False)
print(f"Pairwise cosine similarity saved → {out_csv}")

# ---------------------------------------------------------------------------
# 10.  Boxplot  (tick_labels fallback for older MPL)
# ---------------------------------------------------------------------------
plt.figure(figsize=(8, 3))
try:
    plt.boxplot([sim_00_03, sim_00_07, sim_03_07],
                tick_labels=["0.0 vs 0.3", "0.0 vs 0.7", "0.3 vs 0.7"],
                patch_artist=True,
                boxprops=dict(facecolor='lightblue'))
except TypeError:  # Matplotlib < 3.9
    plt.boxplot([sim_00_03, sim_00_07, sim_03_07],
                labels=["0.0 vs 0.3", "0.0 vs 0.7", "0.3 vs 0.7"],
                patch_artist=True,
                boxprops=dict(facecolor='lightblue'))

plt.title("Semantic Similarity Across GPT Sampling Temperatures")
plt.ylabel("Cosine Similarity")
plt.grid(alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(fig_box, dpi=300)
plt.show()
print(f"Boxplot saved → {fig_box}")

# ---------------------------------------------------------------------------
# 11.  KDE plot (clipped to 0-1 for aesthetics)
# ---------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
for arr, lbl in zip([sim_00_03, sim_00_07, sim_03_07],
                    ["0.0 vs 0.3", "0.0 vs 0.7", "0.3 vs 0.7"]):
    sns.kdeplot(arr, label=lbl, fill=True, clip=(0, 1),
                linewidth=2, alpha=0.4)

plt.title("Cosine Similarity KDE Distribution (Temperature Pairs)")
plt.xlabel("Cosine Similarity")
plt.ylabel("Density")
plt.xlim(0, 1)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(fig_kde, dpi=300)
plt.show()
print(f"KDE plot saved → {fig_kde}")
