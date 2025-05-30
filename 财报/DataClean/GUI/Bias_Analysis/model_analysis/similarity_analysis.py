import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
import matplotlib.pyplot as plt
import os

# ========== Configuration ==========
gpt_csv = r"C:\Users\99235\Desktop\thesisB\财报\DataClean\GUI\screenshots\BHPAnnualReport2024\QA_Temp07\bhp_qa_output.csv"
gemini_csv = r"C:\Users\99235\Desktop\thesisB\财报\DataClean\GUI\screenshots\BHPAnnualReport2024\QA_Temp07_gemini\bhp_qa_output.csv"

output_csv = r"C:\Users\99235\Desktop\thesisB\财报\DataClean\GUI\Bias_Analysis\BHP\aligment_similarity_mpnetV2\qa_semantic_alignment_mpnetV2.csv"
output_fig = r"C:\Users\99235\Desktop\thesisB\财报\DataClean\GUI\Bias_Analysis\BHP\aligment_similarity_mpnetV2\semantic_diff_hist_mpnetV2.png"

# ========== Step 1: Load model ==========
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
# model = SentenceTransformer("intfloat/e5-large-v2")



# ========== Step 2: Read Q&A ==========
df_gpt = pd.read_csv(gpt_csv).dropna()
df_gemini = pd.read_csv(gemini_csv).dropna()

df_gpt["QA"] = df_gpt["Question"].astype(str) + " " + df_gpt["Answer"].astype(str)
df_gemini["QA"] = df_gemini["Question"].astype(str) + " " + df_gemini["Answer"].astype(str)

gpt_texts = df_gpt["QA"].tolist()
gemini_texts = df_gemini["QA"].tolist()

# ========== Step 3: Generate embeddings ==========
gpt_embeddings = model.encode(gpt_texts, convert_to_tensor=True, show_progress_bar=True)
gemini_embeddings = model.encode(gemini_texts, convert_to_tensor=True, show_progress_bar=True)

# ========== Step 4: Matching and similarity calculation ==========
similarities = []
matches = []

for i, gem_emb in enumerate(gemini_embeddings):   # Cosine similarity
    cos_scores = util.cos_sim(gem_emb, gpt_embeddings)[0]
    best_score = float(torch.max(cos_scores))
    best_idx = int(torch.argmax(cos_scores))
    similarities.append(best_score)
    matches.append(gpt_texts[best_idx])

print("Average similarity:", np.mean(similarities))
print("Standard deviation:", np.std(similarities))
print("Proportion with similarity > 0.9:", np.mean(np.array(similarities) > 0.9))

# ========== Step 5: Output results ==========
df_result = df_gemini.copy()
df_result["Matched_GPT_QA"] = matches
df_result["Cosine_Similarity"] = similarities
df_result.to_csv(output_csv, index=False)
print(f"Matched results written to: {output_csv}")

# ========== Step 6: Visualization ==========
plt.figure(figsize=(8, 5))
plt.hist(similarities, bins=20, color='skyblue', edgecolor='black')
plt.title("Cosine Similarity Distribution (Gemini → GPT)")
plt.xlabel("Cosine Similarity")
plt.ylabel("Frequency")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(output_fig)
plt.show()
print(f"Similarity histogram saved to: {output_fig}")
