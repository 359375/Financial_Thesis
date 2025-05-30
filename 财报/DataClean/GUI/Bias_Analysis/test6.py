import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import os
import hdbscan

# --------- 用户自行配置路径 ------------------
gpt_csv = r"C:\Users\99235\Desktop\thesisB\财报\DataClean\GUI\screenshots\BHPAnnualReport2024\QA_Temp07\bhp_qa_output.csv"
gemini_csv = r"C:\Users\99235\Desktop\thesisB\财报\DataClean\GUI\screenshots\BHPAnnualReport2024\QA_Temp07_gemini\bhp_qa_output.csv"
save_dir = r"/DataClean/GUI/Bias_Analysis/model_analysis/bhp/aligment_similarity_e5/aligment_similarity_hdbscan"
os.makedirs(save_dir, exist_ok=True)

# --------- 1. 读取问答 ------------------
df_gpt = pd.read_csv(gpt_csv, encoding="utf-8").dropna()
df_gem = pd.read_csv(gemini_csv, encoding="utf-8").dropna()

# QA合并 & 词组前处理
df_gpt["QA"] = df_gpt["Question"].astype(str) + " " + df_gpt["Answer"].astype(str)
df_gem["QA"] = df_gem["Question"].astype(str) + " " + df_gem["Answer"].astype(str)
df_gpt["Question_only"] = df_gpt["Question"].astype(str)

# --------- 2. 语义嵌入 ------------------
model = SentenceTransformer("intfloat/e5-large-v2")

emb_gpt_qa = model.encode(df_gpt["QA"].tolist(), convert_to_tensor=True, show_progress_bar=True)
emb_gpt_q = model.encode(df_gpt["Question_only"].tolist(), convert_to_tensor=True, show_progress_bar=True)
emb_gem = model.encode(df_gem["QA"].tolist(), convert_to_tensor=True, show_progress_bar=True)

# --------- 3. Gemini QA 匹配 GPT QA ------------------
best_idx = []
best_sim = []
for gem_vec in emb_gem:
    sims = util.cos_sim(gem_vec, emb_gpt_qa)[0]
    idx = int(sims.argmax())
    best_idx.append(idx)
    best_sim.append(float(sims[idx]))

df_match = df_gem.copy()
df_match["Matched_GPT_QA"] = df_gpt["QA"].iloc[best_idx].values
df_match["Matched_GPT_Index"] = best_idx
df_match["Cosine_Similarity"] = best_sim

# --------- 4. HDBSCAN 分频聚类 ------------------
clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
cluster_labels = clusterer.fit_predict(emb_gpt_q.cpu().numpy())
df_gpt["Cluster"] = cluster_labels
df_match["Cluster"] = df_gpt["Cluster"].iloc[best_idx].values

# --------- 5. 给为各主题计算均相似度 ----------
theme_stats = df_match[df_match["Cluster"] != -1].groupby("Cluster")["Cosine_Similarity"].agg(["count", "mean"]).reset_index()
theme_stats.rename(columns={"count": "Pairs", "mean": "Avg_Sim"}, inplace=True)

# --------- 6. TF-IDF 抽取各主题关键词 ----------
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X = vectorizer.fit_transform(df_gpt["Question_only"])
terms = np.array(vectorizer.get_feature_names_out())
top_terms = []
for c in theme_stats["Cluster"]:
    idx_in_cluster = np.where(cluster_labels == c)[0]
    tfidf_mean = X[idx_in_cluster].mean(axis=0).A1
    top_idx = tfidf_mean.argsort()[-5:][::-1]
    top_terms.append(", ".join(terms[top_idx]))
theme_stats["Keywords"] = top_terms

# --------- 7. 可视化 ----------
plt.figure(figsize=(10, 5))
plt.bar(theme_stats["Cluster"].astype(str), theme_stats["Avg_Sim"])
plt.ylabel("Average Cosine Similarity")
plt.xlabel("Cluster")
plt.title("Average GPT–Gemini Similarity per HDBSCAN Theme")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "avg_similarity_per_theme_hdbscan.png"))
plt.close()

# --------- 8. 结果保存 ----------
df_match.to_csv(os.path.join(save_dir, "qa_match_with_theme_hdbscan.csv"), index=False, encoding='utf_8_sig')
theme_stats.to_csv(os.path.join(save_dir, "theme_stats_summary_hdbscan.csv"), index=False, encoding='utf_8_sig')

print("[✓] 添加了 HDBSCAN 聚类分析效果")
