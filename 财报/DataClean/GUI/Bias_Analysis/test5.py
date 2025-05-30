import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import os
# --------- 用户自行配置路径 ------------------
gpt_csv = r"C:\Users\99235\Desktop\thesisB\财报\DataClean\GUI\screenshots\BHPAnnualReport2024\QA_Temp07\bhp_qa_output.csv"
gemini_csv = r"C:\Users\99235\Desktop\thesisB\财报\DataClean\GUI\screenshots\BHPAnnualReport2024\QA_Temp07_gemini\bhp_qa_output.csv"

# save_dir = r"C:\Users\99235\Desktop\thesisB\财报\/DataClean/GUI/Bias_Analysis/bhp/aligment_similarity_mpnetV2/aligment_similarity_cluster"


save_dir = r"/DataClean/GUI/Bias_Analysis/model_analysis/bhp/aligment_similarity_e5/aligment_similarity_cluster"

os.makedirs(save_dir, exist_ok=True)

# --------- 1. 读取问答 ------------------
df_gpt = pd.read_csv(gpt_csv, encoding="utf-8").dropna()
df_gem = pd.read_csv(gemini_csv, encoding="utf-8").dropna()


df_gpt["QA"] = df_gpt["Question"].astype(str) + " " + df_gpt["Answer"].astype(str)
df_gem["QA"] = df_gem["Question"].astype(str) + " " + df_gem["Answer"].astype(str)

# --------- 2. 语义嵌入 ------------------
# model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
model = SentenceTransformer("intfloat/e5-large-v2")
emb_gpt = model.encode(df_gpt["QA"].tolist(), convert_to_tensor=True, show_progress_bar=True)
emb_gem = model.encode(df_gem["QA"].tolist(), convert_to_tensor=True, show_progress_bar=True)

# --------- 3. Gemini QA 匹配到 GPT QA ------------------
best_idx = []
best_sim = []


for gem_vec in emb_gem:
    sims = util.cos_sim(gem_vec, emb_gpt)[0]
    idx = int(sims.argmax())
    best_idx.append(idx)
    best_sim.append(float(sims[idx]))


df_match = df_gem.copy()
df_match["Matched_GPT_QA"] = df_gpt["QA"].iloc[best_idx].values
df_match["Matched_GPT_Index"] = best_idx
df_match["Cosine_Similarity"] = best_sim


# 只用于聚类和关键词提取的 QA 字段
df_gpt["Question_only"] = df_gpt["Question"].astype(str)
# 用这个做聚类嵌入
emb_gpt = model.encode(df_gpt["Question_only"].tolist(), convert_to_tensor=True, show_progress_bar=True)


# --------- 4. 对 GPT QA 做主题聚类 -------------
#   自动选择 k（2..10）以 Silhouette Score 最高为准
sil_scores = {}
for k in range(2, 100):
    km = KMeans(n_clusters=k, random_state=42)
    lbl = km.fit_predict(emb_gpt.cpu().numpy())
    sil = silhouette_score(emb_gpt.cpu().numpy(), lbl)
    sil_scores[k] = sil



best_k = max(sil_scores, key=sil_scores.get)

# --------- Step 4: 打印每个 k 的 Silhouette Score 和最终选择的 k ----------
print("Silhouette Scores for each k:")
for k, score in sil_scores.items():
    print(f"k = {k}: Silhouette Score = {score:.4f}")

print(f"\n[✓] 最佳聚类数为 k = {best_k}，对应的 Silhouette Score = {sil_scores[best_k]:.4f}")



# 重新聚类用最佳 k
kmeans = KMeans(n_clusters=best_k, random_state=42)

cluster_labels = kmeans.fit_predict(emb_gpt.cpu().numpy())


df_gpt["Cluster"] = cluster_labels

# --------- 5. 为匹配表加入主题标签 ----------
df_match["Cluster"] = df_gpt["Cluster"].iloc[best_idx].values

# --------- 6. 每主题统计平均相似度 ----------
theme_stats = df_match.groupby("Cluster")["Cosine_Similarity"].agg(["count", "mean"]).reset_index()
theme_stats.rename(columns={"count": "Pairs", "mean": "Avg_Sim"}, inplace=True)

# --------- 7. 为每个主题提取关键词 ----------
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)

X = vectorizer.fit_transform(df_gpt["Question_only"])

terms = np.array(vectorizer.get_feature_names_out())
top_terms = []
for c in range(best_k):
    idx_in_cluster = np.where(cluster_labels == c)[0]
    tfidf_mean = X[idx_in_cluster].mean(axis=0).A1
    top_idx = tfidf_mean.argsort()[-5:][::-1]
    top_terms.append(", ".join(terms[top_idx]))
theme_stats["Keywords"] = top_terms


# --------- 8. 可视化 -------------
# 8a 柱状图
plt.figure(figsize=(10, 5))
plt.bar(theme_stats["Cluster"].astype(str), theme_stats["Avg_Sim"])
plt.ylabel("Average Cosine Similarity")
plt.xlabel("Cluster")
plt.title("Average GPT–Gemini Similarity per Semantic Theme")
plt.tight_layout()
bar_path = os.path.join(save_dir, "avg_similarity_per_theme.png")
plt.savefig(bar_path)
plt.close()

# 8b 雷达图
angles = np.linspace(0, 2 * np.pi, best_k, endpoint=False).tolist()
values = theme_stats["Avg_Sim"].tolist()
values += values[:1]
angles += angles[:1]

plt.figure(figsize=(6, 6))
ax = plt.subplot(111, polar=True)
ax.plot(angles, values, "o-", linewidth=2)
ax.fill(angles, values, alpha=0.25)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(theme_stats["Cluster"].astype(str) + ": " + theme_stats["Keywords"], fontsize=8)
ax.set_title("Theme‑level Semantic Agreement (Radar)")
plt.tight_layout()
radar_path = os.path.join(save_dir, "theme_similarity_radar.png")
plt.savefig(radar_path)
plt.close()

# --------- 9. 保存结果 -----------
output_csv = os.path.join(save_dir, "qa_match_with_theme.csv")

df_match.to_csv(output_csv, index=False, encoding='utf_8_sig')

theme_stats.to_csv(os.path.join(save_dir, "theme_stats_summary.csv"), index=False)

print(os.path.join(save_dir, "theme_stats_summary.csv"))
