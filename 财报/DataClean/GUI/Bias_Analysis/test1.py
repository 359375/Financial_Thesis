import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np

# === 1. 加载数据 ===
data_path = r"C:\Users\99235\Desktop\thesisB\财报\DataClean\GUI\screenshots\BHPAnnualReport2024\QA_Temp07\bhp_qa_output.csv"
df = pd.read_csv(data_path)
texts = df["Answer"].astype(str).tolist()

# === 2. TF-IDF 向量化 ===
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(texts)

# === 3. KMeans 聚类 ===
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(X)
df["Cluster"] = labels

# === 4. Silhouette Score 衡量聚类质量 ===
sil_score = silhouette_score(X, labels)
print(f"Silhouette Score: {sil_score:.4f}")

# === 5. 每个簇的关键词提取 ===
terms = vectorizer.get_feature_names_out()
top_keywords_per_cluster = []
for i in range(n_clusters):
    center = kmeans.cluster_centers_[i]
    top_indices = center.argsort()[::-1][:5]
    keywords = [terms[j] for j in top_indices]
    top_keywords_per_cluster.append(", ".join(keywords))

# === 6. 构建雷达图数据 ===
cluster_counts = df["Cluster"].value_counts().sort_index().tolist()
angles = np.linspace(0, 2 * np.pi, n_clusters, endpoint=False).tolist()
cluster_counts += cluster_counts[:1]
angles += angles[:1]
labels = top_keywords_per_cluster + [top_keywords_per_cluster[0]]

# === 7. 绘制雷达图 ===
plt.figure(figsize=(8, 6))
ax = plt.subplot(111, polar=True)
ax.plot(angles, cluster_counts, "o-", linewidth=2)
ax.fill(angles, cluster_counts, alpha=0.25)
ax.set_xticks(angles)
ax.set_xticklabels(labels, fontsize=8)
ax.set_title("TF-IDF Topic Clustering Radar Chart", y=1.1)
plt.tight_layout()
plt.show()
