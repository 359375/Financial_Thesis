import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

# === 配置你的 CSV 路径 ===
csv_path = r"C:\Users\99235\Desktop\thesisB\财报\DataClean\GUI\screenshots\BHPAnnualReport2024\QA_Temp07\bhp_qa_output.csv"
# csv_path = r"C:\Users\99235\Desktop\thesisB\财报\DataClean\GUI\screenshots\BHPAnnualReport2024\QA_Temp07_gemini\bhp_qa_output.csv"

# === 读取数据 ===
df = pd.read_csv(csv_path)
texts = df["Answer"].dropna().tolist()

# === Step 1: 文本嵌入 ===
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(texts, show_progress_bar=True)

# === Step 2: 聚类（KMeans） ===
num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
labels = kmeans.fit_predict(embeddings)

# === Step 3: 降维（用于可视化） ===
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)

# === Step 4: 聚类质量评估 ===
sil_score = silhouette_score(embeddings, labels)

# === Step 5: 每个聚类提取关键词 ===
cluster_keywords = []
vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
X = vectorizer.fit_transform(texts)
terms = np.array(vectorizer.get_feature_names_out())

for i in range(num_clusters):
    cluster_indices = np.where(labels == i)[0]
    cluster_tfidf = X[cluster_indices].mean(axis=0).A1
    top_indices = cluster_tfidf.argsort()[-5:][::-1]
    top_keywords = terms[top_indices]
    cluster_keywords.append(", ".join(top_keywords))

# === Step 6: 可视化 ===
plt.figure(figsize=(8, 6))
colors = plt.cm.get_cmap('Set2', num_clusters)

for i in range(num_clusters):
    cluster_points = reduced_embeddings[np.array(labels) == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i}: {cluster_keywords[i]}', s=40, alpha=0.7)

plt.title(f"Semantic Clustering (SBERT + KMeans + PCA)\nSilhouette Score: {sil_score:.3f}")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.legend(loc='best', fontsize='small')
plt.tight_layout()

# 保存图像
plt_path = os.path.join(os.path.dirname(csv_path), "semantic_clusters_annotated.png")
plt.savefig(plt_path)
plt_path
