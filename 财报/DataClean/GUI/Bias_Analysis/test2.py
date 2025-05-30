from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# === 配置你的 CSV 路径 ===
csv_path = r"C:\Users\99235\Desktop\thesisB\财报\DataClean\GUI\screenshots\BHPAnnualReport2024\QA_Temp07_gemini\bhp_qa_output.csv"

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

# === Step 4: 可视化 ===
plt.figure(figsize=(8, 6))
colors = plt.cm.get_cmap('Set2', num_clusters)

for i in range(num_clusters):
    cluster_points = reduced_embeddings[np.array(labels) == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i}', s=40, alpha=0.7)

plt.title("Semantic Clustering (SBERT + KMeans + PCA)")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.legend()
plt.tight_layout()

plt_path = os.path.join(os.path.dirname(csv_path), "semantic_clusters.png")
plt.savefig(plt_path)
plt_path
