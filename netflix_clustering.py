import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("netflix_cleaned.csv")

# Combine 'title' and 'listed_in' for clustering
df['content'] = df['title'] + ' ' + df['listed_in']

df = pd.read_csv("netflix_cleaned.csv")

df['content'] = df['title'] + ' ' + df['listed_in']

# Vectorize text features
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['content'])

# Clustering with K-Means
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
df['cluster'] = kmeans.fit_predict(tfidf_matrix)

# Reduce to 2D for plotting
pca = PCA(n_components=2, random_state=42)
reduced = pca.fit_transform(tfidf_matrix.toarray())

df['pca_x'] = reduced[:, 0]
df['pca_y'] = reduced[:, 1]

# Visualize the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='pca_x', y='pca_y', hue='cluster', palette='Set2')
plt.title("Netflix Content Clustering with K-Means")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend(title='Cluster')
plt.grid(True)
plt.tight_layout()
plt.show()

# See some samples from each cluster
for i in range(k):
    print(f"\n🎬 Cluster {i} Samples:")
    print(df[df['cluster'] == i][['title', 'type', 'listed_in']].head(3))
