import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load data
titles = pd.read_csv("data/soc-redditHyperlinks-title.tsv", delimiter="\t")
body = pd.read_csv("data/soc-redditHyperlinks-body.tsv", delimiter="\t")

# Combine dataframes
all_data = pd.concat([titles, body])

# Extract 'POST_PROPERTIES' as a list feature
post_properties = all_data['PROPERTIES'].apply(eval)  # Assuming 'PROPERTIES' is a string representation of a list

# Convert list feature to DataFrame with each value in the list as a separate column
post_properties_df = pd.DataFrame(post_properties.tolist(), columns=[f"feature_{i}" for i in range(86)])

# Preprocess the features (scaling)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(post_properties_df)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
pca_features = pca.fit_transform(scaled_features)

print(pca.components_)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42)
cluster_labels = kmeans.fit_predict(scaled_features)

# Plot the clusters
plt.figure(figsize=(8, 6))
for cluster in np.unique(cluster_labels):
    plt.scatter(pca_features[cluster_labels == cluster, 0], 
                pca_features[cluster_labels == cluster, 1],
                label=f'Cluster {cluster}')

plt.title('KMeans Clustering with PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.show()
