import networkx as nx
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer



def get_centrality(G):
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    eigenvector_centrality = nx.eigenvector_centrality(G)

    max_degree_nodes = max(degree_centrality, key=degree_centrality.get)
    max_degree_value = degree_centrality[max_degree_nodes]
    max_betweenness_nodes = max(betweenness_centrality, key=betweenness_centrality.get)
    max_betweenness_value = betweenness_centrality[max_betweenness_nodes]
    max_closeness_nodes = max(closeness_centrality, key=closeness_centrality.get)
    max_closeness_value = closeness_centrality[max_closeness_nodes]
    max_eigenvector_nodes = max(eigenvector_centrality, key=eigenvector_centrality.get)
    max_eigenvector_value = eigenvector_centrality[max_eigenvector_nodes]

    print("Node with largest degree centrality:", max_degree_nodes)
    print("Degree centrality value:", max_degree_value)
    print()
    print("Node with largest betweenness centrality:", max_betweenness_nodes)
    print("Betweenness centrality value:", max_betweenness_value)
    print()
    print("Node with largest closeness centrality:", max_closeness_nodes)
    print("Closeness centrality value:", max_closeness_value)
    print()
    print("Node with largest eigenvector centrality:", max_eigenvector_nodes)
    print("Eigenvector centrality value:", max_eigenvector_value)


def cluster(df):
    text_data = df['TEXT']
    vectorizer = TfidfVectorizer(max_features=1000)  # You can adjust max_features as needed
    tfidf_matrix = vectorizer.fit_transform(text_data)
    pca = PCA(n_components=2)
    tfidf_pca = pca.fit_transform(tfidf_matrix.toarray())

    num_clusters = 5  # Adjust the number of clusters as needed
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(tfidf_pca)
    cluster_labels = kmeans.labels_
    df['Cluster'] = cluster_labels

    plt.figure(figsize=(8, 6))
    plt.scatter(tfidf_pca[:, 0], tfidf_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.5)
    plt.title('K-means Clustering with PCA')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(label='Cluster')
    plt.show()

def keyword_finder(df, n=10):
    text_data = df['TEXT']
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(text_data)
    feature_names = tfidf_vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()

    # Initialize a defaultdict to store the total TF-IDF scores for each word
    keyword_scores = defaultdict(float)

    # Aggregate TF-IDF scores for each word across all posts
    for i, post in enumerate(text_data):
        for idx, score in enumerate(tfidf_scores[i]):
            keyword = feature_names[idx].lower()  # Convert to lowercase for case insensitivity
            keyword_scores[keyword] += score

    # Sort the words based on their total TF-IDF scores
    sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)

    # Return the top N keywords
    top_keywords = sorted_keywords[:n]
    return top_keywords

############## MAIN ############## 

df = pd.read_csv('data/out.csv')
df = df.dropna()
G = nx.from_pandas_edgelist(df, 'SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT', create_using=nx.DiGraph())
top_keywords = keyword_finder(df, n=10)
for keyword, score in top_keywords:
    print(f"Keyword: {keyword}, Total TF-IDF Score: {score}")

