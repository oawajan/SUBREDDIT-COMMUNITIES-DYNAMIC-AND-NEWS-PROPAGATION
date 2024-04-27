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
    text_data = df['text']
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
    text_data = df['text']
    word_list = [
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', "aren't", 'as', 'at',
    'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', "can't", 'com', 'could',
    "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during', 'each', 'few', 'for',
    'from', 'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having', 'he', "he'd", "he'll", "he's",
    'her', 'here', "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's", 'https', 'i', "i'd", "i'll",
    "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself', 'just', 'like', 'me', 'more',
    'most', 'my', 'myself', 'no', 'nor', 'not', 'now', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought',
    'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', "shan't", 'she', "she'd", "she'll", "she's", 'should',
    "shouldn't", 'so', 'some', 'such', 'than', 'that', "that's", 'the', 'their', 'theirs', 'them', 'themselves',
    'then', 'there', "there's", 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through',
    'to', 'too', 'under', 'until', 'up', 'url', 'very', 'was', "wasn't", 'we', "we'd", "we'll", "we're", "we've",
    'were', "weren't", 'what', "what's", 'when', "when's", 'where', "where's", 'which', 'while', 'who', "who's",
    'whom', 'why', "why's", 'with', "won't", 'would', "wouldn't", 'www', 'you', "you'd", "you'll", "you're", "you've",
    'your', 'yours', 'yourself', 'yourselves', 'things', 'new', 'many', 'anyone', 'feel', 'guy', 'take'
    'reddit', 'subreddit', 'post', 'comment', 'comments', 'thread', 'threads', 'upvote', 'downvote', 'karma',
    'user', 'users', 'account', 'accounts', 'moderator', 'moderators', 'subscribers', 'subscribe', 'subscribed',
    'subscribe', 'unsubscribe', 'unsubscribe', 'repost', 'reposts', 'crosspost', 'crossposted', 'crossposting',
    'original', 'source', 'sources', 'link', 'links', 'website', 'websites', 'article', 'articles', 'blog', 'blogs',
    'video', 'videos', 'image', 'images', 'picture', 'pictures', 'photo', 'photos', 'gif', 'gifs', 'title', 'titles',
    'text', 'body', 'content', 'post', 'posts', 'thread', 'threads', 'comment', 'comments', 'reply', 'replies',
    'discussion', 'discussions', 'question', 'questions', 'answer', 'answers', 'ask', 'asking', 'asked', 'answered',
    'opinion', 'opinions', 'thought', 'thoughts', 'idea', 'ideas', 'thoughtful', 'help', 'advice', 'suggestion',
    'suggestions', 'recommendation', 'recommendations', 'tip', 'tips', 'trick', 'tricks', 'tip', 'tips', 'guide',
    'guides', 'tutorial', 'tutorials', 'information', 'info', 'news', 'update', 'updates', 'announcement',
    'announcements', 'event', 'events', 'announcement', 'announcements', 'important', 'community', 'communities',
    'user', 'users', 'member', 'members', 'audience', 'readers', 'viewer', 'viewers', 'follower', 'followers',
    'subscriber', 'subscribers', 'participant', 'participants', 'visitor', 'visitors', 'guest', 'guests',
    'membership', 'experience', 'insight', 'insights', 'perspective', 'perspectives', 'viewpoint', 'viewpoints',
    'suggestion', 'suggestions', 'feedback', 'commentary', 'discussion', 'discussions', 'interaction', 'interactions',
    'engagement', 'conversation', 'conversations', 'dialogue', 'dialogues', 'chat', 'chats', 'message', 'messages',
    'reply', 'replies', 'response', 'responses', 'interaction', 'interactions', 'thread', 'threads', 'topic',
    'topics', 'theme', 'themes', 'subject', 'subjects', 'content', 'content', 'post', 'posts', 'submission',
    'submissions', 'submission', 'submissions', 'share', 'sharing', 'sharing', 'sharing', 'read', 'reading', 'readership',
    'reader', 'readers', 'view', 'viewing', 'viewer', 'viewers', 'viewing', 'viewings', 'visit', 'visiting', 'visitors',
    'visit', 'visits', 'traffic', 'visitor', 'visitors', 'traffic', 'visit', 'visits', 'visit', 'visits', 'click',
    'clicks', 'click', 'clicks', 'vote', 'voting', 'voted', 'votes', 'like', 'likes', 'liked', 'liking', 'dislike',
    'dislikes', 'disliked', 'disliking', 'upvote', 'upvotes', 'upvoted', 'upvoting', 'downvote', 'downvotes',
    'downvoted', 'downvoting', 'subscribe', 'subscribes', 'subscribed', 'subscribing', 'unsubscribe', 'unsubscribes',
    'unsubscribed', 'unsubscribing', 'reply', 'replies', 'replied', 'replying', 'mention', 'mentions', 'mentioned',
    'mentioning', 'tag', 'tags', 'tagged', 'tagging', 'follow', 'follows', 'http', 'https', 'com', 'org', 'www', 
    'reddit', 'comment', 'post', 'gt', 'one', 'get', 'don', 'comments', 've', 'amp', 'thread', 'even', 'will', 'people',
    'know', 're', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'll', 'shan', 'shouldn', 'wasn',
    'weren', 'won', 'wouldn', 'context', 'time', 'think','imgur', 'also', 'really', 'see', 'want', 'np', 'make', 'got',
    'made', 'go', 'edit', 'something','much', 'good', 'right', 'us', 'way', 'someone', 'first', 'going', 'game', 'back',
    'sub', 'well', 'thing', 'still', 'posted', 'said', 'say', 'day','need', 'anything', 'another', 'please', 'look', 'lot',
    'find', 'take', 'two', 'use', 'let', 'every', 'found', 'around', 'sure', 'since', 'never', 'actually', 'guys',
    'ago', 
    ]

    tfidf_vectorizer = TfidfVectorizer(stop_words=word_list, max_features=1000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(text_data)
    feature_names = tfidf_vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()

    # Initialize a defaultdict to store the total TF-IDF scores for each word
    keyword_scores = defaultdict(float)

    # Aggregate TF-IDF scores for each word across all posts
    for i, post in enumerate(text_data):
        for idx, score in enumerate(tfidf_scores[i]):
            keyword = feature_names[idx].lower()  # Convert to lowercase for case insensitivity
            if keyword not in word_list:  # Ignore NLTK stopwords
                keyword_scores[keyword] += score

    # Sort the words based on their total TF-IDF scores
    sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)

    # Return the top N keywords
    top_keywords = sorted_keywords[:n]
    return top_keywords

############## MAIN ############## 

df = pd.read_csv('data/out_V2.csv',low_memory=False)
df = df.dropna()
# subset_data = df.sample(frac=0.01, random_state=42)

# cluster(subset_data)
G = nx.from_pandas_edgelist(df, 'SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT', create_using=nx.DiGraph())
top_keywords = keyword_finder(df, n=20)
print('\n')
for keyword, score in top_keywords:
    print(f"Keyword: {keyword}, Total TF-IDF Score: {score}")
print('\n')
