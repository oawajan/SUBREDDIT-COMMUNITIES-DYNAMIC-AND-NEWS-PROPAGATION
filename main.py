import numpy as np
import pandas as pd
import networkx as nx
from community import community_louvain 
import matplotlib.pyplot as plt


def plotcommubnities(df, number):
    # print top 20
    bars = df.value_counts('SOURCE_SUBREDDIT')
    plt.bar(bars.keys()[:number], bars.values[:number], width=1, label="overall", edgecolor='black')
    plt.title(f"Top {number} Active Subreddits")
    plt.xlabel("Communities")
    plt.ylabel("Frequency")
    plt.tick_params(axis='x', labelrotation=90)
    plt.tight_layout()
    plt.legend()
    plt.show()

    # print top 20 positive subreddits
    positive = df[df['LINK_SENTIMENT'] == 1].value_counts('SOURCE_SUBREDDIT')
    plt.bar(positive.keys()[:number], positive.values[:number], width=1, label="Positive", edgecolor='black')
    plt.title(f"Top {number} Positive Subreddits")
    plt.xlabel("Communities")
    plt.ylabel("Frequency")
    plt.tick_params(axis='x', labelrotation=90)
    plt.tight_layout()
    plt.legend()
    plt.show()

    # print top 20 positive subreddits
    positive = df[df['LINK_SENTIMENT'] == -1].value_counts('SOURCE_SUBREDDIT')
    plt.bar(positive.keys()[:number], positive.values[:number], width=1, label="Negative", edgecolor='black')
    plt.title(f"Top {number} Negative Subreddits")
    plt.xlabel("Communities")
    plt.ylabel("Frequency")
    plt.tick_params(axis='x', labelrotation=90)
    plt.tight_layout()
    plt.legend()
    plt.show()

    # print top 20 overall/positive/negative including sentiment

    bars = df.value_counts('SOURCE_SUBREDDIT')
    positive = df[df['LINK_SENTIMENT'] == 1].value_counts('SOURCE_SUBREDDIT')
    negative = df[df['LINK_SENTIMENT'] == -1].value_counts('SOURCE_SUBREDDIT')
    width = 0.5
    plt.bar(bars.keys()[:number], bars.values[:number], width=width, label="overall", edgecolor='#008fd5')
    plt.bar(negative.keys()[:number], negative.values[:number], width=width, label="negative", edgecolor='#ffffff')
    plt.bar(positive.keys()[:number], positive.values[:number], width=width, label="positive", edgecolor='#eeeeee')
    plt.title(f"Top {number}")
    plt.xlabel("Communities")
    plt.ylabel("Frequency")
    plt.tick_params(axis='x', labelrotation=90)
    plt.tight_layout()
    plt.legend()
    plt.show()

    # print sentiment details on top 20 active subreddits
    subreddits = []
    positive = []
    negative = []
    indexes = np.arange(0, number)
    width = 0.30
    subset_bars =df.value_counts('SOURCE_SUBREDDIT')[:number]
    for subreddit in list(subset_bars.keys()):
        # pos = subset_data[subset_data['SOURCE_SUBREDDIT'] == subreddit].value_counts('LINK_SENTIMENT')
        positive.append(df[(df['SOURCE_SUBREDDIT'] == subreddit) &
                                    (df['LINK_SENTIMENT'] == 1)].shape[0])
        negative.append(df[(df['SOURCE_SUBREDDIT'] == subreddit) &
                                    (df['LINK_SENTIMENT'] == -1)].shape[0])
        subreddits.append(subreddit)

    plt.bar(indexes - width, subset_bars.values[:number], width=width, label="overall", edgecolor='#008fd5')
    plt.bar(indexes, negative, width=width, label="negative", edgecolor='#ffffff')
    plt.bar(indexes + width, positive, width=width, label="positive", edgecolor='#eeeeee')
    plt.xticks(ticks=indexes, labels=subreddits)
    plt.tick_params(axis='x', labelrotation=90)
    plt.xlabel("Communities")
    plt.ylabel("Frequency")
    plt.title(f"Detailed Top {number}")
    plt.style.use("fivethirtyeight")
    plt.tight_layout()
    plt.legend()
    plt.show()


def plotposts(df, number=20) -> str:
    # print top 20
    bars = df.value_counts('POST_ID')
    plt.bar(bars.keys()[:number], bars.values[:number], width=1, label="overall", edgecolor='black')
    plt.title(f"Top {number} Propagated Posts")
    plt.xlabel("Posts_ID")
    plt.ylabel("Frequency")
    plt.tick_params(axis='x', labelrotation=90)
    plt.tight_layout()
    plt.show()
    return bars.keys()[0]


def plotpostgraph(df, POST_ID) -> None:
    post_events = df[df['POST_ID'] == POST_ID]
    print(len(post_events['SOURCE_SUBREDDIT'].unique()))
    print(len(post_events['TARGET_SUBREDDIT'].unique()))
    print(len(post_events))
    source = post_events['SOURCE_SUBREDDIT'].unique()[0]
    post_graph = nx.DiGraph()
    post_graph.add_node(source)
    for target in post_events['TARGET_SUBREDDIT']:
        post_graph.add_edge(source, target)
        post_graph[source][target]['weight'] = post_events[(post_events['SOURCE_SUBREDDIT'] == source) &
                                                           (post_events['TARGET_SUBREDDIT'] == target)
                                                           ]['LINK_SENTIMENT'].values[0]
    nx.draw(post_graph, with_labels=True, node_size=500)
    labels = {}
    for e in post_graph.edges:
        labels[e] = post_graph.edges[e]['weight']
    pos = nx.spring_layout(post_graph, k=10)
    nx.draw_networkx_edge_labels(G, edge_labels=labels, pos=pos)
    plt.show()


def printtimeline(df, POST_ID):
    print(POST_ID)
    post_events = df[df['POST_ID'] == POST_ID]
    post_events['TIMESTAMP'] = pd.to_datetime(post_events['TIMESTAMP'])
    # print(post_events['TIMESTAMP'].unique())
    print(post_events[['TIMESTAMP', 'SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT']])
    # print(len(df['TIMESTAMP'].unique()))
    # print(post_events[:5][['SOURCE_SUBREDDIT', 'TARGET_SUBBREDDIT', 'TIMESTAMP']])


# Load data
titles = pd.read_csv("data/soc-redditHyperlinks-title.tsv", delimiter="\t")
body = pd.read_csv("data/soc-redditHyperlinks-body.tsv", delimiter="\t")

# Combine dataframes
all_data = pd.concat([titles, body])

# Sample a subset of the data
subset_data = all_data.sample(frac=0.001, random_state=0)  #### Change the fraction to change subset size ####


# Create directed graph
G = nx.from_pandas_edgelist(subset_data, 'SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT', create_using=nx.DiGraph())

# Convert to undirected graph
G_undirected = G.to_undirected()

# Run Louvain community detection
partition = community_louvain.best_partition(G_undirected)

#text based readout of the clusters
# communities = {}
# for node, community_id in partition.items():
#     if community_id not in communities:
#         communities[community_id] = []
#     communities[community_id].append(node)
#
# for community_id, nodes in communities.items():
#     if len(nodes)>2:
#         print(f"Community {community_id}:")
#         print(f"Number of nodes: {len(nodes)}")
#         print("Nodes:",nodes)
#         #### get the first node in each cluster just as an example ####
#         row_entry = all_data.loc[all_data['SOURCE_SUBREDDIT'] == nodes[0]].iloc[0]
#         print("Row entry:")
#         print(row_entry,'\n')
#         #### if You wish to pr
#
#         int out all the info about all the nodes in each cluster ####
#     for node in nodes:
#         print(f"Node {node}:")
#         # Check if there are rows matching the condition in all_data
#         if not all_data[all_data['SOURCE_SUBREDDIT'] == node].empty:
#             # Get the row entry corresponding to the node from all_data
#             row_entry = all_data.loc[all_data['SOURCE_SUBREDDIT'] == node].iloc[0]
#             print("Row entry:")
#             print(row_entry)
#         else:
#             print("No matching row entry found.")
#         print()



# Visualize the graph with community colors
##### This takes forever and is hard to understand. run at your own risk #####
# pos = nx.spring_layout(G)  # positions for all nodes
# plt.figure(figsize=(10, 10))
# # nodes
# nx.draw_networkx_nodes(G, pos, node_size=20, cmap=plt.cm.RdYlBu, node_color=list(partition.values()))
# # edges
# nx.draw_networkx_edges(G, pos, alpha=0.3)
# plt.show()

# plotcommubnities(all_data, 10)
top_post = plotposts(all_data, 20)
# printtimeline(all_data, top_post)
plotpostgraph(all_data, top_post)
# print(len(all_data['POST_ID'].unique()), len(all_data['TIMESTAMP'].unique()))
