import pandas as pd
import networkx as nx
from community import community_louvain 
import matplotlib.pyplot as plt

# Load data
titles = pd.read_csv("data/soc-redditHyperlinks-title.tsv", delimiter="\t")
body = pd.read_csv("data/soc-redditHyperlinks-body.tsv", delimiter="\t")

# Combine dataframes
all_data = pd.concat([titles, body])

# Sample a subset of the data
subset_data = all_data.sample(frac=0.1, random_state=42)  #### Change the fraction to change subset size ####


# Create directed graph
G = nx.from_pandas_edgelist(subset_data, 'SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT', create_using=nx.DiGraph())

# Convert to undirected graph
G_undirected = G.to_undirected()

# Run Louvain community detection
partition = community_louvain.best_partition(G_undirected)

#text based readout of the clusters
communities = {}
for node, community_id in partition.items():
    if community_id not in communities:
        communities[community_id] = []
    communities[community_id].append(node)

for community_id, nodes in communities.items():
    if len(nodes)>2:
        print(f"Community {community_id}:")
        print(f"Number of nodes: {len(nodes)}")
        print("Nodes:",nodes)
        #### get the first node in each cluster just as an example ####
        row_entry = all_data.loc[all_data['SOURCE_SUBREDDIT'] == nodes[0]].iloc[0]
        print("Row entry:")
        print(row_entry,'\n')
        #### if You wish to print out all the info about all the nodes in each cluster ####
        # for node in nodes:
        #     print(f"Node {node}:")
        #     # Check if there are rows matching the condition in all_data
        #     if not all_data[all_data['SOURCE_SUBREDDIT'] == node].empty:
        #         # Get the row entry corresponding to the node from all_data
        #         row_entry = all_data.loc[all_data['SOURCE_SUBREDDIT'] == node].iloc[0]
        #         print("Row entry:")
        #         print(row_entry)
        #     else:
        #         print("No matching row entry found.")
        print()




# Visualize the graph with community colors
##### This takes forever and is hard to understand. run at your own risk #####
# pos = nx.spring_layout(G)  # positions for all nodes
# plt.figure(figsize=(10, 10))
# # nodes
# nx.draw_networkx_nodes(G, pos, node_size=20, cmap=plt.cm.RdYlBu, node_color=list(partition.values()))
# # edges
# nx.draw_networkx_edges(G, pos, alpha=0.3)
# plt.show()
