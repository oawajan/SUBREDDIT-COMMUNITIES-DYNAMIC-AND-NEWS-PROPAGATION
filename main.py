import numpy as np
import pandas as pd
import networkx as nx
from community import community_louvain
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pygraphviz as pgv
import os
#from "1_graph_tool".all import *


# def gt_graph(df) -> None:
#     comm_graph = Graph(directed=True)
#     communities = df['SOURCE_SUBREDDIT'].unique()
#     for community_1 in communities:
#         events = df[(df['SOURCE_SUBREDDIT'] == community_1)]
#         comm_graph.add_vertex(community_1)
#         for target in events['TARGET_SUBREDDIT']:
#             comm_graph.add_vertex(target)
#             sentiment_counts = events[(events['SOURCE_SUBREDDIT'] == community_1) &
#                                       (events['TARGET_SUBREDDIT'] == target)
#                                       ]['LINK_SENTIMENT'].value_counts()
#             comm_graph.add_edge(community_1, target)
#     graph_draw(comm_graph)
#


# def areas(df) -> dict:
#     areas_dic = dict()
#     area_c = 1
#     node_list = []
#     communities = df['SOURCE_SUBREDDIT'].value_counts()[:200].keys().unique()
#     for community in communities:
#         if community not in node_list:
#             print("adding node to discovered bins")
#             node_list.append(community)
#             areas_dic[area_c] = list()
#             areas_dic[area_c].append(community)
#             events = df[(df['SOURCE_SUBREDDIT'] == community)]
#             targets = events['TARGET_SUBREDDIT'].unique()
#             for target in targets:
#                 node_list.append(target)



def pgv_graph(df) -> None:
    comm_graph = pgv.AGraph(directed=True, nodesep=10.0, ranksep=30.0, overlap=False, splines='true')
    comm_graph.node_attr["shape"] = "circle"
    comm_graph.node_attr["color"] = "orange"
    comm_graph.node_attr["style"] = "filled"
    nodelist = []
    communities = df['SOURCE_SUBREDDIT'].value_counts()[:200].keys().unique()
    #area_list = areas(df)
    option = ['green', 'red', 'blue']

    for community_1 in communities:
        events = df[(df['SOURCE_SUBREDDIT'] == community_1)]
        if not community_1 in nodelist:
            label = community_1
            height = width = np.log10(len(events))
            if 0.2 > height:
                height = width = 0.2
                label = ''
            nodelist.append(community_1)
            comm_graph.add_node(community_1, label=label, height=height, width=width)
        targets = events['TARGET_SUBREDDIT'].unique()
        for target in targets:
            if not target in nodelist:
                count = len(df[(df['TARGET_SUBREDDIT'] == community_1)])
                height = width = np.log10(count)
                label = target
                if 0.2 > height:
                    height = width = 0.2
                    label = ''
                comm_graph.add_node(target, label=label, height=height, width=width)
                nodelist.append(target)
            sentiment_counts = events[(events['SOURCE_SUBREDDIT'] == community_1) &
                                      (events['TARGET_SUBREDDIT'] == target)
                                      ]['LINK_SENTIMENT'].value_counts()
            x1 = y1 = x2 = y2 =0
            length = float()
            color = ''
            pos = float()
            if len(sentiment_counts) > 1:
                if sentiment_counts.values[0] > sentiment_counts.values[1]:
                    weight = sentiment_counts.keys()[0]
                    if weight == 1:
                        color = option[0]
                        length = 1
                        x1 = np.log10(sentiment_counts.values[0])
                        y1 = np.log10(sentiment_counts.values[0])
                        x2 = np.log10(sentiment_counts.values[0] * 1.5)
                        y2 = np.log10(sentiment_counts.values[0] * 1.5)
                    else:
                        color = option[1]
                        x1 = np.log10(sentiment_counts.values[1])
                        y1 = np.log10(sentiment_counts.values[1])
                        x2 = np.log10(sentiment_counts.values[1] * 5.5)
                        y2 = np.log10(sentiment_counts.values[1] * 5.5)
                        length = 6
                else:
                    weight = 0
                    color = option[2]
                    x1 = np.log10(sentiment_counts.values[0] * 2.5)
                    y1 = np.log10(sentiment_counts.values[0] * 2.5)
                    x2 = np.log10((sentiment_counts.values[0] - sentiment_counts.values[1]) * 2.5)
                    y2 = np.log10((sentiment_counts.values[0] - sentiment_counts.values[1]) * 2.5)
                    length = 3
            else:
                weight = sentiment_counts.keys()[0]
                if weight == 1:
                    length = 1
                    x1 = np.log10(sentiment_counts.values[0])
                    y1 = np.log10(sentiment_counts.values[0])
                    x2 = np.log10(sentiment_counts.values[0] * 2.5)
                    y2 = np.log10(sentiment_counts.values[0] * 2.5)
                    color = option[0]
                else:
                    length = 6
                    x1 = np.log10(sentiment_counts.values[0])
                    y1 = np.log10(sentiment_counts.values[0])
                    x2 = np.log10(sentiment_counts.values[0] * 5.5)
                    y2 = np.log10(sentiment_counts.values[0] * 5.5)
                    color = option[1]

            node = comm_graph.get_node(community_1)
            node.attr["pos"] = f"{-(float(x1) - 7000) / 10.0:f},{(float(y1) - 2000) / 10.0:f}"
            node = comm_graph.get_node(target)
            node.attr["pos"] = f"{-(float(x2) - 7000) / 10.0:f},{(float(y2) - 2000) / 10.0:f}"
            comm_graph.add_edge(community_1, target, color=color, len=length)

    comm_graph.layout("fdp")
    comm_graph.draw("image.png")
    comm_graph.draw("image.ps")


def plotcorrelations(df):
    overall = []
    positive = []
    negative = []
    communinties = df.value_counts('SOURCE_SUBREDDIT')
    for community in communinties:
        overall.append(len(df[df['SOURCE_SUBREDDIT'] == community]))
        positive.append(len(df[(df['SOURCE_SUBREDDIT'] == community) &
                               (df['LINK_SENTIMENT'] == 1)]))
        positive.append(len(df[(df['SOURCE_SUBREDDIT'] == community) &
                               (df['LINK_SENTIMENT'] == -1)]))
    x_axis = np.arange(0, len(communinties))
    plt.plot(x_axis=x_axis, data=overall, label="overall", color="blue")
    plt.plot(x_axis=x_axis, data=positive, label="positive", color="green")
    # plt.plot(overall, negative, label="negative", color="red")

    plt.title("Correlation line plot")
    plt.tight_layout()
    plt.legend()
    plt.show()

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
    subset_bars = df.value_counts('SOURCE_SUBREDDIT')[:number]
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

    graphcommubnities(df, bars.keys()[0])


def graphallcommubnities(df):
    # communities = df['SOURCE_SUBREDDIT'].unique()
    communities = df['SOURCE_SUBREDDIT'].value_counts()[:200].keys().unique()
    option = ['g', 'r', 'b']
    colors = []
    node_sizes = []

    comm_graph = nx.DiGraph()
    for community_1 in communities:
        comm_graph.add_node(community_1)
        events = df[(df['SOURCE_SUBREDDIT'] == community_1)]
        for target in events['TARGET_SUBREDDIT']:
            comm_graph.add_node(target)
            comm_graph.add_edge(community_1, target)

            sentiment_counts = events[(events['SOURCE_SUBREDDIT'] == community_1) &
                                      (events['TARGET_SUBREDDIT'] == target)
                                      ]['LINK_SENTIMENT'].value_counts()
            comm_graph[community_1][target]['node_size'] = sentiment_counts.sum() * 500
            comm_graph[community_1][target]['node_width'] = sentiment_counts.values.sum() * 500
            if len(sentiment_counts) > 1:
                if sentiment_counts.values[0] > sentiment_counts.values[1]:
                    weight = sentiment_counts.keys()[0]
                    if weight == 1:
                        colors.append(option[0])
                    else:
                        colors.append(option[1])
                else:
                    weight = 0
                    colors.append(option[2])
            else:
                weight = sentiment_counts.keys()[0]
                if weight == 1:
                    colors.append(option[0])
                else:
                    colors.append(option[1])
    plt.figure(3, figsize=(250, 250))
    pos = nx.spring_layout(G, k=10.5, iterations=20)
    nx.draw_random(comm_graph, #with_labels=True,
                   edge_color=colors,
                   width=1,
                   linewidths=1,
                   node_size=50,
                   alpha=0.9)
    red_patch = mpatches.Patch(color='red', label='Conflicting')
    blue_patch = mpatches.Patch(color='blue', label='Neutral')
    green_patch = mpatches.Patch(color='green', label='Friendly')
    plt.legend(handles=[red_patch, blue_patch, green_patch], loc=2, prop={'size': 15})
    plt.tight_layout()
    plt.show()


def graphcommubnities(df, source):
    communities = df[df['SOURCE_SUBREDDIT'] == source]
    comm_graph = nx.DiGraph()
    comm_graph.add_node(source)
    option = ['g', 'r', 'b']
    colors = []
    # add link sentiment as edge weight
    labels = {}
    for target in communities['TARGET_SUBREDDIT']:
        comm_graph.add_node(target)
        comm_graph.add_edge(source, target)
        sentiment_counts = communities[(communities['SOURCE_SUBREDDIT'] == source) &
                                       (communities['TARGET_SUBREDDIT'] == target)
                                       ]['LINK_SENTIMENT'].value_counts()
        if len(sentiment_counts) > 1:
            if sentiment_counts.values[0] > sentiment_counts.values[1]:
                weight = sentiment_counts.keys()[0]
                if weight == 1:
                    colors.append(option[0])
                else:
                    colors.append(option[1])
            else:
                weight = 0
                colors.append(option[2])
        else:
            weight = sentiment_counts.keys()[0]
            if weight == 1:
                colors.append(option[0])
            else:
                colors.append(option[1])

        comm_graph[source][target]['weight'] = weight

    plt.figure(3, figsize=(15, 15))
    pos = nx.spring_layout(comm_graph, k=10)
    nx.draw(comm_graph, with_labels=True,
            edge_color=colors,
            width=5,
            linewidths=1,
            node_size=8000,
            alpha=0.9)
    nx.draw_networkx_edge_labels(comm_graph, edge_labels=labels, pos=pos)
    red_patch = mpatches.Patch(color='red', label='Conflicting')
    blue_patch = mpatches.Patch(color='blue', label='Neutral')
    green_patch = mpatches.Patch(color='green', label='Friendly')
    plt.legend(handles=[red_patch, blue_patch, green_patch], loc=2, prop={'size': 23})
    plt.tight_layout()
    plt.show()


def plotposts(df, number=20):
    # print top 20
    bars = df.value_counts('POST_ID')
    plt.bar(bars.keys()[:number], bars.values[:number], width=1, label="overall", edgecolor='black')
    plt.title(f"Top {number} Propagated Posts")
    plt.xlabel("Posts_ID")
    plt.ylabel("Frequency")
    plt.tick_params(axis='x', labelrotation=90)
    plt.tight_layout()
    plt.show()

    graphpost(all_data, bars.keys()[0])

    # print top 20 positive
    bars = df[df['LINK_SENTIMENT'] == 1].value_counts('POST_ID')
    plt.bar(bars.keys()[:number], bars.values[:number], width=1, label="overall", edgecolor='black')
    plt.title(f"Top {number} positive Propagated Posts")
    plt.xlabel("Posts_ID")
    plt.ylabel("Frequency")
    plt.tick_params(axis='x', labelrotation=90)
    plt.tight_layout()
    plt.show()

    graphpost(all_data, bars.keys()[0])

    # print top 20 negative
    bars = df[df['LINK_SENTIMENT'] == -1].value_counts('POST_ID')
    plt.bar(bars.keys()[:number], bars.values[:number], width=1, label="overall", edgecolor='black')
    plt.title(f"Top {number} negative Propagated Posts")
    plt.xlabel("Posts_ID")
    plt.ylabel("Frequency")
    plt.tick_params(axis='x', labelrotation=90)
    plt.tight_layout()
    plt.show()

    graphpost(all_data, bars.keys()[0])
    return bars.keys()[0]


def graphpost(df, POST_ID) -> None:
    post_events = df[df['POST_ID'] == POST_ID]
    source = post_events['SOURCE_SUBREDDIT'].unique()[0]
    post_graph = nx.DiGraph()
    post_graph.add_node(source)
    color_list = []
    # add coloring to edges based on sentiment
    option = ['g', 'r']
    colors = []
    # add link sentiment as edge weight
    labels = {}

    for target in post_events['TARGET_SUBREDDIT']:
        post_graph.add_edge(source, target)
        post_graph[source][target]['weight'] = post_events[(post_events['SOURCE_SUBREDDIT'] == source) &
                                                           (post_events['TARGET_SUBREDDIT'] == target)
                                                           ]['LINK_SENTIMENT'].values[0]
    for e in post_graph.edges:
        labels[e] = post_graph.edges[e]['weight']
        if labels[e] == 1:
            colors.append(option[0])
        else:
            colors.append(option[1])

    plt.figure(3, figsize=(50, 50))
    pos = nx.spring_layout(post_graph, k=10)
    nx.draw(post_graph, with_labels=True,
            edge_color=colors,
            width=1,
            linewidths=1,
            node_size=5000,
            alpha=0.9)
    nx.draw_networkx_edge_labels(post_graph, edge_labels=labels, pos=pos)
    plt.show()


# Load data
titles = pd.read_csv("data/soc-redditHyperlinks-title.tsv", delimiter="\t")
body = pd.read_csv("data/soc-redditHyperlinks-body.tsv", delimiter="\t")

# Combine dataframes
all_data = pd.concat([titles, body])

# Sample a subset of the data
subset_data = all_data.sample(frac=0.009, random_state=0)  #### Change the fraction to change subset size ####

# Create directed graph
G = nx.from_pandas_edgelist(subset_data, 'SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT', create_using=nx.DiGraph())

# Convert to undirected graph
G_undirected = G.to_undirected()

# Run Louvain community detection
partition = community_louvain.best_partition(G_undirected)

# text based readout of the clusters
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


# plotcommubnities(subset_data, 10)
# top_post = plotposts(subset_data, 20)
# printtimeline(all_data, top_post)
# print(len(all_data['POST_ID'].unique()), len(all_data['TIMESTAMP'].unique()))
#graphallcommubnities(subset_data)
# plotcorrelations(all_data)
pgv_graph(subset_data)
#gt_graph(subset_data)
plt.close()
