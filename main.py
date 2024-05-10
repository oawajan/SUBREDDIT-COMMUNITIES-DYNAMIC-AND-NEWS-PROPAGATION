import numpy as np
import pandas as pd
import networkx as nx
from community import community_louvain
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pygraphviz as pgv
import math
from datetime import datetime


def TopPosts(df,n=0) -> dict:
    freq = dict()
    for index, post in df.iterrows():
        targets = post["TARGET_POST_IDS"]
        for id in targets:
            if id not in freq.keys():
                freq[id] = 0
            else:
                freq[id] += 1
    freq = dict(sorted(freq.items(), key=lambda item:item[1]))
    print(freq)
    return freq


def findPosts(ID,df) -> list:
    results = list()
    for index, row in df.iterrows():
        if ID in row["TARGET_POST_IDS"]:
            results.append(row["POST_ID"])
    filtered = list()
    for item in results:
        if item not in filtered:
            filtered.append(item)
    return filtered


def load_ds():
    df = pd.read_csv("data\\out.csv", low_memory=False)
    df = df.drop(columns=["AUTHOR", "TEXT", "URL", "LINK"], axis=1)
    df1 = df["TARGET_POST_IDS"].apply(update)
    df1 = pd.concat([df1, df[["TIMESTAMP", "POST_ID",
                              "SOURCE_SUBREDDIT", "TARGET_SUBREDDITS"]]], axis=1)
    df1.dropna()
    df1 = df1[df1["TIMESTAMP"] != "TIMESTAMP"]
    df = df1["TIMESTAMP"].apply(convert_to_datetime)
    df = pd.concat([df, df1[["TARGET_POST_IDS",
                             "POST_ID", "SOURCE_SUBREDDIT", "TARGET_SUBREDDITS"]]], axis=1)
    df.dropna()
    #Posts = TopPosts(df)
    #PostID = next(iter(Posts))
    #print(PostID)
    InformationCascades(df,PostID="3l5xpg") #3l4t1t, 3l5xpg,5bibaj,6116dt


def InformationCascades(df, PostID='4asjoo'):
    explored_nodes = list()
    labels = {}
    post_list = list()
    plt.figure(figsize=(30, 30))
    casc_graph = nx.DiGraph()
    post_list.append(PostID)
    initial = df[df['POST_ID'] == PostID]["SOURCE_SUBREDDIT"].values[-1]
    labels[initial] = initial
    casc_graph.add_node(initial)
    pos = dict()
    pos[initial] = [0, 0]
    while not (len(post_list) == 0):
        print(f"Exploring {post_list[0]}")
        print(f"Queue size {len(post_list)}")
        print("#"*20)
        explored_nodes.append(post_list[0])
        results = findPosts(post_list[0], df)
        if not (len(results) == 0):
            source = df[df['POST_ID'] == post_list[0]]["SOURCE_SUBREDDIT"].values[-1]
            p = Points(len(results), pos[source][0],
                       pos[source][1], r=1, c=1)
            i = 0
            for result in results:
                if (result not in explored_nodes) and (results not in post_list):
                    target = df[df['POST_ID'] == result]["SOURCE_SUBREDDIT"].values[-1]

                    if target not in pos.keys():
                        print(f"adding node {target}")
                        casc_graph.add_node(target)
                        pos[target] = p[i]
                        labels[target] = target
                    post_list.append(result)
                    casc_graph.add_edge(source, target)
                    i += 1
        post_list.pop(0)

    nx.draw_networkx_nodes(casc_graph, label=labels, pos=pos,
                           node_color="orange", node_size=500)
    nx.draw_networkx_labels(casc_graph, labels=labels,
                            pos=pos, font_size=6)
    nx.draw_networkx_edges(casc_graph, pos=pos)

    plt.show()


def convert_to_datetime(timestamp):
    return datetime.strptime(timestamp, '%Y-%m-%d %H:%M')


def update(TARGET_POST_IDS):
    return eval(TARGET_POST_IDS)


def GetPostID(df, n=0):
     return df.value_counts('POST_ID').keys()[:n]


def Points(n=1, x_start=0, y_start=0, r=3, c=2) -> list:

    if n == 1:
        return [[x_start+1, y_start]]

    pi = math.pi
    points = []

    if 2 == c:
        for i in range(n):
            angle = 2 * pi * i / n
            x = (math.cos(angle) * r) + x_start
            y = (math.sin(angle) * r) + y_start
            points.append([x, y])
        return points

    else:
        for i in range(n):
            angle = 1 * pi * i / n
            x = (math.cos(angle) * r) + x_start
            y = (math.sin(angle) * r) + y_start
            points.append([x, y])

        for point in points:
            if 0 > point[0]:
                point[0] = point[0] * -1
                point[1] = point[1] * -1
        print(points)
        return points


def normalize_data(data):
    # Find the minimum and maximum values in the data
    data_min = min(data)
    data_max = max(data)

    # Calculate the range of the data
    data_range = data_max - data_min
    if data_range == 0:
        data_range = 100
    # Calculate the scaling factor
    scale_factor = (750 - 100) / data_range

    # Normalize the data using linear scaling
    normalized_data = [int((x - data_min) * scale_factor + 40) for x in data]

    return normalized_data


def clustered_graph(df) -> None:
    communities = df['SOURCE_SUBREDDIT'].value_counts()[:20].index
    option = ['g', 'r', 'b']
    colors = []
    node_sizes = []
    nodelist = []
    labels = {}
    comm_graph = nx.DiGraph()
    f_pos = dict()
    for community_1 in communities:
        events = df[df['SOURCE_SUBREDDIT'] == community_1][:30]
        if community_1 not in nodelist:
            label = community_1
            height = width = len(events)
            if height < 20:
                height = width = 30
                label = " "
            else:
                height = width = len(events) * 100
            weight = events['LINK_SENTIMENT'].value_counts().keys()[0]
            if weight == 1:
                colors.append(option[0])
            else:
                colors.append(option[1])

            comm_graph.add_node(community_1)
            labels[community_1] = label
            node_sizes.append(height)
            nodelist.append(community_1)

        for target in events['TARGET_SUBREDDIT']:
            sentiment_counts = events[events['TARGET_SUBREDDIT'] == target
                                      ]['LINK_SENTIMENT'].value_counts()
            if target not in nodelist:
                label = target
                height = width = sentiment_counts.sum()
                if height < 20:
                    height = width = 30
                    label = " "
                else:
                    height = width = sentiment_counts.sum() * 100
                comm_graph.add_node(target)
                labels[target] = label
                node_sizes.append(height)
                nodelist.append(target)

            comm_graph.add_edge(community_1, target)

            if len(sentiment_counts) > 1:
                if sentiment_counts.values[0] > sentiment_counts.values[1]:
                    weight = sentiment_counts.keys()[0]
                    if weight == 1:
                        colors.append(option[0])
                    else:
                        colors.append(option[1])
                else:
                    colors.append(option[2])
            else:
                weight = sentiment_counts.keys()[0]
                if weight == 1:
                    colors.append(option[0])
                else:
                    colors.append(option[1])
    parts = nx.community.louvain_partitions(comm_graph, seed=0)
    count = 0
    partitions = []
    for _ in parts:
        count = len(_)
        partitions.append(_)
    print(partitions)
    grids = int(np.ceil(np.sqrt(count)))
    x_positions = np.linspace(-20, 20, grids)
    y_positions = np.linspace(-20, 20, grids)
    positions = [[x, y] for x in x_positions for y in y_positions]
    plt.figure(figsize=(20, 20))
    pos = nx.spring_layout(comm_graph, k=1)
    for node in comm_graph.nodes():
        if node not in pos:
            print(f"Node {node} is missing from the positions dictionary.")
            pos[node] = [0, 0]
    i = 0
    for parts in partitions:
        for nodes in parts:
            sub_node_sizes = []
            sub_node_labels = {}
            sub_node_colors = []
            print(f"Nodes: \n {nodes}")
            if isinstance(nodes, int):
                nodes = [nodes]
            if nodes:
                subgraph = comm_graph.subgraph(nodes)
                for node in nodes:
                    idx = list(comm_graph.nodes()).index(node)
                    sub_node_sizes.append(node_sizes[idx])
                    sub_node_labels[node] = labels[node]


                subgraph_pos = dict()
                points = Points(len(subgraph.nodes()), positions[i][0],
                                positions[i][1], r=3, c=2)
                j=0
                for n in subgraph.nodes():
                    subgraph_pos[n] = points[j]
                    f_pos[n] = points[j]
                    j += 1

                _pos = subgraph_pos
                print(f"subgraph_pos: {subgraph_pos}")
                i += 1
    nx.draw_networkx_nodes(comm_graph, pos=f_pos, nodelist=comm_graph.nodes(),
                                       node_size=normalize_data(node_sizes),
                                       node_color='orange', edgecolors='grey')
    nx.draw_networkx_labels(comm_graph, pos=f_pos, labels=labels, font_size=3)
    nx.draw_networkx_edges(comm_graph, pos=f_pos, alpha=0.5, edge_color=colors)

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(".\\graphs\\Clustered_communities.png", format="PNG", bbox_inches='tight')
    plt.show()


def clustered_graph2(df) -> None:
    communities = df['SOURCE_SUBREDDIT'].value_counts()[:20].index
    option = ['g', 'r', 'b']
    colors = []
    node_sizes = []
    nodelist = []
    labels = {}
    comm_graph = nx.DiGraph()
    for community_1 in communities:
        events = df[df['SOURCE_SUBREDDIT'] == community_1][:20]
        if community_1 not in nodelist:
            label = community_1
            height = width = len(events)
            if height < 20:
                height = width = 30
                label = " "
            else:
                height = width = len(events) * 100
            weight = events['LINK_SENTIMENT'].value_counts().keys()[0]
            if weight == 1:
                colors.append(option[0])
            else:
                colors.append(option[1])

            comm_graph.add_node(community_1)
            labels[community_1] = label
            node_sizes.append(height)
            nodelist.append(community_1)

        for target in events['TARGET_SUBREDDIT']:
            sentiment_counts = events[events['TARGET_SUBREDDIT'] == target
                                      ]['LINK_SENTIMENT'].value_counts()
            if target not in nodelist:
                label = target
                height = width = sentiment_counts.sum()
                if height < 20:
                    height = width = 30
                    label = " "
                else:
                    height = width = sentiment_counts.sum() * 100
                comm_graph.add_node(target)
                labels[target] = label
                node_sizes.append(height)
                nodelist.append(target)

            comm_graph.add_edge(community_1, target)

            if len(sentiment_counts) > 1:
                if sentiment_counts.values[0] > sentiment_counts.values[1]:
                    weight = sentiment_counts.keys()[0]
                    if weight == 1:
                        colors.append(option[0])
                    else:
                        colors.append(option[1])
                else:
                    colors.append(option[2])
            else:
                weight = sentiment_counts.keys()[0]
                if weight == 1:
                    colors.append(option[0])
                else:
                    colors.append(option[1])
    parts = nx.community.louvain_partitions(comm_graph, seed=0)
    count = 0
    partitions = []
    for _ in parts:
        count = len(_)
        partitions.append(_)
    print(partitions)
    grids = int(np.ceil(np.sqrt(count)))
    x_positions = np.linspace(-30, 30, grids)
    y_positions = np.linspace(-30, 30, grids)
    positions = [[x, y] for x in x_positions for y in y_positions]
    plt.figure(figsize=(20, 20))
    pos = nx.spring_layout(comm_graph, k=1)
    for node in comm_graph.nodes():
        if node not in pos:
            print(f"Node {node} is missing from the positions dictionary.")
            pos[node] = [0, 0]
    i = 0
    for parts in partitions:
        for nodes in parts:
            sub_node_sizes = []
            sub_node_labels = {}
            sub_node_colors = []
            print(f"Nodes: \n {nodes}")
            if isinstance(nodes, int):
                nodes = [nodes]
            if nodes:
                subgraph = comm_graph.subgraph(nodes)
                for node in nodes:
                    idx = list(comm_graph.nodes()).index(node)
                    sub_node_sizes.append(node_sizes[idx])
                    sub_node_labels[node] = labels[node]


                subgraph_pos = dict()
                points = Points(len(subgraph.nodes()), positions[i][0], positions[i][1])
                j=0
                for n in subgraph.nodes():
                    subgraph_pos[n] = points[j]
                    j += 1

                _pos = subgraph_pos
                print(f"subgraph_pos: {subgraph_pos}")
                nx.draw_networkx_nodes(subgraph, pos=_pos, nodelist=nodes,
                                       node_size=normalize_data(sub_node_sizes),
                                       node_color='orange', edgecolors='grey')
                nx.draw_networkx_labels(subgraph, pos=_pos, labels=sub_node_labels, font_size=6)
                nx.draw_networkx_edges(subgraph, pos=_pos, alpha=0.5, edge_color=colors)
                i += 1
    plt.axis('off')
    plt.tight_layout()


def pgv_graph(df) -> None:
    comm_graph = pgv.AGraph(directed=True, nodesep=10.0, ranksep=30.0,
                            overlap=False, splines='true')
    comm_graph.node_attr["shape"] = "circle"
    comm_graph.node_attr["color"] = "orange"
    comm_graph.node_attr["style"] = "filled"
    nodelist = []
    communities = df['SOURCE_SUBREDDIT'].value_counts()[:200].keys().unique()
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
                    x2 = np.log10((sentiment_counts.values[0]
                                   - sentiment_counts.values[1]) * 2.5)
                    y2 = np.log10((sentiment_counts.values[0]
                                   - sentiment_counts.values[1]) * 2.5)
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
            node.attr["pos"] = (f"{-(float(x1) - 7000) / 10.0:f},"
                                f"{(float(y1) - 2000) / 10.0:f}")
            node = comm_graph.get_node(target)
            node.attr["pos"] = (f"{-(float(x2) - 7000) / 10.0:f},"
                                f"{(float(y2) - 2000) / 10.0:f}")
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
    plt.bar(bars.keys()[:number], bars.values[:number], width=1,
            label="overall", edgecolor='black')
    plt.title(f"Top {number} Active Subreddits")
    plt.xlabel("Communities")
    plt.ylabel("Frequency")
    plt.tick_params(axis='x', labelrotation=90)
    plt.tight_layout()
    plt.legend()
    plt.show()

    # print top 20 positive subreddits
    positive = df[df['LINK_SENTIMENT'] == 1].value_counts('SOURCE_SUBREDDIT')
    plt.bar(positive.keys()[:number], positive.values[:number], width=1,
            label="Positive", edgecolor='black')
    plt.title(f"Top {number} Positive Subreddits")
    plt.xlabel("Communities")
    plt.ylabel("Frequency")
    plt.tick_params(axis='x', labelrotation=90)
    plt.tight_layout()
    plt.legend()
    plt.show()

    # print top 20 positive subreddits
    positive = df[df['LINK_SENTIMENT'] == -1].value_counts('SOURCE_SUBREDDIT')
    plt.bar(positive.keys()[:number], positive.values[:number], width=1,
            label="Negative", edgecolor='black')
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
    plt.bar(bars.keys()[:number], bars.values[:number], width=width,
            label="overall", edgecolor='#008fd5')
    plt.bar(negative.keys()[:number], negative.values[:number], width=width,
            label="negative", edgecolor='#ffffff')
    plt.bar(positive.keys()[:number], positive.values[:number], width=width,
            label="positive", edgecolor='#eeeeee')
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

    plt.bar(indexes - width, subset_bars.values[:number], width=width,
            label="overall", edgecolor='#008fd5')
    plt.bar(indexes, negative, width=width, label="negative",
            edgecolor='#ffffff')
    plt.bar(indexes + width, positive, width=width, label="positive",
            edgecolor='#eeeeee')
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
    plt.bar(bars.keys()[:number], bars.values[:number], width=1,
            label="overall", edgecolor='black')
    plt.title(f"Top {number} Propagated Posts")
    plt.xlabel("Posts_ID")
    plt.ylabel("Frequency")
    plt.tick_params(axis='x', labelrotation=90)
    plt.tight_layout()
    plt.show()

    graphpost(all_data, bars.keys()[0])

    # print top 20 positive
    bars = df[df['LINK_SENTIMENT'] == 1].value_counts('POST_ID')
    plt.bar(bars.keys()[:number], bars.values[:number], width=1,
            label="overall", edgecolor='black')
    plt.title(f"Top {number} positive Propagated Posts")
    plt.xlabel("Posts_ID")
    plt.ylabel("Frequency")
    plt.tick_params(axis='x', labelrotation=90)
    plt.tight_layout()
    plt.show()

    graphpost(all_data, bars.keys()[0])

    # print top 20 negative
    bars = df[df['LINK_SENTIMENT'] == -1].value_counts('POST_ID')
    plt.bar(bars.keys()[:number], bars.values[:number], width=1,
            label="overall", edgecolor='black')
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
        post_graph[source][target]['weight'] = post_events[
            (post_events['SOURCE_SUBREDDIT'] == source) &
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
subset_data = all_data.sample(frac=1, random_state=0)  #### Change the fraction to change subset size ####

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
# graphallcommubnities(subset_data)
# plotcorrelations(all_data)
# clustered_graph(all_data)
# gt_graph(subset_data)
# print(Points(n=10, x_start=0.0, y_start=0))
load_ds()