# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def Project_name():
    print(f'SUBREDDIT-COMMUNITIES-DYNAMIC-AND-NEWS-PROPAGATION')

def Members_names():
    print(f'Omar and Drew')


def readdata() -> pd.DataFrame:
    titles = pd.read_csv("data/soc-redditHyperlinks-title.tsv", delimiter="\t")
    body = pd.read_csv("data/soc-redditHyperlinks-body.tsv", delimiter="\t")
    return titles, body


if __name__ == '__main__':
    df1, df2 = readdata()
    print(len(df1[df1["SOURCE_SUBREDDIT"] == "conspiracy"]))
