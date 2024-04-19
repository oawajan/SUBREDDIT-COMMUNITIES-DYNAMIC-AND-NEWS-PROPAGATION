import pandas as pd
import sys


# Load body DataFrame
titles = pd.read_csv("data/soc-redditHyperlinks-title.tsv", delimiter="\t")
body = pd.read_csv("data/soc-redditHyperlinks-body.tsv", delimiter="\t")

all_data = pd.concat([titles, body])
all_data=all_data.drop('TIMESTAMP', axis='columns')

actual_data = pd.read_csv('data/zst_as_csv.csv')

def extract_post_id(link):
    return link.split('/')[6]

actual_data['POST_ID'] = actual_data['link'].apply(extract_post_id)

merged_df = pd.merge(all_data, actual_data, on='POST_ID', how='left')

new_column_names = {0: 'AUTHOR', 1: 'TITLE', 2: 'SCORE', 3:'TIMESTAMP',4: 'LINK', 5: 'TEXT', 6: 'URL'}
merged_df=merged_df.rename(columns=new_column_names)
# merged_df=merged_df.drop(['index','PROPERTIES',''],axis='columns')

merged_df.to_csv('data/out_V2.csv', index=True, mode='a') 