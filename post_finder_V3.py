import pandas as pd

titles = pd.read_csv("data/soc-redditHyperlinks-title.tsv", delimiter="\t")
body = pd.read_csv("data/soc-redditHyperlinks-body.tsv", delimiter="\t")

stan_data = pd.concat([titles, body])
stan_data=stan_data.drop('TIMESTAMP', axis='columns')

actual_data=pd.read_csv('data/zst_as_csv.csv')

def extract_post_id(link):
    return link.split('/')[6]

actual_data['POST_ID'] = actual_data['link'].apply(extract_post_id)

matching_rows = []

for index, row in actual_data.iterrows():
    post_id = row['POST_ID']
    if post_id in stan_data['POST_ID'].values:
        matching_rows.append(row)

matching_df = pd.DataFrame(matching_rows)
merged_df = pd.merge(matching_df, stan_data, on='POST_ID', how='inner')
merged_df.to_csv('data/out_V3.csv', index=True, mode='a') 
