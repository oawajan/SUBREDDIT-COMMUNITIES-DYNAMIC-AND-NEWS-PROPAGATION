import csv
import re
import pandas as pd
import sys

def iterate_through_lines_csv(csv_file):
    matching_post_id = []
    lines=[]
    target_post_ids=[]
    with open(csv_file, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)
        try:
            for line in reader:
                post_ids = line[4].split('/')[6]  # Extract post IDs
                links = find_links_in_string(line[5])  # Extract links from the 6th column
                target_post_ids_for_current_post = [extract_post_id(link) for link in links]
                if find_links_in_string(line[5]):  # If links are present in the text  # If links are present in the text
                    matching_post_id.append(post_ids)  # Store only the post IDs
                    lines.append(line)
                    target_post_ids.append(target_post_ids_for_current_post)
        except csv.Error as e:
            print('Error:', e)
    return matching_post_id, lines, target_post_ids

def find_links_in_string(input_string):
    # Regular expression pattern to match URLs
    url_pattern = r'(https?://www\.reddit\.com/\S+)'
    # Find all matches of URLs in the input string
    matches = re.findall(url_pattern, input_string)
    # Return the list of URLs found
    return matches

def extract_post_id(url):
    # Split the URL and get the post ID
    parts = url.split('/')
    return parts[-3] if len(parts) >= 7 else None

############ MAIN ############
maxInt = sys.maxsize

while True:
    # decrease the maxInt value by factor 10 
    # as long as the OverflowError occurs.
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)


# Load body DataFrame
titles = pd.read_csv("data/soc-redditHyperlinks-title.tsv", delimiter="\t")
body = pd.read_csv("data/soc-redditHyperlinks-body.tsv", delimiter="\t")

all_data = pd.concat([titles, body])

all_data=all_data.drop('TIMESTAMP', axis='columns')

csv_file = 'data/zst_as_csv.csv'
matching_post_ids, lines, target_post_ids= iterate_through_lines_csv(csv_file)

actual_df=pd.DataFrame(lines)
actual_df['POST_ID'] = matching_post_ids
actual_df['TARGET_POST_IDS'] = target_post_ids

matching_posts = all_data[all_data['POST_ID'].isin(matching_post_ids)].reset_index()

merged_df = pd.merge(matching_posts, actual_df, on='POST_ID', how='left')

new_column_names = {0: 'AUTHOR', 1: 'TITLE', 2: 'SCORE', 3:'TIMESTAMP',4: 'LINK', 5: 'TEXT', 6: 'URL'}
merged_df=merged_df.rename(columns=new_column_names)
merged_df=merged_df.drop(['index','PROPERTIES','LINK'],axis='columns')

merged_df.to_csv('data/out.csv', index=True, mode='a') 
