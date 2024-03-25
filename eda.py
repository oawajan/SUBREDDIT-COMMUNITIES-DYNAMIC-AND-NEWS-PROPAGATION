# Exploratory Data Analysis 
import pandas as pd

# Load data
titles = pd.read_csv("data/soc-redditHyperlinks-title.tsv", delimiter="\t")
body = pd.read_csv("data/soc-redditHyperlinks-body.tsv", delimiter="\t")

# Combine dataframes
all_data = pd.concat([titles, body])

# Most common SOURCE_SUBREDDIT
most_common_source_subreddits = all_data['SOURCE_SUBREDDIT'].value_counts().head(n=10)
print(f"Most common SOURCE_SUBREDDIT: {most_common_source_subreddits}")

# Most common TARGET_SUBREDDIT
most_common_target_subreddits = all_data['TARGET_SUBREDDIT'].value_counts().head(n=10)
print(f"\nMost common TARGET_SUBREDDIT: {most_common_target_subreddits}")

most_common_post_id=all_data['POST_ID'].value_counts().head(n=10)
print(f"\nMost common POST_ID: {most_common_post_id}")

# Number of positive post labels
positive_post_labels = all_data[all_data['LINK_SENTIMENT'] == 1]['LINK_SENTIMENT'].count()
print(f"\nNumber of positive post labels: {positive_post_labels}")

# Number of negative post labels
negative_post_labels = all_data[all_data['LINK_SENTIMENT'] == -1]['LINK_SENTIMENT'].count()
print(f"Number of negative post labels: {negative_post_labels}")

# Convert TIMESTAMP column to datetime format
all_data['TIMESTAMP'] = pd.to_datetime(all_data['TIMESTAMP'])
# Calculate the average TIMESTAMP
average_timestamp = all_data['TIMESTAMP'].mean()
print(f"Average TIMESTAMP: {average_timestamp}")

#filtering the post id to the most common poster: 4asjoos
filtered_data = all_data[all_data['POST_ID'] == '3a3uk8s']
# Count the number of positive post labels
positive_labels_counts = filtered_data[filtered_data['LINK_SENTIMENT'] == 1]['LINK_SENTIMENT'].count()
print(f"\nNumber of positive post labels from 4asjoos: {positive_labels_counts}")