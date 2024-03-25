#Linear model prediting link sentiment based on the source and target nodes
#only test the on the most popular source node and the 10 most popular target nodes for size reasons 
import pandas as pd
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def create_lm(subset_data_encoded):
    X = subset_data_encoded.drop(columns=['LINK_SENTIMENT', 'POST_ID', 'TIMESTAMP','PROPERTIES'])  # Drop non-feature columns
    y = subset_data_encoded['LINK_SENTIMENT']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train a logistic regression model
    model = linear_model.LogisticRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

def create_rand_forest(subset_data_encoded):
    X = subset_data_encoded.drop(columns=['LINK_SENTIMENT','POST_ID','TIMESTAMP','PROPERTIES'])
    y = subset_data_encoded['LINK_SENTIMENT']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize Random Forest Classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42,max_depth=3)

    # Train the classifier
    rf_classifier.fit(X_train, y_train)

    # Predict on the test set
    y_pred = rf_classifier.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))



# Load data
titles = pd.read_csv("data/soc-redditHyperlinks-title.tsv", delimiter="\t")
body = pd.read_csv("data/soc-redditHyperlinks-body.tsv", delimiter="\t")

# Combine dataframes
all_data = pd.concat([titles, body])

# Get the 10 most common source and target subreddits
top_source_subreddits = all_data['SOURCE_SUBREDDIT'].value_counts().head(1).index
top_target_subreddits = all_data['TARGET_SUBREDDIT'].value_counts().head(10).index

# Filter data to only include rows with the 10 most common source or target subreddits
filtered_data = all_data[(all_data['SOURCE_SUBREDDIT'].isin(top_source_subreddits)) & 
                         (all_data['TARGET_SUBREDDIT'].isin(top_target_subreddits))]

subset_data = filtered_data.sample(frac=1, random_state=42)  #### Change the fraction to change subset size ####

# Extract 'POST_PROPERTIES' as a list feature
post_properties = subset_data['PROPERTIES'].apply(eval)  # Assuming 'POST_PROPERTIES' is a string representation of a list

# Convert list feature to DataFrame with each value in the list as a separate column
post_properties_df = pd.DataFrame(post_properties.tolist(), columns=[f"feature_{i}" for i in range(86)])

# Reset index of post_properties_df
post_properties_df.reset_index(drop=True, inplace=True)
subset_data.reset_index(drop=True, inplace=True)

# Append post_properties_df to all_data
subset_data = pd.concat([subset_data, post_properties_df], axis=1)

# One-hot encode categorical features
subset_data_encoded = pd.get_dummies(subset_data, columns=['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT'])
# subset_data_encoded = pd.get_dummies(subset_data, columns=['SOURCE_SUBREDDIT'])

# Remove common prefixes from column names
subset_data_encoded.columns = subset_data_encoded.columns.str.replace('SOURCE_SUBREDDIT_', '').str.replace('TARGET_SUBREDDIT_', '')

# Merge columns with the same names
subset_data_encoded = subset_data_encoded.groupby(level=0, axis=1).sum()

# Display encoded data
# print(subset_data_encoded.drop(columns=['LINK_SENTIMENT','PROPERTIES','POST_ID']).head())

# create_lm(subset_data_encoded)
create_rand_forest(subset_data_encoded)