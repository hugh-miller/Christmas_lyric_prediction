import csv
import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_predict
import matplotlib.pyplot as plt

############################################################################
### Data sources - we have used resources from the million song dataset
#http://millionsongdataset.com/
# this includes 
#  - word counts from the musicXmatch dataset, mxm_dataset.db,  http://millionsongdataset.com/musixmatch/
#  - song tags from the last.fm dataset, lastfm_tags.db,  http://millionsongdataset.com/lastfm/ 
#  - track names from the first additional dataset, unique_tracks.txt http://millionsongdataset.com/pages/getting-dataset/
# Main paper reference is
#Thierry Bertin-Mahieux, Daniel P.W. Ellis, Brian Whitman, and Paul Lamere. 
#The Million Song Dataset. In Proceedings of the 12th International Society
#for Music Information Retrieval Conference (ISMIR 2011), 2011.

############################################################################
### Import data
# Step 1: Load data from SQLite databases
cnx = sqlite3.connect('data/mxm_dataset.db')
predictor_data = pd.read_sql_query("SELECT * FROM lyrics", cnx)
cnx.commit()
cnx.close()


# Get all christmas tags 
def sanitize(tag):
    """
    sanitize a tag so it can be included or queried in the db
    """
    tag = tag.replace("'","''")
    return tag

tag = 'christmas'
print 'We get all tracks for the tag: %s' % tag
sql = "SELECT tids.tid FROM tid_tag, tids, tags WHERE tids.ROWID=tid_tag.tid AND tid_tag.tag=tags.ROWID AND tags.tag='%s'" % sanitize(tag)

cnx2 = sqlite3.connect('data/lastfm_tags.db')
res = cnx2.execute(sql)
christ_list1 = res.fetchall()

# Get a response vector
response_data  = pd.DataFrame(christ_list1, columns=['track_id'])
response_data['response'] = 1

cnx2.commit()
cnx2.close()

# Step 2: Transform the data from long to wide format
data_pivot = predictor_data.pivot(index="track_id", columns="word", values="count").fillna(0)

# Step 2: Merge predictor and response data on track_id
data_pivot2 = pd.merge(data_pivot, response_data, on="track_id", how="left")
data_pivot2["response"].fillna(0, inplace=True)


# Step 4: Sample down the majority class (0 response)
positive_data = data_pivot2[data_pivot2["response"] == 1]
negative_data = data_pivot2[data_pivot2["response"] == 0]
negative_data = resample(negative_data, n_samples=int(0.1 * len(negative_data)), random_state=42)

# Combine positive and downsampled negative data
balanced_data = pd.concat([positive_data, negative_data])
balanced_data.shape

# Step 5: Split the data into training and testing sets
X = balanced_data.drop("response", axis=1)
X = X.drop("track_id", axis=1)
y = balanced_data["response"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
eval_set = [(X_train, y_train), (X_test, y_test)]

# Step 6: Train an XGBoost model
#model = xgb.XGBClassifier()
model = xgb.XGBClassifier(n_estimators=30, objective='binary:logistic', eval_metric='logloss',use_label_encoder=False)
model.fit(X_train, y_train, eval_set= eval_set)

# Step 7: Make predictions and evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(report)


# plot learn curve
results = model.evals_result()
epochs = len(results['validation_0']['logloss'])
x_axis = range(0, epochs)
fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
ax.legend()
plt.ylabel('AUC')
plt.title('XGBoost AUC')
plt.show()


# Step 8: Cross-validation for predictions
y_pred_cv = cross_val_predict(model, X, y, cv=5, method='predict_proba')
y_pred_cv[:, 1]

# Find the 10 track_id values with the highest predicted values
top_10_track_ids = balanced_data['track_id'].values[np.argsort(-y_pred_cv[:, 1])[:10]]

low_10_track_ids = balanced_data['track_id'].values[np.argsort(y_pred_cv[:, 1])[:10]]

# Display the top 10 track_id values
print("Top 10 track_id values with the highest predicted values:")
print(top_10_track_ids)

# Import song names 

# Define the path to the 'unique_tracks.txt' file
unique_tracks_file = 'data/unique_tracks.txt'

# Create a dictionary to store artist and song information
track_info = {}

# Read data from the 'unique_tracks.txt' file and store it in the dictionary
with open(unique_tracks_file, 'r', encoding='utf-8') as file:
    for line in file:
        parts = line.strip().split('<SEP>')
        if len(parts) == 4:
            track_id, other_id, artist_name, song_name = parts
            track_info[track_id] = {'artist_name': artist_name, 'song_name': song_name}


# Define the path to the output CSV file
output_csv_file = 'top_10_track_info.csv'

# Create a list to store the data
output_data = []

# Find song names and artists for the top_10_track_ids and add them to the output_data
for track_id in top_10_track_ids:
    if track_id in track_info:
        artist_name = track_info[track_id]['artist_name']
        song_name = track_info[track_id]['song_name']
        probability = y_pred_cv[:, 1][balanced_data['track_id'] == track_id][0]  # Find the corresponding probability
        output_data.append([track_id, artist_name, song_name, probability])
    else:
        print(f"Track ID {track_id} not found in 'unique_tracks.txt'")

# Write the data to the output CSV file
with open(output_csv_file, 'w', newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile)
    # Write the header
    csv_writer.writerow(['track_id', 'artist_name', 'song_name', 'predicted_probability'])
    # Write the data
    csv_writer.writerows(output_data)

print(f"Data has been written to {output_csv_file}")




# Define the path to the output CSV file for variable importance
output_variable_importance_csv = 'top_10_variable_importance.csv'

# Get the feature importance from the trained XGBoost model
importance = model.feature_importances_
feature_names = X.columns

# Sort the features by importance in descending order
sorted_indices = np.argsort(importance)[::-1]

# Select the top 10 important features
top_10_features = [feature_names[i] for i in sorted_indices[:50]]
top_10_importance = [importance[i] for i in sorted_indices[:50]]

# Write the top 10 variable importance data to a CSV file
with open(output_variable_importance_csv, 'w', newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile)
    # Write the header
    csv_writer.writerow(['Variable Name', 'Importance Score'])
    # Write the data
    csv_writer.writerows(zip(top_10_features, top_10_importance))


### Histogram of predicted probabilities
# Create a DataFrame to store the predicted probabilities and responses
predictions_df = pd.DataFrame({
    'TrackID': balanced_data['track_id'],
    'PredictedProbability': y_pred_cv[:, 1],
    'Response': y
})

# Define the bin edges based on specific predicted probability values
bin_edges = [i / 100 for i in range(1, 101)]  # e.g., [0.01, 0.02, 0.03, ..., 1.0]

# Create bins using cut with specific bin edges
predictions_df['Bin'] = pd.cut(predictions_df['PredictedProbability'], bins=bin_edges, right=False)

# Calculate counts for each bin and response
bin_counts = predictions_df.groupby(['Bin', 'Response']).size().unstack(fill_value=0)

# Export the table to a CSV file
bin_counts.to_csv('histogram.csv', index=True)

print("Data has been exported to 'histogram.csv'")




######### Lowest probabilities of false negatives ####################
# Create a DataFrame to store the predicted probabilities, responses, artist names, and song names
predictions_df = pd.DataFrame({
    'TrackID': balanced_data['track_id'],
    'PredictedProbability': y_pred_cv[:, 1],
    'Response': y
})

# Merge the artist_name and song_name from the 'track_info' dictionary
predictions_df['ArtistName'] = [track_info.get(track_id, {}).get('artist_name', '') for track_id in predictions_df['TrackID']]
predictions_df['SongName'] = [track_info.get(track_id, {}).get('song_name', '') for track_id in predictions_df['TrackID']]

# Filter for positive responses and sort by predicted probability in ascending order
positive_responses = predictions_df[predictions_df['Response'] == 1]
sorted_positive_responses = positive_responses.sort_values(by='PredictedProbability')

# Get the 10 tracks with the lowest predicted probability and positive response
top_10_lowest_positive_probabilities = sorted_positive_responses.head(10)

# Export the data to a CSV file
top_10_lowest_positive_probabilities.to_csv('lowest_positive_predictions.csv', index=False)
