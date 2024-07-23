import pandas as pd
from sklearn.cluster import KMeans
import joblib

# Load the CSV file
df = pd.read_csv('Train/clustering.csv')

# Select the feature columns
features = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']
X = df[features]

# Create and fit the K-means model
kmeans = KMeans(n_clusters=4, random_state=0)
kmeans.fit(X)

# Predict clusters for each data point
df['cluster'] = kmeans.predict(X)

# Save the results to a new CSV file
df.to_csv('Classifier/describing_vectors_clustered2.csv', index=False)

# Save the trained model
joblib.dump(kmeans, 'kmeans.pkl')
print('Model has been saved successfully.')

