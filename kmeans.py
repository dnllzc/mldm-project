# All imports
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import silhouette_score

# Loading the dataset
data = pd.read_csv("archive/breed_traits.csv")

# Definining columns
numerical_cols = ['Affectionate With Family', 'Good With Young Children', 'Good With Other Dogs',
                  'Shedding Level', 'Coat Grooming Frequency', 'Drooling Level',
                  'Openness To Strangers', 'Playfulness Level', 'Watchdog/Protective Nature',
                  'Adaptability Level', 'Trainability Level', 'Energy Level', 'Barking Level',
                  'Mental Stimulation Needs', 'Coat Type', 'Coat Length']

# Create a preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols)])

# Define the pipeline with preprocessor and KMeans model
# Tested with different number of clusters, 2 seems to be the best based on silhouette score
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('kmeans', KMeans(n_clusters=2, random_state=42))])

# Fit the pipeline to the data
pipeline.fit(data)

# Predict the cluster labels
labels = pipeline.predict(data)

# Adding the cluster labels to the DataFrame
data['Cluster'] = labels

# Printing the counts of each cluster
print(data['Cluster'].value_counts())

# Calculating the silhouette score
silhouette_avg = silhouette_score(data[numerical_cols], labels)
print("Silhouette Score:", silhouette_avg)