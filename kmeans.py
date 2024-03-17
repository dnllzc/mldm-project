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
                  'Shedding Level', 'Coat Grooming Frequency', 'Drooling Level', 'Coat Type', 'Coat Length',
                  'Openness To Strangers', 'Playfulness Level', 'Watchdog/Protective Nature',
                  'Adaptability Level', 'Trainability Level', 'Energy Level', 'Barking Level',
                  'Mental Stimulation Needs']

# Create a preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols)])


# Looking for the best number of clusters using silhouette score
# The best silhouette score and the number of clusters will be printed
# And later on be used to define the pipeline
best_score = -1
best_num_cl = 2

for num_cl in range(2, 190):
    # Define the pipeline with preprocessor and KMeans model
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('kmeans', KMeans(n_clusters=num_cl, random_state=42))])

    # Fit the pipeline to the data
    pipeline.fit(data)

    # Predict the cluster labels
    labels = pipeline.predict(data)

    # Adding the cluster labels to the DataFrame
    data['Cluster'] = labels

    # Calculating the silhouette score
    silhouette_avg = silhouette_score(data[numerical_cols], labels)
    print("Silhouette Score:", silhouette_avg, "Clusters:", num_cl)

    if silhouette_avg > best_score:
        best_score = silhouette_avg
        best_num_cl = num_cl


print("Best silhouette score:", best_score, "Best number of clusters:", best_num_cl)


# Define the pipeline with preprocessor and KMeans model
# Picking the best number of clusters from the previous step
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('kmeans', KMeans(n_clusters=best_num_cl, random_state=42))])

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