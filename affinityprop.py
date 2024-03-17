# All imports
import pandas as pd
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Loading the dataset
data = pd.read_csv("archive/breed_traits.csv")

# Defining columns
numerical_cols = ['Affectionate With Family', 'Good With Young Children', 'Good With Other Dogs',
                  'Shedding Level', 'Coat Grooming Frequency', 'Drooling Level', 'Coat Type', 'Coat Length',
                  'Openness To Strangers', 'Playfulness Level', 'Watchdog/Protective Nature',
                  'Adaptability Level', 'Trainability Level', 'Energy Level', 'Barking Level',
                  'Mental Stimulation Needs']

# Create a preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols)])

# Define the pipeline with preprocessor and Affinity Propagation model
# Tested with different random_state values, all had the same silhouette score so 42 is kept
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('affinity_propagation', AffinityPropagation(random_state=42))])

# Fit the pipeline to the data
pipeline.fit(data)

# Predict the cluster labels
labels = pipeline.named_steps['affinity_propagation'].labels_

# Adding the cluster labels to the DataFrame
data['Cluster'] = labels

# Printing the counts of each cluster
print(data['Cluster'].value_counts())

# Calculating the silhouette score
silhouette_avg = silhouette_score(data[numerical_cols], labels)
print("Silhouette Score:", silhouette_avg)
