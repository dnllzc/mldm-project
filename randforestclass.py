import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Loadin and preprocessing the dataset
data = pd.read_csv("archive/breed_traits.csv")
