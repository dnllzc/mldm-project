import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Loadin and preprocessing the dataset
data = pd.read_csv("archive/breed_traits.csv")

# Separating features (X) and target variable (y)
x = data.drop('Breed', axis=1)  # Features
y = data['Breed']  # Target variable

# Splitting the dataset into training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Training the model
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(x_train.values, y_train.values)

# Model evaluation
y_pred = rf_classifier.predict(x_test.values)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Making predictions
new_data = [[5,5,4,3,2,2,1,1,4,4,4,4,5,5,3,5]]
predicted_breed = rf_classifier.predict(new_data)
print("Predicted Breed:", predicted_breed)
