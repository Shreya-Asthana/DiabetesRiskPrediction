from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from joblib import dump, load
import pandas as pd
import json
from sklearn.preprocessing import MinMaxScaler
import joblib
import numpy as np

# Adjust the Python path to include the parent directory
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ml.trainer import df_train as train, df_test as test

# Ensure the Outcome column is dropped
columns_to_drop = ['Outcome', 'Risk_Level']
X_train = train.drop(columns=columns_to_drop)
y_train = train['Outcome']
X_test = test.drop(columns=columns_to_drop)
y_test = test['Outcome']

# Best Parameters for GradientBoosting
params = {
    'learning_rate': 0.2,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'min_samples_split': 5,
    'n_estimators': 100,
    'subsample': 0.9
}

# Instantiate the Gradient Boosting Classifier
model = GradientBoostingClassifier(**params)

# Fit the model on the training data
model.fit(X_train, y_train)

# Save the model
dump(model, 'ml/best_model.joblib')

# Load the training data for scaler
train_df = pd.read_csv("data/train_smote.csv")
X_train = train_df.drop(columns=columns_to_drop)

# Create and save the scaler
scaler = MinMaxScaler()
scaler.fit(X_train)
joblib.dump(scaler, 'ml/scaler.joblib')

# Save model parameters to JSON
model_params = model.get_params()
with open('ml/best_model_params.json', 'w') as json_file:
    json.dump(model_params, json_file)




""" # Make predictions on the test data
predictions = model.predict(X_test)

# Print predictions and true labels for the test set
for i in range(len(X_test)):
    print(f'True: {y_test.iloc[i]}, Predicted: {predictions[i]}')

# Evaluate the model's performance
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy:.2f}') """
