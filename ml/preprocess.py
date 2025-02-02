import pandas as pd
import os
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from collections import Counter

def fill_mean(df, column, target):
    # Calculate mean values for each target group, excluding zeros
    mean_values = df[df[column] != 0].groupby(target)[column].mean().round().to_dict()

    # Apply function with if-else condition
    def replace_zero(row):
        if row[column] == 0:  # If the value is zero
            return mean_values.get(row[target], row[column])  # Replace with group mean
        else:
            return row[column]  # Keep original value

    df[column] = df.apply(replace_zero, axis=1)
    
    return df

def get_count(df,column):
    """Returns count of distinct values with label in that column to check if data is biased"""
    return df[column].value_counts()
    
def apply_smote(df, target_column, test_size=0.2, random_state=42):
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Print class distribution before SMOTE
    print("Before SMOTE:", Counter(y_train))

    # Apply SMOTE
    smote = SMOTE(sampling_strategy="auto", random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Print class distribution after SMOTE
    print("After SMOTE:", Counter(y_train_resampled))

    return X_train_resampled, X_test, y_train_resampled, y_test

def create_scaler(X_train):
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler

# adding a risk level column based on the target column's stats
def risk(df, target):
    # Compute mean and standard deviation of the target column
    mean_value = df[target].mean()
    std_dev = df[target].std()

    # Define a function to assign risk levels
    def assign_risk(value):
        if value >= mean_value + std_dev:
            return "High"
        elif value >= mean_value:
            return "Medium"
        else:
            return "Low"

    # Apply function to create a new 'Risk_Level' column
    df["Risk_Level"] = df[target].apply(assign_risk)
    
    return df

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ml.commons import read_csv

# use data/diabetes.csv
df = read_csv("data", "diabetes.csv")
print(df.head())

# Handling Missing Values
filled_insulin=fill_mean(df,"Insulin","Outcome")
filled_glucose=fill_mean(filled_insulin,"Glucose","Outcome")
filled_BMI=fill_mean(filled_glucose,"BMI","Outcome")
filled_Bp=fill_mean(filled_BMI,"BloodPressure","Outcome")

df_filled=fill_mean(filled_Bp,"SkinThickness","Outcome")
print(df_filled.head())

#biasness
print(get_count(df_filled,"Outcome")) 
# 0    500, 1    268 --> data is biased towards non-diabetic patients

# Applying Smote for Handling Biasness
target_column = "Outcome" 
X_train, X_test, y_train, y_test = apply_smote(df_filled, target_column)

# Combine X and y into one DataFrame
train_df = pd.DataFrame(X_train)
risk(train_df,"Glucose")
train_df[target_column] = y_train  # Add target column

test_df = pd.DataFrame(X_test)
risk(test_df,"Glucose")
test_df[target_column] = y_test  # Add target column

# Save the new datasets
train_df.to_csv("data/train_smote.csv", index=False)
test_df.to_csv("data/test.csv", index=False)
