from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import sys
import os

def scale_selected_columns(df, columns_to_scale):
    df = df.copy()  # Avoid modifying the original DataFrame
    scaler = MinMaxScaler()
    
    # Scale only the selected columns
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
    
    return df

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ml.commons import read_csv

# Use absolute path for the data directory
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

df_train = read_csv(data_dir, 'train_smote.csv')
df_test = read_csv(data_dir, 'test.csv')
columns_to_scale = ['Pregnancies','Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
train_scaled = scale_selected_columns(df_train, columns_to_scale)
test_scaled = scale_selected_columns(df_test, columns_to_scale)
train_scaled.to_csv(os.path.join(data_dir, 'scaled_train.csv'), index=False)
test_scaled.to_csv(os.path.join(data_dir, 'scaled_test.csv'), index=False)
print(train_scaled)
