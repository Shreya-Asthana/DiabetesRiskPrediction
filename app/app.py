from flask import Flask, render_template, request, flash
import joblib
import pandas as pd
import sys
import os
import logging

# Adjust the Python path to include the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ml.trainer import df_train as train, df_test as test

app = Flask(__name__)
app.secret_key = '4f3c2b1a0e9d8c7b6a5e4f3c2b1a0e9d'  # Required for flashing messages

# Load the model and scaler once at startup
model = joblib.load('ml/best_model.joblib')
scaler = joblib.load('ml/scaler.joblib')

def predict(input_data):
    # Define the feature names
    feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    
    # Convert input data to DataFrame with feature names
    input_df = pd.DataFrame([input_data], columns=feature_names)
    
    # Convert all columns to float
    input_df = input_df.astype(float)
    
    # Scale the input data
    scaled_data = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(scaled_data)
    
    # Return "Diabetic" or "Non-Diabetic" based on prediction
    return "Diabetic" if prediction[0] == 1 else "Non-Diabetic"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    try:
        # Extract input values safely
        form_data = request.form

        required_keys = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        
        # Ensure all required keys are present
        if not all(key in form_data and form_data[key] for key in required_keys):
            flash("All input fields are required.")
            return render_template('index.html')

        # Convert input data to float
        data = {key: float(form_data[key]) for key in required_keys}

        # Make prediction
        prediction = predict(data)
        
        return render_template('index.html', prediction=prediction,data=data)

    except ValueError as e:
        flash(f"Invalid input: {e}")
    except Exception as e:
        flash(f"Unexpected error: {e}")
    
    return render_template('index.html')

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    app.run(debug=True)
