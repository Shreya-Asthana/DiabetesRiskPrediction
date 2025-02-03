# DiabetesRiskPrediction
The goal was to build a predictive model to assess the risk of diabetes in females based on various health metrics.

## 🔍 Project Overview:

### 🔹 Data Preparation:
Utilized datasets like diabetes.csv for health metric analysis.
Performed Exploratory Data Analysis (EDA) using histograms, scatter plots, and correlation heatmaps.
### 🔹 Modeling Techniques:
Implemented Gradient Boosting, Logistic Regression, Random Forest, SVC, and XGBoost.
Applied Grid Search Cross-Validation for hyperparameter tuning to improve performance.
### 🔹 Deployment:
Built a Simple Flask web application to serve the model.
Containerized the app using Docker for scalability and seamless deployment.
### 🔹 Prediction Process:
Designed a user-friendly form for inputting health metrics.
The model processes input data and predicts diabetes risk with high accuracy.
### 🔹 Results:
Gradient Boosting delivered the best performance with 89.73% accuracy.
Optimized hyperparameters:
Learning Rate: 0.2
Max Depth: 5
Min Samples Leaf: 2
Min Samples Split: 2
Estimators: 100
Subsample: 0.9
