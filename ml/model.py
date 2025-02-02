import json
import xgboost as xgb  # New import for XGBoost
from trainer import df_train as train, df_test as test
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

columns_to_drop = ['Outcome', 'Risk_Level']
X_train = train.drop(columns=columns_to_drop)
y_train = train['Outcome']
X_test = test.drop(columns=columns_to_drop)
y_test = test['Outcome']

# Define the models and their parameter grids
models = {
    'GradientBoosting': (GradientBoostingClassifier(), {
        'n_estimators': [100, 150],
        'learning_rate': [0.1, 0.2],
        'max_depth': [3, 5],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'subsample': [0.8, 0.9],
    }),
    'LogisticRegression': (LogisticRegression(max_iter=200), {
        'C': [0.01, 0.1, 1, 10],
        'solver': ['liblinear', 'saga'],
    }),
    'RandomForest': (RandomForestClassifier(), {
        'n_estimators': [100, 150],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
    }),
    'SVC': (SVC(), {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
    }),
    'XGBoost': (xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'), {  # New XGBoost model
        'n_estimators': [100, 150],
        'max_depth': [3, 5],
        'learning_rate': [0.1, 0.2],
    }),
}

results = {}  # Dictionary to store results

# Iterate through models and perform grid search
for model_name, (model, param_grid) in models.items():
    print(f"Training {model_name}...")
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    # Display the best parameters found
    print(f"Best Parameters for {model_name}:", grid_search.best_params_)
    
    # Access the best model from grid search
    best_model = grid_search.best_estimator_
    
    # Evaluate the best model
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy of the best model ({model_name}): {accuracy:.4f}")
    
    # Classification Report
    report = classification_report(y_test, y_pred, output_dict=True)
    results[model_name] = {
        'accuracy': accuracy,
        'classification_report': report
    }
    
    # Confusion Matrix
    print(f"\nConfusion Matrix for {model_name}:\n", confusion_matrix(y_test, y_pred))

# Save results to JSON file
with open('model_results.json', 'w') as json_file:
    json.dump(results, json_file, indent=4)
    
# By the results we can see GradientBoost is perfroming the best.
# and the Best Parameters for GradientBoosting: {'learning_rate': 0.2, 'max_depth': 5, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 100, 'subsample': 0.9}