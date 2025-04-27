
from breast_cancer_ml.config import PROCESSED_DATA_DIR
from breast_cancer_ml import config  
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV, train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import sys
sys.path.append('C:/Users/Maryl/breast_cancer_classifier/breast_cancer_ml/modeling')
from utilities_nested_cv import plot_nestedcv_model_accuracy, save_best_nestedcv_params
import os


# Assuming config and processed data path are correctly set
from breast_cancer_ml import config

# Load the processed data
processed_path = config.PROCESSED_DATA_DIR / "df_feat_eng.csv"
df = pd.read_csv(processed_path)
nestedcv_report_file_path = "reports/nestedcv_evaluation_report.txt"

sys.stdout = open(nestedcv_report_file_path, 'w')


# Prepare X and y
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Define preprocessing pipeline
preprocessor = Pipeline([
    ('scaler', StandardScaler())
])

# Random Forest model pipeline
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, max_depth=7, min_samples_split=6, 
                                         min_samples_leaf=55, random_state=42))
])

# XGBoost model pipeline
xgb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(learning_rate=0.3, n_estimators=150, min_samples_split=4, 
                                 min_samples_leaf=2, max_depth=5, random_state=42))
])

# Random Forest parameter grid for hyperparameter tuning
rf_param_grid = {
    'classifier__n_estimators': [50, 100, 150],
    'classifier__max_depth': [5, 7, 9],
    'classifier__min_samples_split': [2, 6, 10],
    'classifier__min_samples_leaf': [1, 5, 10]
}

# XGBoost parameter grid for hyperparameter tuning
xgb_param_grid = {
    'classifier__n_estimators': [50, 100, 150],
    'classifier__max_depth': [3, 5, 7],
    'classifier__learning_rate': [0.01, 0.1, 0.3],
    'classifier__min_samples_split': [2, 4, 6]
}

# Nested cross-validation function
def nested_cross_validation(pipeline, param_grid, X, y, outer_cv=5, inner_cv=3):
    """
    Perform nested cross-validation for model evaluation.
    """
    X = np.array(X)
    y = np.array(y)
    
    # Outer cross-validation split
    outer_cv_split = KFold(n_splits=outer_cv, shuffle=True, random_state=42)
    
    outer_scores = []  # To store the scores of each outer fold
    best_params_list = []  # To store best params of each fold
    all_models = []  # To store models of each fold

    # Outer loop
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv_split.split(X)):
        print(f"\nOuter Fold {fold_idx + 1}/{outer_cv}")
        
        # Split data for current fold
        X_train_fold, X_test_fold = X[train_idx], X[test_idx]
        y_train_fold, y_test_fold = y[train_idx], y[test_idx]
        
        # Grid search for hyperparameter tuning on training data
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=inner_cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=0,
            refit=True
        )
        
        grid_search.fit(X_train_fold, y_train_fold)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        all_models.append(best_model)
        best_params_list.append(best_params)
        
        # Evaluate model on test data
        y_pred = best_model.predict(X_test_fold)
        fold_score = accuracy_score(y_test_fold, y_pred)
        outer_scores.append(fold_score)
        
        print(f"  Best parameters: {best_params}")
        print(f"  Validation accuracy: {grid_search.best_score_:.4f}")
        print(f"  Test accuracy: {fold_score:.4f}")

    mean_accuracy = np.mean(outer_scores)
    std_accuracy = np.std(outer_scores)

    print(f"\nNested CV Results:")
    print(f"Mean Test Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Test Accuracy Range: [{min(outer_scores):.4f}, {max(outer_scores):.4f}]")
    
    return {
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'all_scores': outer_scores,
        'best_params': best_params_list,
        'models': all_models
    }

# Run nested CV for Random Forest model
print("\nEvaluating Random Forest with Nested Cross-Validation")
rf_results = nested_cross_validation(rf_pipeline, rf_param_grid, X_train, y_train)

# Run nested CV for XGBoost model
print("\nEvaluating XGBoost with Nested Cross-Validation")
xgb_results = nested_cross_validation(xgb_pipeline, xgb_param_grid, X_train, y_train)

# Compare model performances
print("\nModel Comparison:")
print(f"Random Forest: {rf_results['mean_accuracy']:.4f} ± {rf_results['std_accuracy']:.4f}")
print(f"XGBoost: {xgb_results['mean_accuracy']:.4f} ± {xgb_results['std_accuracy']:.4f}")

# plot_nestedcv_model_accuracy(rf_results, xgb_results)

# Define save paths
plot_save_path = "reports/nestedcv_model_comparison_plot.png"
params_save_path = "reports/nestedcv_best_parameters.txt"

# Define function to get most common parameters
def get_most_common_params(param_lists):
    # Count parameter occurrences
    param_counts = {}
    
    for params in param_lists:
        param_tuple = tuple(sorted(params.items()))
        if param_tuple not in param_counts:
            param_counts[param_tuple] = 0
        param_counts[param_tuple] += 1
    
    # Get most common parameter combination
    most_common_params = dict(max(param_counts.items(), key=lambda x: x[1])[0])
    return most_common_params

# Call the plotting function with all three required arguments
plot_nestedcv_model_accuracy(rf_results, xgb_results, plot_save_path)

# Get most common hyperparameters
rf_best_params = get_most_common_params(rf_results['best_params'])
xgb_best_params = get_most_common_params(xgb_results['best_params'])

# Now call the save function with all three arguments
save_best_nestedcv_params(rf_best_params, xgb_best_params, params_save_path)

# Print the parameters for the report
print("\nBest Random Forest parameters:", rf_best_params)
print("Best XGBoost parameters:", xgb_best_params)




# Get most common hyperparameters
# rf_best_params = save_best_nestedcv_params(rf_results['best_params'])
# xgb_best_params = save_best_nestedcv_params(xgb_results['best_params'])

# print("\nBest Random Forest parameters:", rf_best_params)
# print("Best XGBoost parameters:", xgb_best_params)


sys.stdout.close()


