import logging
import pickle
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from pathlib import Path
import json
from tqdm import tqdm
import typer
import matplotlib.pyplot as plt

from breast_cancer_ml.config import MODELS_DIR, PROCESSED_DATA_DIR, SPLITS_DATA_DIR, REPORTS_DIR

def load_best_params(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Load the data splits
with open('data/splits/X_train_20250427_011019.pkl', 'rb') as f:
    X_train = pickle.load(f)

with open('data/splits/X_test_20250427_011019.pkl', 'rb') as f:
    X_test = pickle.load(f)

with open('data/splits/y_train_20250427_011019.pkl', 'rb') as f:
    y_train = pickle.load(f)

with open('data/splits/y_test_20250427_011019.pkl', 'rb') as f:
    y_test = pickle.load(f)




app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    splits_path: Path = SPLITS_DATA_DIR / "data.splits.py",
    params_path: Path = REPORTS_DIR / "nestedcv_best_parameters.json" 

    # -----------------------------------------
    ):

    with open(params_path, 'r') as f:
        best_params = json.load(f)
    
    print(f"Best Parameters: {best_params}")


# Logger setup
logger = logging.getLogger(__name__)

def train_and_save_models(X_train, y_train, X_test, y_test, best_params, model_paths):
    def clean_params(params):
        clean = {}
        for k, v in params.items():
            if k.startswith('classifier__'):
                clean[k.replace('classifier__', '')] = v
            else:
                clean[k] = v
        return clean

    # Get the best parameters for each model
    rf_params = clean_params(best_params.get('RandomForest', {}))
    xgb_params = clean_params(best_params.get('XGBoost', {}))

    # Create a pipeline for RandomForest with StandardScaler
    rf_pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Apply scaling to the features
        ('classifier', RandomForestClassifier(**rf_params))  # Random Forest classifier
    ])
    
    # Create a pipeline for XGBoost with StandardScaler
    xgb_pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Apply scaling to the features
        ('classifier', XGBClassifier(**xgb_params))  # XGBoost classifier
    ])
    
    # Train the Random Forest model
    logger.info("Training Random Forest model...")
    rf_pipeline.fit(X_train, y_train)
    
    # Make predictions using the test data for Random Forest
    y_pred_rf = rf_pipeline.predict(X_test)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    print(f"Random Forest Model Accuracy: {accuracy_rf:.4f}")
    
    # Save the Random Forest model to the specified path
    with open(model_paths['rf'], 'wb') as f:
        pickle.dump(rf_pipeline, f)
    logger.info("Random Forest model training and evaluation complete.")

    # Train the XGBoost model
    logger.info("Training XGBoost model...")
    xgb_pipeline.fit(X_train, y_train)
    
    # Make predictions using the test data for XGBoost
    y_pred_xgb = xgb_pipeline.predict(X_test)
    accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
    print(f"XGBoost Model Accuracy: {accuracy_xgb:.4f}")
    
    # Save the XGBoost model to the specified path
    with open(model_paths['xgb'], 'wb') as f:
        pickle.dump(xgb_pipeline, f)
    logger.info("XGBoost model training and evaluation complete.")
    return accuracy_rf, accuracy_xgb

    
def plot_multiple_accuracies(accuracies, model_names, save_path):
    plt.figure(figsize=(8, 5))
    bars = plt.bar(model_names, accuracies, color=['orange', 'green'])
    plt.ylim(0, 1.05)
    plt.title("Model Test Accuracies")
    plt.ylabel("Accuracy")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add text labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()    
    
    
    
    # Load the best parameters from the JSON file


    # def train_random_forest(X_train, y_train, random_state=0):
    #     forest = RandomForestClassifier(random_state=random_state)
    #     forest.fit(X_train, y_train)
    #     return forest
    # logger.info("Training some model...")
    # for i in tqdm(range(10), total=10):
    #     if i == 5:
    #         logger.info("Something happened for iteration 5.")
    # logger.success("Modeling training complete.")
    # -----------------------------------------


if __name__ == "__main__":

# Paths where the models will be saved
    model_paths = {
        'rf': MODELS_DIR / "predict" / "random_forest_model.pkl",
        'xgb': MODELS_DIR / "predict" / "xgboost_model.pkl"
    }

    best_params = load_best_params('C:/Users/Maryl/breast_cancer_classifier/reports/nestedcv_best_parameters.json')  # path to your json file
    accuracy_rf, accuracy_xgb = train_and_save_models(X_train, y_train, X_test, y_test, best_params, model_paths)

    plot_multiple_accuracies(
        accuracies=[accuracy_rf, accuracy_xgb],
        model_names=["Random Forest", "XGBoost"],
        save_path="reports/figures/model_test_accuracies.png"
    )