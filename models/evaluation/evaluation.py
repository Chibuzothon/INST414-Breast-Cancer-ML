import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss, precision_score, recall_score, f1_score, roc_curve, auc
import shap
from sklearn.metrics import ConfusionMatrixDisplay
import pandas as pd

from breast_cancer_ml import config
Path = "C:/Users/Maryl/breast_cancer_classifier/breast_cancer_ml"
PROJ_ROOT = Path(os.getcwd())
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJ_ROOT / "models"
PREDICT_MODELS_DIR = MODELS_DIR / "predict"
DATA_DIR = PROJ_ROOT / "data"
SPLITS_DATA_DIR = DATA_DIR / "splits"

# Define path for the evaluation report
evaluation_report_path = REPORTS_DIR / "model_evaluation.txt"

# Open a file to save the outputs
report_file = open(evaluation_report_path, "w")

# Helper function to both print and save
def print_and_save(text):
    print(text)
    report_file.write(text + "\n")


# def get_scaled_data(model_pipeline, X):
#     """
#     Extracts the scaler from the pipeline and transforms X.
#     """
#     scaler = model_pipeline.named_steps['scaler']
#     X_scaled = scaler.transform(X)
#     return X_scaled

# Function to extract scaled data from the pipeline for any model
def get_scaled_data_from_pipeline(model_pipeline, X):
    """
    Extracts the scaler from the pipeline (if it exists) and transforms X.
    Handles both Random Forest and XGBoost models.
    """
    if 'scaler' in model_pipeline.named_steps:
        # If scaler exists in the pipeline, apply it
        scaler = model_pipeline.named_steps['scaler']
        X_scaled = scaler.transform(X)
    else:
        # If there's no scaler in the pipeline, return the data as is (e.g., for XGBoost)
        X_scaled = X
    return X_scaled



# Load the Random Forest and XGBoost models from pickle files
rf_model_path = PREDICT_MODELS_DIR / 'random_forest_model.pkl'  # Specify the path to your RF pickle file
xgb_model_path = PREDICT_MODELS_DIR / 'xgboost_model.pkl'  # Specify the path to your XGBoost pickle file


with open(rf_model_path, 'rb') as rf_file:
    final_rf_pipeline = pickle.load(rf_file)

with open(xgb_model_path, 'rb') as xgb_file:
    final_xgb_pipeline = pickle.load(xgb_file)

# After loading the models, you can now proceed with predictions or evaluations as needed
print("Random Forest and XGBoost models have been successfully loaded!")


with open(SPLITS_DATA_DIR / "X_train.pkl", 'rb') as f:
    X_train = pickle.load(f)

with open(SPLITS_DATA_DIR / "X_test.pkl", 'rb') as f:
    X_test = pickle.load(f)

with open(SPLITS_DATA_DIR / "y_train.pkl", 'rb') as f:
    y_train = pickle.load(f)

with open(SPLITS_DATA_DIR / "y_test.pkl", 'rb') as f:
    y_test = pickle.load(f)

# Apply scaling to training data using the Random Forest pipeline
X_train_scaled_rf = get_scaled_data_from_pipeline(final_rf_pipeline, X_train)

# Apply scaling to training data using the XGBoost pipeline (if needed)
X_train_scaled_xgb = get_scaled_data_from_pipeline(final_xgb_pipeline, X_train)

# X_train_scaled = get_scaled_data(rf_model, X_train)

# Check the data shapes to ensure they loaded correctly
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")



# Ensure the figures directory exists
figures_dir = "reports/figures"
# os.makedirs(figures_dir, exist_ok=True)

# Make predictions
y_pred_rf = final_rf_pipeline.predict(X_test)
y_pred_xgb = final_xgb_pipeline.predict(X_test)

y_pred_rf_proba = final_rf_pipeline.predict_proba(X_test)[:, 1]  # Get probs for class 1
y_pred_xgb_proba = final_xgb_pipeline.predict_proba(X_test)[:, 1]  # Get probs for class 1
threshold = 0.5
y_pred_rf_binary = (y_pred_rf_proba >= threshold).astype(int)
y_pred_xgb_binary = (y_pred_xgb_proba >= threshold).astype(int)

# Print Accuracy and Classification Report
print_and_save("\nRandom Forest Test Results:")
print_and_save(f"Accuracy: {accuracy_score(y_test, y_pred_rf_binary):.4f}")
print_and_save(classification_report(y_test, y_pred_rf_binary))

print_and_save("\nXGBoost Test Results:")
print_and_save(f"Accuracy: {accuracy_score(y_test, y_pred_xgb_binary):.4f}")
print_and_save(classification_report(y_test, y_pred_xgb_binary))

# Print Log Loss
rf_loss = log_loss(y_test, y_pred_rf_binary)
xgb_loss = log_loss(y_test, y_pred_xgb_binary)

print_and_save(f"Random Forest Log Loss: {rf_loss:.4f}")
print_and_save(f"XGBoost Log Loss: {xgb_loss:.4f}")

# Confusion Matrices
print_and_save("\nRandom Forest Confusion Matrix:")
print_and_save(confusion_matrix(y_test, y_pred_rf_binary))

print_and_save("\nXGBoost Confusion Matrix:")
print_and_save(confusion_matrix(y_test, y_pred_xgb_binary))

report_file.close()
# Save Confusion Matrices as Images
cm_rf = confusion_matrix(y_test, y_pred_rf_binary)
cm_xgb = confusion_matrix(y_test, y_pred_xgb_binary)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf)
disp_rf.plot(ax=ax1, cmap='Blues', values_format='d')
ax1.set_title('Random Forest Confusion Matrix')

disp_xgb = ConfusionMatrixDisplay(confusion_matrix=cm_xgb)
disp_xgb.plot(ax=ax2, cmap='Blues', values_format='d')
ax2.set_title('XGBoost Confusion Matrix')

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "confusion_matrices.png"))
plt.close()

# Precision, Recall, and F1 Score
rf_precision = precision_score(y_test, y_pred_rf_binary)
rf_recall = recall_score(y_test, y_pred_rf_binary)
rf_f1 = f1_score(y_test, y_pred_rf_binary)

xgb_precision = precision_score(y_test, y_pred_xgb_binary)
xgb_recall = recall_score(y_test, y_pred_xgb_binary)
xgb_f1 = f1_score(y_test, y_pred_xgb_binary)

# Save Performance Comparison Bar Plot
fig, ax = plt.subplots(figsize=(10, 6))
barWidth = 0.25
positions1 = np.arange(3)
positions2 = [x + barWidth for x in positions1]

rf_bars = ax.bar(positions1, [rf_precision, rf_recall, rf_f1], barWidth, color='orange', label='Random Forest')
xgb_bars = ax.bar(positions2, [xgb_precision, xgb_recall, xgb_f1], barWidth, color='green', label='XGBoost')

# Add labels to the bars
for bars in [rf_bars, xgb_bars]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f'{height:.3f}', ha='center', va='bottom')

ax.set_ylim(0, 1.05)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Performance Comparison', fontsize=14)
ax.set_xticks([r + barWidth / 2 for r in range(3)])
ax.set_xticklabels(['Precision', 'Recall', 'F1 Score'], fontsize=11)
ax.legend(fontsize=11)
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "performance_comparison.png"))
plt.close()

# ROC Curve
rf_fpr, rf_tpr, _ = roc_curve(y_test, y_pred_rf_binary)
xgb_fpr, xgb_tpr, _ = roc_curve(y_test, y_pred_xgb_binary)

rf_auc = auc(rf_fpr, rf_tpr)
xgb_auc = auc(xgb_fpr, xgb_tpr)

plt.figure(figsize=(10, 8))
plt.plot(rf_fpr, rf_tpr, color='orange', lw=2, label=f'Random Forest (AUC = {rf_auc:.3f})')
plt.plot(xgb_fpr, xgb_tpr, color='green', lw=2, label=f'XGBoost (AUC = {xgb_auc:.3f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "roc_curve_comparison.png"))
plt.close()

# SHAP Summary Plots
X_train_scaled_xgb = final_xgb_pipeline.named_steps['preprocessor'].transform(X_train)
if hasattr(X_train_scaled_xgb, "toarray"):
    X_train_scaled_xgb = X_train_scaled_xgb.toarray()

xgb_model = final_xgb_pipeline.named_steps['classifier']
explainer_xgb = shap.TreeExplainer(xgb_model)
shap_values_xgb = explainer_xgb.shap_values(X_train_scaled_xgb)

plt.figure(figsize=(10, 8))
plt.title("XGBoost - SHAP Summary Plot")
shap.summary_plot(shap_values_xgb, X_train_scaled_xgb, feature_names=X_train.columns.tolist())
plt.tight_layout()
plt.savefig(FIGURES_DIR / "xgb_shap_summary_plot.png")
plt.close()

# Random Forest SHAP Summary Plot
X_train_scaled_rf = final_rf_pipeline.named_steps['preprocessor'].transform(X_train)
if hasattr(X_train_scaled_rf, "toarray"):
    X_train_scaled_rf = X_train_scaled_rf.toarray()

rf_model = final_rf_pipeline.named_steps['classifier']
explainer_rf = shap.TreeExplainer(rf_model)
shap_values_rf = explainer_rf.shap_values(X_train_scaled_rf)

plt.figure(figsize=(20, 20))
shap.summary_plot(shap_values_rf, X_train_scaled_rf, feature_names=X_train.columns.tolist())
plt.title("Random Forest - SHAP Summary Plot")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "rf_shap_summary_plot.png")
plt.close()