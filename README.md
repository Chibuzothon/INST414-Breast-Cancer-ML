# Breast_Cancer_Classifier
## Description
This is a classification machine leanring model that determines if a breast cancer tumor is malignant or benign. This model is trained on a dataset that is in a CSV format. The dataset contains information on the tumor. This project aims to use use Machine Leanring as a tool that Physicians can use to validate breast cancer screening results. Although mammograms have a high accuracy in detecting breast cancer, it has rates of type 1 and type 2 errors. These errors can make breast cancer screening results inaccurate. In the case of type 1 errors, a patient could have a tumor that gets diagnosed as cancer when no cancer is present. This can result in patients having unnecessary follow-up up visits and having to spend money on treatment. Type 2 errors occur when the patient has breast cancer but the mammogram screening misses it. This can result in the breast cancer getting worse and the patient not getting immediate treatment. Using the ML model as a tool to validate breast cancer screening results can help reduce the occurrences of type 1 and type 2 errors. Overall it can create a more accurate and reliable approach to diagnosing and treating breast cancer.


## Dependencies
### Programming Languages
- Python 
### ML Frameworks
- Scikit-Learn
- XGboost

### Libraries
- import os  
- import pandas as pd  
- import numpy as np  
- import matplotlib.pyplot as plt  
- import seaborn as sns  
- from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  
- from sklearn.metrics import precision_score, recall_score, f1_score  
- from sklearn.metrics import roc_curve, auc  
- from sklearn.model_selection import KFold, GridSearchCV, train_test_split, cross_val_score  
- from sklearn.preprocessing import StandardScaler  
- import shap  

## Intructions

### Setting up the environment 
- Use pip to install all the dependencies
      - For example pip install pandas as pd
- import the libraires 

### Running the data processing pipeline
1. Load the dataset
2. Use standard Scaler to standardize the values
3. Use mapping to set target feature values of benign to 0 and malignant to 1
4. Use train_test_split to split the data into training and testing sets 

### Training the Models
1. Train the model using the Scikit-Learn's classifiers
   - Random Forrest
   - XGBoost
2. Perform hyperparameter tuning with GridSerachCV
3. use cross_val_score and Kfold for model validation

### Evaluating the models
Use evaluation metrics such as the accuracy, precision, recall, F1-score, and ROC- AUC to assess the model performance
1. Compute performance metrics
- Accuracy, Precision, Recall, and F1-score
- Generate confusion matricies
- Plot ROC curse and calculate AUC scores
3. Use SHAP for model explainability analysis 


### Reproducing results

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Machine Learning Classification model that is able to detect if a breast cancer tumor is malignant or benign

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         breast_cancer_ml and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── breast_cancer_ml   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes breast_cancer_ml a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

