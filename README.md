# Breast_Cancer_Classifier
## Description
This is a classification machine leanring model that determines if a breast cancer tumor is malignant or benign. This model is trained on a dataset that is in a CSV format. The dataset contains information on the tumor. This project aims to use use Machine Leanring as a tool that Physicians can use to validate breast cancer screening results. Although mammograms have an 87 percent accuracy in detecting breast cancer in women who have breast cancer (Komen) there are rates of false positive and false negative results, which can make screening results inaccurate. Fasle-positive results occur when a patient has breast cancer, but the mammogram doesn't detect it. 20 percent of all breast cancer that is present during screenings gets missed by mammograms (national Cancer Institute). A false positive result occurs when there is an abnormal tissue that gets diagnosed as cancer when no cancer is present. The likelihood of a woman having a false positive result increases as they have more mammograms. After one mammogram the percent of having false positives is 7-12.  Additionally, false positive results are common for young women, women with dense breast tissue, women who have had breast biopsies, and women who have breast cancer that runs in the family (National Cancer Institute). Consequently, type one errors, can result in a doctor scheduling more follow-up visits and running more tests. This can be expensive and time-consuming. On average it costs patients $527 for breast care after a type one error. (Chubak et al., 2010). The chance of having a false positive result is 50-60, after 10 annual mammograms (Komen). 

Using the ML model as a tool can help to create a more accurate and reliable approach to diagnosing and  treating breast cancer.
## Dependencies

## Intructions

### Setting up the environment 

### Running the data processing pipeline

### Evaluating the models



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

