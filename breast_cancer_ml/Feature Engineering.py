

# from breast_cancer_ml.config import PROCESSED_DATA_DIR

# app = typer.Typer()


# @app.command()
# def main(
#     # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
#     input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
#     output_path: Path = PROCESSED_DATA_DIR / "features.csv",
#     # -----------------------------------------
# ):
#     from sklearn.preprocessing import StandardScaler

#     # def preprocess_features(X_train, X_test):
#     #     scaler = StandardScaler()
#     #     X_train_scaled = scaler.fit_transform(X_train)  # Fit to training data and transform it
#     #     X_test_scaled = scaler.transform(X_test)
#     #     return X_train_scaled, X_test_scaled, scaler
    
    

    
    
#     logger.info("Generating features from dataset...")
#     for i in tqdm(range(10), total=10):
#         if i == 5:
#             logger.info("Something happened for iteration 5.")
#     logger.success("Features generation complete.")
#     # -----------------------------------------


# if __name__ == "__main__":
#     app()


from breast_cancer_ml.config import PROCESSED_DATA_DIR

# app = typer.Typer()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
from breast_cancer_ml import config  

def load_data():
    # data = pd.read_csv(config.INTERIM_DATA_DIR / "dataset_numerical_target.csv")

    data_path = config.INTERIM_DATA_DIR / "dataset_numerical_target.csv"  # adjust filename
    data = pd.read_csv(data_path)
    return data

def run_random_forest_feature_importance(X, y, feature_names, save_fig=False, fig_name="feature_importance.png", title = None, save_df=False, df_name="feature_importance.csv"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    forest = RandomForestClassifier(random_state=0)
    forest.fit(X_train, y_train)

    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(range(X_train.shape[1]), importances[indices])
    plt.xticks(range(X_train.shape[1]), [feature_names[i] for i in indices], rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importances')
    plt.tight_layout()

    if title:
        plt.title(title)
    else:
        plt.title('Feature Importances')

    plt.tight_layout()

    if save_fig:
        fig_dir = config.REPORTS_DIR / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig_path = fig_dir / fig_name
        plt.savefig(fig_path, dpi=750)
        print(f"Feature importance plot saved to {fig_path}")
    
    plt.show()

    feature_importance_df = pd.DataFrame({
        'Feature': [feature_names[i] for i in indices],
        'Importance': importances[indices]
    })
    
    if save_df:
        df_dir = config.REPORTS_DIR / "figures"
        df_dir.mkdir(parents=True, exist_ok=True)
        df_path = df_dir / df_name
        feature_importance_df.to_csv(df_path, index=False)
        print(f"Feature importance DataFrame saved to {df_path}")

    return feature_importance_df

def main():
    data = load_data()
    
    # Original features
    X = data.drop("diagnosis", axis=1)
    y = data["diagnosis"]
    feature_names = X.columns.tolist()
    
    orig_feat_imp_df = run_random_forest_feature_importance(X, y, feature_names, save_fig=True, fig_name="original_feature_importance.png", save_df=True, df_name="original_feature_importance.csv")
    orig_feat_imp_df
    print("Top 10 most important original features:")
    print(orig_feat_imp_df.head(10))

        # Save Top 10 original features to text
    with open(config.REPORTS_DIR / "top10_original_features.txt", "w") as f:
        f.write("Top 10 most important original features:\n")
        f.write(orig_feat_imp_df.head(10).to_string(index=False))

    # Load PCA dataset
    data_pca_path = config.INTERIM_DATA_DIR /"pca_features.csv"  # or whatever your PCA filename is
    data_pca = pd.read_csv(data_pca_path)

    y_pca = data_pca.pop("diagnosis")
    X_pca = data_pca
    pca_feature_names = X_pca.columns.tolist()

    pca_feat_imp_df = run_random_forest_feature_importance(X_pca, y_pca, pca_feature_names, save_fig=True, fig_name="pca_feature_importance.png", title="Feature Importances (Dataset with PCA)", save_df=True, df_name="pca_feature_importance.csv")
    print("Top 10 most important features with PCA:")
    print(pca_feat_imp_df.head(10))

    # Step 1: Set a threshold   
    feat_imp_threshold = 0.01

    # Step 2: Get important features above the threshold
    imp_feats = pca_feat_imp_df[pca_feat_imp_df["Importance"] > feat_imp_threshold]["Feature"].tolist()

    # Step 3: See how many features passed the threshold
    imp_feats_count = len(imp_feats)
    print(f"Number of important PCA features (Feature Importance > {feat_imp_threshold}): {imp_feats_count}")

    print(imp_feats)

# Save the important feature names into the reports folder
    imp_feats_path = config.REPORTS_DIR / "important_pca_features.txt"
    with open(imp_feats_path, "w") as f:
        for feature in imp_feats:
            f.write(feature + "\n")
    print(f"Important PCA features list saved to {imp_feats_path}")
    # Step 4: Create a new DataFrame with those features
    df = data_pca[imp_feats].copy()  # add .copy() to avoid warning

    # Step 5: Add diagnosis column back
    df["diagnosis"] = y_pca

    # Step 6: Save the DataFrame into the PROCESSED folder
    processed_path = config.PROCESSED_DATA_DIR / "df_feat_eng.csv"
    df.to_csv(processed_path, index=False)

    print(f"Important PCA features dataset saved to {processed_path}")


    with open(config.REPORTS_DIR / "top10_pca_features.txt", "w") as f:
        f.write("Top 10 most important features with PCA:\n")
        f.write(pca_feat_imp_df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()




# @app.command()
# def main(
#     input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
#     output_path: Path = PROCESSED_DATA_DIR / f"features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
# ):
#     logger.info(f"Reading data from {input_path}")
#     data = pd.read_csv(input_path)

#     )

#     logger.success("Feature generation with PCA complete.")


# if __name__ == "__main__":
#     app()