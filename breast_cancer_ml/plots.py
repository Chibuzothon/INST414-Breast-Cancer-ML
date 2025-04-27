from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from breast_cancer_ml.config import FIGURES_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = FIGURES_DIR / "plot.png",
    # -----------------------------------------
):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    def plot_feature_importances(model, feature_names, X_train):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Plot the feature importances
        plt.figure(figsize=(10, 6))
        plt.bar(range(X_train.shape[1]), importances[indices])
        plt.xticks(range(X_train.shape[1]), [feature_names[i] for i in indices], rotation=90)
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.title('Feature Importances')
        plt.tight_layout()
        
        # Return the DataFrame of feature importances
        feat_imp_df = pd.DataFrame({
            'Feature': [feature_names[i] for i in indices],
            'Importance': importances[indices]
        })
        
        return feat_imp_df
    
    
    logger.info("Generating plot from data...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Plot generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
