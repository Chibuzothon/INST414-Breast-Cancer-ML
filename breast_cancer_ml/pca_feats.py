import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path
import sys
from pathlib import Path
import pandas as pd
from breast_cancer_ml.config import REPORTS_DIR


def apply_pca_to_feature_groups(data, n_components=2, plot=True, save=True, filename="pca_features.csv"):
    """
    Applies PCA to mean, worst, and SE feature groups in the dataset,
    and optionally saves the resulting DataFrame.

    Parameters:
        data (pd.DataFrame): Original breast cancer dataset.
        n_components (int): Number of PCA components per group.
        plot (bool): Whether to plot PCA scatter plots.
        save (bool): Whether to save the resulting DataFrame to CSV.
        filename (str): Output CSV file name (default: "pca_features.csv").

    Returns:
        pd.DataFrame: The PCA-enhanced DataFrame.
    """
    data_pca = data.copy()

    feature_groups = {
        "Means": ['diagnosis','radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
                  'smoothness_mean', 'compactness_mean', 'concavity_mean',
                  'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean'],

        "Worse": ['diagnosis','radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
                  'smoothness_worst', 'compactness_worst', 'concavity_worst',
                  'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'],

        "Standard Error (SE)": ['diagnosis','radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
               'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
               'fractal_dimension_se']
    }

    for key, features in feature_groups.items():
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(data[features])
        pca_cols = [f"PC{i+1}_{key}" for i in range(n_components)]
        data_pca[pca_cols] = X_pca

        if plot:
            plt.figure(figsize=(10, 8))
            sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=data['diagnosis'])
            plt.title(f'PCA of {key} Features')
            plt.xlabel('First Principal Component')
            plt.ylabel('Second Principal Component')
            # plt.show()

        # if save:
        #     config.REPORTS_DIR
        #     fig_dir = REPORTS_DIR / "figures"
        #     fig_dir.mkdir(parents=True, exist_ok=True)
        #     fig_path = fig_dir / "pca_projection.png"
        #     plt.savefig(fig_path)
        #     print(f"PCA plot saved to {fig_path}")

    # Save the file if requested
        if save:
            fig_dir = REPORTS_DIR / "figures"
            fig_dir.mkdir(parents=True, exist_ok=True)
            fig_path = fig_dir / f"pca_projection_{key}.png"  # Unique file per group
            plt.savefig(fig_path, dpi = 750)
            print(f"PCA plot saved to {fig_path}")
            plt.show()

        
        # if save:
        # interim_dir = PROJ_ROOT / "data" / "interim"
        # interim_dir.mkdir(parents=True, exist_ok=True)  # Make sure the directory exists
        # full_path = interim_dir / filename
        # data_pca.to_csv(full_path, index=False)
        # print(f"PCA-enhanced dataset saved to: {full_path.resolve()}")

    return data_pca


sys.path.append(str(Path.cwd().parents[0]))
from breast_cancer_ml import config
data = pd.read_csv(config.INTERIM_DATA_DIR / "dataset_numerical_target.csv")


apply_pca_to_feature_groups(data, n_components=2, plot=True, save=True, filename="pca_features.csv")


