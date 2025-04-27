from pathlib import Path
import pandas as pd
from loguru import logger
from tqdm import tqdm
import typer
from pathlib import Path
import sys



sys.path.append(str(Path(__file__).resolve().parents[1]))

from breast_cancer_ml.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "breast-cancer.csv",
    # output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    data = pd.read_csv(input_path)
    
    X = data.drop("diagnosis", axis = 1)
    y = data["diagnosis"]

    from sklearn.model_selection import train_test_split
    X_orig_train, X_orig_test, y_orig_train, y_orig_test = train_test_split(X, y, stratify=y, random_state=42)


    logger.info("Processing dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Processing dataset complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
