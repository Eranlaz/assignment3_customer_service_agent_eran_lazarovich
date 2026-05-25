import pandas as pd

from src.config import DATA_PATH


def load_dataset() -> pd.DataFrame:
    """
    Load the Bitext customer service dataset from disk.

    Returns:
        A pandas DataFrame containing the customer service dataset.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}. "
            "Place the CSV file in the data/ folder and name it bitext_customer_service.csv."
        )

    return pd.read_csv(DATA_PATH)
