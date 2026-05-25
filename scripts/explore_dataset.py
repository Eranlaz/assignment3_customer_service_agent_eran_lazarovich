from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/bitext_customer_service.csv")


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}. "
            "Put the CSV file in the data folder and rename it to bitext_customer_service.csv."
        )

    df = pd.read_csv(DATA_PATH)

    print("=" * 80)
    print("BASIC DATASET INFO")
    print("=" * 80)
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print()

    print("=" * 80)
    print("COLUMN NAMES")
    print("=" * 80)
    for column in df.columns:
        print("-", column)
    print()

    print("=" * 80)
    print("FIRST 3 ROWS")
    print("=" * 80)
    print(df.head(3).to_string())
    print()

    print("=" * 80)
    print("MISSING VALUES")
    print("=" * 80)
    print(df.isna().sum())


if __name__ == "__main__":
    main()
