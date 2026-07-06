"""Downloads the Kaggle Credit Card Fraud Detection dataset (creditcard.csv).

The original dataset is hosted on Kaggle at mlg-ulb/creditcardfraud and requires
Kaggle authentication to fetch via API. This script instead pulls the identical
CSV from a public GitHub mirror so the project can be reproduced without a
Kaggle account.
"""

from pathlib import Path
from urllib.request import urlretrieve

DATA_URL = (
    "https://raw.githubusercontent.com/nsethi31/"
    "Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv"
)
OUTPUT_PATH = Path(__file__).parent / "creditcard.csv"
EXPECTED_ROWS = 284807


def download() -> Path:
    if OUTPUT_PATH.exists():
        print(f"Dataset already present at {OUTPUT_PATH}, skipping download.")
        return OUTPUT_PATH

    print(f"Downloading dataset from {DATA_URL} ...")
    urlretrieve(DATA_URL, OUTPUT_PATH)
    print(f"Saved to {OUTPUT_PATH}")
    return OUTPUT_PATH


def verify(path: Path) -> None:
    with open(path, "rb") as f:
        line_count = sum(1 for _ in f)
    # +1 for the header row
    if line_count != EXPECTED_ROWS + 1:
        raise ValueError(
            f"Expected {EXPECTED_ROWS + 1} lines (incl. header), got {line_count}. "
            "The downloaded file may be corrupted or the mirror may have changed."
        )
    print(f"Verified {EXPECTED_ROWS} transaction rows.")


if __name__ == "__main__":
    csv_path = download()
    verify(csv_path)
