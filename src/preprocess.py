from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


RAW_DATA_PATH = Path("data/raw/creditcard.csv")
PROCESSED_DIR = Path("data/processed")
TARGET_COLUMN = "Class"


def load_data(input_path: Path) -> pd.DataFrame:
    return pd.read_csv(input_path)


def clean_data(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].astype(int)
    return cleaned


def split_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_frame, temp_frame = train_test_split(
        frame,
        test_size=0.30,
        random_state=42,
        stratify=frame[TARGET_COLUMN],
    )
    validation_frame, test_frame = train_test_split(
        temp_frame,
        test_size=0.50,
        random_state=42,
        stratify=temp_frame[TARGET_COLUMN],
    )
    return train_frame, validation_frame, test_frame


def scale_features(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_columns = [column for column in train_frame.columns if column != TARGET_COLUMN]
    scaler = StandardScaler()

    train_scaled = train_frame.copy()
    validation_scaled = validation_frame.copy()
    test_scaled = test_frame.copy()

    train_scaled[feature_columns] = scaler.fit_transform(train_frame[feature_columns])
    validation_scaled[feature_columns] = scaler.transform(validation_frame[feature_columns])
    test_scaled[feature_columns] = scaler.transform(test_frame[feature_columns])

    return train_scaled, validation_scaled, test_scaled


def save_outputs(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_frame.to_csv(PROCESSED_DIR / "train.csv", index=False)
    validation_frame.to_csv(PROCESSED_DIR / "validation.csv", index=False)
    test_frame.to_csv(PROCESSED_DIR / "test.csv", index=False)

    summary = pd.DataFrame(
        [
            {"split": "train", "rows": len(train_frame), "fraud_rate": train_frame[TARGET_COLUMN].mean()},
            {"split": "validation", "rows": len(validation_frame), "fraud_rate": validation_frame[TARGET_COLUMN].mean()},
            {"split": "test", "rows": len(test_frame), "fraud_rate": test_frame[TARGET_COLUMN].mean()},
        ]
    )
    summary.to_csv(PROCESSED_DIR / "summary.csv", index=False)


def run_preprocessing(input_path: Path) -> None:
    frame = load_data(input_path)
    cleaned = clean_data(frame)
    train_frame, validation_frame, test_frame = split_data(cleaned)
    train_scaled, validation_scaled, test_scaled = scale_features(train_frame, validation_frame, test_frame)
    save_outputs(train_scaled, validation_scaled, test_scaled)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess the credit card fraud dataset.")
    parser.add_argument(
        "--input",
        type=Path,
        default=RAW_DATA_PATH,
        help="Path to the raw CSV dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_preprocessing(args.input)


if __name__ == "__main__":
    main()
