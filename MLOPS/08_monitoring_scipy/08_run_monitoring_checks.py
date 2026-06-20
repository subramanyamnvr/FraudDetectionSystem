from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parents[2]
MLOPS_ROOT = REPO_ROOT / "MLOPS"

DATA_FILE = MLOPS_ROOT / "01_data_versioning_dvc" / "01_creditcard_dataset.csv"
REFERENCE_FEATURES_FILE = MLOPS_ROOT / "02_feature_store_feast" / "feature_repo" / "data" / "02_fraud_feature_source.parquet"
MODEL_METADATA_FILE = MLOPS_ROOT / "06_model_artifacts_joblib" / "06_model_metadata.json"
MODEL_FILE = MLOPS_ROOT / "06_model_artifacts_joblib" / "06_best_model.joblib"
SCALER_FILE = MLOPS_ROOT / "06_model_artifacts_joblib" / "06_scaler.joblib"

OUTPUT_DIR = CURRENT_FILE.parent
CURRENT_BATCH_FILE = OUTPUT_DIR / "08_current_scored_batch.csv"
DRIFT_REPORT_FILE = OUTPUT_DIR / "08_drift_report.csv"
SUMMARY_FILE = OUTPUT_DIR / "08_monitoring_summary.json"


def load_artifacts() -> tuple[list[str], object, object]:
    with open(MODEL_METADATA_FILE, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    return metadata["feature_columns"], model, scaler


def build_current_batch(feature_columns: list[str]) -> pd.DataFrame:
    data = pd.read_csv(DATA_FILE)
    data.columns = [column.lower() for column in data.columns]

    current_batch = data.tail(40).copy().reset_index(drop=True)

    # Introduce a small shift so the monitoring step can demonstrate drift clearly.
    current_batch["amount"] = current_batch["amount"] * 1.35
    current_batch["v14"] = current_batch["v14"] * 1.15
    current_batch["v17"] = current_batch["v17"] * 0.85

    return current_batch[feature_columns + ["class"]]


def score_current_batch(current_batch: pd.DataFrame, feature_columns: list[str], model: object, scaler: object) -> pd.DataFrame:
    scaled_features = scaler.transform(current_batch[feature_columns])
    current_batch = current_batch.copy()
    current_batch["prediction"] = model.predict(scaled_features)
    current_batch["fraud_probability"] = model.predict_proba(scaled_features)[:, 1]
    return current_batch


def calculate_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    rows = []

    for column in feature_columns:
        reference_series = reference_df[column].dropna()
        current_series = current_df[column].dropna()
        ks_statistic, p_value = ks_2samp(reference_series, current_series)

        reference_mean = float(reference_series.mean())
        current_mean = float(current_series.mean())
        mean_gap = abs(current_mean - reference_mean)
        std_base = float(reference_series.std(ddof=0)) or 1.0
        normalized_mean_shift = mean_gap / std_base

        rows.append(
            {
                "feature": column,
                "reference_mean": round(reference_mean, 4),
                "current_mean": round(current_mean, 4),
                "reference_std": round(float(reference_series.std(ddof=0)), 4),
                "current_std": round(float(current_series.std(ddof=0)), 4),
                "ks_statistic": round(float(ks_statistic), 4),
                "p_value": round(float(p_value), 6),
                "normalized_mean_shift": round(float(normalized_mean_shift), 4),
                "drift_detected": bool(p_value < 0.05 or normalized_mean_shift > 0.3),
            }
        )

    return pd.DataFrame(rows).sort_values(
        by=["drift_detected", "normalized_mean_shift", "ks_statistic"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def build_summary(drift_report: pd.DataFrame, scored_batch: pd.DataFrame) -> dict[str, object]:
    drifted_features = drift_report[drift_report["drift_detected"]]["feature"].tolist()
    prediction_rate = float(scored_batch["prediction"].mean())
    avg_probability = float(scored_batch["fraud_probability"].mean())

    return {
        "monitoring_status": "warning" if drifted_features else "healthy",
        "checked_rows": int(len(scored_batch)),
        "drifted_feature_count": int(len(drifted_features)),
        "drifted_features": drifted_features[:10],
        "average_fraud_probability": round(avg_probability, 4),
        "predicted_fraud_rate": round(prediction_rate, 4),
        "note": "In production, this current batch would come from live BentoML request logs or a feature pipeline.",
    }


def main() -> None:
    feature_columns, model, scaler = load_artifacts()
    reference_df = pd.read_parquet(REFERENCE_FEATURES_FILE)
    current_batch = build_current_batch(feature_columns)
    scored_batch = score_current_batch(current_batch, feature_columns, model, scaler)
    drift_report = calculate_drift(reference_df, scored_batch, feature_columns)
    summary = build_summary(drift_report, scored_batch)

    scored_batch.to_csv(CURRENT_BATCH_FILE, index=False)
    drift_report.to_csv(DRIFT_REPORT_FILE, index=False)

    with open(SUMMARY_FILE, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("Monitoring files created:")
    print(f"- {CURRENT_BATCH_FILE}")
    print(f"- {DRIFT_REPORT_FILE}")
    print(f"- {SUMMARY_FILE}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

