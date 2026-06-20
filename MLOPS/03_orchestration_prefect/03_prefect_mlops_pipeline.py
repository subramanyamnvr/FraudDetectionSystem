from __future__ import annotations

import importlib.util
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from dotenv import load_dotenv
from feast import FeatureStore
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parents[2]
MLOPS_ROOT = REPO_ROOT / "MLOPS"

SETUP_DIR = MLOPS_ROOT / "00_project_setup"
DATA_DIR = MLOPS_ROOT / "01_data_versioning_dvc"
FEATURE_REPO_DIR = MLOPS_ROOT / "02_feature_store_feast" / "feature_repo"
FEATURE_DATA_DIR = FEATURE_REPO_DIR / "data"
TRACKING_DIR = MLOPS_ROOT / "04_experiment_tracking_mlflow"
ARTIFACT_DIR = MLOPS_ROOT / "06_model_artifacts_joblib"

load_dotenv(TRACKING_DIR / ".env")
load_dotenv(REPO_ROOT / ".env", override=False)
os.environ.setdefault("PREFECT_SERVER_ANALYTICS_ENABLED", "false")

from prefect import flow, get_run_logger, task

DATA_FILE = DATA_DIR / "01_creditcard_dataset.csv"
PARAMS_FILE = CURRENT_FILE.with_name("03_pipeline_params.yaml")
FEATURE_DEFINITIONS_FILE = FEATURE_REPO_DIR / "02_fraud_feature_definitions.py"
FEATURE_SOURCE_FILE = FEATURE_DATA_DIR / "02_fraud_feature_source.parquet"
ONLINE_FEATURES_FILE = FEATURE_DATA_DIR / "02_latest_online_features.json"
MODEL_PATH = ARTIFACT_DIR / "06_best_model.joblib"
SCALER_PATH = ARTIFACT_DIR / "06_scaler.joblib"
METADATA_PATH = ARTIFACT_DIR / "06_model_metadata.json"
RESULTS_PATH = ARTIFACT_DIR / "06_model_results.csv"
DEFAULT_MLFLOW_DB = TRACKING_DIR / "04_mlflow_tracking.db"
FEATURE_VIEW_NAME = "fraud_transaction_features"
FEATURE_SERVICE_NAME = "fraud_feature_service_v1"
FEATURE_COLUMNS = [f"v{index}" for index in range(1, 29)] + ["amount"]


def load_params() -> dict[str, Any]:
    with open(PARAMS_FILE, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)["pipeline"]


def import_feast_objects() -> list[Any]:
    spec = importlib.util.spec_from_file_location("fraud_feature_definitions", FEATURE_DEFINITIONS_FILE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return [
        module.transaction_entity,
        module.fraud_transaction_features,
        module.fraud_feature_service_v1,
    ]


def make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    return value


@task
def load_and_prepare_data(sample_size: int, random_state: int) -> pd.DataFrame:
    data = pd.read_csv(DATA_FILE)

    fraud = data[data["Class"] == 1].sample(sample_size // 2, random_state=random_state)
    non_fraud = data[data["Class"] == 0].sample(sample_size // 2, random_state=random_state)

    prepared = pd.concat([fraud, non_fraud], ignore_index=True)
    prepared = prepared.sample(frac=1, random_state=random_state).reset_index(drop=True)
    prepared.columns = [column.lower() for column in prepared.columns]
    prepared = prepared.drop_duplicates().reset_index(drop=True)
    prepared["transaction_id"] = range(10001, 10001 + len(prepared))
    prepared["event_timestamp"] = pd.date_range(
        start="2024-01-01 00:00:00",
        periods=len(prepared),
        freq="min",
        tz="UTC",
    )

    return prepared


@task
def build_feast_training_data(data: pd.DataFrame) -> dict[str, Any]:
    FEATURE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    feature_source_df = data[["transaction_id", "event_timestamp", *FEATURE_COLUMNS]].copy()
    feature_source_df.to_parquet(FEATURE_SOURCE_FILE, index=False)

    entity_df = data[["transaction_id", "event_timestamp", "class"]].copy()
    feast_objects = import_feast_objects()
    store = FeatureStore(repo_path=str(FEATURE_REPO_DIR))
    store.apply(feast_objects)

    feature_references = [f"{FEATURE_VIEW_NAME}:{column}" for column in FEATURE_COLUMNS]
    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=feature_references,
    ).to_df()

    store.materialize_incremental(end_date=datetime.now(timezone.utc) + timedelta(minutes=1))
    online_features = store.get_online_features(
        features=feature_references,
        entity_rows=[{"transaction_id": int(entity_df.iloc[0]["transaction_id"])}],
    ).to_dict()

    with open(ONLINE_FEATURES_FILE, "w", encoding="utf-8") as file:
        json.dump(make_json_safe(online_features), file, indent=2)

    return {
        "training_df": training_df,
        "feature_references": feature_references,
        "feature_source_file": str(FEATURE_SOURCE_FILE),
        "online_features_file": str(ONLINE_FEATURES_FILE),
    }


@task
def split_and_scale(training_bundle: dict[str, Any], test_size: float, random_state: int) -> dict[str, Any]:
    training_df = training_bundle["training_df"]
    x_data = training_df[FEATURE_COLUMNS]
    y_data = training_df["class"]

    x_train, x_test, y_train, y_test = train_test_split(
        x_data,
        y_data,
        test_size=test_size,
        random_state=random_state,
        stratify=y_data,
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return {
        "x_train_scaled": x_train_scaled,
        "x_test_scaled": x_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "feature_columns": FEATURE_COLUMNS,
        "scaler": scaler,
        "feature_references": training_bundle["feature_references"],
        "feature_source_file": training_bundle["feature_source_file"],
        "online_features_file": training_bundle["online_features_file"],
    }


@task
def train_and_select_model(split_data: dict[str, Any], random_state: int) -> dict[str, Any]:
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "SVC": SVC(probability=True, random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=random_state),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=random_state,
        ),
    }

    models["Stacking Classifier"] = StackingClassifier(
        estimators=[
            ("lr", LogisticRegression(max_iter=1000, random_state=random_state)),
            ("svc", SVC(probability=True, random_state=random_state)),
            ("rf", RandomForestClassifier(n_estimators=100, random_state=random_state)),
        ],
        final_estimator=LogisticRegression(max_iter=1000, random_state=random_state),
    )

    results = []
    trained_models = {}

    for name, model in models.items():
        model.fit(split_data["x_train_scaled"], split_data["y_train"])
        trained_models[name] = model

        predictions = model.predict(split_data["x_test_scaled"])
        probabilities = model.predict_proba(split_data["x_test_scaled"])[:, 1]

        results.append(
            {
                "Model": name,
                "Accuracy": round(accuracy_score(split_data["y_test"], predictions), 4),
                "Precision": round(precision_score(split_data["y_test"], predictions, zero_division=0), 4),
                "Recall": round(recall_score(split_data["y_test"], predictions, zero_division=0), 4),
                "F1 Score": round(f1_score(split_data["y_test"], predictions, zero_division=0), 4),
                "ROC AUC": round(roc_auc_score(split_data["y_test"], probabilities), 4),
            }
        )

    results_df = pd.DataFrame(results).sort_values(
        by=["F1 Score", "ROC AUC"],
        ascending=False,
    ).reset_index(drop=True)
    best_model_name = results_df.iloc[0]["Model"]

    return {
        "results_df": results_df,
        "best_model_name": best_model_name,
        "best_model": trained_models[best_model_name],
    }


@task
def package_outputs(model_data: dict[str, Any], split_data: dict[str, Any], sample_size: int) -> dict[str, Any]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model_data["best_model"], MODEL_PATH)
    joblib.dump(split_data["scaler"], SCALER_PATH)
    model_data["results_df"].to_csv(RESULTS_PATH, index=False)

    metadata = {
        "model_name": model_data["best_model_name"],
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "sample_size": sample_size,
        "target_column": "class",
        "feature_columns": split_data["feature_columns"],
        "feature_references": split_data["feature_references"],
        "feature_view_name": FEATURE_VIEW_NAME,
        "feature_service_name": FEATURE_SERVICE_NAME,
        "feature_source_file": split_data["feature_source_file"],
        "online_features_file": split_data["online_features_file"],
        "metrics": model_data["results_df"].iloc[0].to_dict(),
        "model_file": str(MODEL_PATH),
        "scaler_file": str(SCALER_PATH),
        "results_file": str(RESULTS_PATH),
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    return metadata


@task
def track_with_mlflow(metadata: dict[str, Any], model_data: dict[str, Any], sample_size: int) -> None:
    default_tracking_uri = f"sqlite:///{DEFAULT_MLFLOW_DB.as_posix()}"
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", default_tracking_uri)
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "fraud-detection-demo")
    registered_model_name = os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "fraud-classifier-demo")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=model_data["best_model_name"]):
        mlflow.log_param("sample_size", sample_size)
        mlflow.log_param("best_model_name", model_data["best_model_name"])
        mlflow.log_param("target_column", "class")
        mlflow.log_param("feature_view_name", FEATURE_VIEW_NAME)
        mlflow.log_param("feature_service_name", FEATURE_SERVICE_NAME)

        mlflow.log_metric("accuracy", float(metadata["metrics"]["Accuracy"]))
        mlflow.log_metric("precision", float(metadata["metrics"]["Precision"]))
        mlflow.log_metric("recall", float(metadata["metrics"]["Recall"]))
        mlflow.log_metric("f1_score", float(metadata["metrics"]["F1 Score"]))
        mlflow.log_metric("roc_auc", float(metadata["metrics"]["ROC AUC"]))

        mlflow.log_artifact(str(MODEL_PATH))
        mlflow.log_artifact(str(SCALER_PATH))
        mlflow.log_artifact(str(METADATA_PATH))
        mlflow.log_artifact(str(RESULTS_PATH))
        mlflow.log_artifact(str(FEATURE_SOURCE_FILE))
        mlflow.log_artifact(str(ONLINE_FEATURES_FILE))

        mlflow.sklearn.log_model(
            sk_model=model_data["best_model"],
            name="model",
            serialization_format="cloudpickle",
            registered_model_name=registered_model_name,
        )


@flow(name="fraud-detection-mlops-flow", log_prints=True)
def fraud_detection_flow() -> None:
    logger = get_run_logger()
    params = load_params()

    logger.info("Step 1 - Loading and preparing data")
    data = load_and_prepare_data(params["sample_size"], params["random_state"])

    logger.info("Step 2 - Building Feast training features")
    training_bundle = build_feast_training_data(data)

    logger.info("Step 3 - Splitting data and scaling features")
    split_data = split_and_scale(training_bundle, params["test_size"], params["random_state"])

    logger.info("Step 4 - Training models and selecting the best one")
    model_data = train_and_select_model(split_data, params["random_state"])

    logger.info("Step 5 - Packaging model artifacts")
    metadata = package_outputs(model_data, split_data, params["sample_size"])

    logger.info("Step 6 - Tracking the run in MLflow")
    track_with_mlflow(metadata, model_data, params["sample_size"])

    logger.info("Best model: %s", metadata["model_name"])
    logger.info("Artifacts saved in: %s", ARTIFACT_DIR)


if __name__ == "__main__":
    fraud_detection_flow()

