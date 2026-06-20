from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from dotenv import load_dotenv
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

load_dotenv()
os.environ.setdefault("PREFECT_SERVER_ANALYTICS_ENABLED", "false")

from prefect import flow, get_run_logger, task

DATA_FILE = "creditcard.csv"
OUTPUT_DIR = Path("packaged_model")
MODEL_PATH = OUTPUT_DIR / "best_model.joblib"
SCALER_PATH = OUTPUT_DIR / "scaler.joblib"
METADATA_PATH = OUTPUT_DIR / "model_metadata.json"
RESULTS_PATH = OUTPUT_DIR / "model_results.csv"


def load_params() -> dict:
    with open("params.yaml", "r", encoding="utf-8") as file:
        return yaml.safe_load(file)["pipeline"]


@task
def load_and_prepare_data(sample_size: int, random_state: int) -> pd.DataFrame:
    data = pd.read_csv(DATA_FILE)

    fraud = data[data["Class"] == 1].sample(sample_size // 2, random_state=random_state)
    non_fraud = data[data["Class"] == 0].sample(sample_size // 2, random_state=random_state)

    prepared = pd.concat([fraud, non_fraud], ignore_index=True)
    prepared = prepared.sample(frac=1, random_state=random_state).reset_index(drop=True)
    prepared.columns = [column.lower() for column in prepared.columns]
    prepared = prepared.drop_duplicates().reset_index(drop=True)

    return prepared


@task
def split_and_scale(data: pd.DataFrame, test_size: float, random_state: int) -> dict:
    x_data = data.drop("class", axis=1)
    y_data = data["class"]

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
        "feature_columns": x_data.columns.tolist(),
        "scaler": scaler,
    }


@task
def train_and_select_model(split_data: dict, random_state: int) -> dict:
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
def package_outputs(model_data: dict, split_data: dict, sample_size: int) -> dict:
    OUTPUT_DIR.mkdir(exist_ok=True)

    joblib.dump(model_data["best_model"], MODEL_PATH)
    joblib.dump(split_data["scaler"], SCALER_PATH)

    model_data["results_df"].to_csv(RESULTS_PATH, index=False)

    metadata = {
        "model_name": model_data["best_model_name"],
        "saved_at": datetime.now().isoformat(),
        "sample_size": sample_size,
        "target_column": "class",
        "feature_columns": split_data["feature_columns"],
        "metrics": model_data["results_df"].iloc[0].to_dict(),
        "model_file": str(MODEL_PATH),
        "scaler_file": str(SCALER_PATH),
        "results_file": str(RESULTS_PATH),
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    return metadata


@task
def track_with_mlflow(metadata: dict, model_data: dict, sample_size: int) -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "fraud-detection-demo")
    registered_model_name = os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "fraud-classifier-demo")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=model_data["best_model_name"]):
        mlflow.log_param("sample_size", sample_size)
        mlflow.log_param("best_model_name", model_data["best_model_name"])
        mlflow.log_param("target_column", "class")

        mlflow.log_metric("accuracy", float(metadata["metrics"]["Accuracy"]))
        mlflow.log_metric("precision", float(metadata["metrics"]["Precision"]))
        mlflow.log_metric("recall", float(metadata["metrics"]["Recall"]))
        mlflow.log_metric("f1_score", float(metadata["metrics"]["F1 Score"]))
        mlflow.log_metric("roc_auc", float(metadata["metrics"]["ROC AUC"]))

        mlflow.log_artifact(str(MODEL_PATH))
        mlflow.log_artifact(str(SCALER_PATH))
        mlflow.log_artifact(str(METADATA_PATH))
        mlflow.log_artifact(str(RESULTS_PATH))

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

    logger.info("Loading and preparing data")
    data = load_and_prepare_data(params["sample_size"], params["random_state"])

    logger.info("Splitting data and scaling features")
    split_data = split_and_scale(data, params["test_size"], params["random_state"])

    logger.info("Training models and selecting the best one")
    model_data = train_and_select_model(split_data, params["random_state"])

    logger.info("Packaging model artifacts")
    metadata = package_outputs(model_data, split_data, params["sample_size"])

    logger.info("Tracking the run in MLflow")
    track_with_mlflow(metadata, model_data, params["sample_size"])

    logger.info("Best model: %s", metadata["model_name"])
    logger.info("Artifacts saved in: %s", OUTPUT_DIR)


if __name__ == "__main__":
    fraud_detection_flow()
