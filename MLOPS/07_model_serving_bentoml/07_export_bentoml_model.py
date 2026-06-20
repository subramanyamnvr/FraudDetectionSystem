from __future__ import annotations

import json
import os
from pathlib import Path

import bentoml
import cloudpickle
import joblib

CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parents[2]
MLOPS_ROOT = REPO_ROOT / "MLOPS"
ARTIFACT_DIR = MLOPS_ROOT / "06_model_artifacts_joblib"

MODEL_PATH = ARTIFACT_DIR / "06_best_model.joblib"
SCALER_PATH = ARTIFACT_DIR / "06_scaler.joblib"
METADATA_PATH = ARTIFACT_DIR / "06_model_metadata.json"
DEFAULT_MODEL_NAME = "fraud_classifier_bundle"


def export_model() -> bentoml.Tag:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    model_name = os.getenv("BENTOML_MODEL_NAME", DEFAULT_MODEL_NAME)
    bundle = {
        "model": model,
        "scaler": scaler,
        "feature_columns": metadata["feature_columns"],
        "metadata": metadata,
    }

    with bentoml.models.create(
        name=model_name,
        module="fraud_detection_bento_bundle",
        context=bentoml.models.ModelContext(
            framework_name="custom-python-bundle",
            framework_versions={"bentoml": bentoml.__version__},
        ),
        metadata={
            "source_model_name": metadata["model_name"],
            "sample_size": metadata["sample_size"],
            "feature_count": len(metadata["feature_columns"]),
        },
    ) as bento_model:
        bundle_path = Path(bento_model.path_of("model_bundle.pkl"))
        with open(bundle_path, "wb") as file:
            cloudpickle.dump(bundle, file)

    return bento_model.tag


if __name__ == "__main__":
    tag = export_model()
    print(f"BentoML model saved: {tag}")

