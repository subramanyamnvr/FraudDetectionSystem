from __future__ import annotations

import os
from typing import Any

import bentoml
import cloudpickle
import pandas as pd
from pydantic import BaseModel, Field

MODEL_TAG = os.getenv("BENTOML_MODEL_TAG", "fraud_classifier_bundle:latest")
MODEL_REFERENCE = bentoml.models.get(MODEL_TAG)

with open(MODEL_REFERENCE.path_of("model_bundle.pkl"), "rb") as file:
    MODEL_BUNDLE = cloudpickle.load(file)

MODEL = MODEL_BUNDLE["model"]
SCALER = MODEL_BUNDLE["scaler"]
FEATURE_COLUMNS = MODEL_BUNDLE["feature_columns"]
MODEL_METADATA = MODEL_BUNDLE["metadata"]


class FraudRequest(BaseModel):
    v1: float
    v2: float
    v3: float
    v4: float
    v5: float
    v6: float
    v7: float
    v8: float
    v9: float
    v10: float
    v11: float
    v12: float
    v13: float
    v14: float
    v15: float
    v16: float
    v17: float
    v18: float
    v19: float
    v20: float
    v21: float
    v22: float
    v23: float
    v24: float
    v25: float
    v26: float
    v27: float
    v28: float
    amount: float = Field(..., ge=0)


@bentoml.service(name="fraud-detection-service")
class FraudDetectionService:
    @bentoml.api
    def health(self) -> dict[str, str]:
        return {
            "status": "ok",
            "model_name": MODEL_METADATA["model_name"],
        }

    @bentoml.api
    def predict(self, records: list[FraudRequest]) -> list[dict[str, Any]]:
        frame = pd.DataFrame([record.model_dump() for record in records], columns=FEATURE_COLUMNS)
        scaled_features = SCALER.transform(frame)
        predictions = MODEL.predict(scaled_features).tolist()
        probabilities = MODEL.predict_proba(scaled_features)[:, 1].tolist()

        results = []
        for prediction, probability in zip(predictions, probabilities):
            results.append(
                {
                    "prediction": int(prediction),
                    "fraud_probability": round(float(probability), 4),
                }
            )

        return results
