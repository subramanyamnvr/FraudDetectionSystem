from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import yaml
from dotenv import load_dotenv
from mlflow import MlflowClient

CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parents[2]
MLOPS_ROOT = REPO_ROOT / "MLOPS"
TRACKING_DIR = MLOPS_ROOT / "04_experiment_tracking_mlflow"
RETRAINING_DIR = MLOPS_ROOT / "09_retraining_trigger_policy"

DECISION_FILE = RETRAINING_DIR / "09_retraining_decision.json"
POLICY_FILE = CURRENT_FILE.with_name("11_promotion_policy.yaml")
RESULT_FILE = CURRENT_FILE.with_name("11_promotion_result.json")
DEFAULT_MLFLOW_DB = TRACKING_DIR / "04_mlflow_tracking.db"


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_policy(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)["promotion_policy"]


def build_client() -> tuple[MlflowClient, str, str]:
    load_dotenv(TRACKING_DIR / ".env")
    load_dotenv(REPO_ROOT / ".env", override=False)

    default_tracking_uri = f"sqlite:///{DEFAULT_MLFLOW_DB.as_posix()}"
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", default_tracking_uri)
    model_name = os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "fraud-classifier-demo")

    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient(tracking_uri=tracking_uri), tracking_uri, model_name


def get_latest_version(client: MlflowClient, model_name: str):
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        return None
    return max(versions, key=lambda version: int(version.version))


def promote_or_block(decision: dict, policy: dict) -> dict:
    client, tracking_uri, model_name = build_client()
    latest_version = get_latest_version(client, model_name)

    if latest_version is None:
        return {
            "promotion_status": "blocked",
            "reason": "no registered MLflow model version was found",
            "tracking_uri": tracking_uri,
            "registered_model_name": model_name,
        }

    retraining_required = bool(decision["should_retrain"])
    allow_override = bool(policy["allow_promotion_when_retraining_required"])

    if retraining_required and not allow_override:
        client.set_model_version_tag(model_name, latest_version.version, "promotion_status", "blocked")
        client.set_model_version_tag(model_name, latest_version.version, "promotion_reason", "retraining_required")

        return {
            "promotion_status": "blocked",
            "reason": "monitoring requested retraining, so promotion is blocked",
            "tracking_uri": tracking_uri,
            "registered_model_name": model_name,
            "candidate_version": latest_version.version,
            "candidate_run_id": latest_version.run_id,
            "requested_alias": policy["alias_name"],
        }

    client.set_registered_model_alias(model_name, policy["alias_name"], latest_version.version)
    client.set_model_version_tag(model_name, latest_version.version, "promotion_status", "approved")
    client.set_model_version_tag(model_name, latest_version.version, "promotion_note", policy["promotion_stage_note"])
    client.set_model_version_tag(
        model_name,
        latest_version.version,
        "promoted_at",
        datetime.now(timezone.utc).isoformat(),
    )

    return {
        "promotion_status": "approved",
        "tracking_uri": tracking_uri,
        "registered_model_name": model_name,
        "promoted_version": latest_version.version,
        "promoted_run_id": latest_version.run_id,
        "alias_name": policy["alias_name"],
        "note": "The MLflow model alias now points to the version approved for serving.",
    }


def main() -> None:
    decision = load_json(DECISION_FILE)
    policy = load_policy(POLICY_FILE)
    result = promote_or_block(decision, policy)

    with open(RESULT_FILE, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    print("Promotion result created:")
    print(f"- {RESULT_FILE}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
