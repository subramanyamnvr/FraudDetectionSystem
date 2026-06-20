from __future__ import annotations

import json
from pathlib import Path

import yaml

CURRENT_FILE = Path(__file__).resolve()
MLOPS_ROOT = CURRENT_FILE.parents[1]
MONITORING_DIR = MLOPS_ROOT / "08_monitoring"

MONITORING_SUMMARY_FILE = MONITORING_DIR / "08_monitoring_summary.json"
POLICY_FILE = CURRENT_FILE.with_name("09_retraining_policy.yaml")
DECISION_FILE = CURRENT_FILE.with_name("09_retraining_decision.json")


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_policy(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)["retraining_policy"]


def decide_retraining(summary: dict, policy: dict) -> dict:
    reasons = []

    if summary["drifted_feature_count"] >= policy["max_drifted_features"]:
        reasons.append(
            f"drifted_feature_count {summary['drifted_feature_count']} is above threshold {policy['max_drifted_features']}"
        )

    if policy["monitoring_warning_triggers_retraining"] and summary["monitoring_status"] == "warning":
        reasons.append("monitoring status is warning")

    if summary["average_fraud_probability"] >= policy["max_average_fraud_probability"]:
        reasons.append(
            "average fraud probability "
            f"{summary['average_fraud_probability']} is above threshold {policy['max_average_fraud_probability']}"
        )

    should_retrain = len(reasons) > 0

    return {
        "should_retrain": should_retrain,
        "decision": "retrain" if should_retrain else "keep_current_model",
        "reasons": reasons,
        "checked_rows": summary["checked_rows"],
        "drifted_feature_count": summary["drifted_feature_count"],
        "monitoring_status": summary["monitoring_status"],
        "recommended_command": policy["retraining_command"] if should_retrain else "",
        "note": "In production, this decision can trigger a scheduled pipeline or approval workflow.",
    }


def main() -> None:
    summary = load_json(MONITORING_SUMMARY_FILE)
    policy = load_policy(POLICY_FILE)
    decision = decide_retraining(summary, policy)

    with open(DECISION_FILE, "w", encoding="utf-8") as file:
        json.dump(decision, file, indent=2)

    print("Retraining decision created:")
    print(f"- {DECISION_FILE}")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
