from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

CURRENT_FILE = Path(__file__).resolve()
MLOPS_ROOT = CURRENT_FILE.parents[1]
PROMOTION_DIR = MLOPS_ROOT / "11_deployment_promotion_mlflow"

PROMOTION_RESULT_FILE = PROMOTION_DIR / "11_promotion_result.json"
POLICY_FILE = CURRENT_FILE.with_name("12_rollout_policy.yaml")
ROLLOUT_RESULT_FILE = CURRENT_FILE.with_name("12_rollout_plan.json")


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_policy(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)["rollout_policy"]


def build_rollout_plan(promotion_result: dict, policy: dict) -> dict:
    rollout_steps = []

    if promotion_result["promotion_status"] != "approved" and policy["require_promotion_approval"]:
        return {
            "rollout_status": "blocked",
            "strategy_name": policy["strategy_name"],
            "serving_platform": policy["serving_platform"],
            "reason": "model promotion was not approved, so rollout cannot start",
            "source_promotion_status": promotion_result["promotion_status"],
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }

    rollout_steps.append(
        {
            "step": 1,
            "environment": policy["initial_environment"],
            "traffic_percentage": 0,
            "action": "deploy BentoML service and verify health endpoints",
        }
    )

    for index, percentage in enumerate(policy["canary_percentages"], start=2):
        rollout_steps.append(
            {
                "step": index,
                "environment": "production",
                "traffic_percentage": percentage,
                "action": f"shift {percentage}% traffic to alias {policy['production_alias']}",
            }
        )

    return {
        "rollout_status": "ready",
        "strategy_name": policy["strategy_name"],
        "serving_platform": policy["serving_platform"],
        "registered_model_name": promotion_result.get("registered_model_name", ""),
        "alias_name": policy["production_alias"],
        "candidate_version": promotion_result.get("promoted_version", promotion_result.get("candidate_version", "")),
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "rollout_steps": rollout_steps,
        "note": "In production, each step would be tied to health checks, metrics review, and rollback rules.",
    }


def main() -> None:
    promotion_result = load_json(PROMOTION_RESULT_FILE)
    policy = load_policy(POLICY_FILE)
    rollout_plan = build_rollout_plan(promotion_result, policy)

    with open(ROLLOUT_RESULT_FILE, "w", encoding="utf-8") as file:
        json.dump(rollout_plan, file, indent=2)

    print("Rollout plan created:")
    print(f"- {ROLLOUT_RESULT_FILE}")
    print(json.dumps(rollout_plan, indent=2))


if __name__ == "__main__":
    main()
