# 12 Rollout Strategy

This step prepares how an approved model will be rolled out to users.

Restaurant story:
- the recipe is approved, but the restaurant still does not serve it to every customer all at once
- first the team tests it in a smaller setting, then slowly increases how many customers receive it
- this reduces the risk of a bad release

What this step does:
- reads the promotion result from step 11
- reads a rollout policy with environment and traffic-shift rules
- blocks rollout if promotion was not approved
- otherwise prepares a staged rollout plan for BentoML serving

Files here:
- `12_rollout_policy.yaml`
- `12_prepare_rollout.py`
- `12_rollout_plan.json`

How it fits:
- step 11 decides whether the model may be promoted
- step 12 decides how that approved model should be introduced safely
- this is where ideas like staging, canary rollout, and blue-green deployment become practical

Run:
- `python MLOPS/12_rollout_strategy_bentoml/12_prepare_rollout.py`

Output:
- `12_rollout_plan.json` shows whether rollout is blocked or ready, and the staged traffic plan
