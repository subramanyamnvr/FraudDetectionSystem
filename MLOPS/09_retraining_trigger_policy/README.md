# 09 Retraining Trigger

This step decides whether the model should be retrained.

Restaurant story:
- the manager has already noticed that some served plates are changing
- now the head chef decides whether to cook a fresh batch using the original recipe process
- this is the point where monitoring becomes action

What this step does:
- reads the monitoring summary from step 8
- reads a simple retraining policy with thresholds
- decides whether the current model can stay in production or should be retrained
- writes the decision to a JSON file

Files here:
- `09_retraining_policy.yaml`
- `09_decide_retraining.py`
- `09_retraining_decision.json`

How it fits:
- step 8 tells us what is changing
- step 9 decides what to do about it
- if retraining is needed, the recommended next action is to run the Prefect pipeline again

Run:
- `python MLOPS/09_retraining_trigger_policy/09_decide_retraining.py`

Output:
- `09_retraining_decision.json` clearly says whether retraining is required and why

