# 11 Deployment Promotion

This step decides whether a registered MLflow model version can be promoted for serving.

Restaurant story:
- the kitchen has cooked the food, the serving counter is running, and monitoring has checked quality
- now the restaurant owner chooses whether this batch becomes the official dish served to every customer
- this is the approval gate before full production use

What this step does:
- reads the retraining decision from step 9
- reads the latest registered model version from MLflow
- blocks promotion if monitoring says retraining is required
- otherwise assigns an MLflow alias such as `champion` to the approved version
- writes the result to a JSON file

Files here:
- `11_promotion_policy.yaml`
- `11_promote_model.py`
- `11_promotion_result.json`

How it fits:
- step 10 automates validation
- step 11 is the final approval gate for production promotion
- MLflow aliases make it easy to say which registered version is the current approved model

Run:
- `python MLOPS/11_deployment_promotion_mlflow/11_promote_model.py`

Output:
- `11_promotion_result.json` says whether promotion was approved or blocked
