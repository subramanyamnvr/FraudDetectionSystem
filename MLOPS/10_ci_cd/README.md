# 10 CI CD

This step automates the MLOps lifecycle in GitHub Actions.

Restaurant story:
- until now the team has been cooking, serving, checking, and deciding by hand
- CI/CD is the restaurant's daily operations board that makes sure the same checklist runs every time
- this reduces mistakes and keeps the workflow repeatable

What this step does:
- runs the training pipeline automatically on push, pull request, or manual trigger
- exports the BentoML model bundle
- runs monitoring checks
- runs the retraining decision logic
- smoke tests the BentoML service before changes are trusted

Files here:
- `.github/workflows/10_mlops_ci_cd.yml`

How it fits:
- steps 1 to 9 define the full lifecycle
- step 10 makes that lifecycle automatic
- now the repo can check whether code changes still produce a working model and service

Workflow jobs:
- `run-mlops-lifecycle` trains, tracks, packages, monitors, and uploads artifacts
- `smoke-test-bentoml-service` starts the BentoML API and checks `/readyz`, `/docs.json`, and `/predict`

When it runs:
- on every push to `main`
- on every pull request
- when started manually with GitHub Actions `workflow_dispatch`
