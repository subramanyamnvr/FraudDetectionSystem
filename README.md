# Fraud Detection MLOps Platform

This project is a beginner-friendly, end-to-end MLOps implementation built around a fraud detection use case. The goal is not only to train a classification model, but to show how a model moves through the full lifecycle: data versioning, feature management, orchestration, experiment tracking, packaging, serving, monitoring, retraining decisions, promotion, and rollout planning.

The project uses a simple fraud dataset and a compact modeling workflow so the MLOps concepts stay easy to follow. Instead of stopping at a notebook, the repository shows how the same model can be governed like a production system with open-source tools.

## What This Project Demonstrates

- reproducible data and pipeline versioning with `DVC`
- consistent training and serving features with `Feast`
- orchestrated ML workflows with `Prefect`
- experiment tracking and model registry with `MLflow`
- model packaging and API serving with `BentoML`
- post-deployment monitoring with statistical drift checks
- retraining decisions based on monitoring outputs
- promotion control using MLflow model aliases
- rollout planning using a staged deployment strategy
- CI/CD automation using GitHub Actions

## Project Story

I used the idea of a restaurant chain called **Model Masala** to make the lifecycle easier to understand:

- data is the ingredients
- features are the standard masalas
- training is the kitchen workflow
- MLflow is the recipe logbook
- BentoML is the serving counter
- monitoring is customer feedback and kitchen quality checks
- retraining is deciding when to cook a fresh batch
- promotion and rollout are deciding which dish becomes the official menu item and how it is introduced safely

## Repository Structure

The lifecycle is organized in numbered folders so each stage is easy to explain:

- `01_data_versioning_dvc`
- `02_feature_store_feast`
- `03_orchestration_prefect`
- `04_experiment_tracking_mlflow`
- `05_pipeline_versioning_dvc`
- `06_model_artifacts_joblib`
- `07_model_serving_bentoml`
- `08_monitoring_scipy`
- `09_retraining_trigger_policy`
- `10_ci_cd_github_actions`
- `11_deployment_promotion_mlflow`
- `12_rollout_strategy_bentoml`

More detail for each step is available in [MLOPS/README.md](MLOPS/README.md).

## Core Files

- `FraudClassifier.ipynb`: notebook version for learning the model workflow
- `requirements.txt`: project dependencies
- `MLOPS/03_orchestration_prefect/03_prefect_mlops_pipeline.py`: end-to-end pipeline
- `MLOPS/07_model_serving_bentoml/07_bentoml_service.py`: BentoML inference service

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full training pipeline:

```bash
python MLOPS/03_orchestration_prefect/03_prefect_mlops_pipeline.py
```

Reproduce the tracked DVC stage:

```bash
dvc repro MLOPS/05_pipeline_versioning_dvc/dvc.yaml:run_full_mlops_cycle
```

Start the BentoML service:

```bash
python MLOPS/07_model_serving_bentoml/07_export_bentoml_model.py
bentoml serve MLOPS/07_model_serving_bentoml/07_bentoml_service.py:FraudDetectionService
```

Run monitoring and policy checks:

```bash
python MLOPS/08_monitoring_scipy/08_run_monitoring_checks.py
python MLOPS/09_retraining_trigger_policy/09_decide_retraining.py
python MLOPS/11_deployment_promotion_mlflow/11_promote_model.py
python MLOPS/12_rollout_strategy_bentoml/12_prepare_rollout.py
```

## Why This Repo Is Useful

Many ML projects show only model training. This one is designed to help explain the bigger picture:

- how models are made reproducible
- how serving and training stay aligned
- how monitoring affects retraining
- how promotion and rollout decisions are controlled

It is meant to be simple enough for beginners, but structured enough to reflect how real MLOps systems are designed.
