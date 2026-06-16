# MLOps Fraud Risk Scoring Platform - Requirements

## 1. Project Summary

Build a production-style machine learning platform that detects fraud or high-risk transactions in near real time. The project should demonstrate the full MLOps lifecycle: data ingestion, feature engineering, training, evaluation, model registry, deployment, monitoring, retraining, governance, and reproducibility.

This project should feel like a serious ML systems portfolio piece, not a toy notebook. The final system should support batch training and real-time inference, with clear observability and operational controls.

## 2. Core Goal

The platform should answer one question well:

**Given a transaction or event, how risky is it, and what action should the business take?**

Examples of actions:
- approve
- flag for review
- require additional verification
- block

## 3. Primary Use Case

Recommended domain: **payments fraud detection**.

Why this domain:
- easy to explain to recruiters
- naturally includes imbalanced classification
- has clear business impact
- supports online scoring, drift monitoring, and threshold tuning
- maps well to real MLOps practices

Optional alternate domains:
- credit risk scoring
- account takeover detection
- SaaS churn risk
- marketplace abuse detection

## 4. Success Criteria

The project is successful if it demonstrates all of the following:
- A repeatable data pipeline that creates training and inference features.
- An offline training pipeline with experiment tracking.
- A model registry with versioned models and promotion rules.
- A real-time prediction service with low-latency scoring.
- A batch scoring job for historical backfills or nightly runs.
- Monitoring for data drift, feature drift, and prediction quality.
- Retraining triggers or a retraining workflow.
- Explainability for individual predictions.
- Strong engineering hygiene: tests, containers, CI, docs, and reproducibility.

## 5. Functional Requirements

### 5.1 Data Ingestion

- Ingest historical labeled data from CSV, Parquet, SQL, or public fraud datasets.
- Ingest streaming or near-real-time events through an API or queue simulation.
- Store raw data in a durable location.
- Separate raw, cleaned, curated, training, and inference-ready datasets.

### 5.2 Data Validation

- Validate schema, null rates, ranges, categorical values, and duplicate records.
- Detect label leakage and obvious target contamination.
- Fail the pipeline or quarantine bad data when validation thresholds are violated.
- Produce a validation report for every dataset run.

### 5.3 Feature Engineering

- Create reusable features for both offline training and online inference.
- Support time-aware features such as rolling counts, velocity metrics, and recency features.
- Keep feature definitions centralized to avoid training-serving skew.
- Track feature versions and feature lineage.

### 5.4 Training Pipeline

- Train at least one baseline model and one stronger model.
- Support class imbalance handling.
- Run cross-validation or time-based validation where appropriate.
- Log hyperparameters, metrics, and artifacts for every run.
- Save the full training context: code version, dataset version, feature version, and environment details.

### 5.5 Evaluation

- Evaluate using fraud-friendly metrics, not just accuracy.
- Required metrics:
  - precision
  - recall
  - F1
  - ROC-AUC
  - PR-AUC
  - confusion matrix
  - calibration or threshold performance
- Compare model versions against a baseline.
- Choose an operating threshold based on business tradeoffs.

### 5.6 Model Registry

- Register model versions with metadata.
- Store training data version, feature set version, metrics, and explanation artifacts.
- Support model promotion states such as:
  - staging
  - production
  - archived
- Only allow promotion when evaluation gates are satisfied.

### 5.7 Serving

- Provide a real-time inference API.
- Provide a batch inference job.
- Return a prediction score, decision label, model version, and explanation summary.
- Keep the API fast and stateless where possible.
- Support safe fallback behavior if the model service is unavailable.

### 5.8 Monitoring

- Monitor prediction volume, latency, error rate, and service uptime.
- Monitor feature drift and data drift.
- Monitor score distribution changes.
- Track post-deployment performance when labels become available later.
- Emit alerts when thresholds are exceeded.

### 5.9 Retraining

- Support scheduled retraining.
- Support retraining triggered by drift, performance decay, or new labeled data.
- Compare challenger models against the current production model.
- Require promotion approval or automated quality gates before deployment.

### 5.10 Explainability

- Generate explanations for individual predictions.
- Show top contributing features for each score.
- Expose explanation artifacts in API responses or a dashboard.
- Make explanations understandable to a non-ML stakeholder.

## 6. MLOps Lifecycle Coverage

The project should explicitly demonstrate these MLOps concepts:

- problem framing
- data collection
- labeling strategy
- data cleaning
- feature engineering
- exploratory analysis
- dataset versioning
- training and validation
- experiment tracking
- hyperparameter tuning
- model registry
- CI/CD for ML
- deployment to staging and production
- monitoring and alerting
- drift detection
- feedback loops
- retraining and redeployment
- rollback strategy
- reproducibility
- governance and auditability

## 7. Suggested System Architecture

### 7.1 Components

- `data/` for raw and processed datasets
- `pipelines/` for ETL, feature generation, and training workflows
- `training/` for model training and evaluation code
- `registry/` or managed registry for model versions
- `serving/` for real-time API and batch scoring
- `monitoring/` for drift, metrics, and alerts
- `dashboard/` for operational visibility
- `infra/` for container and deployment configuration

### 7.2 Recommended Stack

Use a practical stack that looks modern in 2026:

- Python
- FastAPI for inference API
- scikit-learn, XGBoost, LightGBM, or CatBoost for strong tabular baselines
- MLflow for experiment tracking and model registry
- Feast or a lightweight feature store pattern for feature reuse
- Great Expectations or similar for data validation
- Evidently or comparable tooling for drift monitoring
- Docker for local reproducibility
- GitHub Actions for CI
- PostgreSQL or SQLite for metadata in early stages
- Redis or a queue only if async workflows are needed
- Streamlit or a simple React dashboard for demo visibility

## 8. Model Scope

Start with tabular data.

Recommended model progression:
1. Logistic regression baseline.
2. Tree-based model such as XGBoost, LightGBM, or CatBoost.
3. Optional calibrated ensemble.

The project should show why the better model wins in business terms, not just metric terms.

## 9. Data Requirements

The dataset should have:
- transaction or event identifier
- timestamp
- entity identifiers such as user, merchant, device, or account
- numeric features
- categorical features
- label indicating fraud or risk outcome
- enough imbalance to make the problem realistic

If using public data, document:
- source
- license
- preprocessing steps
- label definition
- time split strategy

## 10. Validation and Quality Gates

Add quality gates at each major stage.

Examples:
- training data must pass schema validation
- missing values must stay below a threshold
- evaluation metrics must exceed baseline
- drift on critical features must stay below threshold
- model latency must stay below service budget
- model promotion must require passing all gates

## 11. Testing Requirements

### 11.1 Unit Tests

- feature computation
- preprocessing logic
- metric calculations
- threshold decision logic
- API request/response validation

### 11.2 Integration Tests

- training pipeline end-to-end on a small sample
- model registration and retrieval
- serving endpoint returns valid predictions
- monitoring pipeline processes sample drift data

### 11.3 Data Tests

- schema consistency
- data range checks
- duplicate detection
- label distribution checks

## 12. Observability Requirements

The project should expose:
- request counts
- inference latency
- error counts
- model version usage
- drift metrics
- score distribution
- approval/decline rate
- false positive/false negative trend when labels arrive

Add a dashboard or at least a structured report that makes the system easy to demo.

## 13. Deployment Requirements

- Everything should run locally with Docker.
- The project should support one-command startup for the core stack.
- The inference API should be containerized.
- Training should be reproducible from a CLI or pipeline entrypoint.
- Production-like config should use environment variables.

Optional but strong portfolio upgrade:
- deploy inference API to a cloud service
- expose a public demo endpoint
- add GitHub Actions for tests and build verification

## 14. Security and Governance

- Do not store secrets in source control.
- Use environment variables or secret managers.
- Log model decisions and important metadata for auditability.
- Keep a clear record of model versions and promotion history.
- Consider basic fairness checks if the dataset contains sensitive proxies.

## 15. Documentation Requirements

The repo should include:
- project overview
- architecture diagram
- local setup instructions
- data source documentation
- training instructions
- serving instructions
- monitoring explanation
- model evaluation summary
- retraining strategy
- limitations and future work

## 16. Portfolio Story

The README should tell a clear story:
- what business problem is being solved
- why the problem matters
- what makes the ML system production-like
- how the MLOps lifecycle is represented
- how to run the demo
- what technical choices were made and why

This matters as much as the code.

## 17. MVP Scope

Minimum viable version:
- one dataset
- one baseline model
- one stronger model
- offline training pipeline
- model registry
- FastAPI scoring endpoint
- one monitoring report
- one simple dashboard
- Docker-based local setup

## 18. Stretch Goals

If time allows, add:
- online feature store
- streaming ingestion simulation
- automated retraining
- champion/challenger comparison
- SHAP explanations
- model cards
- data lineage tracking
- canary deployment or shadow deployment
- alerting to Slack or email

## 19. Suggested Milestones

### Milestone 1
- define the problem, dataset, and success metrics
- build the repository structure
- add data validation

### Milestone 2
- implement feature engineering
- train baseline and improved models
- add experiment tracking

### Milestone 3
- register model versions
- expose inference API
- create batch scoring path

### Milestone 4
- add monitoring and drift detection
- add explanations
- add retraining workflow

### Milestone 5
- harden deployment
- add tests and CI
- polish documentation and dashboard

## 20. Acceptance Criteria

The project is ready for portfolio use when:
- a reviewer can clone and run it locally
- the training pipeline runs end to end
- the model can be scored through an API
- model metadata is tracked and reproducible
- monitoring outputs are visible
- the README clearly explains the ML lifecycle
- the system demonstrates real MLOps, not just model training

## 21. Recommended Repo Name

Possible names:
- `fraud-mlops-platform`
- `risk-score-lab`
- `real-time-risk-engine`
- `ml-risk-ops`

Recommended choice: **`fraud-mlops-platform`**

