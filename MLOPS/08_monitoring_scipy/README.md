# 08 Monitoring

This step checks whether the served data still looks like the data used in training.

Restaurant story:
- the dish is already being served
- now the restaurant manager tastes random plates and checks whether the ingredients still match the original recipe
- if the taste starts changing, the kitchen knows it may be time to retrain

What this step does:
- loads the training-era reference features from step 2
- creates a production-like current batch from fresh records
- scores that batch with the saved model from step 6
- checks each feature for drift using simple statistical tests
- writes clear monitoring outputs for review

Files here:
- `08_run_monitoring_checks.py`
- `08_current_scored_batch.csv`
- `08_drift_report.csv`
- `08_monitoring_summary.json`

How it fits:
- BentoML serves predictions in step 7
- BentoML also exposes service health and metrics endpoints such as `/readyz` and `/metrics`
- this step adds model-input monitoring so we can see whether the incoming data shape is changing

Run:
- `python MLOPS/08_monitoring_scipy/08_run_monitoring_checks.py`

Outputs:
- `08_current_scored_batch.csv` contains the current batch with predictions
- `08_drift_report.csv` shows drift statistics for each feature
- `08_monitoring_summary.json` gives a short summary you can explain easily

