# 07 Model Serving

This step serves the trained model as an API with BentoML.

Restaurant story:
- BentoML is the serving counter of the restaurant
- the kitchen has already cooked and packed the dish, and BentoML now delivers it consistently to each customer request

Files here:
- `07_export_bentoml_model.py`
- `07_bentoml_service.py`

How it fits:
- step 6 creates the packed model files
- this step loads those files into BentoML
- BentoML then exposes a prediction endpoint that an app or another service can call

Run order:
- `python MLOPS/07_model_serving/07_export_bentoml_model.py`
- `bentoml serve MLOPS/07_model_serving/07_bentoml_service.py:FraudDetectionService`

Useful endpoints after startup:
- `GET /docs.json`
- `POST /predict`
- `GET /readyz`
- `GET /metrics`
