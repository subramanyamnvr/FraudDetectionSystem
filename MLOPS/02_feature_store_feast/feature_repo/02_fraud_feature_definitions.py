from datetime import timedelta

from feast import Entity, FeatureService, FeatureView, Field, FileSource
from feast.types import Float32, Int64

transaction_entity = Entity(name="transaction", join_keys=["transaction_id"])

fraud_transaction_source = FileSource(
    name="fraud_transaction_source",
    path="data/02_fraud_feature_source.parquet",
    timestamp_field="event_timestamp",
)

fraud_transaction_features = FeatureView(
    name="fraud_transaction_features",
    entities=[transaction_entity],
    ttl=timedelta(days=30),
    schema=[
        Field(name="v1", dtype=Float32),
        Field(name="v2", dtype=Float32),
        Field(name="v3", dtype=Float32),
        Field(name="v4", dtype=Float32),
        Field(name="v5", dtype=Float32),
        Field(name="v6", dtype=Float32),
        Field(name="v7", dtype=Float32),
        Field(name="v8", dtype=Float32),
        Field(name="v9", dtype=Float32),
        Field(name="v10", dtype=Float32),
        Field(name="v11", dtype=Float32),
        Field(name="v12", dtype=Float32),
        Field(name="v13", dtype=Float32),
        Field(name="v14", dtype=Float32),
        Field(name="v15", dtype=Float32),
        Field(name="v16", dtype=Float32),
        Field(name="v17", dtype=Float32),
        Field(name="v18", dtype=Float32),
        Field(name="v19", dtype=Float32),
        Field(name="v20", dtype=Float32),
        Field(name="v21", dtype=Float32),
        Field(name="v22", dtype=Float32),
        Field(name="v23", dtype=Float32),
        Field(name="v24", dtype=Float32),
        Field(name="v25", dtype=Float32),
        Field(name="v26", dtype=Float32),
        Field(name="v27", dtype=Float32),
        Field(name="v28", dtype=Float32),
        Field(name="amount", dtype=Float32),
    ],
    online=True,
    source=fraud_transaction_source,
    tags={"team": "fraud-detection", "step": "02_feature_store_feast"},
)

fraud_feature_service_v1 = FeatureService(
    name="fraud_feature_service_v1",
    features=[fraud_transaction_features],
)

