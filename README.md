# RecoMart End-to-End Data Management Pipeline for Recommendations

This repository implements Assignment I: a modular data management pipeline for an e-commerce recommendation system. It covers problem formulation, ingestion, raw storage, validation, preparation, transformation, feature management, data versioning/lineage, model training, and orchestration.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.orchestration.pipeline
python src/inference.py
```

The submitted ZIP already includes generated sample data, logs, validation reports, feature-store materialization, model artifact, and experiment metadata.

## Folder Structure

- `src/ingestion`: CSV and REST API ingestion with retries and logging.
- `src/validation`: automated data quality checks and PDF/JSON report generation.
- `src/preparation`: cleaning, encoding, normalization, and EDA plots.
- `src/features`: user, item, user-item, and co-occurrence feature transformations.
- `src/feature_store`: custom versioned feature registry and retrieval demo.
- `src/training`: SVD collaborative filtering model, ranking metrics, and MLflow-like metadata.
- `src/orchestration`: runnable DAG runner plus Airflow/Prefect templates.
- `data/raw`: partitioned immutable raw lake by source/type/timestamp.
- `data/prepared`, `data/transformed`, `data/feature_store`: curated datasets, warehouse, and versioned features.
- `reports`: PDF/JSON reports, EDA figures, feature retrieval samples, and model metrics.
- `docs`: SQL schema, feature metadata, lineage/versioning workflow, and demo video script.

## Model

The training stage builds a collaborative filtering recommender using truncated SVD over the user-item interaction matrix. It evaluates Precision@10, Recall@10, and NDCG@10 on a time-based holdout and stores artifacts in `models/` and `mlruns/<run_id>/`.

## Notes

The REST API source is implemented as an offline-safe `mock://recomart/products` endpoint backed by a JSON API response file, so the project is reproducible without internet access.
