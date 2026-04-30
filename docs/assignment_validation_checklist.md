# Assignment Validation Checklist

| Requirement | Status | Evidence |
|---|---:|---|
| Problem formulation PDF | Complete | `reports/problem_formulation_report.pdf` |
| At least two data sources | Complete | CSV interactions and REST API product metadata |
| Automated ingestion with retries/logs | Complete | `src/ingestion/ingest.py`, `logs/ingestion.log` |
| Raw storage partitioned by source/type/timestamp | Complete | `data/raw/source=*/type=*/dt=*/hour=*` |
| Data profiling and validation | Complete | `src/validation/validate.py`, `reports/data_quality_report.pdf` |
| Cleaning, encoding, normalization, EDA | Complete | `src/preparation/prepare.py`, `reports/figures/*` |
| Feature engineering and warehouse | Complete | `src/features/build_features.py`, `docs/sql_schema.sql`, SQLite warehouse |
| Feature store with metadata and versioned retrieval | Complete | `src/feature_store/registry.py`, `docs/feature_metadata.md` |
| Data versioning and lineage | Complete | `dvc.yaml`, `docs/versioning_lineage.md` |
| Model training and evaluation | Complete | `src/training/train.py`, `reports/model_performance.json`, `mlruns/*` |
| Orchestration DAG/code/logs | Complete | `src/orchestration/pipeline.py`, `dags/recomart_airflow_dag.py`, `reports/orchestration_run.json` |
| Documentation and demo | Complete | `README.md`, `reports/final_pipeline_report.pdf`, `docs/demo_video_script.md` |
