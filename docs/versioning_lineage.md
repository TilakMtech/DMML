# Data Versioning and Lineage Workflow

This project uses a DVC-compatible structure. In an environment with DVC installed, run:

```bash
dvc init
dvc add data/raw data/prepared data/transformed data/feature_store
git add data/*.dvc .gitignore dvc.yaml dvc.lock
git commit -m "Version RecoMart datasets and pipeline outputs"
```

Each pipeline stage writes metadata to `reports/*.json` and logs to `logs/*.log`. The lineage chain is:

1. `data/incoming` synthetic/reproducible sources
2. `data/raw/source=<source>/type=<type>/dt=<date>/hour=<hour>` ingested immutable partitions
3. `data/prepared` cleaned and encoded datasets
4. `data/transformed` warehouse-ready feature tables and SQLite database
5. `data/feature_store/version=<timestamp>` versioned feature snapshot
6. `models` and `mlruns/<run_id>` model artifact and experiment metadata

Raw and transformed data snapshots are versioned by DVC using `dvc.yaml` and can be pushed to remote storage.
