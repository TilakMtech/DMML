# 5-10 Minute Demo Video Script

1. Introduce RecoMart and the business objective: personalized product recommendations to improve engagement and cross-sell.
2. Show the repository structure and explain the modular pipeline stages.
3. Run `python -m src.orchestration.pipeline` and show the successful orchestration log.
4. Open `data/raw` to show partitioned raw storage by source, type, date, and hour.
5. Open `reports/data_quality_report.pdf` and discuss validation checks and issue handling.
6. Show EDA plots in `reports/figures`: event distribution, item popularity, and sparsity heatmap.
7. Show `docs/sql_schema.sql`, feature tables in SQLite, and `docs/feature_metadata.md`.
8. Demonstrate feature retrieval using `reports/feature_retrieval_users.csv` and `reports/feature_retrieval_items.csv`.
9. Open `reports/model_performance.json` and `mlruns/<run_id>/meta.json` to show experiment tracking.
10. Run `python src/inference.py` to display recommendations for a sample user.
