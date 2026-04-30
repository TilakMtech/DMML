"""Airflow DAG for RecoMart pipeline. Copy into AIRFLOW_HOME/dags and adjust PROJECT_ROOT."""
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime
PROJECT_ROOT = '/path/to/recomart_assignment'
with DAG('recomart_recommendation_pipeline', start_date=datetime(2025,1,1), schedule='@daily', catchup=False, tags=['dm4ml','recommendation']) as dag:
    generate = BashOperator(task_id='generate_sample_data', bash_command=f'cd {PROJECT_ROOT} && python -m src.utils.generate_sample_data')
    ingest = BashOperator(task_id='ingest_raw_data', bash_command=f'cd {PROJECT_ROOT} && python -m src.ingestion.ingest')
    validate = BashOperator(task_id='validate_data_quality', bash_command=f'cd {PROJECT_ROOT} && python -m src.validation.validate')
    prepare = BashOperator(task_id='prepare_clean_eda', bash_command=f'cd {PROJECT_ROOT} && python -m src.preparation.prepare')
    features = BashOperator(task_id='transform_features', bash_command=f'cd {PROJECT_ROOT} && python -m src.features.build_features')
    feature_store = BashOperator(task_id='materialize_feature_store', bash_command=f'cd {PROJECT_ROOT} && python -m src.feature_store.registry')
    train = BashOperator(task_id='train_evaluate_model', bash_command=f'cd {PROJECT_ROOT} && python -m src.training.train')
    generate >> ingest >> validate >> prepare >> features >> feature_store >> train
