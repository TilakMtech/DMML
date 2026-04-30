"""Optional Prefect flow. Install prefect and run: prefect deploy or python this file."""
from prefect import flow, task
import subprocess, sys
TASKS=['src.utils.generate_sample_data','src.ingestion.ingest','src.validation.validate','src.preparation.prepare','src.features.build_features','src.feature_store.registry','src.training.train']
@task(retries=2)
def run(module):
    subprocess.check_call([sys.executable, '-m', module])
@flow(name='recomart-recommendation-pipeline')
def recomart_flow():
    for module in TASKS:
        run(module)
if __name__ == '__main__': recomart_flow()
