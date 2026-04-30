from __future__ import annotations
import subprocess, sys
from pathlib import Path
from src.utils.common import ROOT, setup_logger, save_json, utc_now
LOG=setup_logger('orchestration','logs/orchestration.log')
TASKS=[
 ('generate_sample_data','src.utils.generate_sample_data'),
 ('ingest','src.ingestion.ingest'),
 ('validate','src.validation.validate'),
 ('prepare','src.preparation.prepare'),
 ('features','src.features.build_features'),
 ('feature_store','src.feature_store.registry'),
 ('train','src.training.train')]

def run_task(name, module):
    LOG.info('START task=%s', name)
    res=subprocess.run([sys.executable,'-m',module], cwd=ROOT, text=True, capture_output=True)
    (ROOT/f'logs/{name}.stdout.log').write_text(res.stdout, encoding='utf-8')
    (ROOT/f'logs/{name}.stderr.log').write_text(res.stderr, encoding='utf-8')
    if res.returncode != 0:
        LOG.error('FAIL task=%s stderr=%s', name, res.stderr[-1000:])
        raise RuntimeError(f'{name} failed')
    LOG.info('SUCCESS task=%s', name)

def main():
    started=utc_now(); completed=[]
    for name, module in TASKS:
        run_task(name,module); completed.append(name)
    save_json({'orchestrator':'lightweight DAG runner; Airflow-compatible task order documented','started_at':started,'finished_at':utc_now(),'status':'success','tasks':completed}, 'reports/orchestration_run.json')
    LOG.info('PIPELINE SUCCESS')
if __name__=='__main__': main()
