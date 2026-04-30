from __future__ import annotations
import shutil, time, json
from pathlib import Path
import pandas as pd
from src.utils.common import ROOT, load_config, ensure_dirs, run_partition, setup_logger, save_json, utc_now

LOG = setup_logger('ingestion', 'logs/ingestion.log')

def retry(fn, tries=3, delay=1.0):
    last = None
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except Exception as e:
            last = e
            LOG.warning('attempt=%s failed error=%s', attempt, e)
            time.sleep(delay * attempt)
    raise last

def ingest_csv():
    cfg = load_config(); src = ROOT / cfg['ingestion']['batch_interactions_csv']
    part = run_partition(); dest_dir = ROOT / cfg['raw_root'] / 'source=csv' / 'type=interactions' / part
    ensure_dirs(dest_dir)
    dest = dest_dir / 'interactions.csv'
    def copy():
        if not src.exists(): raise FileNotFoundError(src)
        pd.read_csv(src, nrows=5)
        shutil.copy2(src, dest)
        return dest
    out = retry(copy)
    LOG.info('CSV ingestion success source=%s dest=%s', src, out)
    return str(out.relative_to(ROOT))

def ingest_products_api():
    cfg = load_config(); part = run_partition()
    dest_dir = ROOT / cfg['raw_root'] / 'source=rest_api' / 'type=products' / part
    ensure_dirs(dest_dir)
    dest = dest_dir / 'products.json'
    def fetch():
        # Offline-safe REST API ingestion. mock:// reads a JSON response captured from a REST-like endpoint.
        url = cfg['ingestion']['product_api_url']
        if url.startswith('mock://'):
            data = json.loads((ROOT / 'data/incoming/mock_products_api.json').read_text(encoding='utf-8'))
        else:
            import requests
            resp = requests.get(url, timeout=15); resp.raise_for_status(); data = resp.json()
        dest.write_text(json.dumps(data, indent=2), encoding='utf-8')
        return dest
    out = retry(fetch)
    LOG.info('REST API ingestion success dest=%s', out)
    return str(out.relative_to(ROOT))

def main():
    ensure_dirs('logs')
    outputs = {'run_ts': utc_now(), 'raw_files': [ingest_csv(), ingest_products_api()]}
    save_json(outputs, 'reports/ingestion_manifest.json')
    LOG.info('Ingestion manifest written')
    return outputs

if __name__ == '__main__':
    main()
