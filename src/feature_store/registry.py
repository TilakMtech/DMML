"""
Custom feature store registry for the RecoMart recommendation pipeline.

This module implements a lightweight feature store that stores versioned
feature snapshots and metadata. It enables consistent feature retrieval for
training and inference.

Key responsibilities:
- Register engineered feature tables
- Store feature metadata
- Maintain versioned feature snapshots
- Provide retrieval examples for users and items
- Support reproducibility across training and inference
"""
from __future__ import annotations
import shutil, json
from pathlib import Path
import pandas as pd
from src.utils.common import ROOT, ensure_dirs, save_json, utc_now, setup_logger
LOG=setup_logger('feature_store','logs/feature_store.log')
FEATURE_TABLES=['user_features','item_features','user_item_features','item_cooccurrence']

def materialize(version=None):
    version=version or utc_now().replace(':','').replace('-','').replace('T','_').replace('Z','')
    base=ROOT/'data/feature_store'/f'version={version}'
    ensure_dirs(base)
    metadata={'version':version,'created_at':utc_now(),'features':[]}
    for table in FEATURE_TABLES:
        src=ROOT/f'data/transformed/{table}.csv'; dest=base/f'{table}.csv'
        shutil.copy2(src,dest)
        df=pd.read_csv(dest)
        metadata['features'].append({'table':table,'source':str(src.relative_to(ROOT)),'path':str(dest.relative_to(ROOT)),'row_count':len(df),'columns':list(df.columns)})
    save_json(metadata, base/'feature_metadata.json')
    save_json({'latest_version':version,'latest_path':str(base.relative_to(ROOT))}, 'data/feature_store/LATEST.json')
    LOG.info('Feature store materialized version=%s', version)
    return version

def retrieve(entity_type, ids, features=None, version='latest'):
    latest=json.loads((ROOT/'data/feature_store/LATEST.json').read_text()) if version=='latest' else {'latest_path':f'data/feature_store/version={version}'}
    base=ROOT/latest['latest_path']
    table='user_features' if entity_type=='user' else 'item_features'
    key='user_id' if entity_type=='user' else 'item_id'
    df=pd.read_csv(base/f'{table}.csv')
    out=df[df[key].isin(ids)]
    if features: out=out[[key]+features]
    return out

def demo():
    version=materialize()
    users=pd.read_csv(ROOT/'data/transformed/user_features.csv').user_id.head(3).tolist()
    items=pd.read_csv(ROOT/'data/transformed/item_features.csv').item_id.head(3).tolist()
    retrieve('user', users).to_csv(ROOT/'reports/feature_retrieval_users.csv', index=False)
    retrieve('item', items).to_csv(ROOT/'reports/feature_retrieval_items.csv', index=False)
    return version
if __name__=='__main__': demo()
