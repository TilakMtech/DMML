"""
Feature engineering module for the RecoMart recommendation pipeline.

This module transforms prepared user-item events into features suitable for
recommendation algorithms. It creates user-level, item-level, user-item-level,
and co-occurrence features, then stores them in transformed datasets and a
structured warehouse.

Key responsibilities:
- Build user activity features
- Build item popularity and engagement features
- Build user-item affinity features
- Generate item co-occurrence features
- Store transformed outputs for training and feature-store registration
"""
from __future__ import annotations
import sqlite3, pandas as pd, numpy as np
from pathlib import Path
from src.utils.common import ROOT, ensure_dirs, setup_logger, save_json, utc_now
LOG=setup_logger('features','logs/features.log')

def main():
    """
    Executes the feature engineering stage.

    Reads prepared interaction data, creates recommendation features, writes
    transformed datasets, and updates the structured feature warehouse.
    """
    ensure_dirs('data/transformed')
    events=pd.read_csv(ROOT/'data/prepared/user_item_events.csv')
    events['event_ts']=pd.to_datetime(events['event_ts'], utc=True)
    user_features=events.groupby('user_id').agg(
        user_event_count=('item_id','count'), user_unique_items=('item_id','nunique'),
        user_avg_strength=('interaction_strength','mean'), user_total_spend=('price','sum'),
        user_last_event_ts=('event_ts','max')).reset_index()
    item_features=events.groupby('item_id').agg(
        item_event_count=('user_id','count'), item_unique_users=('user_id','nunique'),
        item_avg_strength=('interaction_strength','mean'), item_purchase_count=('event_type', lambda s: int((s=='purchase').sum())),
        item_avg_price=('price','mean')).reset_index()
    ui=events.groupby(['user_id','item_id']).agg(
        ui_event_count=('event_type','count'), ui_max_strength=('interaction_strength','max'), ui_last_event_ts=('event_ts','max')).reset_index()
    # Co-occurrence item similarity from users who interacted with pairs.
    basket=events[['user_id','item_id']].drop_duplicates().assign(v=1).pivot_table(index='user_id', columns='item_id', values='v', fill_value=0)
    co=basket.T.dot(basket)
    co = pd.DataFrame(co.to_numpy(copy=True), index=co.index, columns=co.columns)
    for i in range(len(co)):
        co.iat[i, i] = 0
    pairs=[]
    for item in co.index:
        for other, score in co.loc[item].nlargest(5).items():
            if score>0: pairs.append({'item_id':item,'similar_item_id':other,'cooccurrence_score':int(score)})
    co_features=pd.DataFrame(pairs)
    for name,df in [('user_features',user_features),('item_features',item_features),('user_item_features',ui),('item_cooccurrence',co_features)]:
        df.to_csv(ROOT/f'data/transformed/{name}.csv', index=False)
    db=ROOT/'data/transformed/recomart_warehouse.sqlite'
    with sqlite3.connect(db) as conn:
        for name,df in [('user_features',user_features),('item_features',item_features),('user_item_features',ui),('item_cooccurrence',co_features)]:
            df.to_sql(name, conn, if_exists='replace', index=False)
    save_json({'run_ts':utc_now(),'warehouse':str(db.relative_to(ROOT)),'feature_tables':{'user_features':len(user_features),'item_features':len(item_features),'user_item_features':len(ui),'item_cooccurrence':len(co_features)}}, 'reports/feature_summary.json')
    LOG.info('Feature engineering complete')
if __name__=='__main__': main()
