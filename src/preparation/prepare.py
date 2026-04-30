from __future__ import annotations
import glob
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from src.utils.common import ROOT, load_config, setup_logger, save_json, ensure_dirs, utc_now
LOG=setup_logger('preparation','logs/preparation.log')

def latest(pattern):
    files=sorted(glob.glob(str(ROOT/pattern)))
    if not files: raise FileNotFoundError(pattern)
    return Path(files[-1])

def main():
    ensure_dirs('data/prepared','reports/figures')
    interactions=pd.read_csv(latest('data/raw/source=csv/type=interactions/dt=*/hour=*/interactions.csv'))
    products=pd.read_json(latest('data/raw/source=rest_api/type=products/dt=*/hour=*/products.json'))
    interactions=interactions.drop_duplicates().copy()
    interactions['event_ts']=pd.to_datetime(interactions['event_ts'], errors='coerce', utc=True)
    interactions['price']=pd.to_numeric(interactions['price'], errors='coerce')
    interactions['quantity']=pd.to_numeric(interactions['quantity'], errors='coerce').fillna(1).clip(lower=1)
    interactions=interactions.dropna(subset=['user_id','item_id','event_type','event_ts'])
    interactions=interactions[interactions.price >= 0]
    interactions['rating']=pd.to_numeric(interactions['rating'], errors='coerce')
    # Implicit feedback target: purchase=5, cart=3, view=1; explicit ratings preserved.
    interactions['interaction_strength']=interactions['rating'].fillna(interactions['event_type'].map({'view':1,'cart':3,'purchase':5,'rating':4})).astype(float)
    interactions['event_date']=interactions['event_ts'].dt.date.astype(str)
    interactions['event_hour']=interactions['event_ts'].dt.hour
    products=products.drop_duplicates('item_id').copy()
    for col in ['price','sentiment_score','popularity_score']:
        products[col]=pd.to_numeric(products[col], errors='coerce')
    products=products.dropna(subset=['item_id','category','brand'])
    scaler=MinMaxScaler()
    products[['price_norm','sentiment_norm','popularity_norm']]=scaler.fit_transform(products[['price','sentiment_score','popularity_score']])
    products_encoded=pd.get_dummies(products, columns=['category','brand'], prefix=['cat','brand'])
    prepared=interactions.merge(products, on='item_id', how='left', suffixes=('','_product'))
    interactions.to_csv(ROOT/'data/prepared/interactions_clean.csv', index=False)
    products.to_csv(ROOT/'data/prepared/products_clean.csv', index=False)
    products_encoded.to_csv(ROOT/'data/prepared/products_encoded.csv', index=False)
    prepared.to_csv(ROOT/'data/prepared/user_item_events.csv', index=False)
    # EDA plots
    plt.figure(figsize=(7,4)); interactions['event_type'].value_counts().plot(kind='bar'); plt.title('Interaction distribution by event type'); plt.tight_layout(); plt.savefig(ROOT/'reports/figures/interaction_distribution.png'); plt.close()
    plt.figure(figsize=(7,4)); interactions['item_id'].value_counts().head(20).plot(kind='bar'); plt.title('Top 20 item popularity'); plt.tight_layout(); plt.savefig(ROOT/'reports/figures/item_popularity_top20.png'); plt.close()
    matrix=interactions.pivot_table(index='user_id', columns='item_id', values='interaction_strength', aggfunc='max', fill_value=0)
    sparsity=1 - (np.count_nonzero(matrix.values)/(matrix.shape[0]*matrix.shape[1]))
    plt.figure(figsize=(6,5)); plt.imshow(matrix.iloc[:40,:40], aspect='auto'); plt.title('User-item matrix sample heatmap'); plt.xlabel('Items'); plt.ylabel('Users'); plt.tight_layout(); plt.savefig(ROOT/'reports/figures/sparsity_heatmap.png'); plt.close()
    save_json({'run_ts':utc_now(),'clean_interactions':len(interactions),'clean_products':len(products),'user_item_sparsity':round(float(sparsity),4),'plots':['reports/figures/interaction_distribution.png','reports/figures/item_popularity_top20.png','reports/figures/sparsity_heatmap.png']},'reports/preparation_summary.json')
    LOG.info('Preparation complete interactions=%s products=%s sparsity=%s', len(interactions), len(products), sparsity)
if __name__=='__main__': main()
