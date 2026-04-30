from __future__ import annotations
import numpy as np, pandas as pd
from joblib import load
from src.utils.common import ROOT

def recommend_for_user(user_id: str, top_k: int = 10):
    model=load(ROOT/'models/svd_recommender.joblib')
    users,items=model['users'],model['items']
    if user_id not in users:
        # cold-start fallback: popular items
        feats=pd.read_csv(ROOT/'data/transformed/item_features.csv')
        return feats.sort_values(['item_purchase_count','item_event_count'], ascending=False).item_id.head(top_k).tolist()
    idx=users.index(user_id)
    scores=model['user_embeddings'][idx].dot(model['item_embeddings'].T)
    return [items[i] for i in np.argsort(-scores)[:top_k]]

if __name__ == '__main__':
    print(recommend_for_user('U0001', 10))
