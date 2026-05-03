"""
Inference module for the RecoMart recommendation pipeline.

This module loads the trained recommendation model and generates Top-K product
recommendations for a given user.

Key responsibilities:
- Load trained SVD recommender artifact
- Accept user ID and Top-K value as inputs
- Rank candidate items by predicted relevance
- Return personalized recommendations
"""
from __future__ import annotations
import numpy as np, pandas as pd
from joblib import load
from src.utils.common import ROOT

def recommend_for_user(user_id: str, top_k: int = 10):
    """
    Generates Top-K recommendations for a user.

    Args:
        user_id: User identifier for whom recommendations are generated.
        top_k: Number of recommendations to return.

    Returns:
        List of recommended item IDs ranked by predicted relevance.
    """
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
