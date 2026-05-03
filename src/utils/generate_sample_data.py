"""
Sample data generation module for the RecoMart recommendation pipeline.

This module creates synthetic but reproducible datasets for user interactions
and product metadata. It supports offline execution of the full pipeline
without requiring external systems.

Key responsibilities:
- Generate user-item interaction records
- Generate product metadata
- Write input files under data/incoming
- Ensure reproducibility using a fixed random seed
"""

from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
from datetime import datetime, timedelta, timezone
from src.utils.common import ROOT, ensure_dirs, set_seed, load_config

EVENTS = ['view', 'cart', 'purchase', 'rating']
CATEGORIES = ['electronics', 'home', 'fashion', 'books', 'sports', 'beauty']
BRANDS = ['Nova', 'Aster', 'ZenCo', 'UrbanLeaf', 'Pulse', 'Orion']

def generate(n_users=240, n_items=120, n_events=3500):
    cfg = load_config(); set_seed(cfg['random_seed'])
    ensure_dirs('data/incoming')
    users = [f'U{u:04d}' for u in range(1, n_users+1)]
    items = [f'I{i:04d}' for i in range(1, n_items+1)]
    user_pref = {u: np.random.choice(CATEGORIES, p=np.random.dirichlet(np.ones(len(CATEGORIES)))) for u in users}
    products = []
    for item in items:
        cat = np.random.choice(CATEGORIES)
        price = round(float(np.random.lognormal(3.4, .55)), 2)
        products.append({
            'item_id': item, 'category': cat, 'brand': np.random.choice(BRANDS), 'price': price,
            'sentiment_score': round(float(np.clip(np.random.normal(.68, .18), 0, 1)), 3),
            'popularity_score': round(float(np.random.beta(2, 4)), 3)
        })
    pdf = pd.DataFrame(products)
    rows = []
    start = datetime.now(timezone.utc) - timedelta(days=90)
    for _ in range(n_events):
        u = np.random.choice(users)
        if np.random.rand() < .65:
            candidates = pdf[pdf.category == user_pref[u]].item_id.values
            item = np.random.choice(candidates if len(candidates) else items)
        else:
            item = np.random.choice(items)
        prod = pdf.loc[pdf.item_id == item].iloc[0]
        event = np.random.choice(EVENTS, p=[.55, .18, .15, .12])
        rating = np.nan if event != 'rating' else int(np.random.choice([1,2,3,4,5], p=[.04,.08,.18,.38,.32]))
        qty = 1 if event != 'purchase' else int(np.random.choice([1,2,3], p=[.78,.17,.05]))
        ts = start + timedelta(minutes=int(np.random.randint(0, 90*24*60)))
        rows.append({'user_id': u, 'item_id': item, 'event_type': event, 'rating': rating, 'quantity': qty,
                     'price': prod.price, 'event_ts': ts.strftime('%Y-%m-%dT%H:%M:%SZ'), 'channel': np.random.choice(['web','mobile'])})
    df = pd.DataFrame(rows)
    # Add controlled quality issues to demonstrate validation/remediation.
    df = pd.concat([df, df.iloc[:7]], ignore_index=True)
    df.loc[3, 'event_ts'] = 'bad_timestamp'
    df.loc[8, 'price'] = -5
    df.to_csv(ROOT / 'data/incoming/interactions.csv', index=False)
    pdf.to_json(ROOT / 'data/incoming/mock_products_api.json', orient='records', indent=2)
    print('Generated data/incoming/interactions.csv and mock_products_api.json')

if __name__ == '__main__':
    generate()
