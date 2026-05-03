# Feature Metadata Registry

| Table | Entity | Features | Source | Purpose |
|---|---|---|---|---|
| user_features | user_id | user_event_count, user_unique_items, user_avg_strength, user_total_spend, user_last_event_ts | prepared user-item events | User activity and recency signals |
| item_features | item_id | item_event_count, item_unique_users, item_avg_strength, item_purchase_count, item_avg_price | prepared user-item events | Item popularity and conversion signals |
| user_item_features | user_id, item_id | ui_event_count, ui_max_strength, ui_last_event_ts | prepared user-item events | User-item affinity for ranking |
| item_cooccurrence | item_id, similar_item_id | cooccurrence_score | user-item baskets | Similar item candidates |

Feature store versions are stored under `data/feature_store/version=<timestamp>/`. `LATEST.json` points to the current materialized version. Retrieval is supported for training and inference through `src/feature_store/registry.py`.

Features are used consistently across training and inference to ensure no feature skew.
