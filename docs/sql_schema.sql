CREATE TABLE user_features (
  user_id TEXT PRIMARY KEY,
  user_event_count INTEGER,
  user_unique_items INTEGER,
  user_avg_strength REAL,
  user_total_spend REAL,
  user_last_event_ts TEXT
);
CREATE TABLE item_features (
  item_id TEXT PRIMARY KEY,
  item_event_count INTEGER,
  item_unique_users INTEGER,
  item_avg_strength REAL,
  item_purchase_count INTEGER,
  item_avg_price REAL
);
CREATE TABLE user_item_features (
  user_id TEXT,
  item_id TEXT,
  ui_event_count INTEGER,
  ui_max_strength REAL,
  ui_last_event_ts TEXT,
  PRIMARY KEY (user_id, item_id)
);
CREATE TABLE item_cooccurrence (
  item_id TEXT,
  similar_item_id TEXT,
  cooccurrence_score INTEGER,
  PRIMARY KEY (item_id, similar_item_id)
);
