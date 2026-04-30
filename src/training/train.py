from __future__ import annotations

import uuid
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from joblib import dump
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from src.utils.common import ROOT, load_config, ensure_dirs, setup_logger, save_json, utc_now

LOG = setup_logger("training", "logs/training.log")


def precision_recall_ndcg_at_k(test_items, ranked, k):
    top = ranked[:k]
    hits = [1 if i in test_items else 0 for i in top]

    precision = sum(hits) / k
    recall = sum(hits) / max(len(test_items), 1)

    dcg = sum(h / np.log2(idx + 2) for idx, h in enumerate(hits))
    ideal = sum(1 / np.log2(idx + 2) for idx in range(min(len(test_items), k)))
    ndcg = dcg / ideal if ideal else 0

    return precision, recall, ndcg


def main():
    cfg = load_config()
    ensure_dirs("models", "mlruns", "reports")

    mlflow.set_tracking_uri(f"file:{ROOT / 'mlruns'}")
    mlflow.set_experiment("RecoMart Recommendation Experiment")

    events_path = ROOT / "data/prepared/user_item_events.csv"
    events = pd.read_csv(events_path)
    events["event_ts"] = pd.to_datetime(events["event_ts"], utc=True)

    cutoff = events["event_ts"].max() - pd.Timedelta(days=cfg["model"]["test_days"])
    train = events[events.event_ts < cutoff]
    test = events[events.event_ts >= cutoff]

    mat = train.pivot_table(
        index="user_id",
        columns="item_id",
        values="interaction_strength",
        aggfunc="max",
        fill_value=0,
    )

    users = list(mat.index)
    items = list(mat.columns)

    n_components = min(
        cfg["model"]["n_components"],
        max(2, min(mat.shape) - 1),
    )

    with mlflow.start_run(run_name="svd_recommender") as run:
        mlflow.log_param("model_type", "Collaborative Filtering - TruncatedSVD")
        mlflow.log_param("n_components", n_components)
        mlflow.log_param("top_k", cfg["model"]["top_k"])
        mlflow.log_param("test_days", cfg["model"]["test_days"])
        mlflow.log_param("random_seed", cfg["random_seed"])
        mlflow.log_param("training_dataset", str(events_path))
        mlflow.log_param("train_rows", int(len(train)))
        mlflow.log_param("test_rows", int(len(test)))

        svd = TruncatedSVD(
            n_components=n_components,
            random_state=cfg["random_seed"],
        )

        user_emb = svd.fit_transform(csr_matrix(mat.values))
        item_emb = svd.components_.T
        scores = user_emb.dot(item_emb.T)

        k = cfg["model"]["top_k"]
        metrics = []

        user_to_idx = {u: i for i, u in enumerate(users)}
        test_pos = (
            test[test.interaction_strength >= 3]
            .groupby("user_id")["item_id"]
            .apply(set)
            .to_dict()
        )
        train_seen = train.groupby("user_id")["item_id"].apply(set).to_dict()

        for u, truth in test_pos.items():
            if u not in user_to_idx:
                continue

            s = scores[user_to_idx[u]].copy()

            for seen in train_seen.get(u, set()):
                if seen in items:
                    s[items.index(seen)] = -np.inf

            ranked = [items[i] for i in np.argsort(-s)]
            p, r, n = precision_recall_ndcg_at_k(truth, ranked, k)

            metrics.append(
                {
                    "user_id": u,
                    "precision_at_k": p,
                    "recall_at_k": r,
                    "ndcg_at_k": n,
                }
            )

        mdf = pd.DataFrame(metrics)

        aggregate = {
            "precision_at_10": float(mdf.precision_at_k.mean()) if len(mdf) else 0,
            "recall_at_10": float(mdf.recall_at_k.mean()) if len(mdf) else 0,
            "ndcg_at_10": float(mdf.ndcg_at_k.mean()) if len(mdf) else 0,
            "evaluated_users": int(len(mdf)),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
        }

        model = {
            "users": users,
            "items": items,
            "user_embeddings": user_emb,
            "item_embeddings": item_emb,
            "explained_variance_ratio": svd.explained_variance_ratio_,
        }

        model_path = ROOT / "models/svd_recommender.joblib"
        user_metrics_path = ROOT / "reports/model_user_metrics.csv"
        performance_path = ROOT / "reports/model_performance.json"
        metadata_path = ROOT / "reports/model_metadata.json"

        dump(model, model_path)
        mdf.to_csv(user_metrics_path, index=False)
        save_json(aggregate, performance_path)

        for metric_name, metric_value in aggregate.items():
            if isinstance(metric_value, (int, float)):
                mlflow.log_metric(metric_name, metric_value)

        mlflow.sklearn.log_model(svd, artifact_path="svd_estimator")
        mlflow.log_artifact(str(model_path), artifact_path="model_artifacts")
        mlflow.log_artifact(str(performance_path), artifact_path="reports")
        mlflow.log_artifact(str(user_metrics_path), artifact_path="reports")
        mlflow.log_artifact(str(events_path), artifact_path="datasets")

        run_id = run.info.run_id

        model_metadata = {
            "tracking_tool": "MLflow",
            "experiment_name": "RecoMart Recommendation Experiment",
            "run_id": run_id,
            "created_at": utc_now(),
            "model_name": "SVD Collaborative Filtering Recommender",
            "model_type": "Matrix Factorization / TruncatedSVD",
            "parameters": {
                "n_components": n_components,
                "top_k": cfg["model"]["top_k"],
                "test_days": cfg["model"]["test_days"],
                "random_seed": cfg["random_seed"],
            },
            "metrics": aggregate,
            "artifacts": {
                "model_file": "models/svd_recommender.joblib",
                "performance_report": "reports/model_performance.json",
                "user_metrics": "reports/model_user_metrics.csv",
                "mlflow_tracking_dir": "mlruns/",
            },
            "data_lineage": {
                "training_dataset": "data/prepared/user_item_events.csv",
                "feature_inputs": [
                    "data/transformed/user_features.csv",
                    "data/transformed/item_features.csv",
                    "data/transformed/user_item_features.csv",
                ],
            },
        }

        save_json(model_metadata, metadata_path)
        mlflow.log_artifact(str(metadata_path), artifact_path="reports")

        LOG.info("Training complete metrics=%s mlflow_run_id=%s", aggregate, run_id)


if __name__ == "__main__":
    main()
