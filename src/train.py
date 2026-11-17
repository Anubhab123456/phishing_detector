"""Train & compare models, save the best one to models/model.joblib.

Usage:
    python -m src.train                    # synthetic dataset
    python -m src.train --csv my_data.csv  # your own dataset (url,label)
"""
import argparse
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.dataset import generate_dataset
from src.features import FEATURE_NAMES, features_to_vector

MODEL_PATH = "models/model.joblib"
METRICS_PATH = "models/metrics.json"


def build_matrix(urls):
    return np.vstack([features_to_vector(u) for u in urls])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="CSV with columns: url,label (1=phishing, 0=legit)")
    ap.add_argument("--n", type=int, default=6000, help="synthetic URLs per class")
    args = ap.parse_args()

    os.makedirs("models", exist_ok=True)
    if args.csv:
        df = pd.read_csv(args.csv)[["url", "label"]].dropna()
        print(f"Loaded {len(df)} rows from {args.csv}")
    else:
        df = generate_dataset(args.n)
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/urls.csv", index=False)
        print(f"Generated synthetic dataset: {len(df)} URLs")

    print("Extracting features ...")
    t0 = time.time()
    X = build_matrix(df["url"])
    y = df["label"].astype(int).values
    print(f"  {X.shape[0]} x {X.shape[1]} features in {time.time() - t0:.1f}s")

    urls_arr = df["url"].astype(str).to_numpy(dtype=object)
    X_tr, X_te, y_tr, y_te, u_tr, u_te = train_test_split(
        X, y, urls_arr, test_size=0.2, stratify=y, random_state=42)

    candidates = {
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                                n_jobs=-1, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    }

    results, fitted = {}, {}
    for name, model in candidates.items():
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        proba = model.predict_proba(X_te)[:, 1]
        results[name] = {
            "accuracy": accuracy_score(y_te, pred),
            "precision": precision_score(y_te, pred),
            "recall": recall_score(y_te, pred),
            "f1": f1_score(y_te, pred),
            "roc_auc": roc_auc_score(y_te, proba),
            "confusion_matrix": confusion_matrix(y_te, pred).tolist(),
        }
        fitted[name] = model
        print(f"{name:20s} acc={results[name]['accuracy']:.4f} f1={results[name]['f1']:.4f} "
              f"auc={results[name]['roc_auc']:.4f}")

    best_name = max(results, key=lambda k: results[k]["f1"])
    print(f"\nBest model: {best_name}")

    rf = fitted["Random Forest"]
    importance = dict(sorted(zip(FEATURE_NAMES, rf.feature_importances_.tolist()),
                             key=lambda kv: -kv[1]))

    # Baseline = mean feature vector of legitimate URLs (for local explanations)
    baseline = X_tr[y_tr == 0].mean(axis=0)

    # Held-out sample for the live-stream simulation
    rng = np.random.RandomState(0)
    idx = rng.choice(len(u_te), size=min(300, len(u_te)), replace=False)
    sample = [{"url": u_te[i], "label": int(y_te[i])} for i in idx]

    joblib.dump({"model": fitted[best_name], "model_name": best_name,
                 "feature_names": FEATURE_NAMES, "baseline": baseline}, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump({"best_model": best_name, "results": results,
                   "feature_importance": importance, "stream_sample": sample,
                   "n_train": int(len(y_tr)), "n_test": int(len(y_te))}, f, indent=2)
    print(f"Saved {MODEL_PATH} and {METRICS_PATH}")


if __name__ == "__main__":
    main()
