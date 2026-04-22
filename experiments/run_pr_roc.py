"""Build a joint PR/ROC figure for LSTM-AE and Isolation Forest on the
synthetic tasmax benchmark (seed 42), reporting AUPRC and AUROC for both.

Output: pr_roc_results/fig_pr_roc.png + metrics.json + metrics.csv
"""
from __future__ import annotations

import os
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    average_precision_score,
    roc_auc_score,
)

PROJECT_DIR = Path(r"C:\Users\meowcolate\Desktop\climate anomaly detection")
sys.path.insert(0, str(PROJECT_DIR))

from config.config import AppConfig
from src.data_loader import ClimateDataLoader
from models.lstm_autoencoder import LSTMAutoencoder
from models.isolation_forest import IsolationForestDetector

OUT_DIR = Path(__file__).resolve().parent / "pr_roc_results"
OUT_DIR.mkdir(exist_ok=True)

SEED = 42


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        tf.keras.backend.clear_session()
    except Exception:
        pass


def main() -> int:
    cfg = AppConfig()
    cfg.RANDOM_STATE = SEED
    loader = ClimateDataLoader(cfg)
    ds = loader.create_synthetic_dataset("tasmax")
    labels = loader.extract_labels(ds)
    series = loader.extract_time_series(ds, "tasmax")
    series = loader.handle_missing_values(series)
    prepared = loader.prepare_lstm_data(series, labels=labels)

    X_train = prepared["X_train"]
    X_val = prepared["X_val"]
    X_test = prepared["X_test"]
    test_labels = np.asarray(prepared["test_labels"]).astype(int)
    test_dates = prepared.get("test_dates")
    train_dates = prepared.get("train_dates")

    print(f"Test windows: {len(X_test)} (anomalous: {int(test_labels.sum())})")

    # --- LSTM scores (reconstruction MSE on test set) ---
    print("Training LSTM (seed 42)...")
    _set_seed(SEED)
    model = LSTMAutoencoder(cfg)
    model.output_prefix = "prroc_lstm"
    model.build_model((X_train.shape[1], X_train.shape[2]))
    model.train(X_train, X_val, verbose=0, save_checkpoint=False)
    lstm_scores = np.asarray(model.reconstruction_errors(X_test)["mse"])
    print(f"  LSTM score range: [{lstm_scores.min():.4f}, {lstm_scores.max():.4f}]")

    # --- IF scores (decision_function-based) ---
    print("Fitting Isolation Forest (seed 42)...")
    det = IsolationForestDetector(cfg)
    res = det.fit_and_detect(
        X_train, X_test,
        train_dates=train_dates,
        test_dates=test_dates,
        feature_mode="flatten_only",
    )
    if_scores = np.asarray(res["raw_scores"])  # higher = more anomalous
    print(f"  IF score range: [{if_scores.min():.4f}, {if_scores.max():.4f}]")

    # --- Curves and areas ---
    lstm_p, lstm_r, _ = precision_recall_curve(test_labels, lstm_scores)
    if_p, if_r, _ = precision_recall_curve(test_labels, if_scores)
    lstm_auprc = float(average_precision_score(test_labels, lstm_scores))
    if_auprc = float(average_precision_score(test_labels, if_scores))

    lstm_fpr, lstm_tpr, _ = roc_curve(test_labels, lstm_scores)
    if_fpr, if_tpr, _ = roc_curve(test_labels, if_scores)
    lstm_auroc = float(roc_auc_score(test_labels, lstm_scores))
    if_auroc = float(roc_auc_score(test_labels, if_scores))

    prevalence = float(test_labels.mean())
    print(f"LSTM AUPRC={lstm_auprc:.3f}  AUROC={lstm_auroc:.3f}")
    print(f"IF   AUPRC={if_auprc:.3f}  AUROC={if_auroc:.3f}")
    print(f"Positive prevalence in test set: {prevalence:.3f}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    ax = axes[0]
    ax.plot(lstm_r, lstm_p, color="#1f77b4", lw=2,
            label=f"LSTM AE (AUPRC={lstm_auprc:.3f})")
    ax.plot(if_r, if_p, color="#d62728", lw=2,
            label=f"Isolation Forest (AUPRC={if_auprc:.3f})")
    ax.axhline(prevalence, color="grey", ls="--", lw=1,
               label=f"Random baseline ({prevalence:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("(a) Precision-Recall Curve")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    ax = axes[1]
    ax.plot(lstm_fpr, lstm_tpr, color="#1f77b4", lw=2,
            label=f"LSTM AE (AUROC={lstm_auroc:.3f})")
    ax.plot(if_fpr, if_tpr, color="#d62728", lw=2,
            label=f"Isolation Forest (AUROC={if_auroc:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=1, label="Random baseline (0.500)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("(b) ROC Curve")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    out_png = OUT_DIR / "fig_pr_roc.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png}")

    # --- Persist numbers ---
    summary = {
        "seed": SEED,
        "prevalence": prevalence,
        "n_test_windows": int(len(test_labels)),
        "n_test_anomalous": int(test_labels.sum()),
        "lstm": {"auprc": lstm_auprc, "auroc": lstm_auroc},
        "isolation_forest": {"auprc": if_auprc, "auroc": if_auroc},
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame([
        {"model": "LSTM Autoencoder", "auprc": lstm_auprc, "auroc": lstm_auroc},
        {"model": "Isolation Forest", "auprc": if_auprc, "auroc": if_auroc},
    ]).to_csv(OUT_DIR / "metrics.csv", index=False)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
