"""Run W3 (5-seed LSTM/IF sweep + threshold percentile sweep) and
W4 (clean vs contaminated LSTM training) experiments.

Writes JSON + CSV artefacts next to this script.
"""
from __future__ import annotations

import os
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(r"C:\Users\meowcolate\Desktop\climate anomaly detection")
sys.path.insert(0, str(PROJECT_DIR))

from config.config import AppConfig
from src.data_loader import ClimateDataLoader
from models.lstm_autoencoder import LSTMAutoencoder
from models.isolation_forest import IsolationForestDetector


OUT_DIR = Path(__file__).resolve().parent / "w3w4_results"
OUT_DIR.mkdir(exist_ok=True)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        tf.keras.backend.clear_session()
    except Exception:
        pass


def _copy_config(base: AppConfig, **overrides) -> AppConfig:
    new = AppConfig()
    for k, v in base.__dict__.items():
        setattr(new, k, v)
    for k, v in overrides.items():
        setattr(new, k, v)
    return new


def _metrics(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    return {
        "model": name,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision, "recall": recall,
        "f1_score": f1, "accuracy": accuracy,
        "anomaly_count": int(y_pred.sum()),
    }


def _summary_stats(rows: list[dict], fields: list[str]) -> dict:
    out = {}
    for f in fields:
        vals = np.array([r[f] for r in rows], dtype=float)
        out[f"{f}_mean"] = float(vals.mean())
        out[f"{f}_std"] = float(vals.std(ddof=0))
    out["n_seeds"] = len(rows)
    return out


def main() -> int:
    cfg = AppConfig()
    loader = ClimateDataLoader(cfg)
    ds = loader.create_synthetic_dataset("tasmax")
    labels = loader.extract_labels(ds)
    series = loader.extract_time_series(ds, "tasmax")
    series = loader.handle_missing_values(series)
    prepared = loader.prepare_lstm_data(series, labels=labels)

    X_train = prepared["X_train"]
    X_val = prepared["X_val"]
    X_test = prepared["X_test"]
    train_labels = prepared["train_labels"]
    test_labels = prepared["test_labels"]

    print(f"Train windows: {len(X_train)} (anomalous: {int(train_labels.sum())})")
    print(f"Val windows: {len(X_val)}")
    print(f"Test windows: {len(X_test)} (anomalous: {int(test_labels.sum())})")

    # Clean training split: drop any training window that contains an anomalous day.
    clean_mask = (train_labels == 0)
    X_train_clean = X_train[clean_mask]
    print(f"Clean training windows: {len(X_train_clean)}")

    SEEDS = [42, 84, 126, 168, 210]
    THRESH_PCTS = [90, 92, 95, 97, 99]

    lstm_rows = []       # 5-seed on contaminated training (default)
    lstm_clean_rows = [] # 5-seed on clean training (normal-only)
    thresh_rows = []     # threshold sweep on seed 42 contaminated model

    for seed in SEEDS:
        print(f"\n=== LSTM seed {seed} (contaminated training) ===")
        t0 = time.time()
        _set_seed(seed)
        sub_cfg = _copy_config(cfg, RANDOM_STATE=seed)
        model = LSTMAutoencoder(sub_cfg)
        model.output_prefix = f"w3_seed{seed}"
        model.build_model((X_train.shape[1], X_train.shape[2]))
        model.train(X_train, X_val, verbose=0, save_checkpoint=False)
        errs = model.reconstruction_errors(X_test)["mse"]
        thr = float(np.percentile(errs, 95))
        preds = errs > thr
        m = _metrics("LSTM Autoencoder", test_labels, preds)
        m.update({"seed": seed, "threshold_percentile": 95, "training_mode": "contaminated",
                  "threshold": thr, "train_time_sec": round(time.time() - t0, 1)})
        lstm_rows.append(m)
        print(f"  F1={m['f1_score']:.4f} P={m['precision']:.4f} R={m['recall']:.4f} "
              f"time={m['train_time_sec']}s")

        # For seed 42: reuse reconstruction errors for threshold sweep.
        if seed == 42:
            for pct in THRESH_PCTS:
                th = float(np.percentile(errs, pct))
                p = errs > th
                mm = _metrics("LSTM Autoencoder", test_labels, p)
                mm.update({"seed": seed, "threshold_percentile": pct, "threshold": th})
                thresh_rows.append(mm)
                print(f"  [sweep] pct={pct} F1={mm['f1_score']:.4f}")

    for seed in SEEDS:
        print(f"\n=== LSTM seed {seed} (clean training) ===")
        t0 = time.time()
        _set_seed(seed)
        sub_cfg = _copy_config(cfg, RANDOM_STATE=seed)
        model = LSTMAutoencoder(sub_cfg)
        model.output_prefix = f"w4_clean_seed{seed}"
        model.build_model((X_train_clean.shape[1], X_train_clean.shape[2]))
        model.train(X_train_clean, X_val, verbose=0, save_checkpoint=False)
        errs = model.reconstruction_errors(X_test)["mse"]
        thr = float(np.percentile(errs, 95))
        preds = errs > thr
        m = _metrics("LSTM Autoencoder", test_labels, preds)
        m.update({"seed": seed, "threshold_percentile": 95, "training_mode": "clean",
                  "threshold": thr, "train_time_sec": round(time.time() - t0, 1)})
        lstm_clean_rows.append(m)
        print(f"  F1={m['f1_score']:.4f} P={m['precision']:.4f} R={m['recall']:.4f} "
              f"time={m['train_time_sec']}s")

    # IF 5-seed sweep at default contamination 0.10.
    if_rows = []
    for seed in SEEDS:
        print(f"\n=== IF seed {seed} ===")
        sub_cfg = _copy_config(cfg, RANDOM_STATE=seed)
        det = IsolationForestDetector(sub_cfg)
        res = det.fit_and_detect(
            X_train, X_test,
            train_dates=prepared.get("train_dates"),
            test_dates=prepared.get("test_dates"),
            feature_mode="flatten_only",
        )
        m = _metrics("Isolation Forest", test_labels, res["anomalies"])
        m.update({"seed": seed, "contamination": cfg.CONTAMINATION})
        if_rows.append(m)
        print(f"  F1={m['f1_score']:.4f} P={m['precision']:.4f} R={m['recall']:.4f}")

    # Save raw rows.
    pd.DataFrame(lstm_rows).to_csv(OUT_DIR / "w3_lstm_5seeds_contaminated.csv", index=False)
    pd.DataFrame(lstm_clean_rows).to_csv(OUT_DIR / "w4_lstm_5seeds_clean.csv", index=False)
    pd.DataFrame(thresh_rows).to_csv(OUT_DIR / "w3_lstm_threshold_sweep.csv", index=False)
    pd.DataFrame(if_rows).to_csv(OUT_DIR / "w3_if_5seeds.csv", index=False)

    fields = ["precision", "recall", "f1_score", "accuracy"]
    summary = {
        "lstm_contaminated_5seeds": _summary_stats(lstm_rows, fields),
        "lstm_clean_5seeds": _summary_stats(lstm_clean_rows, fields),
        "if_5seeds": _summary_stats(if_rows, fields),
        "lstm_threshold_sweep_seed42": thresh_rows,
        "seeds": SEEDS,
        "threshold_percentiles": THRESH_PCTS,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nDone. Summary:")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
