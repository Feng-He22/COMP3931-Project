---
name: Experiment change
about: Propose or track an experiment, ablation, metric, or reproducibility change
title: "experiment: "
labels: experiment
assignees: ""
---

## Objective

Track a reproducibility or evaluation change for the Climate Anomaly Detection project.

Current established experiment questions include:

- Does the LSTM Autoencoder remain stable across five random seeds on synthetic `tasmax` labels?
- How sensitive is the LSTM threshold to percentiles `{90, 92, 95, 97, 99}`?
- How do LSTM reconstruction scores compare with Isolation Forest scores under AUPRC/AUROC?
- Do detected `tasmax` anomalies align with externally curated event windows?

## Scope

- Variables: `rainfall`, `tasmax`, `tasmin`, or `all`
- Models: `LSTM Autoencoder`, `Isolation Forest`, or both
- Data source: HadUK-Grid NetCDF under `data/raw/hadukgrid_60km_last10y/`, synthetic `tasmax` benchmark, or `data/external/tasmax_event_windows.csv`
- Main pipeline outputs:
  - `results/metrics/<variable>_anomaly_results.csv`
  - `results/metrics/<variable>_model_metrics.csv`
  - `results/metrics/<variable>_summary.txt`
- Fairness-ablation outputs:
  - `results/metrics/tasmax_fairness_window_sweep.csv`
  - `results/metrics/tasmax_fairness_if_feature_ablation.csv`
  - `results/metrics/tasmax_fairness_lstm_seed_runs.csv`
  - `results/metrics/tasmax_fairness_lstm_seed_summary.csv`
- Reproducibility script outputs:
  - `experiments/w3w4_results/w3_lstm_5seeds_contaminated.csv`
  - `experiments/w3w4_results/w3_if_5seeds.csv`
  - `experiments/w3w4_results/w3_lstm_threshold_sweep.csv`
  - `experiments/w3w4_results/w4_lstm_5seeds_clean.csv`
  - `experiments/pr_roc_results/metrics.csv`
  - `experiments/pr_roc_results/fig_pr_roc.png`

## Validation Plan

Use the validation command that matches the change:

- Unit tests: `python -m pytest -q`
- Single-variable smoke test: `python main.py --variable tasmax --no-plots`
- All-variable smoke test: `python main.py --variable all --skip-lstm --no-plots`
- Fairness ablation: `python main.py --run-fairness-ablation`
- Event alignment: `python main.py --run-event-alignment`
- Seed and threshold experiments: `python experiments/run_additional_experiments.py`
- PR/ROC comparison: `python experiments/run_pr_roc.py`

Baseline PR/ROC reference from the existing results:

- LSTM AUPRC: `0.572`
- LSTM AUROC: `0.761`
- Isolation Forest AUPRC: `0.319`
- Isolation Forest AUROC: `0.721`
- Random-baseline AUPRC: `0.179`

## Version-Control Notes

Commit source code, tests, and lightweight reproducibility evidence. Keep raw data, large model artifacts, local report drafts, and bundle backups out of Git.

Expected commit areas:

- Code: `main.py`, `src/`, `models/`, `config/`, `experiments/`
- Tests: `tests/`
- Lightweight evidence: selected CSV, JSON, TXT, and PNG files under `experiments/` or `results/metrics/`
- Documentation: `README.md`, `CHANGELOG.md`, `docs/version-control.md`
