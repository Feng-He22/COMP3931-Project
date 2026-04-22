# Climate Anomaly Detection

A comparative study of unsupervised anomaly detection (LSTM Autoencoder vs Isolation Forest) on HadUK-Grid daily climate data, covering three variables:

- `rainfall`
- `tasmax`
- `tasmin`

The pipeline reads NetCDF files, aggregates gridded observations into a single daily series, prepares sliding windows, runs anomaly detectors, and saves metrics, figures, and model artifacts. It also includes two analysis extensions:

- a fairness-style ablation package on synthetic `tasmax` labels
- an external event-window alignment analysis for real `tasmax` anomaly outputs

## Main Capabilities

- Load HadUK-Grid NetCDF files from `data/raw/hadukgrid_60km_last10y`
- Aggregate spatial dimensions into a unified daily time series
- Apply layered preprocessing with short-gap linear interpolation, seasonal gap filling, edge filling, and training-only outlier clipping
- Build sliding windows for sequence-based anomaly detection
- Run an `LSTM Autoencoder` and an `Isolation Forest`
- Switch between standard loading and Dask-backed lazy loading
- Export processed series, anomaly tables, plots, summaries, and saved models
- Run single-variable analysis, all-variable batch analysis, fairness ablations, and event alignment reports

## Project Layout

- `config/config.py`: central configuration, directory paths, model settings, ablation settings, and event-alignment parameters
- `src/data_loader.py`: NetCDF loading, Dask support, layered preprocessing, label extraction, leakage-safe sequence preparation, and processed-data export
- `models/lstm_autoencoder.py`: LSTM autoencoder model build, training, checkpointing, and anomaly scoring
- `models/isolation_forest.py`: Isolation Forest training, anomaly scoring, and optional rolling plus seasonal feature construction
- `src/anomaly_detector.py`: end-to-end pipeline orchestration, fairness ablations, event alignment, metrics writing, and aggregate summaries
- `src/visualization.py`: time-series, training-history, anomaly, and metric-summary figures
- `tests/`: unit tests for loaders, detectors, and event alignment logic

## Installation

Create an environment and install dependencies:

```bash
pip install -r requirements.txt
```

## Command-Line Usage

Run the default `tasmax` pipeline:

```bash
python main.py
```

Run one variable explicitly:

```bash
python main.py --variable rainfall
```

Run all supported variables:

```bash
python main.py --variable all
```

Use Dask-backed loading for larger NetCDF collections:

```bash
python main.py --variable tasmax --data-backend dask
```

Run only Isolation Forest and skip figure generation:

```bash
python main.py --variable tasmin --skip-lstm --no-plots
```

Disable synthetic fallback if raw data are missing:

```bash
python main.py --variable tasmax --no-synthetic-fallback
```

Run the synthetic fairness ablation package for `tasmax`:

```bash
python main.py --run-fairness-ablation
```

Run external event-window alignment for `tasmax`:

```bash
python main.py --run-event-alignment
```

## Data And Analysis Notes

### Raw climate data

Place HadUK-Grid NetCDF files under:

- `data/raw/hadukgrid_60km_last10y/rainfall`
- `data/raw/hadukgrid_60km_last10y/tasmax`
- `data/raw/hadukgrid_60km_last10y/tasmin`

### External event windows

The event-alignment workflow expects a CSV such as:

- `data/external/tasmax_event_windows.csv`

The table should include event metadata and date windows, including fields such as:

- `event_id`
- `event_name`
- `start_date`
- `end_date`
- `window_basis`
- `source_org`
- `source_title`
- `source_url`
- `label_strength`
- `notes`

These event windows are treated as weak-label references for evaluation summaries, not exact point-wise ground truth.

## Outputs

### Processed data

Processed daily series are written to `data/processed/`, for example:

- `rainfall_series.csv`
- `tasmax_series.csv`
- `tasmin_series.csv`

### Metrics and summaries

Main pipeline runs write files such as:

- `results/metrics/<variable>_anomaly_results.csv`
- `results/metrics/<variable>_model_metrics.csv`
- `results/metrics/<variable>_summary.txt`
- `results/metrics/all_variables_model_metrics.csv`
- `results/metrics/all_variables_summary.txt`

Fairness ablation runs write:

- `results/metrics/tasmax_fairness_window_sweep.csv`
- `results/metrics/tasmax_fairness_if_feature_ablation.csv`
- `results/metrics/tasmax_fairness_lstm_seed_runs.csv`
- `results/metrics/tasmax_fairness_lstm_seed_summary.csv`
- `results/metrics/tasmax_fairness_ablation_summary.txt`

Event alignment writes:

- `results/metrics/tasmax_event_alignment_by_event.csv`
- `results/metrics/tasmax_event_alignment_summary.csv`
- `results/metrics/tasmax_event_alignment_summary.txt`

### Figures

Figures are saved in `results/figures/`, including:

- time-series plots
- training-history plots
- anomaly-detection plots
- metric-summary plots

### Saved models

Model artifacts are saved in `results/models/`, including:

- `<variable>_best_lstm_autoencoder.h5`
- `<variable>_lstm_autoencoder.h5`
- `<variable>_isolation_forest.pkl`

## Reproducibility Experiments (`experiments/`)

Additional scripts supporting the dissertation's §4.3 fairness-ablation and threshold-free comparison results. All runs use the synthetic `tasmax` benchmark (725-window hold-out, 130 anomalous windows, 17.93% prevalence).

- `experiments/run_additional_experiments.py` — 5-seed robustness sweep for LSTM Autoencoder and Isolation Forest, plus a percentile threshold sweep `{90, 92, 95, 97, 99}` for the LSTM, plus a W4 clean-vs-contaminated training control. Writes to `experiments/w3w4_results/`:
  - `w3_lstm_5seeds_contaminated.csv`, `w3_if_5seeds.csv`
  - `w3_lstm_threshold_sweep.csv`
  - `w4_lstm_5seeds_clean.csv`
  - `summary.json`, `run.log`
- `experiments/run_pr_roc.py` — threshold-free comparison (continuous scores): sweeps LSTM reconstruction MSE and Isolation Forest `-decision_function` over all thresholds, and reports AUPRC and AUROC for both methods against the random-baseline AUPRC (test-set prevalence). Writes to `experiments/pr_roc_results/`:
  - `fig_pr_roc.png` — joint PR / ROC figure (seed 42)
  - `metrics.json`, `metrics.csv`, `run.log`

Reported numbers (seed 42): LSTM AUPRC 0.572 / AUROC 0.761; Isolation Forest AUPRC 0.319 / AUROC 0.721; random-baseline AUPRC 0.179.

To reproduce:

```bash
python experiments/run_additional_experiments.py
python experiments/run_pr_roc.py
```

## Typical Workflow

1. Place NetCDF files in the raw-data folders.
2. Run `python main.py --variable tasmax` or `python main.py --variable all`.
3. Inspect processed series in `data/processed/`.
4. Review metrics and summaries in `results/metrics/`.
5. Review figures in `results/figures/`.
6. Optionally run `--run-fairness-ablation` or `--run-event-alignment` for additional analysis.
