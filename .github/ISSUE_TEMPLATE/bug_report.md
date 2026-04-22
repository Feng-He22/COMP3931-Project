---
name: Bug report
about: Report incorrect behavior in the pipeline, models, tests, or documentation
title: "fix: "
labels: bug
assignees: ""
---

## Problem

The climate anomaly pipeline produced incorrect behavior in one of the supported workflows:

- `python main.py --variable rainfall`
- `python main.py --variable tasmax`
- `python main.py --variable tasmin`
- `python main.py --variable all`
- `python main.py --run-fairness-ablation`
- `python main.py --run-event-alignment`
- `python experiments/run_additional_experiments.py`
- `python experiments/run_pr_roc.py`

## Steps To Reproduce

1. Create the project environment with `pip install -r requirements.txt`.
2. Place HadUK-Grid NetCDF files under `data/raw/hadukgrid_60km_last10y/<variable>/`, or rely on the synthetic fallback for `tasmax`.
3. Run the failing command, for example `python main.py --variable tasmax --data-backend standard --no-plots`.
4. Inspect generated files under `data/processed/`, `results/metrics/`, `results/figures/`, or `experiments/*_results/`.

## Expected Behavior

The command should complete without an exception and write the expected project outputs, such as:

- `data/processed/tasmax_series.csv`
- `results/metrics/tasmax_anomaly_results.csv`
- `results/metrics/tasmax_model_metrics.csv`
- `results/metrics/tasmax_summary.txt`
- `experiments/pr_roc_results/metrics.csv`
- `experiments/w3w4_results/summary.json`

## Actual Behavior

Observed behavior should include the exact exception, wrong metric, missing file, or unexpected output. For example:

- import error while collecting tests
- missing NetCDF time coordinate
- empty anomaly-results CSV
- mismatched event-alignment count in `results/metrics/tasmax_event_alignment_summary.csv`
- unexpected AUPRC/AUROC values in `experiments/pr_roc_results/metrics.csv`

## Environment

- Python version: 3.10
- Operating system: Windows / Linux / macOS
- Data backend: standard or dask
- Variable: rainfall, tasmax, tasmin, or all
- Main dependencies: `numpy`, `pandas`, `xarray`, `tensorflow`, `scikit-learn`, `dask`, `pytest`
- Active branch: `main`

## Evidence

Attach the relevant command output and file path. Useful evidence includes:

- full command line used
- traceback or warning
- first rows of a generated CSV
- affected file path, for example `src/data_loader.py` or `models/isolation_forest.py`
- latest commit hash from `git rev-parse --short HEAD`
