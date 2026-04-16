# Climate Anomaly Detection

This project detects anomalies in HadUK-Grid daily climate data for three variables:

- `rainfall`
- `tasmax`
- `tasmin`

The code reads raw NetCDF files, converts gridded data into unified time series, preprocesses the series, trains anomaly detection models, scores abnormal windows, and saves charts, metrics, and model files for each variable.

## What The Code Does

The codebase provides these core functions:

- Load raw climate data from `data/raw/hadukgrid_60km_last10y`
- Extract one time series per variable from gridded NetCDF data
- Aggregate spatial dimensions automatically
- Fill missing values by interpolation and backfilling
- Standardize the time series and build sliding-window sequences
- Detect anomalies with `LSTM Autoencoder` using reconstruction error
- Detect anomalies with `Isolation Forest` using flattened sequence features
- Save separate outputs for `rainfall`, `tasmax`, and `tasmin`
- Run one variable at a time or all supported variables in one pass

## Processing Flow

The execution flow is:

1. Read the NetCDF files for the selected variable.
2. Reduce spatial dimensions into a single time series.
3. Clean missing values and save the processed series.
4. Split the series into sliding windows.
5. Train the LSTM Autoencoder and compute anomaly scores.
6. Train the Isolation Forest and compute anomaly scores.
7. Save anomaly results, summary files, metrics tables, and figures.
8. When `all` is selected, repeat the same workflow for `rainfall`, `tasmax`, and `tasmin`, then write aggregate summary files.

## Main Modules

### `config/config.py`

Stores project paths, supported variables, output directories, and runtime configuration.

### `src/data_loader.py`

Handles data loading and preprocessing:

- NetCDF file loading
- Variable extraction
- Spatial aggregation
- Missing-value handling
- Sequence construction
- Train, validation, and test splitting
- Processed-series export

### `models/lstm_autoencoder.py`

Implements the LSTM Autoencoder:

- Build encoder and decoder layers
- Train the reconstruction model
- Compute reconstruction errors
- Produce anomaly labels and anomaly scores
- Save trained model checkpoints

### `models/isolation_forest.py`

Implements the Isolation Forest detector:

- Flatten sequence windows into feature vectors
- Train the tree-based detector
- Produce anomaly predictions and scores
- Save the fitted model and scaler

### `src/visualization.py`

Generates result figures:

- Raw time-series plots
- LSTM training-history plots
- Anomaly-detection plots
- Metric summary plots

### `src/anomaly_detector.py`

Coordinates the full pipeline, including preprocessing, model execution, result saving, and multi-variable batch runs.

### `main.py`

Provides the command-line entry point and supports:

- Single-variable runs
- All-variable runs
- Optional LSTM skipping
- Optional figure generation control
- Optional synthetic fallback control

## Outputs

### Processed Data

The pipeline writes processed series such as:

- `data/processed/rainfall_series.csv`
- `data/processed/tasmax_series.csv`
- `data/processed/tasmin_series.csv`

### Figures

For each variable, the pipeline generates:

- a time-series figure
- a training-history figure
- an LSTM anomaly-detection figure

All figures are saved in `results/figures/`.

### Metrics And Summaries

For each variable, the pipeline writes:

- `<variable>_anomaly_results.csv`
- `<variable>_model_metrics.csv`
- `<variable>_summary.txt`

When all variables are processed, it also writes:

- `all_variables_model_metrics.csv`
- `all_variables_summary.txt`

These files are saved in `results/metrics/`.

### Model Files

For each variable, the pipeline saves:

- `<variable>_best_lstm_autoencoder.h5`
- `<variable>_lstm_autoencoder.h5`
- `<variable>_isolation_forest.pkl`

These files are saved in `results/models/`.

## Use Cases

This code is suitable for:

- climate time-series anomaly detection
- rainfall and temperature anomaly screening
- batch analysis of multiple climate variables
- unsupervised anomaly scoring with visual outputs
