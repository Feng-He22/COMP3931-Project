import xarray as xr
import numpy as np
import pandas as pd
import importlib.util

from config.config import AppConfig
from src.data_loader import ClimateDataLoader


def test_synthetic_dataset_contains_signal_and_labels(tmp_path):
    config = AppConfig(project_root=tmp_path)
    loader = ClimateDataLoader(config)

    dataset = loader.create_synthetic_dataset("tasmax")

    assert "tasmax" in dataset.data_vars
    assert "anomaly_label" in dataset.data_vars
    assert int(dataset["anomaly_label"].sum()) > 0


def test_prepare_lstm_data_returns_expected_shapes(tmp_path):
    config = AppConfig(project_root=tmp_path)
    loader = ClimateDataLoader(config)
    dataset = loader.create_synthetic_dataset("tasmax")
    series = loader.extract_time_series(dataset, "tasmax")
    labels = loader.extract_labels(dataset)

    prepared = loader.prepare_lstm_data(series, labels=labels)

    assert prepared["X_train"].shape[1:] == (config.SEQUENCE_LENGTH, 1)
    assert prepared["X_val"].shape[1:] == (config.SEQUENCE_LENGTH, 1)
    assert prepared["X_test"].shape[1:] == (config.SEQUENCE_LENGTH, 1)
    assert len(prepared["test_dates"]) == prepared["X_test"].shape[0]
    assert prepared["test_labels"].shape[0] == prepared["X_test"].shape[0]


def test_prepare_lstm_data_respects_custom_sequence_length(tmp_path):
    config = AppConfig(project_root=tmp_path)
    loader = ClimateDataLoader(config)
    dataset = loader.create_synthetic_dataset("tasmax")
    series = loader.extract_time_series(dataset, "tasmax")
    labels = loader.extract_labels(dataset)

    prepared = loader.prepare_lstm_data(series, labels=labels, sequence_length=14)

    assert prepared["sequence_length"] == 14
    assert prepared["X_train"].shape[1:] == (14, 1)
    assert prepared["X_test"].shape[1:] == (14, 1)


def test_handle_missing_values_uses_linear_seasonal_and_edge_fill(tmp_path):
    config = AppConfig(project_root=tmp_path)
    loader = ClimateDataLoader(config)
    dates = pd.date_range("2020-01-01", periods=370, freq="D")
    series = pd.Series(np.full(len(dates), 10.0, dtype=np.float32), index=dates, name="tasmax")

    series.iloc[0] = np.nan
    series.iloc[10:12] = np.nan
    series.iloc[100:103] = np.nan
    series.loc[pd.Timestamp("2021-04-10")] = 25.0
    series.loc[pd.Timestamp("2021-04-11")] = 26.0
    series.loc[pd.Timestamp("2021-04-12")] = 27.0

    filled, report = loader.handle_missing_values(series, return_report=True)

    assert report["linear_interpolation_count"] == 2
    assert report["seasonal_interpolation_count"] == 3
    assert report["edge_fill_count"] == 1
    assert report["missing_remaining_count"] == 0
    assert float(filled.iloc[0]) == 10.0
    assert float(filled.iloc[10]) == 10.0
    assert float(filled.iloc[11]) == 10.0
    assert float(filled.loc[pd.Timestamp("2020-04-10")]) == 25.0
    assert float(filled.loc[pd.Timestamp("2020-04-11")]) == 26.0
    assert float(filled.loc[pd.Timestamp("2020-04-12")]) == 27.0


def test_handle_missing_values_rejects_pervasive_missingness(tmp_path):
    config = AppConfig(project_root=tmp_path)
    config.MAX_MISSING_RATIO = 0.2
    loader = ClimateDataLoader(config)
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    series = pd.Series([np.nan, np.nan, np.nan, 1.0, np.nan, np.nan, 2.0, np.nan, 3.0, np.nan], index=dates)

    try:
        loader.handle_missing_values(series)
    except ValueError as exc:
        assert "missing ratio" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected pervasive missingness to raise ValueError.")


def test_prepare_lstm_data_fits_scaler_on_training_only(tmp_path):
    config = AppConfig(project_root=tmp_path)
    loader = ClimateDataLoader(config)
    dates = pd.date_range("2023-01-01", periods=12, freq="D")
    series = pd.Series([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1000.0], index=dates)

    prepared = loader.prepare_lstm_data(series, sequence_length=3)

    assert np.isclose(float(prepared["scaler"].mean_[0]), 3.5)
    assert prepared["preprocessing_report"]["training_outlier_clip_count"] == 0


def test_prepare_lstm_data_clips_training_outliers_only(tmp_path):
    config = AppConfig(project_root=tmp_path)
    config.TRAINING_OUTLIER_ZSCORE = 1.5
    loader = ClimateDataLoader(config)
    dates = pd.date_range("2023-01-01", periods=12, freq="D")
    series = pd.Series([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1000.0, 8.0, 9.0, 10.0, 11.0], index=dates)

    prepared = loader.prepare_lstm_data(series, sequence_length=3)

    assert prepared["preprocessing_report"]["training_outlier_clip_count"] > 0
    assert prepared["preprocessing_report"]["training_outlier_clip_upper"] < 1000.0


def test_dask_backend_loads_multi_file_dataset(tmp_path):
    if importlib.util.find_spec("dask") is None:
        return

    config = AppConfig(project_root=tmp_path)
    variable_dir = config.RAW_DATA_DIR / config.DATASET_DIRNAME / "tasmax"
    variable_dir.mkdir(parents=True)

    ds1 = xr.Dataset({"tasmax": (("time",), [1.0, 2.0])}, coords={"time": ["2023-01-01", "2023-01-02"]})
    ds2 = xr.Dataset({"tasmax": (("time",), [3.0, 4.0])}, coords={"time": ["2023-01-03", "2023-01-04"]})
    ds1.to_netcdf(variable_dir / "tasmax_part1.nc")
    ds2.to_netcdf(variable_dir / "tasmax_part2.nc")

    config.DATA_BACKEND = "dask"
    loader = ClimateDataLoader(config)
    dataset = loader.load_variable_dataset("tasmax")
    series = loader.extract_time_series(dataset, "tasmax")

    assert len(series) == 4
    assert float(series.iloc[-1]) == 4.0
