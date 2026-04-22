from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.preprocessing import StandardScaler


class ClimateDataLoader:
    """Data loading and preprocessing utilities for climate anomaly detection."""

    def __init__(self, config) -> None:
        self.config = config
        self.scalers: dict[str, StandardScaler] = {}

    def _get_data_backend(self) -> str:
        backend = getattr(self.config, "DATA_BACKEND", "standard")
        if backend not in {"standard", "dask"}:
            raise ValueError("DATA_BACKEND must be either 'standard' or 'dask'.")
        return backend

    @staticmethod
    def _is_dask_backed(data_array: xr.DataArray) -> bool:
        module_name = type(data_array.data).__module__.lower()
        return "dask" in module_name

    @staticmethod
    def _open_dataset(file_path: Path, *, chunks: Optional[dict[str, int]] = None) -> xr.Dataset:
        engines = (None, "h5netcdf", "scipy")
        errors: list[str] = []

        for engine in engines:
            kwargs = {} if engine is None else {"engine": engine}
            if chunks is not None:
                kwargs["chunks"] = chunks
            engine_name = engine or "default"
            try:
                return xr.open_dataset(file_path, **kwargs)
            except Exception as exc:  # pragma: no cover
                errors.append(f"{engine_name}: {exc}")

        raise OSError(f"Unable to open {file_path.name}. Tried backends: {' | '.join(errors)}")

    def _open_mfdataset(self, nc_files: list[Path], *, chunks: dict[str, int]) -> xr.Dataset:
        engines = ("h5netcdf", None, "scipy")
        errors: list[str] = []

        for engine in engines:
            kwargs = {
                "combine": "by_coords",
                "parallel": getattr(self.config, "DASK_PARALLEL", False),
                "chunks": chunks,
                "data_vars": "minimal",
                "coords": "minimal",
                "compat": "override",
            }
            if engine is not None:
                kwargs["engine"] = engine
            engine_name = engine or "default"
            try:
                return xr.open_mfdataset(nc_files, **kwargs)
            except Exception as exc:  # pragma: no cover
                errors.append(f"{engine_name}: {exc}")

        raise OSError(f"Unable to load NetCDF files with Dask. Tried backends: {' | '.join(errors)}")

    def load_variable_dataset(self, variable_name: str) -> xr.Dataset:
        variable_path = self.config.get_variable_path(variable_name)
        nc_files = sorted(variable_path.glob("*.nc"))
        backend = self._get_data_backend()

        if not nc_files:
            raise FileNotFoundError(f"No NetCDF files found in {variable_path}")

        if self.config.MAX_FILES_TO_LOAD is not None:
            nc_files = nc_files[: self.config.MAX_FILES_TO_LOAD]

        if backend == "dask":
            chunks = dict(getattr(self.config, "DASK_CHUNKS", {"time": 365}))
            if len(nc_files) == 1:
                return self._open_dataset(nc_files[0], chunks=chunks).sortby("time")
            return self._open_mfdataset(nc_files, chunks=chunks).sortby("time")

        datasets: list[xr.Dataset] = []
        for file_path in nc_files:
            try:
                datasets.append(self._open_dataset(file_path))
            except OSError:
                continue

        if not datasets:
            raise OSError(f"Unable to load any NetCDF files for {variable_name}.")

        if len(datasets) == 1:
            return datasets[0].sortby("time")

        combined = xr.concat(datasets, dim="time", data_vars="minimal", coords="minimal", compat="override")
        return combined.sortby("time")

    def load_dataset(self, variable_name: str, allow_synthetic: bool = True) -> tuple[xr.Dataset, str]:
        try:
            return self.load_variable_dataset(variable_name), "raw"
        except Exception as exc:
            if not allow_synthetic:
                raise

            dataset = self.create_synthetic_dataset(variable_name)
            dataset.attrs["fallback_reason"] = str(exc)
            self.save_synthetic_snapshot(dataset, variable_name)
            return dataset, "synthetic"

    def create_synthetic_dataset(self, variable_name: str) -> xr.Dataset:
        rng = np.random.default_rng(self.config.RANDOM_STATE)
        dates = pd.date_range("2014-01-01", "2023-12-31", freq="D")
        t = np.arange(len(dates), dtype=np.float32)

        seasonal = 10.0 * np.sin(2 * np.pi * t / 365.25)
        trend = 0.015 * t / 365.25
        noise = rng.normal(0, 1.75, len(dates))
        values = 15 + seasonal + trend + noise

        anomaly_label = np.zeros(len(dates), dtype=np.int8)
        anomaly_indices = rng.choice(len(dates), size=max(24, len(dates) // 90), replace=False)
        anomaly_label[anomaly_indices] = 1
        values[anomaly_indices] += rng.normal(8.0, 2.0, size=len(anomaly_indices)) * rng.choice(
            np.array([-1.0, 1.0]),
            size=len(anomaly_indices),
        )

        return xr.Dataset(
            data_vars={
                variable_name: ("time", values.astype(np.float32)),
                "anomaly_label": ("time", anomaly_label),
            },
            coords={"time": dates},
            attrs={"source": "synthetic"},
        )

    def save_synthetic_snapshot(self, dataset: xr.Dataset, variable_name: str) -> Path:
        output_path = self.config.SYNTHETIC_DATA_DIR / f"{variable_name}_synthetic.csv"
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(dataset["time"].values),
                "value": dataset[variable_name].values,
                "anomaly_label": dataset["anomaly_label"].values,
            }
        )
        frame.to_csv(output_path, index=False)
        return output_path

    @staticmethod
    def extract_time_series(dataset: xr.Dataset, variable_name: str) -> pd.Series:
        if variable_name not in dataset.data_vars:
            variable_name = next(iter(dataset.data_vars))

        data_array = dataset[variable_name]
        spatial_dims = [dim for dim in data_array.dims if dim != "time"]
        if spatial_dims:
            data_array = data_array.mean(dim=spatial_dims, skipna=True)

        # Materialise lazily loaded arrays before converting to pandas.
        if ClimateDataLoader._is_dask_backed(data_array):
            data_array = data_array.compute()

        series = data_array.to_series()
        series.name = variable_name
        return series.sort_index()

    @staticmethod
    def extract_labels(dataset: xr.Dataset) -> Optional[pd.Series]:
        if "anomaly_label" not in dataset.data_vars:
            return None

        series = dataset["anomaly_label"].to_series().sort_index().astype(int)
        series.name = "anomaly_label"
        return series

    def _seasonal_fill_value(self, series: pd.Series, timestamp: pd.Timestamp) -> float:
        max_year_distance = int(getattr(self.config, "SEASONAL_FILL_MAX_YEAR_DISTANCE", 2))
        candidates: list[float] = []

        for year_offset in range(1, max_year_distance + 1):
            for direction in (-1, 1):
                try:
                    candidate_timestamp = timestamp.replace(year=timestamp.year + direction * year_offset)
                except ValueError:
                    if timestamp.month == 2 and timestamp.day == 29:
                        candidate_timestamp = timestamp.replace(year=timestamp.year + direction * year_offset, day=28)
                    else:
                        continue

                if candidate_timestamp in series.index:
                    candidate_value = series.loc[candidate_timestamp]
                    if pd.notna(candidate_value):
                        candidates.append(float(candidate_value))

        if candidates:
            return float(np.mean(candidates))

        same_calendar_day = series[
            (series.index.month == timestamp.month) & (series.index.day == timestamp.day) & series.notna()
        ]
        if not same_calendar_day.empty:
            return float(same_calendar_day.median())

        return np.nan

    def handle_missing_values(self, series: pd.Series, *, return_report: bool = False):
        series = series.sort_index()
        report: dict[str, float | int | bool] = {
            "missing_original_count": int(series.isna().sum()),
            "missing_original_ratio": float(series.isna().mean()) if len(series) else 0.0,
            "linear_interpolation_count": 0,
            "seasonal_interpolation_count": 0,
            "edge_fill_count": 0,
            "missing_remaining_count": 0,
        }

        if report["missing_original_count"] == 0:
            return (series, report) if return_report else series

        max_missing_ratio = float(getattr(self.config, "MAX_MISSING_RATIO", 1.0))
        if report["missing_original_ratio"] > max_missing_ratio:
            raise ValueError(
                f"Series has missing ratio {report['missing_original_ratio']:.3f}, "
                f"which exceeds the configured maximum of {max_missing_ratio:.3f}."
            )

        filled = series.astype(np.float32).copy()
        linear_limit = int(getattr(self.config, "SHORT_GAP_LINEAR_LIMIT", 2))
        missing_mask = filled.isna().to_numpy()
        gap_start: int | None = None

        for position, is_missing in enumerate(np.append(missing_mask, False)):
            if is_missing and gap_start is None:
                gap_start = position
                continue

            if gap_start is None or is_missing:
                continue

            gap_end = position
            gap_length = gap_end - gap_start
            is_edge_gap = gap_start == 0 or gap_end == len(filled)
            if not is_edge_gap and gap_length <= linear_limit:
                segment = filled.iloc[gap_start - 1 : gap_end + 1].interpolate(method="linear")
                filled.iloc[gap_start:gap_end] = segment.iloc[1:-1]
                report["linear_interpolation_count"] += gap_length

            gap_start = None

        seasonal_fill_count = 0
        if filled.isna().any() and isinstance(filled.index, pd.DatetimeIndex):
            missing_timestamps = list(filled.index[filled.isna()])
            for timestamp in missing_timestamps:
                position = filled.index.get_indexer([timestamp])[0]
                if position == 0 or position == len(filled) - 1:
                    continue
                seasonal_value = self._seasonal_fill_value(filled, timestamp)
                if pd.notna(seasonal_value):
                    filled.at[timestamp] = seasonal_value
                    seasonal_fill_count += 1
        report["seasonal_interpolation_count"] = seasonal_fill_count

        if filled.isna().any():
            first_valid = filled.first_valid_index()
            last_valid = filled.last_valid_index()
            if first_valid is None or last_valid is None:
                raise ValueError("Series contains only missing values after preprocessing.")

            first_pos = filled.index.get_indexer([first_valid])[0]
            last_pos = filled.index.get_indexer([last_valid])[0]

            if first_pos > 0:
                report["edge_fill_count"] += int(filled.iloc[:first_pos].isna().sum())
                filled.iloc[:first_pos] = filled.iloc[first_pos]
            if last_pos < len(filled) - 1:
                report["edge_fill_count"] += int(filled.iloc[last_pos + 1 :].isna().sum())
                filled.iloc[last_pos + 1 :] = filled.iloc[last_pos]

        report["missing_remaining_count"] = int(filled.isna().sum())
        if report["missing_remaining_count"] > 0:
            raise ValueError("Unresolved missing values remain after preprocessing.")

        return (filled, report) if return_report else filled

    def _clip_training_values(self, values: np.ndarray) -> tuple[np.ndarray, dict[str, float | int | bool]]:
        report: dict[str, float | int | bool] = {
            "training_outlier_clip_enabled": bool(getattr(self.config, "ENABLE_TRAINING_OUTLIER_CLIP", True)),
            "training_outlier_clip_count": 0,
            "training_outlier_clip_lower": np.nan,
            "training_outlier_clip_upper": np.nan,
        }
        if values.size == 0 or not report["training_outlier_clip_enabled"]:
            return values.astype(np.float32, copy=False), report

        zscore_threshold = float(getattr(self.config, "TRAINING_OUTLIER_ZSCORE", 3.0))
        if zscore_threshold <= 0:
            return values.astype(np.float32, copy=False), report

        mean_value = float(np.mean(values))
        std_value = float(np.std(values))
        if std_value == 0.0:
            return values.astype(np.float32, copy=False), report

        lower = mean_value - zscore_threshold * std_value
        upper = mean_value + zscore_threshold * std_value
        clipped = np.clip(values, lower, upper).astype(np.float32, copy=False)

        report["training_outlier_clip_count"] = int(np.count_nonzero((values < lower) | (values > upper)))
        report["training_outlier_clip_lower"] = lower
        report["training_outlier_clip_upper"] = upper
        return clipped, report

    @staticmethod
    def create_sequences(values: np.ndarray, sequence_length: int) -> np.ndarray:
        if len(values) < sequence_length:
            raise ValueError(
                f"Not enough time steps to build sequences: got {len(values)}, need at least {sequence_length}."
            )

        return np.array(
            [values[index : index + sequence_length] for index in range(len(values) - sequence_length + 1)],
            dtype=np.float32,
        )

    @staticmethod
    def create_window_labels(labels: np.ndarray, sequence_length: int) -> np.ndarray:
        return np.array(
            [int(labels[index : index + sequence_length].max()) for index in range(len(labels) - sequence_length + 1)],
            dtype=np.int8,
        )

    def save_processed_series(
        self,
        series: pd.Series,
        variable_name: str,
        labels: Optional[pd.Series] = None,
    ) -> Path:
        output_path = self.config.PROCESSED_DATA_DIR / f"{variable_name}_series.csv"
        frame = pd.DataFrame({"date": pd.to_datetime(series.index), "value": series.values})
        if labels is not None:
            aligned_labels = labels.reindex(series.index).fillna(0).astype(int).values
            frame["anomaly_label"] = aligned_labels
        frame.to_csv(output_path, index=False)
        return output_path

    def prepare_lstm_data(
        self,
        series: pd.Series,
        labels: Optional[pd.Series] = None,
        sequence_length: Optional[int] = None,
    ) -> dict[str, np.ndarray | pd.Index | StandardScaler]:
        series, preprocessing_report = self.handle_missing_values(series, return_report=True)
        series = series.sort_index()
        raw_values = series.to_numpy(dtype=np.float32)
        window_length = sequence_length or self.config.SEQUENCE_LENGTH
        windows_raw = self.create_sequences(raw_values, window_length)
        dates = pd.Index(series.index[window_length - 1 :])
        end_values = raw_values[window_length - 1 :]

        window_labels = None
        if labels is not None:
            aligned_labels = labels.reindex(series.index).fillna(0).astype(int).to_numpy()
            window_labels = self.create_window_labels(aligned_labels, window_length)

        sample_count = len(windows_raw)
        if sample_count < 8:
            raise ValueError("At least 8 sequences are required to create train/validation/test splits.")

        test_count = max(1, int(round(sample_count * self.config.TEST_SIZE)))
        train_val_count = sample_count - test_count
        validation_count = max(1, int(round(train_val_count * self.config.VALIDATION_SPLIT)))
        train_count = train_val_count - validation_count

        if train_count < 1:
            raise ValueError("Training split is empty after applying the configured test and validation ratios.")

        train_slice = slice(0, train_count)
        val_slice = slice(train_count, train_count + validation_count)
        test_slice = slice(train_count + validation_count, sample_count)

        train_series_end = train_count + window_length - 1
        clipped_train_values, clip_report = self._clip_training_values(raw_values[:train_series_end])
        preprocessing_report.update(clip_report)

        scaler = StandardScaler()
        scaler.fit(clipped_train_values.reshape(-1, 1))
        self.scalers[series.name or "series"] = scaler

        train_windows = windows_raw[train_slice]
        if int(preprocessing_report["training_outlier_clip_count"]) > 0:
            lower = float(preprocessing_report["training_outlier_clip_lower"])
            upper = float(preprocessing_report["training_outlier_clip_upper"])
            train_windows = np.clip(train_windows, lower, upper)

        X_train = scaler.transform(train_windows.reshape(-1, 1)).astype(np.float32).reshape(-1, window_length, 1)
        X_val = (
            scaler.transform(windows_raw[val_slice].reshape(-1, 1)).astype(np.float32).reshape(-1, window_length, 1)
        )
        X_test = (
            scaler.transform(windows_raw[test_slice].reshape(-1, 1)).astype(np.float32).reshape(-1, window_length, 1)
        )

        if len(X_val) == 0:
            raise ValueError("Validation split produced zero samples.")

        output: dict[str, np.ndarray | pd.Index | StandardScaler] = {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": end_values[train_slice],
            "y_val": end_values[val_slice],
            "y_test": end_values[test_slice],
            "train_dates": dates[train_slice],
            "val_dates": dates[val_slice],
            "test_dates": dates[test_slice],
            "scaler": scaler,
            "sequence_length": window_length,
            "preprocessing_report": preprocessing_report,
        }

        if window_labels is not None:
            output["train_labels"] = window_labels[train_slice]
            output["val_labels"] = window_labels[val_slice]
            output["test_labels"] = window_labels[test_slice]

        return output
