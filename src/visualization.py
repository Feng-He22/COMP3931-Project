from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class ClimateVisualizer:
    """Visualization utilities for training diagnostics and anomaly inspection."""

    def __init__(self, config) -> None:
        self.config = config
        sns.set_theme(style="whitegrid")
        self.colors = {
            "normal": "#2E86AB",
            "anomaly": "#D64550",
            "secondary": "#5D737E",
            "train": "#4A9C7D",
            "validation": "#F3B700",
        }

    def plot_time_series(self, series: pd.Series, title: str, filename: str) -> None:
        fig, axis = plt.subplots(figsize=(12, 4))
        axis.plot(series.index, series.values, color=self.colors["normal"], linewidth=1)
        axis.set_title(title)
        axis.set_xlabel("Date")
        axis.set_ylabel(series.name or "value")
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        plt.setp(axis.get_xticklabels(), rotation=45)
        fig.tight_layout()
        fig.savefig(self.config.get_output_path("figures", filename), dpi=150)
        plt.close(fig)

    def plot_training_history(self, history, filename: str = "training_history.png") -> None:
        history_dict = getattr(history, "history", None)
        if not history_dict:
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(history_dict.get("loss", []), label="train", color=self.colors["train"])
        axes[0].plot(history_dict.get("val_loss", []), label="validation", color=self.colors["validation"])
        axes[0].set_title("LSTM Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()

        mae_values = history_dict.get("mae")
        val_mae_values = history_dict.get("val_mae")
        if mae_values is not None and val_mae_values is not None:
            axes[1].plot(mae_values, label="train", color=self.colors["train"])
            axes[1].plot(val_mae_values, label="validation", color=self.colors["validation"])
            axes[1].set_title("LSTM MAE")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("MAE")
            axes[1].legend()
        else:
            axes[1].axis("off")

        fig.tight_layout()
        fig.savefig(self.config.get_output_path("figures", filename), dpi=150)
        plt.close(fig)

    def plot_anomaly_detection(
        self,
        dates,
        values,
        anomalies,
        anomaly_scores,
        title: str,
        filename: str,
    ) -> None:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        axes[0].plot(dates, values, color=self.colors["normal"], linewidth=1, label="Observed")
        anomaly_dates = [date for date, is_anomaly in zip(dates, anomalies) if is_anomaly]
        anomaly_values = [value for value, is_anomaly in zip(values, anomalies) if is_anomaly]
        if anomaly_dates:
            axes[0].scatter(anomaly_dates, anomaly_values, color=self.colors["anomaly"], marker="x", s=28, label="Anomaly")
        axes[0].set_title(title)
        axes[0].set_ylabel("Value")
        axes[0].legend()

        axes[1].plot(dates, anomaly_scores, color=self.colors["secondary"], linewidth=1, label="Anomaly score")
        axes[1].axhline(0.95, color=self.colors["anomaly"], linestyle="--", linewidth=1, label="Reference line")
        axes[1].set_xlabel("Date")
        axes[1].set_ylabel("Score")
        axes[1].legend()

        axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.setp(axes[1].get_xticklabels(), rotation=45)

        fig.tight_layout()
        fig.savefig(self.config.get_output_path("figures", filename), dpi=150)
        plt.close(fig)

    def plot_metric_summary(self, metrics_df: pd.DataFrame, filename: str = "metric_summary.png") -> None:
        numeric_columns = ["precision", "recall", "f1_score", "accuracy"]
        if metrics_df.empty or metrics_df[numeric_columns].dropna(how="all").empty:
            return

        plot_df = metrics_df.set_index("model")[numeric_columns]
        fig, axis = plt.subplots(figsize=(10, 5))
        plot_df.plot(
            kind="bar",
            ax=axis,
            color=[self.colors["train"], self.colors["validation"], self.colors["normal"], self.colors["secondary"]],
        )
        axis.set_ylim(0, 1)
        axis.set_ylabel("Score")
        axis.set_title("Model Metrics")
        axis.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(self.config.get_output_path("figures", filename), dpi=150)
        plt.close(fig)
