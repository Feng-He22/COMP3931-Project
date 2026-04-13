from __future__ import annotations

import argparse
import sys

from config.config import AppConfig
from src.anomaly_detector import ClimateAnomalyDetectionSystem


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Climate anomaly detection pipeline")
    parser.add_argument(
        "--variable",
        default="tasmax",
        choices=("rainfall", "tasmax", "tasmin"),
        help="Climate variable to analyse.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of NetCDF files to load per variable.",
    )
    parser.add_argument(
        "--skip-lstm",
        action="store_true",
        help="Skip the LSTM autoencoder and run Isolation Forest only.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip figure generation.",
    )
    parser.add_argument(
        "--no-synthetic-fallback",
        action="store_true",
        help="Fail instead of generating synthetic data when raw data cannot be loaded.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    config = AppConfig()
    if args.max_files is not None:
        config.MAX_FILES_TO_LOAD = args.max_files

    system = ClimateAnomalyDetectionSystem(config)
    outputs = system.run_pipeline(
        variable_name=args.variable,
        allow_synthetic=not args.no_synthetic_fallback,
        make_plots=not args.no_plots,
        run_lstm=not args.skip_lstm,
    )

    print(f"Results saved to: {outputs['results_path']}")
    print(f"Metrics saved to: {outputs['metrics_path']}")
    print(f"Summary saved to: {outputs['summary_path']}")

    for note in outputs["notes"]:
        print(f"Note: {note}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
