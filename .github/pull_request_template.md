## Summary

This change updates the Climate Anomaly Detection project on top of `main`.

Relevant project areas:

- Pipeline entry point: `main.py`
- Core source: `src/data_loader.py`, `src/anomaly_detector.py`, `src/visualization.py`
- Model code: `models/lstm_autoencoder.py`, `models/isolation_forest.py`
- Configuration: `config/config.py`
- Tests: `tests/test_config.py`, `tests/test_data_loader.py`, `tests/test_event_alignment.py`, `tests/test_isolation_forest.py`
- Experiment scripts: `experiments/run_additional_experiments.py`, `experiments/run_pr_roc.py`

## Version-Control Impact

- [ ] Source-code change
- [ ] Documentation-only change
- [ ] Experiment or result artifact change
- [ ] Dependency or environment change
- [ ] Release or tagging change

## Validation

- [ ] `python -m pytest -q`
- [ ] `python main.py --variable tasmax --no-plots`
- [ ] `python main.py --variable all --skip-lstm --no-plots`
- [ ] `python experiments/run_additional_experiments.py`
- [ ] `python experiments/run_pr_roc.py`
- [ ] Not run because the local environment is missing one or more runtime dependencies from `requirements.txt`

## Data And Artifacts

- [ ] No raw data, large model files, local backups, or report drafts included
- [ ] Generated outputs are intentionally included and documented
- [ ] Raw NetCDF inputs remain outside Git under `data/raw/hadukgrid_60km_last10y/`
- [ ] Large model artifacts remain outside Git under `results/models/`
- [ ] Tracked result summaries are limited to reproducibility evidence in `experiments/` or lightweight report-supporting files

## Notes For Review

Reviewers should check that preprocessing remains leakage-safe, that model comparisons use the same test windows where applicable, and that any generated metrics match the documented output paths in `README.md`.
