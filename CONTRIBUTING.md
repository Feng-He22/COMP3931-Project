# Contributing

This repository is maintained as a reproducible climate-anomaly detection
project. The contribution rules below keep the history readable and the
experimental outputs traceable.

## Branch Model

- `main` contains the current stable project state.
- Work should happen on short-lived branches named by purpose:
  - `feature/<short-description>`
  - `fix/<short-description>`
  - `experiment/<short-description>`
  - `docs/<short-description>`
- Branches should be merged only after tests pass and the change is reviewed.

## Commit Style

Use concise, conventional-style commit messages:

```text
feat: add event-alignment summary export
fix: handle missing NetCDF time coordinate
docs: document raw data layout
test: cover isolation forest feature construction
chore: update ignore rules for local artifacts
```

Each commit should represent one logical change. Avoid mixing code, results,
and documentation unless the files are tightly related.

## Pull Request Checklist

Before opening a pull request:

- Explain the purpose of the change.
- Link the related issue or experiment note when applicable.
- Run `python -m pytest -q`.
- Confirm whether generated results, model files, or raw data are intentionally
  included.
- Update `CHANGELOG.md` when the change affects user-visible behavior,
  reproducibility, experiment outputs, or release documentation.

## Release Process

1. Make sure `main` is clean and CI is passing.
2. Update `CHANGELOG.md` by moving relevant `Unreleased` entries into a dated
   version section.
3. Create an annotated tag such as `v0.4.0`.
4. Push the branch and tag to GitHub.
5. Create a GitHub release summarizing code changes, experiment outputs, and
   reproducibility notes.

## Data And Artifact Policy

- Raw HadUK-Grid NetCDF files stay outside Git.
- Large generated models stay outside Git unless explicitly required.
- Small result summaries and figures may be tracked when they support the
  report or a reproducibility claim.
- Local report drafts, backups, virtual environments, caches, and temporary
  files should remain untracked.
