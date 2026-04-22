# Changelog

All notable project changes are recorded here so that code, experiments, and
reported results can be traced back to repository revisions.

The project uses semantic versioning for releases and conventional-style commit
messages for day-to-day development.

## Unreleased

### Added

- Version-control governance documentation covering branch policy, release
  tagging, review expectations, and artifact handling.
- GitHub pull request and issue templates to make changes easier to audit.
- Project-specific PR and issue templates populated with real commands, paths,
  validation steps, and experiment output references.
- Local validation checklist for tests, smoke runs, fairness ablation, and
  PR/ROC experiment reproduction.

## Historical Milestones

These milestones summarize the existing Git history. They are not retroactive
release tags.

- `48bd220` - clarified the README tagline after repository migration.
- `abad916` - added reproducibility experiments, seed sweeps, threshold sweeps,
  and PR/ROC outputs.
- `563d091` - added fairness ablation and event-alignment workflows.
- `43ddc41` - added all-variable runs with split outputs.
- `41b7a1c` - added documentation, tests, and notebook copy.
- `5ecdd7d` - added core anomaly-detection pipeline modules.
- `4590f82` - scaffolded the standard project structure.
- `94437ed` - added the original notebook prototype.
