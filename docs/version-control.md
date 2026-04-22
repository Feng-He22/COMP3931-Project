# Version Control Policy

This document describes how the repository demonstrates controlled,
reproducible software development.

## Repository Remotes

- `origin` points to the active GitHub repository:
  `https://github.com/Feng-He22/COMP3931-Project.git`
- `old-origin` is retained locally only as a migration reference.

## Protected History Expectations

- `main` is treated as the stable branch.
- Force pushes to `main` should be avoided after public sharing.
- Any future history rewrite should be documented in this file or in
  `CHANGELOG.md`.
- Release tags should be annotated and should not be moved once published.

## Traceability

The repository keeps traceability through:

- focused commits with clear prefixes such as `feat`, `fix`, `docs`, `test`,
  and `chore`
- `CHANGELOG.md` for human-readable change summaries
- pull request templates for review evidence
- issue templates for bug reports and experiment changes
- CI checks for repeatable validation
- `.gitignore` rules that separate source code from local data, models,
  virtual environments, report drafts, and backups

## Review Workflow

1. Create a focused branch from `main`.
2. Make one logical change at a time.
3. Run local tests with `python -m pytest -q`.
4. Open a pull request using the template.
5. Confirm that generated artifacts are intentional.
6. Merge only after the change is reviewed and CI passes.

## Release Workflow

Releases should use semantic version tags:

- `MAJOR` for incompatible changes to project structure or command behavior
- `MINOR` for new features, experiments, or analysis workflows
- `PATCH` for bug fixes, documentation corrections, or small maintenance work

Example:

```bash
git tag -a v0.4.0 -m "Release v0.4.0"
git push origin main
git push origin v0.4.0
```

Each release should include:

- important commits or pull requests
- validation command and result
- known limitations
- data and artifact notes

## Backup And Recovery

For major repository operations, create a bundle backup before changing remotes,
rewriting history, or deleting tags:

```bash
git bundle create repo-history-backup.bundle --all
git bundle verify repo-history-backup.bundle
```

Bundle backups are local safety files and should not be committed.
