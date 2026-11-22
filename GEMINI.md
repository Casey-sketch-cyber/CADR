# Instructions for AI agents working on this repository

This repository is under revision after a journal rejection.

Before doing anything:
1. Read `docs/revision/README.md`, then `docs/revision/TASKS.md` (task specs and rules), then `docs/revision/DATA_RECOVERY.md` if your task needs data.
2. Follow the work rules in `docs/revision/TASKS.md` §0:
   - never push to `main`; use one branch and one PR per task, named `<agent>/T<n>-<name>` (e.g. `codex/T1-hazard-posterior`);
   - keep v1 configs and the legacy algorithm reproducible;
   - never commit data, checkpoints or raw traces;
   - include tests and result tables in each PR.
3. Your PRs are reviewed by Claude against `docs/revision/REVIEW_CHECKLIST.md`. If a design decision is not covered by the spec, ask in the PR instead of improvising.

Documents under `docs/revision/` are written in Chinese; code, identifiers, commit messages and PR titles should be in English.
