# Agent guide for paired-INR PD-LK

Read `README.md`, this file, and the task-specific documents before changing the
research code. Machine-specific service guidance belongs outside this repository.

## Project map

- `cameraman/` is the 2D synthetic prototype. Its maintained configs compare
  standard PD-LK (`rho=0`) with relaxed PD-LK (`rho=0.4`).
- `medical/` is the 3D paired-SIREN registration pipeline. The active method is
  pointwise relaxed PD-LK, with `lpba_pdlk_standard.json` as the `rho=0` control.
- `medical/registration/configuration.py` reads current and recognized
  saved pointwise configs in memory. Keep saved run metadata intact;
  actual retired patch/H1 configs are not converted into pointwise methods.
- `Data/` contains the existing local LPBA40 data and is ignored by Git.
  Preserve every internal path and byte. The prepared-data preprocessing
  provenance is unresolved; do not guess or alter it.
- `cameraman/runs/` and `medical/results/` contain local generated artifacts and
  are ignored by Git.
- The 2D fitter supports paired SIRENs and pointwise PD-LK only.
  `cameraman/historical.py` retains small mathematical references for older
  objectives; it is not part of the active pipeline.
- `docs/METHOD.md` is authoritative for equations and coordinate conventions;
  `docs/DEVELOPMENT.md` records implementation state and experiment protocol;
  `docs/RESEARCH_EVOLUTION.md` records the conceptual history; `docs/BEST_RESULTS.md`
  records qualified evidence.

## Research and process rules

- **Do not run model training, hyperparameter sweeps, benchmarks, saved-model
  recovery, or any optimizer step unless the user explicitly asks in the active
  session.** This includes tiny fit smoke tests. Synthetic derivative, loss,
  metric, flow, import, CLI-help and plan-only checks are permitted after auditing
  that the selected test does not train.
- Never upload, stage, commit or otherwise publish medical data, dataset archives,
  generated volumes, checkpoints, result folders, credentials, or machine-local
  config. Cameraman's bundled `cameraman/data/cameraman.jpg` is allowed.
- Keep dataset setup code in `medical/datasets/`; it must refuse to overwrite
  inputs and may only reproduce a documented recipe.
- Keep `rho` as the public relaxation parameter. `rho=0` is the exact standard
  PD-LK control. Preserve derivative units, axis ordering, objective scaling,
  initialization and the distinction between local training velocities and
  global deformation recovery. Read `docs/METHOD.md` before changing formulas.
- Prefer theory and controlled methodology progress over a single benchmark
  improvement. Do not make superiority claims from one seed or gate-failing runs.
- Use fresh output directories and keep all run/result trees ignored.
- The user manages GitHub backups, commits and pushes manually. Never commit
  or push on your own. Leave changes in the working tree for user review; do not
  stage files, create tags or change remotes unless explicitly requested.

## Validation and handover

1. Inspect current file status, running jobs and paths before edits; do not assume
   an earlier process or run state still applies.
2. Audit tests and scripts for optimizer calls, subprocess launches, data writes
   and output creation before running them. Imports and `--help` must not launch
   work.
3. Run the synthetic suites documented in `README.md` after code changes. Never
   use the LPBA files for smoke tests.
   Sweep orchestration tests mock trainer processes and clocks; they are safe
   non-training checks. Keep wheel package discovery restricted to real source
   packages, and include default config JSONs as package data.
4. Check `git check-ignore` on local data and outputs; verify the bundled image is
   eligible for tracking. Inspect candidate files and symlink targets before any
   staging or publication preparation.
5. Update the relevant method, development, research-history or best-results
   document when behavior or evidence changes. Keep those purposes separate.
6. Report what changed, exact checks performed, unverified work and data/output
   preservation status. Mention that training remains unrun unless explicitly
   authorized.
