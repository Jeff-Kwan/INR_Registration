# Four-hour relaxed PD-LK sweep

The sweep is a staged development search. Its latest completed output is recorded
in [the curated results note](../../docs/BEST_RESULTS.md); per-trial files remain
under ignored `medical/results/`.

## Safe inspection

From the repository root, `--help` prints the interface. `--plan-only` writes the
candidate design and summary to a fresh directory without launching the trainer:

```bash
python -m medical.experiments.hyperparameter_sweep --help
python -m medical.experiments.hyperparameter_sweep --plan-only --output /tmp/pdlk-plan-check
```

A real invocation performs model fitting and requires explicit user authorization.
It takes a nonblocking repository-level GPU lock and refuses to reuse an output
folder. Do not start a replacement run as part of documentation or maintenance work.

## Budget and stages

The hard budget is 14,400 seconds. Screening is limited to 3,600 seconds on S02
and S06; confirmation to 2,700 seconds on S02, S04, S06 and S09; full-cohort
comparisons to 6,300 seconds on S02–S10; and 1,800 seconds are reserved for
matched-seed repeats. Stage deadlines are absolute from launch, so unused time
can flow forward. Cost estimates begin from the supplied 600-second full-panel
estimate and adapt from completed candidates. No separate benchmark trial is run.
Every admitted candidate uses 5,000 training steps; partial trials are not ranked.

## Search design

The public coordinates are `c=lk.gradient_weight` and `rho=lk.affine_rho=c/beta`:
`c_eff=c/(1+rho*|grad I|^2)`. The exact standard-PD-LK control is `rho=0`; the
reference relaxed setting is `rho=0.4`. Both are prioritized in confirmation,
full-panel comparisons and reserve repeats. Search values are:

- `rho`: `0`, `0.0004`, `0.004`, `0.04`, `0.4`, `4`, `40`, `400`, `4000`
- gradient weight `c`: `0.25`, `1`, `4`, `16`, `64`; anchors use `1`, `4`, `16`
- loss weight `lambda`: `0.003`, `0.01`, `0.03`, `0.1`, `0.3`, `1`;
  anchored probes use `0.01`, `0.03`, `0.3`
- ridge: `1e-7`, `1e-6`, `1e-5`, `1e-4`; anchors use `1e-6`, `1e-5`
- schedule: constant, linear, cosine
- point count: 4,096 or 8,192
- learning rate: `1e-4`, `2e-4`, `4e-4`; final factor `0.1`, `0.25`, `0.5`
- batch size: 16,384, 32,768, 65,536
- SIREN width: 192, 256, 384; layers: 3, 4, 5; frequency: 10, 20, 30, 40

The candidate list interleaves the base, matched rho controls, anchored
regularization probes and 72 stratified joint settings. It then locally refines
the leading screening configurations. Recovery is fixed throughout. The full
cohort participates in selection, so the results are development evidence rather
than held-out generalization estimates.

Confirmation, full-panel comparisons and reserve repeats deduplicate scientific
settings independently of the subject panel and output-retention flags. Seeds
remain distinct for matched repeats. Runtime grouping also combines seeds.

Ranking requires every subject's endpoint PSNR to meet 30 dB, zero folding, and
finite label metrics. Macro Dice ranks eligible candidates; HD95 breaks ties.
Flow convergence is recorded but not an admission gate. Failures and interrupted
trials consume the same budget.

## Artifact policy

Every candidate saves its config, log, metrics, per-label scores and flow
statistics. The sweep disables volume, frame and velocity files for all stages.
Screening and confirmation also disable checkpoints; full and reserve repeats
retain them. Recovery may use temporary arrays for inverse integration, which
are deleted when the run finishes. Ordinary registration uses its explicit
config settings and may save volumes.

The output includes `plan.json`, trial folders, centralized CSV/JSON summaries,
leaderboards, grouped runtimes, failures, and the best observed full-cohort config
and metrics. `summary/REPORT.md` is the entry point. A single highest-scoring run
is not a seed-robust conclusion; check matched comparisons and flow warnings.
