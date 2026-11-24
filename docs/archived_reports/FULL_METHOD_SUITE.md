# Six full-dataset method trials

The master launcher is `scripts/run_method_suite.py`. No baseline jobs or
additional seeds are included. Seed is 42 for every case.

| Trial | Change from reference |
| --- | --- |
| 01_reference | Global endpoint NCC + N(0,1) moment prior 0.01; PD-LK 0.1; box registration NCC 0.01 |
| 02_target_moments | Replace N(0,1) prior with observed reconstruction-minibatch Gaussian moment target, weight 0.01 |
| 03_registration_003 | Registration NCC weight 0.03 |
| 04_pdlk_003 | PD-LK weight 0.03 |
| 05_context120 | 120 mm crop / 12³ velocity grid, preserving 10 mm cells and nominal physical probe/smoothing scale |
| 06_native_grid | Remove working_grid preprocessing; native voxel grid and fractional crop 0.5 |

All retain paired SIRENs width 256 / four layers / omega 30, 5000 steps,
32768 reconstruction samples and 32768 separate registration particles per
direction, two random probe pairs/cell, two spatial RK2 training half-steps,
and standard brightness-only PCG+RK2 with smoothing 0.01. Physical trials train
at 1 mm and recover at original reference spacing. Trial 6 uses original input
spacing for both training and recovery; crop and smoothing scales therefore
vary with geometry. Full decoder compilation is explicitly false in all six;
compiled derivatives, smoothing and NCC reductions remain enabled.

For trial 02, the penalty per endpoint is KL(N(mu_prediction, var_prediction +
epsilon) || N(mu_observed, var_observed + epsilon)), summed over endpoints:

    0.5 * [(vp + (mp - mt)^2) / vt + log(vt) - log(vp) - 1]

Here vp/vt include epsilon 1e-8. Observed moments are computed on the same
reconstruction minibatch as NCC, within the endpoint sampling domain.
The moment coefficient stays 0.01; its gradient strength need not equal that of
the old prior. The observed-minibatch KL has synthetic value/gradient checks;
no new real-data fitting was launched while preparing this suite.

## Commands

Run from the repository using /venv/main. Preview (no output directory or GPU
worker is created):

```bash
/venv/main/bin/python scripts/run_method_suite.py \
  --output results/full_method_suite --gpus 0 1 2 3
```

**Launch all six complete runs with one call:**

```bash
/venv/main/bin/python scripts/run_method_suite.py \
  --output results/full_method_suite --gpus 0 1 2 3 --run
```

The output root must be new. Optional `--trials` and `--datasets` allow a selected
portion to be launched separately with a new output root. These options do not
resume or overwrite an existing run. Example native-grid-only preview:

```bash
/venv/main/bin/python scripts/run_method_suite.py \
  --trials 06_native_grid --output results/native_grid_full
```

One persistent process per GPU consumes the **same shared queue**, across trial
and dataset boundaries. The next available GPU takes the next
job; trial tails can overlap. Native-mask crop rejection remains active. Case exceptions are recorded and the queue continues; unusable worker processes
are replaced by the existing GPUWorker mechanism. The master exits nonzero if
any job failed. There are no automatic training retries or multiseed runs.

Config snapshots/hashes and all job commands are recorded in plan.json before
workers start. Atomic progress.json
includes running aggregate summaries after each case; final summary.json groups
metrics by trial and dataset with scored/intended counts, failures and PCG
convergence. No deformation maps or warped images are saved.

This queues all six trials without a hard eight-hour cutoff. The 12³ trial is
more expensive and full-run timing is uncertain. Use the progress summaries to
track elapsed work; no experiments were launched during script preparation.
