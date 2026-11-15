# Adaptive Hessian gating screen (2026-11-11)

This recovery-only screen tested gradient gating and optional Hessian confidence on one existing factor-8-trained checkpoint in each prepared cohort. These are the same endpoint weights already evaluated with original PCG and constant enhanced PCG, so the comparison isolates recovery. Checkpoints, dataset files and training results were left unchanged. All four checkpoint endpoint gates passed. No training, threshold sweep, or substep study was run.

## Threshold rule

For each case, `tau_g` was set to the median magnitude of the analytical spatial gradient, and `tau_h` to the median smallest singular value of the Hessian, measured over the full grid at the central interval midpoint `t=8.5/16`. These are one data-derived heuristic setting per case, not calibrated defaults. Derivatives and thresholds use native voxel units; they should not be transferred to another spacing or normalization without calibration. Both adaptive branches retained `c_max=4`, the same PCG controls and midpoint RK2.

## Results

Cells show Dice / HD95 mm. Folding is fraction of voxels with nonpositive Jacobian determinant. Recovery time includes analytical derivatives and adaptive scalar summaries; timing is not a controlled solver-only comparison.

| Cohort / case | Original PCG | Enhanced c=4 | Gradient gate | Gate + Hessian confidence |
|---|---:|---:|---:|---:|
| LPBA / S02 | 0.7067 / 6.40 | 0.7117 / 6.37 | 0.7069 / 6.40 | 0.7068 / 6.40 |
| OASIS / OASIS_0002_0000 | 0.7729 / 3.20 | 0.7704 / 3.09 | 0.7727 / 3.20 | 0.7726 / 3.20 |
| Hippocampus / HippocampusMR_0002_0000 | 0.8145 / 2.21 | 0.8187 / 2.21 | 0.8172 / 2.21 | 0.8170 / 2.21 |
| AbdomenCT / AbdomenCTCT_0002_0000 | 0.5205 / 22.14 | 0.5088 / 22.77 | 0.5134 / 22.36 | 0.5140 / 22.35 |

| Cohort | Solver | Folding fraction | Recovery s | Mean effective c | Max relative residual | All intervals converged |
|---|---|---:|---:|---:|---:|:---:|
| lpba | Original | 0 | 18.7 | — | 2.01e-05 | no |
| lpba | Enhanced | 0 | 59.1 | — | 2e-05 | yes |
| lpba | Gate | 0 | 46.9 | 2.10 | 2.01e-05 | no |
| lpba | Gate + H confidence | 0 | 59.6 | 0.78 | 2.02e-05 | no |
| oasis | Original | 1.29e-05 | 28.8 | — | 2.03e-05 | no |
| oasis | Enhanced | 1.77e-05 | 83.5 | — | 1.97e-05 | yes |
| oasis | Gate | 1.24e-05 | 65.1 | 2.05 | 1.99e-05 | yes |
| oasis | Gate + H confidence | 1.25e-05 | 84.7 | 0.73 | 2e-05 | no |
| hippocampus | Original | 7.63e-06 | 2.1 | — | 1.89e-05 | yes |
| hippocampus | Enhanced | 0.000233 | 5.8 | — | 1.8e-05 | yes |
| hippocampus | Gate | 0.000206 | 3.2 | 2.03 | 1.98e-05 | yes |
| hippocampus | Gate + H confidence | 0.000122 | 2.1 | 0.92 | 1.71e-05 | yes |
| abdomenct | Original | 0.000222 | 22.1 | — | 2e-05 | no |
| abdomenct | Enhanced | 0.000441 | 72.9 | — | 2e-05 | yes |
| abdomenct | Gate | 0.000238 | 56.3 | 2.19 | 1.93e-05 | yes |
| abdomenct | Gate + H confidence | 0.000232 | 76.8 | 0.96 | 2e-05 | yes |

## Findings

Gradient gating stayed close to original PCG across the four cases: it slightly raised LPBA and Hippocampus Dice, left OASIS essentially unchanged, and reduced AbdomenCT Dice. It did not consistently improve HD95 or folding. Gate-only PCG took 47–65 seconds on the three large volumes and 3.2 seconds on Hippocampus; recorded timings include diagnostics and should be treated as approximate.

Hessian confidence barely changed Dice/HD95 on three cases. On Hippocampus it reduced folding from `2.06e-4` with gating alone (and `2.33e-4` with constant c=4) to `1.22e-4`, with Dice changing from `0.8172` to `0.8170`. That is a useful local signal, but one case does not establish a general folding benefit. Confidence also reduced the mean effective c from roughly 2.0–2.2 to 0.73–0.96 and tended to require more PCG iterations on the large grids.

LPBA gate-only and confidence variants both narrowly missed the configured `2e-5` relative residual threshold (max `2.010e-5` and `2.019e-5`). OASIS confidence also narrowly missed (`2.001e-5`). The other five variant runs converged. Endpoint gates remained passed in all eight runs.

The screen does not support promoting either adaptive branch. The thresholds were anchored to a single time point, the cohort sample is one case each, and the adaptive outputs are close to the two fixed solver arms.

Compact metrics and per-interval effective-weight diagnostics are in the companion JSON. Full outputs remain ignored under `results/adaptive_hessian_screen/20261111/`; they contain scalar results only. The four source checkpoint hashes were verified unchanged.
