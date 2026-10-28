# Curated results

These are manually curated observations, not a claim that relaxed PD-LK is
better than standard PD-LK. LPBA metrics use MONAI macro mean Dice and symmetric
95th-percentile surface distance in millimetres. A run is eligible here only if
all endpoint reconstruction checks pass and its folding fraction is zero.

## Relaxed PD-LK four-hour sweep

Artifact directory (ignored locally):
`medical/results/relaxed_pdlk_sweeps/20261028T021814_707988Z/`.
The sweep finished in 3.91 hours, attempted 39 candidates, completed 23, and
recorded 16 failures or interrupted admissions. The nine-subject full cohort is
S02–S10. Screening and confirmation had no persistent volume or model-checkpoint
outputs. All completed ranked candidates passed the endpoint and folding gates.
These historical runs used the original 30 dB reconstruction gate and `1e-5`
flow tolerance; current defaults are 25 dB and `2e-5`. Saved flags and scores
remain unchanged. Flow convergence was not an admission gate; many full-cohort subjects had at
least one interval that missed the configured linear-solver tolerance.

| Stage | Seed | rho | Subjects | Mean Dice | Mean HD95 (mm) | Flow-warning subjects | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| Full, matched control | 42 | 0 | S02–S10 | 0.698714 | 6.3175 | 8/9 | Reference |
| Full, matched candidate | 42 | 0.4 | S02–S10 | 0.698721 | 6.3208 | 8/9 | Effectively tied; tiny Dice increase, slightly worse HD95 |
| Full | 43 | 0 | S02–S10 | 0.700248 | 6.2739 | 7/9 | Best observed individual run, not a stable selection |
| Repeat | 44 | 0 | S02–S10 | 0.699309 | 6.2925 | 9/9 | Control |
| Repeat | 44 | 0.4 | S02–S10 | 0.699120 | 6.3058 | 8/9 | Slightly below control |
| Repeat | 44 | 0.004 | S02–S10 | 0.699042 | 6.2958 | 9/9 | Similar range |

The sweep's ranked full runs kept the remaining training and recovery settings
fixed and varied PD-LK regularization and selected model/training settings during
screening. Since the matched differences are very small, seeds are sparse, and
flow convergence warnings remain, retain `rho=0.4` as the exploratory relaxed
configuration and `rho=0` as the essential control. Do not claim a relaxation
gain from this sweep.

The local `summary/REPORT.md`, `summary/trials.csv`, per-trial configs, logs and
metrics are the provenance for these values. No medical data or generated
volumes belong in this document or in Git.

## Older 3D reference run

The prior standard-PD-LK full-cohort run reported mean Dice `0.6987`, mean HD95
`6.317 mm`, zero folding and endpoint gates passed for all nine subjects. Only
S02 had every flow interval converge to tolerance. This is a useful baseline
reference, not an independent comparison to the later sweep.

## 2D Cameraman

The retained run folders include older H1, metamorphosis, flow and PD-LK
experiments. They are ignored local artifacts with original manifests and
configuration provenance. Their varied endpoint-gate outcomes make them
exploratory; no cross-method winner is promoted here. See
[`historical_methods/CAMERAMAN.md`](historical_methods/CAMERAMAN.md) for the
interpretation limits and the current gate policy.
