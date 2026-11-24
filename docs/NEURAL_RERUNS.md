# Corrected NIR and DNVF replacements

Both corrected reruns completed all 309 cases per method (618 fits, seed 42)
across Neurite-OASIS (49), ACDC (100), LPBA40 (45) and IXI (115).
The corrected results replaced the original NIR and DNVF scores in the
[benchmark report](BASELINES_REPORT.md#corrected-nir-and-dnvf-reruns).

## Corrected settings

- **NIR:** native local NCC9 + negative-Jacobian weight1000; Adam lr1e-4,
  900 requested updates (904 native full-loader-pass steps); Gaussian sigma3,
  128 Fourier vectors with the paper's2π encoding; sine frequency30. The native
  loop is retained with autocast and GradScaler disabled. Convolution and matmul
  use IEEE FP32 (TF32 disabled). Inputs preserve the prepared intensity arrays,
  including LPBA40's white-matter normalization; no independent minmax or clipping.
  Exact source changes are recorded in `baselines/patches/nir-mri.patch`.
- **DNVF:** unchanged released single-scale200-update recipe, source network,
  optimizer, losses, resolution and seed. TF32 convolution/matmul are disabled
  through a scoped precision context. No autocast was used originally or now.

These remain adaptations to the frozen benchmark cohorts/grids; author cohorts,
LR schedule details and our shared scoring differ. All source/configuration hashes
are recorded in the prepared replacement plan. The original invalid fitting was
not a validated paper-reproduction result.

## Controlled DNVF test

IXI pair66, same seed42 and original source/settings, only TF32 disabled:

| Metric | Original | IEEE FP32 |
|---|---:|---:|
| Dice |0.057563|0.740307|
| HD95 |23.658882|3.680531|
| NPJ (%) |3.413493|0|
| SDlogJ |4.908617|0.376419|
| Adapter elapsed seconds |89.98|98.22|

Final native image loss improves from−0.030175 to−0.273317; flow regularization
from0.073512 to0.029615. This controlled test strongly supports TF32 convolution
as the source of collapse in this case; full-cohort impact is not assumed from
one case. [Test metrics](../baselines/results/dnvf_tf32_off_20261116/ixi/dnvf/pair066/metrics.json) ·
[Test plan and precision flags](../baselines/results/dnvf_tf32_off_20261116/plan.json).
The single-case diagnostic does not substitute for the requested full rerun.

## Pre-launch estimate

Requested heuristic: **sum(old case runtimes) ×1.10 ÷8 GPUs**.

| Method | Cases | Old total GPU seconds | Estimate on8 GPUs |
|---|---:|---:|---:|
| NIR |309|6,773.65|15.5 min|
| DNVF |309|19,193.98|44.0 min|
| Combined |618|25,967.63|**59.5 min**|

Budget **about one hour after launch** under that rule. This is a provisional
projection: NIR now uses FP32 local NCC in place of AMP/MSE, so its runtime may
increase by more than10%. The controlled DNVF case's measured increase was about9%.
Eight GPUs are assumed available, one fit per GPU from a shared queue.

## Commands and outputs

Inspect the replacement jobs without launching or modifying outputs:

```bash
bash scripts/rerun_neural_baselines.sh --plan-only
```

The original launch command was:

```bash
bash scripts/rerun_neural_baselines.sh --run --gpus 0 1 2 3 4 5 6 7
```

The existing official neural output directory is reused. `--prepare` explicitly
deletes/replaces only the two selected methods' outputs and resets their scores;
it has already been performed. `--run` refuses duplicate case directories, changed
prepared source/configuration hashes, or GPUs with active compute processes. It
updates progress, summaries and [BASELINES_REPORT.md](BASELINES_REPORT.md) after
each case. Logs, checkpoints and provenance replace the invalid results in place;
no dense maps or warped images are saved.

[Replacement plan](../baselines/results/official_neural_20261116/replacement_plan.json) ·
[Current progress](../baselines/results/official_neural_20261116/progress.json) ·
[Current aggregates](../baselines/results/official_neural_20261116/summary.json).

Preparation checks:19 CPU-only tests passed, including finite native NCC gradients,
exact spectral settings, precision restoration, source verification and deletion
boundaries. The combined618-job plan has zero missing frozen inputs. No NIR optimizer
step or full NIR/DNVF replacement run was executed during preparation.

## Thin-volume NIR completion

The four failed ACDC/NIR cases11,47,61,97 were rerun successfully with the
deterministic adaptive NCC support rule: largest available odd extent up to9
per sampled axis, giving9×9×7 for these cases. The same kernel is used for
all NCC moments, normalization and auxiliary background-mask operations.
Supported9×9×9 losses and gradients are bitwise identical to the preceding
source implementation. No sampler, input preparation or other case was changed.

ACDC/NIR now has **100/100 successful fits**. Its row was directly replaced
in the separate corrected-rerun table in [BASELINES_REPORT.md](BASELINES_REPORT.md).
All **618/618 corrected NIR/DNVF fits now succeed**, zero remaining process failures.
[Supplementary four-case plan](../baselines/results/official_neural_20261116/nir_thin_volume_plan.json) ·
[Four-case completion](../baselines/results/official_neural_20261116/nir_thin_retry_summary.json).
