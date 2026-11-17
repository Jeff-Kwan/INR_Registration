# SIRENPath ablations on Learn2Reg OASIS

The user's seven-run internal ablation design was **launched at 22:40 HKT on 16 November 2026**.
The neural external baseline queue had already completed; its results are preserved. Dataset preparation
is frozen; this launcher reads the released arrays and preparation manifest and
never prepares, rewrites, resamples or expands the cohort.

## Design

All 19 official directed public-validation pairs, fixed 0395–0413 and moving
0396–0414, one seed 42. This is L2R-OASIS, separate from Neurite-OASIS and the
four official-comparison cohorts. See [DATASETS.md](DATASETS.md) and [OASIS.md](OASIS.md).

| Run | Endpoint fitting | Local transport (PD-LK) | Registration (GW-NCC) | Fits |
|---|---|---|---|---:|
| 1. Reconstruction only | MSE | Off | Off | 19 |
| 2. Transport only | MSE | On | Off | 19 |
| 3. Registration only | MSE | Off | On | 19 |
| 4. Both losses | MSE | On | On | 19 |
| 5. SIRENPath | NCC | On | On | 19 |
| 6. Raw intensity crossfade optical flow recovery | No fitting | — | — | 19 no-fit recoveries |
| 7. Canonical reconstructed-intensity crossfade | Reuse run 5 NCC fit | — | — | 19 no-fit recoveries |

**133 evaluations: 95 trained fits and 38 no-training controls.**

Runs 1–5 copy the one canonical configuration: paired SIRENs, 512 channels,
four sine layers, omega 30, output initialization ×30; 5,000 CUDA FP32 captured
updates; AdamW decay 0.001; cosine LR 1e-4→2e-5 without warmup; shared half-size
crop, unchanged reconstruction/particle counts and single-seed sampling policy.
Run 5 is exactly the canonical configuration. Enabled PD-LK has weight 0.1,
c=4 and ridge 1e-6. Enabled registration keeps the 0.03→0.01 cosine schedule.
Turning registration off zeros both its coefficient and schedule start; turning
PD-LK loss off does not disable the velocity solves needed by registration.
When both are off, derivatives and trajectories are omitted from the objective.

For runs 1–4, MSE is **moving MSE + fixed MSE**, with the NCC-specific Gaussian
moment penalty disabled. All five trained runs retain the same inter-SIREN
parameter L2 (0.01) and AdamW decay (0.001). Thus “reconstruction only” means
no transport or registration loss, while these shared regularizers remain.
Run 5 keeps NCC's moment weight 0.005. The 4→5 contrast compares endpoint-fitting
recipes (MSE versus NCC plus its moment anchor), not NCC alone.
MSE is accepted only through the explicit ablation training entry point; the
canonical configuration and ordinary runners still require NCC.

Every run uses the same final frozen stage-3 brightness-flow PCG recovery and
midpoint RK2: 8 intervals × 8 substeps, smoothness 0.01, ridge 1e-6,
500 PCG iterations maximum, tolerance 2e-5, axis cap 256, forward and reverse
integration with warm-started velocity solves. At the released OASIS size the cap
has no effect. No score-based model or case selection is performed.

## Raw crossfade control

Use the same independently minmax-normalized endpoints as the trained models:

`I(x,t) = (1-t) I_moving(x) + t I_fixed(x)`.

Its temporal derivative is exactly `I_fixed - I_moving`. Native-voxel spatial
image derivatives use centered differences internally and first-order one-sided
differences at image boundaries; their crossfade is sampled trilinearly with
border extension. These derivatives replace SIREN derivatives at the same temporal
midpoints. PCG, warm starts, RK2 integration, coefficients, grids and evaluation
are unchanged. No SIREN is constructed and no optimizer is invoked. The control
checkpoint contains provenance only, with no source images, gradients or maps.

The fitted SIRENs continue to use their analytical derivatives; there is no need
to approximate SIREN gradients with finite differences. The raw control tests
whether linear image-space interpolation plus the same recovery can explain
registration quality. A gap also includes the effects of neural reconstruction
and derivative estimation, so it does not isolate weight interpolation alone.

## Canonical reconstruction crossfade control (run 7)

Reuse the **exact corresponding run-5 canonical checkpoint**, without retraining.
Decode moving/fixed SIREN endpoint reconstructions on the original voxel grid,
then apply the same finite-difference gradient machinery and stage-3 recovery as
run 6 to their intensity crossfade. Preserve the fitted signed intensities and
NCC scale/offset: **no minmax renormalization, clipping or intensity calibration**
of decoded reconstructions. Diagnostics compare these reconstructions against
the original normalized images, so **PSNR and Patch NCC equal run 5**, not the
infinite PSNR/exact Patch NCC of raw identity endpoints.

The launcher defers run 7 until the earlier stage has completed, checks the
matching image paths, shape/affine, canonical config and checkpoint model, and
records source-checkpoint SHA-256. Native-voxel geometry is required. A missing
or failed canonical dependency is recorded as a failed dependent evaluation;
no fallback training occurs. Reconstructed volumes, finite-difference fields
and maps remain in memory; the control checkpoint contains only provenance.

This gives a direct **5 versus 7** comparison with identical fitted endpoints:
weight-space interpolation with analytical SIREN derivatives versus intensity
crossfade with finite-difference derivatives. This contrast still changes the
derivative estimator as well as the path. **6 versus 7** shows the effect of
using raw endpoints versus the actual NCC-trained reconstructions.

Run-7 reported runtime is incremental checkpoint loading, decoding, diagnostics,
finite differences, recovery and scoring, with zero new training. The reused
training seconds and a method runtime including that shared fit are also retained
in case metrics (`reused_training_seconds`, `Runtime_with_shared_training`).
For a separate run-7-only evaluation, point to an existing run-5 directory:

```bash
bash scripts/launch_sirenpath_ablations.sh \
  --output results/sirenpath_reconstructed_crossfade \
  --runs 7 --canonical-root results/sirenpath_l2r_oasis_ablations/05_sirenpath \
  --plan-only
```

## Metrics and report

The launcher writes per-case metrics, logs, checkpoints and run provenance,
`progress.json` after every case, `summary.json` on completion and a live
`ABLATIONS_REPORT.md` in the run directory. Aggregates are case means and
population SD, with scored/intended denominators and process failures explicit.

- **PSNR:** whole-grid endpoint PSNR with peak/data range 1 after independent
  minmax normalization; arithmetic mean of moving and fixed dB values. No clipping
  or fitted affine intensity calibration is applied to SIREN output.
- **Patch NCC:** signed Pearson correlation in every fully contained 9×9×9 window,
  stride 1, averaged over windows, then over moving/fixed endpoints. Windows with
  target variance ≤1e-8 are excluded; constant predictions on valid target windows
  score zero. Denominator is max(sqrt(var_prediction × var_target),1e-8). No labels
  or anatomical masks select windows. FP64 moments are computed in bounded slabs.
- **DSC and HD95:** pinned Learn2Reg OASIS label handling and nearest-neighbour
  pullback, labels 1–35, original evaluation grid, HD95 in official voxel units
  (numerically mm at the released 1 mm spacing). Missing/nonfinite distances stay
  null; coverage is retained, never silently filled with zero.
- **NPJ (%):** 100×mean(detJ≤0), central differences on the two-voxel interior.
- **SDlogJ:** std(log(max(detJ,1e-9))) on that same interior, matching the unshifted
  convention used in the main comparisons. The official L2R shifted SDlogJ
  std(log(clip(detJ+3,1e-9,1e9))) is also retained in every case as
  `LogJacDetStd` / `official_L2R_SDlogJ` to preserve author-protocol reporting.
- **Runtime:** worker wall time including input load, training or finite-difference
  setup, reconstruction diagnostics, recovery, scoring and checkpoint serialization.
  Method-level component timings are retained separately; raw training time is zero.

For the raw control, exact endpoints make **PSNR +∞ and Patch NCC 1 by
construction** (if valid nonconstant windows exist). These are not evidence of
registration quality. JSON represents +∞ with null numeric mean and explicit
`positive_infinity` / `infinite_pairs` fields; the Markdown report displays ∞.

## Launching later

Plan all runs without creating outputs, launching workers or touching CUDA:

```bash
bash scripts/launch_sirenpath_ablations.sh \
  --output results/sirenpath_l2r_oasis_ablations --plan-only
```

Optional frozen quick4 plan uses pairs **1,4,8,9**:

```bash
bash scripts/launch_sirenpath_ablations.sh \
  --output results/sirenpath_l2r_oasis_quick4 --subset quick4 --plan-only
```

After the external jobs finish, launch the full study into a fresh output directory:

```bash
bash scripts/launch_sirenpath_ablations.sh \
  --output results/sirenpath_l2r_oasis_ablations \
  --gpus 0 1 2 3 4 5 6 7 --run
```

`--runs 1 2 3 4 5 6 7` and `--pairs` select explicit subsets; default is all seven
runs and all 19 pairs. The launcher refuses selected GPUs with active compute
processes. It copies resolved configurations and records the frozen preparation,
protocol, configuration and relevant source hashes. Persistent GPU workers reset
the seed and create fresh models/optimizers for each fit. Outputs must be outside
Data; dense maps and warped/reconstructed images are never saved. Temporary
recovery scratch is removed on success and exceptions.

Validation performed during preparation is CPU-only synthetic derivative, gradient,
recovery, report and plan testing. No optimizer steps, real-image recovery or
ablation training were run, and no existing job was stopped or modified.

Preparation verification: **27 CPU checks passed; 2 CUDA checks skipped** with
CUDA hidden. The full 133-job plan has zero missing inputs and creates no run
output in plan-only mode. The pre-existing neural baseline tmux launcher remained running.

## Ablation launch — 2026-11-16

Launched all seven SIRENPath ablations at **22:40 HKT on 16 November**, in tmux
session `inr-sirenpath-ablations`, using all eight previously idle RTX 5090 GPUs.
133 evaluations: 95 fresh trained fits and 38 no-training controls. Run 7 is
scheduled after run 5 and reuses its exact per-pair checkpoints. Initial estimate:
**20–35 minutes**, approximately **23:00–23:15 HKT**, subject to first-time
compilation and diagnostic/recovery costs. Baseline investigation remains CPU-only.
Dataset preparation remains frozen.

[Live report](../results/sirenpath_l2r_oasis_ablations_20261116/ABLATIONS_REPORT.md) ·
[Progress](../results/sirenpath_l2r_oasis_ablations_20261116/progress.json) ·
[Plan](../results/sirenpath_l2r_oasis_ablations_20261116/plan.json) ·
[Final summary when complete](../results/sirenpath_l2r_oasis_ablations_20261116/summary.json).

Completion update: **133/133 ablation evaluations completed, zero process failures**, at **22:55 HKT on 16 November 2026**. Final results are in the linked live report and summary.

Completed aggregate scores: [ABLATIONS_REPORT.md](ABLATIONS_REPORT.md).
