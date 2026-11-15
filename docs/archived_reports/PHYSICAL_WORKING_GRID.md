# Experimental 1 mm working grid

This branch adds `medical/configs/physical_working_grid.json`, copied from the
default PD-LK/NCC preset with `working_grid: {"spacing_mm": 1.0, "crop_extent_mm": 80}` added. Omitting
that section retains native-grid execution. The same preprocessing is compatible
with the optional inverse-metric objective. External baselines are unchanged.

The reference geometry is the original prepared fixed image's XYZ shape and
NIfTI affine. To compare resampled versions, supply that original geometry as
`working_grid.reference_geometry`, containing `shape` and `affine`; reuse the
exact same values for every version. Merely inferring a new reference from each
resampled input does not guarantee matching physical bounds. Endpoints must
still share a grid and affine. The official Learn2Reg adapters continue to require
original release inputs and labels; controlled resampled-input comparisons can
exercise preprocessing separately rather than replacing official evaluation data.

The reference axis directions, center, rotations and reflections are preserved.
For each voxel-center extent L=(N-1)*spacing, the working size is ceil(L)+1 at
1 mm spacing, centered on the reference. Extents within 1e-6 mm of an integer
are snapped before rounding. Sheared, nonfinite and degenerate affines are
rejected. Geometry metadata is XYZ; training/recovery tensors remain ZYX with
XYZ vector channels. Normalized SIREN coordinates continue to span the working
voxel centers. All current derivative, regularizer and recovery coefficients are
interpreted in this fixed working-grid geometry; this does not make them
independent of anatomical field of view.

Images use trilinear interpolation with nearest-edge extension. Fine source axes
are antialiased using Gaussian sigma 0.5*sqrt((1/source_spacing)^2-1) in source
voxels, with nearest-edge filtering. Upsampling adds no filter. Already aligned
1 mm images bypass interpolation. Independent min/max normalization follows
resampling. DIR-Lab masks use nearest interpolation with zero outside support;
empty results fail. Labels and landmarks remain evaluation-only on native grids.

Training runs on the working grid. PCG+RK2 recovery uses an independently configured grid (1 mm by default). Map conversion samples
displacement, adds it to the working query coordinate, and transforms the point
back through the affines. Evaluation and Jacobians use the original dataset grid,
and DIR-Lab reports both landmark directions with original lung-mask support.
No images or maps are persisted. Existing recovery scratch cleanup is retained.

Experimental metrics report `reconstruction.working_grid` and
`reconstruction.original_grid` separately. Original-grid PSNR samples the endpoint
network at original voxel centers and uses working-image normalization bounds,
peak 1, with no clipping. Both scopes have separate launcher aggregate keys;
legacy unscoped PSNR keys remain for native runs and are unscored for experimental
runs. Reports include preprocessing, training, reconstruction, recovery and evaluation
time (plus the existing combined recovery/evaluation field), working shape, normalization bounds and resampling details.

Case-runner checkpoints include `working_grid` metadata and the resolved config.
Restore them using `medical.working_grid.restore_checkpoint(payload, device)`
after loading the payload with `torch.load(..., weights_only=True)`. This validates
geometry and normalization, attaches them to the pair, and rejects missing metadata.
`recover` automatically validates attached metadata against the requested ZYX
shape; callers may also pass `working_grid_metadata` explicitly. Use the working
shape, not the original shape. Metadata is part of the checkpoint contract.

## Plan-only quick4 commands

```bash
/venv/main/bin/python -m medical.oasis_experiments --methods ours --subset quick4 \
  --config medical/configs/physical_working_grid.json --gpus 0 1 2 3 \
  --output results/oasis_quick4_physical
/venv/main/bin/python -m medical.abdomen_experiments --methods ours --subset quick4 \
  --config medical/configs/physical_working_grid.json --gpus 0 1 2 3 \
  --output results/abdomenctct_quick4_physical
```

These commands launch no workers. Execution requires a separately requested run
and `--run`. At 1 mm, abdomen has approximately eight times as many recovery
voxels as at 2 mm; recovery strategy changes and experiments are deferred.

## Numerical limits and checks

Consistency means a shared physical working geometry, units and sampling rules.
Interpolation, acquisition blur, lost frequencies, discretized masks and stochastic
optimization prevent exact equality for arbitrary resampled images. Nearest-edge
extension also affects borders; phantom checks distinguish interior errors.

`/venv/main/bin/python -m unittest medical.tests.test_working_grid` runs synthetic
geometry, axis-order, interpolation, masking, displacement conversion,
normalization/reconstruction, checkpoint, sampling/gradient and mocked adapter
checks. No optimizer steps or real-case recovery are required.

Implementation validation: 63 medical checks (56 passed, 7 CUDA opt-ins skipped)
and 22 baseline checks passed using `/venv/main/bin/python`. Both quick4 plans
resolved all inputs, created no experiment directories and launched no workers.
The focused working-grid module contains 12 synthetic/mocked checks. Phantom
interior RMS error is bounded below 0.004 intensity units across 0.5, 1 and 2 mm
sampling; exact geometry and affine-map tolerances are tested separately. No
real-case accuracy, training stability or memory/performance claim is made.

The current experimental preset uses an 80 mm cube for training: 81 voxel
centers per axis, 8 shared-grid cells per axis (10 mm cells), and a central
64 mm particle-start region. `working_grid.crop_extent_mm` overrides fractional
crop sizing; random crop placement is retained. It must be a positive integer
at 1 mm spacing. Volumes smaller than the crop fail rather than silently shrinking
it. Omitting this option retains the configured fractional crop behavior.

## Independent recovery spacing

`working_grid.spacing_mm` fixes training to 1 mm; `crop_extent_mm` controls its
training crop. `working_grid.recovery_spacing_mm` independently controls recovery
and defaults to `"original"`: the frozen reference image's spacing along each
axis, including anisotropic or submillimetre spacing. Explicit finite scalar
overrides >= 1 mm remain available. For example, set it to 2.0 to prepare coarser recovery while leaving training and endpoint PSNR
unchanged. This is configuration only, not authorization to run recovery.

Recovery preserves the full working domain's voxel-center bounds, axis directions
and origin. Each axis uses ceil(extent/requested_spacing) intervals, at least one;
its actual spacing is extent/intervals. Actual spacing can differ slightly by axis
and is at most the requested value. Metadata records both the requested spacing
and exact recovery geometry. SIRENs are queried at normalized coordinates across
those same bounds. Gradient components are converted from recovery-voxel units
to per-mm units; velocity is solved in mm per unit time. PCG uses axis-wise
1/h^2 smoothness weights, including boundary degrees and its preconditioner.
Warm starts remain physical velocities. RK2 and temporary velocity scratch use
recovery-grid coordinate velocities (physical velocity divided by XYZ spacing).
Final displacement conversion uses the recovery affine; original evaluation grids
and training geometry are unchanged. Diagnostics include actual spacing and shape.

Call `recover` with the *training working shape*, as before; metadata selects the
recovery shape internally. Returned maps are on the recovery grid. Use `predictor`
with the same metadata to convert them. Old working-grid metadata without a
recovery entry uses its working grid. Native runs without working-grid metadata
retain their existing voxel-space behavior. PCG algorithm, stopping and integration
schedule are unchanged; coarse recovery is a different discretization and does
not guarantee identical results. No default resolution reduction or recovery
performance experiment is included in this change.

Grid-separation validation: the medical suite ran 68 checks (60 passed, eight
CUDA opt-ins skipped); the weighted and existing compiled PCG parity checks
also passed separately on CUDA. Weighted 2D/3D systems match independently
assembled FP64 solutions, and synthetic translation tests verify derivative,
velocity and displacement units across recovery resolutions. No real-case
recovery or performance benchmark was launched for this change.

## Retained OASIS quick4 result

The random-two-pair 80 mm-crop run completed 4/4 pairs, seed 42, 5000 steps:
mean DSC 0.770455, HD95 2.329283, shifted SDlogJ 0.050633, original-grid
moving/fixed PSNR 34.060/34.378 dB, zero measured folding and 64/64 converged
PCG intervals. Mean end-to-end runtime was 119.83 seconds per pair. Source:
`results/oasis_quick4_physical80_pdlk_20261112_143958/summary.json`.

The discarded three-axis and local-recovery alternatives are recorded only in
[historical notes](archived_reports/discarded_variants_20261112.md); their code
and generated artifacts were removed. Final cleanup validation ran 69 medical
checks (61 passed, eight optional CUDA checks skipped) and 22 baseline checks.

## Historical global reconstruction sampling experiment

In that experiment, endpoint reconstruction drew uniformly from the entire working volume,
independently of the registration crop. Unmasked endpoints share global voxel
indices as before; DIR-Lab endpoints retain independent full-mask sampling.
Registration crop placement has a separate generator (seed + 4000051), so crop
and reconstruction draws cannot perturb each other's random streams. NCC particle
sampling, 80 mm crop, velocity estimation, two Euler updates and recovery are
unchanged. This corrects the previous unintended crop restriction, rather than
adding a new reconstruction ablation mode. Historical cropped-sampling results
remain historical; the changed RNG stream also prevents exact crop-sequence
matching against those runs.

## Current cropped reconstruction and RK2 experiment

The user restored crop-coupled endpoint reconstruction after the global-sampling
experiments. Default sampling and its seed-42 RNG sequence match the original
80 mm cropped run again. Independent DIR-Lab endpoint masks are retained.
Training now uses one midpoint RK2 step of length 0.5 per existing velocity field
(reverse uses -0.5 and reversed field order). Midpoint and endpoint support are
checked; gradients through both velocity queries remain live. Fields retain their
random half-interval times: this improves spatial integration but does not imply
second-order accuracy in time for the continuously varying SIREN trajectory.
Inference PCG+RK2 is unchanged. Structure sampling remains an optional preset.

### Cropped reconstruction + training RK2 quick4 result

Source: `results/oasis_quick4_cropped_rk2_20261112/summary.json`. All four
frozen OASIS pairs completed at seed 42, 5000 steps, PD-LK 0.1 and NCC 0.01.

| Fixed → moving | DSC | HD95 | Original PSNR moving/fixed (dB) |
| --- | ---: | ---: | ---: |
| 0395 → 0396 | 0.780796 | 1.953083 | 34.32 / 34.30 |
| 0398 → 0399 | 0.715427 | 3.396905 | 33.90 / 34.52 |
| 0402 → 0403 | 0.775977 | 2.011565 | 34.01 / 34.63 |
| 0403 → 0404 | 0.804023 | 1.987498 | 34.06 / 34.10 |

Mean DSC 0.769056 versus original cropped Euler 0.770455; HD95 2.337263
versus 2.329283. Mean original moving/fixed PSNR 34.073/34.388 dB.
Mean case runtime 127.94 s versus 119.83 s. All 64 PCG intervals converged;
zero measured folding. Each case had slightly lower DSC than cropped Euler.
No registration improvement was observed in this single-seed quick4 comparison.

### Cropped RK2 with endpoint NCC and moments

Source: `results/oasis_quick4_cropped_rk2_endpoint_ncc_20261112/summary.json`.
Preset: `medical/configs/physical_working_grid_endpoint_ncc.json`. Only the
endpoint objective changed from the preceding cropped RK2 run: sum of endpoint
1-Pearson NCC plus 0.01 Gaussian-moment KL toward mean zero/variance one.
PD-LK 0.1, registration NCC 0.01, seed 42 and 5000 steps retained.

| Pair | DSC | HD95 | Rescaled PSNR moving/fixed (dB) | Folding |
| --- | ---: | ---: | ---: | ---: |
| 0395_to_0396 | 0.817815 | 1.672781 | 22.82 / 22.69 | 0.000821% |
| 0398_to_0399 | 0.787449 | 2.817236 | 23.49 / 21.04 | 0.001844% |
| 0402_to_0403 | 0.823132 | 1.639937 | 22.55 / 21.37 | 0.003596% |
| 0403_to_0404 | 0.832579 | 1.724877 | 20.85 / 22.03 | 0.004495% |

Mean DSC 0.815244 versus MSE+RK2 0.769056; HD95 1.963708 versus 2.337263.
Shifted SDlogJ increased from 0.050545 to 0.079155. All four improved DSC,
but all have small nonzero unshifted folding. All 64 PCG intervals converged.
Mean run time 125.42 s/case, excluding the supplemental rescaling evaluation.

Supplemental PSNR: 22.426/21.783 dB, obtained by mapping each
whole working-volume prediction's min/max to its target's min/max before computing
MSE, with target intensity range as peak. For OASIS this is also the original
1 mm grid and target bounds are [0,1]. This output-range-sensitive metric is
reported separately from raw PSNR (6.011/6.276 dB), which has a different
intensity gauge from the target. Supplemental results are stored as
`rescaled_psnr.json` per case and `rescaled_psnr_summary.json` at run root.
Two streaming endpoint-forward passes save no volume; no additional training or
recovery was performed. A synthetic affine-intensity check validates the formula.

## Training Euler restored after the NCC comparison

At the user's request, training particle trajectories are back to two Euler
half-steps, with reverse field order and negative steps for the reverse direction.
The RK2 training implementation was removed; its experiment results above remain
historical. Inference still uses PCG+RK2. Cropped reconstruction and the optional
endpoint NCC/moment preset remain available. No Euler + endpoint NCC experiment
was launched during this restoration, so the DSC 0.815244 result above applies
only to the completed NCC + training RK2 run.

## RK2 restored for 32k registration-particle comparison (2026-11-13)

The user requested restoration of the exact two-spatial-midpoint-half-step
training implementation to compare against the NCC+RK2 result. Current training
therefore uses RK2 again. Synthetic field, support and derivative checks passed
(15 tests). Inference remains PCG+RK2.

Preset: `medical/configs/physical_working_grid_endpoint_ncc_32k.json`. Source:
`results/oasis_quick4_cropped_rk2_endpoint_ncc_32k_20261113/summary.json`.
The config differs from the previous endpoint NCC+RK2 run only in registration
particles per direction: 4096 → 32768. Reconstruction remains 32768 discrete
voxel samples throughout the crop; both directional registration start sets are
independent continuous samples in the central 64 mm region. Samples are not tied.
Seed 42, 5000 steps, endpoint NCC + moment weight 0.01, PD-LK 0.1, registration
NCC 0.01, grid/probes/crop and recovery settings are retained. Particle RNG draws
and training trajectories naturally change with particle count.

| Pair | DSC | HD95 | Target-range-rescaled PSNR moving/fixed (dB) | Folding |
| --- | ---: | ---: | ---: | ---: |
| 0395_to_0396 | 0.820054 | 1.677693 | 21.85 / 21.04 | 0.000790% |
| 0398_to_0399 | 0.793508 | 2.800925 | 22.53 / 20.39 | 0.001705% |
| 0402_to_0403 | 0.826576 | 1.462807 | 22.67 / 20.33 | 0.005037% |
| 0403_to_0404 | 0.834221 | 1.719685 | 20.88 / 22.38 | 0.004526% |

Mean DSC 0.818590 versus 4k 0.815244; all four pairs improved. HD95
1.915278 versus 1.963708; shifted SDlogJ 0.079323 versus 0.079155.
Mean case runtime 127.82 s versus 125.42 s (supplemental PSNR excluded).
All 64 PCG intervals converged. Mean folding 0.003014%
versus 4k 0.002689%. Target-min/max-matched PSNR moving/fixed
21.983/21.037 dB versus 22.426/21.783 dB; raw PSNR
6.033/6.274 dB. Whole-volume endpoint-forward-only supplemental evaluation
saves scalar summaries and extrema, not images. These are single-seed quick4
development results and do not establish a cohort-wide improvement.

## Tied 32k points: NCC + RK2 quick4 (2026-11-13)

Preset: `medical/configs/physical_working_grid_endpoint_ncc_32k_tied.json`.
Source: `results/oasis_quick4_cropped_rk2_endpoint_ncc_32k_tied_20261113/summary.json`.
All four cases completed seed 42, 5000 steps. The only added configuration is
`training.tie_registration_points: true`, default false. It requires equal endpoint
batch size and registration count, no structure sampler and no endpoint masks.
This option currently rejects masked inputs rather than changing DIR-Lab support.

One shared set of 32768 continuous starts supplies both endpoint reconstructions
and both directional registration trajectories. Targets use trilinear interpolation.
Reconstruction covers the central 64 mm start region rather than discrete voxels
throughout the 80 mm crop, so the experiment changes reconstruction coverage and
interpolation as well as point sharing. It does not isolate covariance of samples.
Crop and particle RNG consumption match the separate-32k control: ordinary crop
voxel draws and both directional start draws are still consumed, then the first
start set is reused. Tests verify shared object identity and interpolation on an
analytic linear field; 16 focused RK2/derivative/sampling checks passed.

| Pair | DSC | HD95 | Target-range-rescaled PSNR moving/fixed (dB) |
| --- | ---: | ---: | ---: |
| 0395_to_0396 | 0.806819 | 1.764583 | 20.55 / 19.93 |
| 0398_to_0399 | 0.778283 | 2.837969 | 21.17 / 19.02 |
| 0402_to_0403 | 0.818425 | 1.683098 | 20.39 / 19.51 |
| 0403_to_0404 | 0.824023 | 1.836394 | 20.00 / 20.66 |

Mean DSC 0.806888 versus separate-32k 0.818590; all four decreased. Mean HD95
2.030511 versus 1.915278; shifted SDlogJ 0.078668 versus 0.079323.
Runtime 127.81 versus 127.82 s/case, supplemental PSNR excluded. Mean folding
0.002166% versus 0.003014%; all 64 PCG intervals converged.
Target-min/max-matched whole-volume PSNR 20.528/19.780 dB versus
21.983/21.037 dB; raw PSNR 6.602/6.773 dB. Scalar supplemental summaries are
saved alongside results, without image artifacts or additional recovery.
Default sampler remains separate. No other datasets or seeds were run.

## Endpoint GW-NCC, separate 32k + RK2 quick4 (2026-11-13)

Preset: `medical/configs/physical_working_grid_endpoint_gwncc_32k.json`.
Source: `results/oasis_quick4_cropped_rk2_endpoint_gwncc_32k_20261113/summary.json`.
Only endpoint similarity changed from the separate-32k NCC+RK2 control.
Reconstruction and registration are separate, with 32768 samples/particles per
endpoint/direction. Endpoint GW-NCC reuses existing global/window 50/50 GW-NCC,
8³ moment bins over the 80 mm reconstruction crop (10 mm bins), sigma 0.75 cells
(nominal 7.5 mm), three-tap reflect-padded smoothing. Both endpoint terms are
summed, retaining the existing 0.01 Gaussian-moment KL toward N(0,1). This is a
fixed output-moment prior, not matching the observed target moments. Box NCC
registration 0.01, PD-LK 0.1, seed 42, 5000 steps, grid and recovery retained.

18 focused synthetic tests passed. Eager and compiled FP32 full endpoint
gradients at actual model width/depth and 32768 samples agreed to relative
L2 3.784e-7 (acceptance 1e-3). No derivative path was detached.

| Pair | DSC | HD95 | Target-range-rescaled PSNR moving/fixed (dB) |
| --- | ---: | ---: | ---: |
| 0395_to_0396 | 0.814211 | 1.736347 | 21.47 / 20.33 |
| 0398_to_0399 | 0.773470 | 2.915789 | 22.15 / 17.72 |
| 0402_to_0403 | 0.822682 | 1.598053 | 19.82 / 19.14 |
| 0403_to_0404 | 0.831071 | 1.809424 | 19.53 / 19.77 |

Mean DSC 0.810359 versus global endpoint NCC 0.818590; all four decreased.
HD95 2.014903 versus 1.915278; shifted SDlogJ 0.073096 versus 0.079323;
mean folding 0.001267% versus 0.003014%. All 64 PCG intervals converged.
Mean case time 139.98 versus 127.82 s (supplemental PSNR excluded).
Target-min/max-matched whole-volume PSNR 20.742/19.242 dB versus
21.983/21.037 dB; raw PSNR 6.818/7.075 dB. Results are single-seed quick4
development evidence; no other datasets or seeds were launched.

## Endpoint GW-NCC 4³ loss grid, separate 32k + RK2 (2026-11-13)

Source: `results/oasis_quick4_cropped_rk2_endpoint_gwncc4_32k_20261113/summary.json`.
Preset: `medical/configs/physical_working_grid_endpoint_gwncc4_32k.json`.
Only `objective.endpoint_gw_grid_size: 4` differs from the prior endpoint GW-NCC
configuration (implicit default 8). The velocity-estimation grid is still 8³,
and registration still uses box NCC. Separate 32k reconstruction/registration,
RK2, signed correlation, global/window 50/50 mixture, sigma 0.75 cells and
three-tap kernel, N(0,1) moment prior weight 0.01, PD-LK 0.1 and registration
NCC 0.01 remain unchanged. Physical loss-bin width is now 20 mm and nominal
Gaussian sigma 15 mm, so this tests both coarser bins and broader windows.
All four frozen cases completed seed 42 and 5000 steps. The 18 focused synthetic
checks passed, including GW-NCC values and gradients at both grid sizes.

| Pair | DSC | HD95 | Target-range-rescaled PSNR moving/fixed (dB) |
| --- | ---: | ---: | ---: |
| 0395_to_0396 | 0.816305 | 1.709512 | 22.39 / 21.23 |
| 0398_to_0399 | 0.786145 | 2.821365 | 21.45 / 18.40 |
| 0402_to_0403 | 0.825525 | 1.598651 | 21.60 / 19.64 |
| 0403_to_0404 | 0.832029 | 1.753707 | 21.13 / 20.44 |

Mean DSC 0.815001 versus 8³ GW-NCC 0.810359 and global NCC 0.818590.
All four improved DSC versus 8³ GW-NCC, but remain below global NCC.
HD95 1.970809 (8³ 2.014903; global 1.915278), shifted SDlogJ 0.074273
(8³ 0.073096; global 0.079323). Mean folding 0.001414%; all 64 PCG
intervals converged. Mean case time 138.04 s versus 8³ 139.98 s and global
127.82 s, supplemental PSNR excluded. Target-min/max-matched whole-volume PSNR
21.642/19.925 dB. Raw PSNR 6.470/6.730 dB is separately retained.
No other datasets, seeds, squared-correlation variant or commits were performed.

## Explicit sampled-window endpoint NCC (2026-11-13)

Preset: `medical/configs/physical_working_grid_endpoint_lncc_32k.json`.
Endpoint `lncc` draws 44 uniformly distributed continuous centers within each
current reconstruction crop, restricted so each complete window fits. Each
window has 9³ samples spaced 1 mm apart (8 mm center-to-center extent), totaling
32076 SIREN evaluations per endpoint. Endpoints share reconstruction coordinates;
registration retains separate 32768 particles per direction and the 8³ velocity
grid. Observed intensities use existing trilinear interpolation. Physical offsets
are converted using working-grid spacing. Endpoint masks and tied particles are
explicitly unsupported for this experimental sampler.

The endpoint loss is 50% batch-global signed NCC plus 50% mean signed window NCC,
summed over endpoints, plus the existing 0.01 N(0,1) output-moment prior. Window
statistics are explicitly centered, with epsilon 1e-8 added to each variance;
constant windows remain in the equal-window average. Full training gradients,
PD-LK 0.1, box registration NCC 0.01, RK2 and recovery are retained.
The usual crop draw is consumed before window sampling; the additional window
center draws change the subsequent crop RNG sequence versus earlier controls.
Window sampling also changes the spatial weighting of the batch-global term.
Thus this first experiment tests a complete window-sampling objective, including
smaller locality scale than the earlier GW variants, rather than isolating only
GW approximation error. The reported batch_size remains the nominal 32768 budget;
actual endpoint count is windows × size³. Defaults and external baselines are unchanged.

Related formulations: [VoxelMorph local correlation](https://web.mit.edu/adalca/www/files/papers/cvpr2018_voxelmorph.pdf)
and [patch-based LNCC in INR registration](https://pmc.ncbi.nlm.nih.gov/articles/PMC10575995/).
Our initial experiment retains signed correlation; it does not reproduce their
complete published objectives or training recipes.

Quick4 completed 4/4: mean DSC 0.755153, HD95 2.445442, shifted SDlogJ
0.0673055, runtime 137.46 s/case. All 64 PCG intervals converged.
Target-range-rescaled whole-volume PSNR moving/fixed 15.764/15.312 dB.
This underperformed global endpoint NCC (DSC 0.818590) and GW-NCC4
(0.815001) on every case. Source: `results/oasis_quick4_cropped_rk2_endpoint_lncc_32k_20261113/summary.json`.

## Abdomen NCC quick4: checkpoint-only 2 mm recovery (2026-11-13)

Same fitted weights from `abdomenctct_quick4_physical_endpoint_ncc_32k_20261113`;
only recovery spacing changed. Physical bounds, derivative units, PCG settings
and original official evaluation grid are retained. No training or PSNR recomputation.

| Recovery | Mean DSC | HD95 (voxels) | SDlogJ | Folding (%) | Recovery seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1mm | 0.343003 | 20.249838 | 0.491463 | 0.314177 | 109.11 |
| 2mm | 0.342967 | 20.254281 | 0.488872 | 0.315939 | 14.67 |

All 64/64 2 mm PCG intervals converged. All temporary directories cleaned.
2 mm timings include fresh-process compilation; 1 mm followed training.
No maps, warped images or duplicate checkpoints saved. Detailed scalar comparison:
`results/abdomenctct_quick4_endpoint_ncc_32k_recovery2mm_20261113/comparison.json`.

Recovery now defaults to original reference spacing in all physical-grid presets.
OASIS remains 1 mm; AbdomenCTCT uses 2 mm; DIR-Lab retains its original
anisotropic spacing. Full working-domain bounds are preserved, so actual steps
may be slightly smaller after rounding. Saved checkpoint metadata stays authoritative;
existing checkpoints retain their recorded recovery geometry unless explicitly overridden.
Native configurations without working_grid already recover on their input grids.

## Abdomen Jacobian tails and physical derivative scales (2026-11-13)

Checkpoint-only diagnostics on all frozen quick4 NCC/MSE fits, 2 mm recovery;
no retraining, scalar outputs only. Determinants use the unchanged official
original-grid stencil and border exclusion. Tail variance shares are contributions
to the full shifted-log variance, centered at its full mean. Positive-only SD
is descriptive, not a replacement official metric.

| Fit | Pair | SDlogJ | detJ <= -3 (%) | Share of log variance (%) | Positive-det-only SD |
| --- | --- | ---: | ---: | ---: | ---: |
| mse | 0001_to_0007 | 0.08629 | 0.00000 | 0.00 | 0.08629 |
| mse | 0001_to_0025 | 0.08858 | 0.00000 | 0.00 | 0.08858 |
| mse | 0007_to_0010 | 0.12375 | 0.00000 | 0.00 | 0.12375 |
| mse | 0019_to_0025 | 0.08295 | 0.00000 | 0.00 | 0.08295 |
| ncc | 0001_to_0007 | 0.30260 | 0.01291 | 68.85 | 0.16781 |
| ncc | 0001_to_0025 | 0.28123 | 0.01022 | 63.11 | 0.17020 |
| ncc | 0007_to_0010 | 0.77852 | 0.11293 | 90.73 | 0.23058 |
| ncc | 0019_to_0025 | 0.59314 | 0.06438 | 89.31 | 0.19021 |

Mean interval gradient-norm RMS: MSE 0.008437, NCC 0.040859 per mm. Temporal RMS: MSE 0.207183, NCC 0.915298. Means equally weight cases and the 16 interval midpoints.
NCC increases brightness-trace/smoothness-diagonal ratio substantially. This supports changed effective regularization but does not isolate causality from different learned trajectories. All 128 PCG intervals converged and all scratch directories cleaned. Detailed quantiles, per-interval scales and script: `results/abdomen_quick4_jacobian_derivative_diagnostic_20261113`.

## NCC abdomen checkpoint recovery smoothing sweep (2026-11-13)

Frozen quick4 fits, same 2 mm recovery; only recovery smoothness changed.
Baseline 0.01: DSC 0.342967, HD95 20.254281, SDlogJ 0.488872, folding 0.315939%.

| Smoothness | DSC | HD95 (voxels) | SDlogJ | Folding (%) | Converged intervals |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.05 | 0.320206 | 20.196013 | 0.150033 | 0.007584 | 64/64 |
| 0.1 | 0.306673 | 20.361450 | 0.108680 | 0.000145 | 64/64 |
| 0.25 | 0.290536 | 20.702410 | 0.078071 | 0.000000 | 16/64 |

Stronger smoothing reduces folding but also lowers mean DSC. No defaults changed.
All scratch directories cleaned; no maps/checkpoints written. Scalar summaries:
`results/abdomen_quick4_ncc_recovery_smoothing_20261113/summary.json`.

## Historical brightness-plus-gradient PCG experiment (2026-11-13)

The removed experimental solver added pointwise gradient constancy to brightness flow on the physical grid. The measurements below describe that historical implementation. Current frozen recovery uses brightness flow, a spatial smoothness penalty, true-residual PCG stopping and midpoint RK2. [Current method](METHOD.md)

Each recovery node uses Q = ggᵀ + c HᵀH and RHS = -(gq + c Hᵀm).
Analytical H and m are evaluated in chunks and discarded after assembling Q/RHS;
full FP32 normal coefficients remain on GPU. There is no cell-probe averaging
or Gaussian smoothing of coefficients. Thus it matches the evidence terms and
weight c=4 used in training, but not training's spatial pooling.

Synthetic validation: independently assembled FP64 matrix/direct solve with
anisotropic spacing; chunking parity; tiny recovery and cleanup; existing PCG
convergence regressions. Ten tests passed, two optional CUDA tests skipped.
An additional executed FP32 CUDA comparison at identical iteration counts gave
eager/compiled relative L2 error 1.626e-5 (threshold 1e-3).

First frozen abdomen quick4 case 0001→0007, existing NCC checkpoint, 2 mm,
smoothness 0.01, ridge 1e-6, 16 intervals and unchanged PCG/RK2 settings:

| Recovery | DSC | HD95 (voxels) | SDlogJ | Folding (%) | Recovery seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Brightness | 0.247819 | 23.173177 | 0.302600 | 0.060414 | 13.02 |
| Brightness + gradient (4) | 0.252016 | 23.048274 | 0.301590 | 0.060225 | 34.13 |

Both recoveries converged in 16/16 intervals. Timing includes fresh-process
compilation, so the 2.62x ratio is not a warm-kernel benchmark. No retraining,
parameter tuning or additional cases were run. Temporary scratch cleaned;
no maps/warped images saved. Source scalar diagnostics and replay script:
`results/abdomen_gradient_recovery_20261113/`. Default remains brightness-only.

## DIR-Lab quick4 NCC with standard recovery (2026-11-13)

Cases 01/03/04/05, best global endpoint NCC separate-32k preset, 1 mm training,
original anisotropic recovery, brightness-only PCG+RK2 and smoothing 0.01.
Initial attempt failed on crops without independent mask support. The sampler used for this historical run
rejected invalid regions (bounded at 1024 draws), requiring both endpoint
masks in the central particle region; unmasked sampling remains unchanged.
Six focused synthetic checks passed. Successful restarted run:
`results/dirlab_quick4_physical_endpoint_ncc_32k_maskedcrop_20261113`.

| Case | Forward continuous TRE (mm) | Reverse continuous TRE (mm) |
| --- | ---: | ---: |
| 01 | 1.4053 | 1.4543 |
| 03 | 2.7903 | 2.8809 |
| 04 | 3.7647 | 3.8431 |
| 05 | 3.7175 | 3.8739 |

Mean forward/reverse continuous TRE 2.9195/3.0131 mm, initial 7.0356 mm. All 64/64 PCG intervals converged; zero measured forward lung folding in each case. Mean recovery 9.83 s.

## DIR-Lab matched MSE quick4 (2026-11-13)

Same mask-valid crops, separate 32k samples, RK2 training and original-spacing
brightness recovery as the NCC run; only endpoint NCC/moment prior changed to MSE.
Mean forward rounded TRE 3.983972 mm (NCC 2.883260); continuous forward/reverse
3.915287/3.993237 mm. All four completed, but 0/64 PCG intervals converged
at 500 iterations, versus NCC 64/64. Thus the registration comparison includes
nonconverged MSE recovery. Zero measured whole-grid and lung folding. Mean
original-grid PSNR moving/fixed 21.504/21.188 dB. Source:
`results/dirlab_quick4_physical_endpoint_mse_32k_maskedcrop_20261113`. No additional recovery was launched.

## Current crop and gradient policy (supersedes earlier masked sampling)

DIR-Lab masks now select crop centers only. Endpoint reconstruction and both
registration directions share the selected crop, using separate samples by
default. All PD-LK grid cells and all registration particles contribute.
Trajectories leaving the crop use border-extended velocity interpolation rather
than being removed from NCC. A full-image boundary penalty applies in cropped
and uncropped execution. See METHOD.md for current details. Earlier experiments
in this document used the earlier masking/trajectory rejection policy.
