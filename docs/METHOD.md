# SIRENPath — canonical method

The canonical recipe is [medical/configs/default.json](../medical/configs/default.json).
This document describes its maintained training and recovery implementation.
[DATASETS.md](DATASETS.md) defines frozen cohorts, prepared inputs, pair directions
and evaluation protocols; [CASE_RUNNER.md](CASE_RUNNER.md) defines execution.

## Representation and initialization

Fit a fresh pair of coordinate SIRENs for each image pair. The moving SIREN
represents time 0 and the fixed SIREN time 1. Each network has **four sine hidden
layers of width 512**, scalar frequency **30** in every sine layer, and one scalar
linear output. Initialize the moving network and copy it exactly to the fixed
network before joint optimization.

First-layer weights and biases are uniform in ±1/input_dimension. Subsequent sine
layers use weights and biases uniform in ±sqrt(6/fan_in)/30. The output weights
use the latter bound multiplied by **30**; the 3D output bias is zero.

Interpolate all corresponding parameters linearly:

`theta(t) = (1-t) theta_moving + t theta_fixed`.

Analytical differentiation of the resulting intensity field gives spatial gradient
`g = grad_x I`, temporal derivative `q = dI/dt`, spatial Hessian `H` and mixed
derivative `m = grad_x q`. Network coordinates are normalized internally, with
chain-rule conversion so derivatives and velocities use **native voxel units**.

Intensity handling follows the frozen dataset adapters:

- Learn2Reg OASIS: independent in-memory min–max scaling to [0,1], without
  percentile clipping.
- IXI and LPBA40: preserve protocol-prepared intensities.
- Neurite-OASIS: preserve released prepared intensities.
- ACDC: preserve its prepared arrays, which already include the protocol's
  independent min–max normalization before cropping.

## Endpoint fitting

Each update samples one shared crop with half the image length along each axis.
The crop origin is sampled uniformly from those that keep the crop inside the
image.

Draw **32,768 voxel indices with replacement** inside the crop and use the same
indices for the two endpoint reconstruction losses. Endpoint NCC is the sum of
`1 - PearsonCorrelation(prediction, image)` for moving and fixed, computed over
these sampled voxels. It is not a dense whole-volume training loss. Correlations
use epsilon **1e-8**.

The moment matching penalty ($L_{\mathrm{mom}}$), weighted **0.005**, anchors each predicted sample
mean and variance toward 0 and 1:

`L_mom = 0.5 * sum_endpoints[mean² + (variance+epsilon) - log(variance+epsilon) - 1]`.

Variances are population variances. The endpoint parameter penalty ($L_\theta$) is the mean
squared difference over all corresponding weights and biases, weighted **0.01**.
This objective term is separate from AdamW weight decay **0.001**.

## Local transport and trajectory-based registration

Draw one scrambled **2D Sobol** point per update. Its coordinates give field
times in `[0,0.5)` and `[0.5,1)`. Crop/reconstruction, particle and query-direction
sampling use separate seeded native CUDA random generators with replayed offsets.
Temporal quadrature uses the scrambled Sobol sequence.

An **8³ cell-centered velocity grid** covers the shared crop. Each cell has two
orthogonal directions, with a symmetric query pair on each direction: four query
sites per cell. Directions are orthogonal in the spacing-aware offset frame.
The physical query radius is **0.2 times the smallest physical cell width**;
convert these offsets back to native voxel offsets for querying the SIRENs.
This use of image spacing does not change the native training grid or units.

Pool the four sites equally and solve one velocity per cell by minimizing

`mean_sites[b (q + g·v)² + c ||m + Hv||²] + eta ||v||²`,

with **b=1**, **c=4**, **eta=1e-6**. Average the minimized energies over cells and
both sampled times to obtain the local transport loss (LT, $L_{\mathrm{LT}}$), weighted **0.1**. The ridge
is applied once per pooled cell solve. Smooth each resulting velocity field with
a separable, reflected three-tap Gaussian, **sigma 0.5 grid cells** per axis.

Registration starts are separate from reconstruction samples: **65,536 independently
drawn continuous particles per direction**, sampled from the central **80%** of
the crop along each axis. Advance with two spatial midpoint-RK2 half-steps of
size **0.5**, using the first field then the second. Reverse trajectories use the
fields in reverse order with half-step size **−0.5**. The two fixed half-step sizes
do not depend on the gap between sampled temporal quadrature points.

Compare observed image intensities at trajectory starts and transported positions.
Gaussian-weighted NCC (GW-NCC) is a **50/50 mixture of global and local Pearson NCC losses**. The local
term bins six particle sufficient statistics into **8³ bins** over the particle
sampling region, applies a three-tap Gaussian with **sigma 0.75 cells**, and
weights local losses by smoothed particle mass. This is separate from the
sigma-0.5 velocity smoothing.

The trajectory-based registration loss ($L_{\mathrm{reg}}$) averages the two directional
similarity losses and their live full-image boundary penalties. Every sampled particle contributes, including crop exits. Image and
velocity interpolation use border extension. The boundary penalty is the mean
squared distance outside the full image in extent-normalized coordinates; crop
exits do not reject particles or gate their losses.

The registration coefficient follows cosine decay from **0.03 at update1** to
**0.01 at update5000**, with `r=(step-1)/4999`:

`registration_weight(step) = 0.01 + 0.5 (0.03-0.01) (1 + cos(pi r))`.

The original default was restored by user instruction on17 November2026;
the constant0.02 trial is retained only as a historical experiment.

## Joint objective and optimization

The objective is

`endpoint_weight * L_NCC + moment_weight * L_mom`
`+ LT_weight * L_LT + registration_weight(step) * L_reg`
`+ parameter_l2 * L_theta`.

Training logs use `local_transport_loss` for $L_{\mathrm{LT}}$ and `registration_loss` for $L_{\mathrm{reg}}$.

All gradients through network derivatives, pooled solves, smoothing and
trajectories remain live. Only logging values detach during training.

Use **one seed-42 fit per pair**, **5,000 updates**, AdamW with betas **(0.9,0.999)**,
weight decay **0.001** and gradient-norm clipping at **1**. The learning rate is
cosine decay **1e-4 → 2e-5**, with **no warmup**:

With `r=(step-1)/4999`:

`lr(step) = 2e-5 + 0.5 (1e-4-2e-5) (1 + cos(pi r))`.

Model and optimizer state are fresh per case. Training uses **CUDA FP32**, highest
float32 matmul precision, and CUDA graphs capturing the loss, backward pass,
clipping and fused AdamW step. Spatial samples and temporal choices refresh each
update. Compilation warmup and graph setup are included in training time.
Numerical eager helpers validate the same equations on synthetic inputs.

## Frozen stage-3 recovery

After fitting, hold the endpoint networks fixed and recover maps without training
gradients. At **8 evenly spaced interval midpoints**, evaluate the full-domain
brightness derivatives `g,q`. Solve the global brightness-flow system

`(g g^T + lambda L + ridge Id) v = -g q`,

where `L` is the native-voxel grid Laplacian, with sampling-spacing corrections
when recovery is capped. Use **lambda=0.01**, **ridge=1e-6**,
maximum **500 PCG iterations** and tolerance **2e-5**. Warm-start each interval's
solve from the preceding velocity. This global brightness-flow solve is separate
from the local velocity solve used in training.

Integrate the forward map with **8 midpoint-RK2 substeps per interval**. Integrate
the inverse using the stored interval velocities in reverse order and negative
time steps. Both integrations use border-extended velocity sampling.

Cap each recovery sampling axis at **256**. Training and prepared image grids
remain unchanged. If capped, spacing ratios enter derivative conversion, PCG
stencils and integration; returned maps are interpolated onto the original grid
in native voxel units. The cap does not reduce any axis in the four official
comparison cohorts. Derivative inference chunks contain **65,536 queries**.
Reject checkpoints requiring unsupported physical/working-grid geometry.

Map names follow model time: forward is moving→fixed; inverse is fixed→moving.
Pulling the moving image onto the fixed grid uses the inverse map.

Recover and evaluate in memory. Retain network checkpoints, resolved settings,
provenance, metrics and timings. Dense maps and warped/reconstructed image volumes
are not persistent artifacts. Remove recovery scratch on success and exceptions.

## Adjustable loss weights

These coefficients may be zero in a copied configuration:

| Component | Configuration key | Canonical value |
|---|---|---:|
| Endpoint reconstruction (NCC) | `objective.endpoint_weight` | 1 |
| Moment matching | `objective.ncc.moment_weight` | 0.005 |
| Local transport loss (LT) | `objective.weight` | 0.1 |
| Brightness constancy residual in the local transport energy | `objective.brightness_weight` | 1 |
| Spatial-gradient constancy residual in the local transport energy | `objective.c` | 4 |
| Trajectory-based registration loss | `objective.registration.weight` | 0.01 final |
| Endpoint parameter penalty | `objective.parameter_l2` | 0.01 |
| AdamW weight decay | `training.optimizer.weight_decay` | 0.001 |
| Recovery PCG smoothness | `recovery.smoothness` | 0.01 |

Zero registration weight disables the entire registration schedule. Zero LT
weight still allows the velocity solves required by registration. If both
losses are off, training omits their derivatives and trajectories. Brightness and
differential residual weights may independently be zero; with both zero, the
ridge solve gives zero local velocity. Endpoint NCC, moment matching and the endpoint parameter penalty are
independent terms. Positive numerical ridges and NCC epsilon remain required.

## Evaluation and internal ablations

Official comparisons use **Neurite-OASIS, ACDC, LPBA40 and IXI**, with the shared
`standardized_v1` Dice, HD95, NPJ (%) and unshifted SDlogJ evaluator. Labels are
evaluation-only; they do not select crops, losses or checkpoints. Other dataset
runners retain their documented author-protocol scoring. Metric definitions,
geometry, label sets and valid-score denominators are specified in
[DATASETS.md](DATASETS.md) and [CASE_RUNNER.md](CASE_RUNNER.md).

The separate [seven-run L2R-OASIS study](SIRENPATH_ABLATIONS.md) uses the same
training/recovery implementation. Runs 1–4 explicitly permit MSE endpoints with
the NCC-specific moment term disabled, varying transport/registration weights.
Run 5 matches the restored canonical recipe (registration cosine0.03→0.01). Run 6 (Raw crossfade) crossfades raw normalized intensities;
run 7 (Fitted crossfade) reuses run-5 checkpoints and crossfades their decoded endpoint intensities
without clipping, renormalizing or retraining. Both controls supply finite-difference
image gradients and exact crossfade temporal derivatives to the same stage-3
PCG/RK2 recovery; trained weight-path SIRENs use analytical derivatives.

The ablation report includes endpoint PSNR/Patch NCC, DSC, HD95, NPJ, SDlogJ and
runtime. Endpoint PSNR uses peak 1 without fitted intensity calibration; NCC-trained
outputs need not match the input scale/offset. Raw-identity endpoints have infinite
PSNR and Patch NCC 1 by construction. The Fitted crossfade shares run 5's
endpoint diagnostics. Its incremental runtime excludes the reused training cost,
which is also recorded separately. [Aggregate ablation results](ABLATIONS_REPORT.md).
