# Generalist sweep — 16 November 2026

Retained results from a shared-configuration sweep on Learn2Reg OASIS, IXI, and LPBA40.

## Question and constraints

Every candidate uses the
same model, losses, sampling, training schedule and recovery settings across cohorts. Dataset adapters retain the prescribed input and scoring conventions. There are no dataset-specific hyperparameters or extra training seeds.

The user allowed eight hours on four RTX 5090 GPUs. The final mean case invocation
must be below 120 seconds, preferably below 90. Invocation time includes fitting,
recovery, scoring and the setup actually incurred by that case. Compiler caches
are reused normally; separate cold-setup measurements are reported rather than
subtracted from the observed total.

## Selection and evaluation protocol

The score-blind selection seed is 20261116. Pair indices are one-based in each
frozen protocol. Screening and expanded validation exclude the previous quick4
pairs, including reverse equivalents, and are disjoint from each other.

| Cohort | Screen | Additional validation | Final cohort |
| --- | --- | --- | ---: |
| OASIS | 12, 15 | 3, 5, 7, 10, 11, 16 | 19 |
| IXI | validation 35, 46 | validation 3, 9, 23, 38, 42, 53 | 115 test |
| LPBA40 | validation 9, 10 | validation 1, 3, 4, 5, 6, 8 | 45 test |

Validation pairs are matched across all four promoted configurations. The shared
configuration was locked before the final fits. The final evaluation includes the
public OASIS quick4 as part of the full public-validation list. IXI and LPBA40
use separate test subjects.

## Execution recipe and explored ranges

All tuning arms use paired width-512 SIRENs, four sine layers, omega 30, output
initialization ×30, 4,000 updates, seed 42, 32,768 reconstruction samples and
32,768 independent registration particles per direction. They use shared crops,
native voxel units, full-gradient shared-grid PD-LK and four stochastic-quarter
Euler training steps. OneCycle peaks at 1e-4 and ends at 2.5e-5; AdamW decay is
0.001. Complete captured training reduces dispatch overhead. No training loss,
solve, smoothing or trajectory gradient is detached.

Frozen inference remains brightness-flow PCG and midpoint RK2. The shared speed
recipe uses eight time intervals, eight RK2 substeps per interval and cap256. The initial control
used 5,000 updates and sixteen recovery intervals with four substeps.

| Variable | Values and schedules explored |
| --- | --- |
| Differential PD-LK coefficient c | 0, 0.25, 1, 2, 4, 8, 16, 64 |
| Endpoint moment coefficient | 0, 0.001, 0.003, 0.01, 0.03, 0.1; selected c interactions |
| PD-LK loss coefficient | 0.01, 0.03, 0.1, 0.3, 1; linear/cosine rise to 0.1 |
| Registration loss coefficient | 0, 0.003, 0.01, 0.03, 0.1; 500/1,000-step rise; joint 250-step delay plus 500-step rise |
| Velocity smoothing | Three-tap sigma 0.4, 0.75, 1, 1.5 cells; full Gaussian box fractions 0.1 constant, 0.2→0.1, 0.4→0.1, 0.6→0.1 |
| Registration similarity | Box NCC and 50/50 global/window GW-NCC; GW weights 0.003 and 0.03 |
| Inter-SIREN L2 coefficient | 0, 0.001, 0.01, 0.1, 1, 10, 100 |

The adaptive coordinate sweep included c/moment interactions.

The L2 term is `lambda × mean((theta_source − theta_target)^2)` over all 790,529
corresponding weight and bias scalars per endpoint. It is a differentiable loss;
the ordinary AdamW decay remains active. It is not paired-mean AdamW decay.
Full Gaussian schedules use precomputed reflected convolution matrices and
retain live velocity gradients, including in captured training.

## Mechanism observations

NCC tolerates changes in predicted intensity mean and contrast, whereas the
derivative losses and fixed inference regularization respond to those changes. Moment matching limits that freedom. This is consistent with a contrast-dependent
training/inference balance; the experiment does not separate those contributions.

Broader training smoothing removes spatial variation from solved velocities.
At an 8³ grid, a box-fraction 0.2→0.1 schedule gives sigma 1.6→0.8 grid cells. The three-tap kernel uses sigma 0.4 cells.

The SIRENs begin with matching parameters, so their corresponding weights are
initially aligned. L2 can limit divergence of their parameter path. In one
matched IXI checkpoint, zero-L2 mean parameter disagreement was 3.51e-6;
lambda0.01 gave 3.55e-6 and lambda100 gave 2.89e-6. The tiny 0.01 penalty did not
demonstrate contraction, despite its screen lead. Expanded validation is needed
to distinguish repeatable benefit from sensitivity to small optimization changes.

Four additional narrow-bandwidth trials were proposed but never dispatched:
cell sigma0.2, full box sigma0.05, 0.2→0.05 and 0.4→0.05. Automatic approval
review rejected the extra service/scheduler intervention; human approval remains
pending. They are excluded from the experiment count and conclusions.

## Expanded validation and locked recipe

The retained validation scores compare the speed control and three GW-NCC arms.

| Configuration | OASIS Dice | IXI Dice | LPBA40 forward Dice |
| --- | ---: | ---: | ---: |
| Speed control | 0.80145 | 0.75044 | 0.71432 |
| GW-NCC, L2 0.01 | 0.79861 | 0.77251 | 0.71886 |
| GW-NCC, L2 100 | 0.79691 | 0.76966 | 0.71866 |
| GW-NCC, L2 0 | 0.79774 | 0.77274 | 0.72019 |

The sweep selected L2 0.01. In the retained validation results its Dice was close
to zero L2 on all three cohorts.

The locked [shared preset](../medical/configs/native_generalist_euler4_4k.json)
matches the resolved winning configuration exactly:

| Setting | Selected value |
| --- | --- |
| PD-LK c | 8 |
| Endpoint NCC moment weight | 0.003 |
| PD-LK loss weight | 0.1, constant |
| Registration similarity/weight | GW-NCC / 0.03, constant |
| Velocity smoothing | Three-tap sigma 0.4 grid cells, constant (5% of each box side at G=8) |
| GW-NCC moment smoothing | Three-tap sigma 0.75 cells, unchanged |
| Inter-SIREN mean squared parameter difference | Lambda 0.01 |
| Training | Width 512, 4,000 steps, four-quarter Euler, full captured gradients |
| Optimizer | AdamW decay 0.001; OneCycle peak 1e-4, final 2.5e-5 |
| Recovery | PCG, eight time intervals, eight RK2 substeps/interval, cap256 |

## Final cohort results and runtime

The table uses population SD across case means. These are descriptive case variations.

| Cohort | Successful / intended | Primary mean ± case SD | Actual seconds/case |
| --- | ---: | ---: | ---: |
| OASIS | 19/19 | 0.80815 ± 0.02104 Dice | 87.1 |
| IXI test | 115/115 | 0.76029 ± 0.02908 Dice | 89.3 |
| LPBA40 test, forward | 45/45 | 0.71845 ± 0.01621 Dice | 87.0 |

OASIS HD95 is 1.89779 ± 0.44069 voxels; official shifted SDlogJ is
0.06718 ± 0.00412. All 152 PCG intervals converged. Twelve cases have nonzero
interior folding; mean fraction is 9.63e-7 and maximum fraction is 3.56e-6.

IXI's primary score is the equal mean of the 17 author groups. Its SD over all
1,955 group/case scores is 0.11039; the table gives SD over the 115 per-case
means. Reverse-map Dice is 0.77321 ± 0.02004. All 920 PCG intervals converged.
Whole-grid folding is nonzero in 93/115 cases, with mean 0.0001845% and maximum
0.0026449%. This is the author's whole-grid `detJ <= 0` convention.

LPBA40 reverse Dice is 0.71984 ± 0.01700; averaging the two directions within
each pair gives **0.71914 ± 0.01640** across 45 pairs. Pooling the 90 directional
scores gives the same mean and SD 0.01662. Forward HD95 is 5.87507 ± 0.43652
voxels; the pooled 90-direction mean is 5.82228 ± 0.42700 voxels. Forward sampled
determinant SD is 0.22810 ± 0.02461; this is not shifted SDlogJ. Forward sampled
folding is nonzero in 17/45 pairs, with mean 0.0001333% and maximum 0.0012000%.
Componentwise inverse-consistency MAE is 0.00535 ± 0.00093 voxels. All 360 PCG
intervals converged. Both maps come from one fit, without an extra swapped-input
optimization.

Case runtime includes incurred setup, training, recovery, scoring and checkpoint
handling. The final OASIS evaluation repeats eight tuning pairs with the same seed
and locked configuration.

| Cohort | Required repeats | Mean absolute primary-score change | Maximum absolute change |
| --- | ---: | ---: | ---: |
| OASIS | 8 | 0.00195 Dice | 0.00587 Dice |

Execution is not bitwise reproducible across these fits. The exact numerical
source has not been isolated. This variation reinforces the qualification on
the small positive-L2 selection advantage. No extra seeds were launched.

The public OASIS pairs outside the tuning selection are reported separately.

| Cohort | Pairs | Primary mean ± case SD | Seconds/case |
| --- | ---: | ---: | ---: |
| OASIS | 11 | 0.81128 ± 0.02455 Dice | 87.3 |

These have a different case mix and are not a matched before/after comparison
or untouched historical test cohorts. Runtime
claims concern the complete-case mean with all incurred setup included, not a
guarantee that every cold invocation is under two minutes. Brain test subjects
remain separate from the validation subjects used for this campaign.

## Protocol details and limitations

True unshifted folding fractions
are reported separately. IXI preserves the author's arrays and atlas direction,
46-class trilinear one-hot warping and equal weighting of 17 anatomical groups;
both per-case and group/case SD are retained. LPBA40 preserves SITReg preparation,
56 labels, masked nearest-label scoring, voxel HD95, the seeded million-point
float64 Jacobian estimate and inverse consistency. Its forward and inverse
scores come from one fit; there is no extra swapped-input fit.

Each fit uses one training seed. Case variability is not seed variability and
does not justify statistical significance claims. The tuning samples are small,
the search is adaptive, and only selected interactions were explored. Public
cohorts used for tuning remain influenced by that selection. No persistent
deformation maps or warped registration images are written; original data remain
unchanged. No commits or pushes are part of this experiment.

The 4k-step recipe was chosen to meet the runtime constraint. There is no matched
full-cohort experiment isolating 4k versus 5k steps or Euler versus RK2 here;
this study does not prove either discretization or update count universally
superior. "Selected" means best among the screened and validated candidates
under the declared criterion, not a global optimum over all possible recipes.

## Evidence

The remaining case records, cohort scores, checkpoint hashes, and resolved configurations are in the local results directory. [Frozen selection](../../results/generalist_sweep_20261116/selection.json) · [Case records](../../results/generalist_sweep_20261116/cases.csv) · [Cohort scores](../../results/generalist_sweep_20261116/analysis.json) · [Checkpoint hashes](../../results/generalist_sweep_20261116/checkpoint_audit.json).
