# Five-cohort generalist sweep — 2026-11-16

**Complete: 778/778 fits succeeded**, including all 234 final evaluation pairs.
The same selected configuration ran on every cohort, with a measured overall
mean of **89.39 seconds per case**. The fitting campaign finished in **5.76 hours**
of the eight-hour budget on four RTX 5090 GPUs. L2 0.01 won the declared
validation criterion; its small advantage over zero L2 remains uncertain.

## Question and constraints

The experiment seeks one per-pair optimization recipe that works across lung CT,
abdominal CT and three brain-registration protocols. Every candidate uses the
same model, losses, sampling, training schedule and recovery settings on all five
cohorts. Dataset adapters retain the prescribed input and scoring conventions.
There are no dataset-specific hyperparameters or extra training seeds.

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
| DIR-Lab | 2, 7 | 8, 9 | 10 |
| OASIS | 12, 15 | 3, 5, 7, 10, 11, 16 | 19 |
| AbdomenCTCT | 14, 16 | 11, 13, 21, 23, 24, 27 | 45 |
| IXI | validation 35, 46 | validation 3, 9, 23, 38, 42, 53 | 115 test |
| LPBA40 | validation 9, 10 | validation 1, 3, 4, 5, 6, 8 | 45 test |

The 26 additional pairs are matched across all four promoted configurations.
One configuration is locked after validation, before the final cohort is run.
The full run contains all 234 entries, including quick4 as part of the complete
public cohorts. Brain test subjects are separate from validation subjects.
Separation in the other protocols is by pair, not necessarily by subject.
Previous full-cohort work has already evaluated the older public medical cohorts;
this experiment does not claim that those cohorts are untouched hidden tests.

Screening ranks equal-weight cohort errors: primary DIR-Lab TRE divided by its
control TRE, and `(1 − Dice)/(1 − control Dice)` for each label cohort. The
selection utility adds `0.2 × max(0, worst cohort error ratio − 1)` to the mean.
The initial 5k-step control supplies screening denominators; expanded validation
uses its matched 4k-step speed control. Raw cohort scores remain available.
Failed or incomplete candidates cannot win. Screening permits cold means below
180 seconds for promotion; validation and the final run require actual means
below 120 seconds. This screening exception is not a claim of final speed.

## Execution recipe and explored ranges

All tuning arms use paired width-512 SIRENs, four sine layers, omega 30, output
initialization ×30, 4,000 updates, seed 42, 32,768 reconstruction samples and
32,768 independent registration particles per direction. They use shared crops,
native voxel units, full-gradient shared-grid PD-LK and four stochastic-quarter
Euler training steps. OneCycle peaks at 1e-4 and ends at 2.5e-5; AdamW decay is
0.001. Complete captured training reduces dispatch overhead. No training loss,
solve, smoothing or trajectory gradient is detached.

Frozen inference remains brightness-flow PCG and midpoint RK2. The shared speed
recipe uses eight time intervals, eight RK2 substeps per interval and cap256.
The cap is inert on the native IXI/LPBA40 grids; DIR-Lab uses the existing
spacing-aware capped recovery and original-grid evaluation. The initial control
used 5,000 updates and sixteen recovery intervals with four substeps, averaging
162.71 seconds over its ten cold cases; the speed control averaged 90.5 seconds.
These changes are disclosed because the original control missed the time target.

| Variable | Values and schedules explored |
| --- | --- |
| Differential PD-LK coefficient c | 0, 0.25, 1, 2, 4, 8, 16, 64 |
| Endpoint moment coefficient | 0, 0.001, 0.003, 0.01, 0.03, 0.1; selected c interactions |
| PD-LK loss coefficient | 0.01, 0.03, 0.1, 0.3, 1; linear/cosine rise to 0.1 |
| Registration loss coefficient | 0, 0.003, 0.01, 0.03, 0.1; 500/1,000-step rise; joint 250-step delay plus 500-step rise |
| Velocity smoothing | Three-tap sigma 0.4, 0.75, 1, 1.5 cells; full Gaussian box fractions 0.1 constant, 0.2→0.1, 0.4→0.1, 0.6→0.1 |
| Registration similarity | Box NCC and 50/50 global/window GW-NCC; GW weights 0.003 and 0.03 |
| Inter-SIREN L2 coefficient | 0, 0.001, 0.01, 0.1, 1, 10, 100 |

This is an adaptive coordinate sweep, including c/moment interactions, rather
than the Cartesian product of every range. Each stage chooses a complete parent
by the stated utility. All 44 configurations, including the two execution
controls, ran the same two screening cases per cohort: 440 fits with no failures.

The L2 term is `lambda × mean((theta_source − theta_target)^2)` over all 790,529
corresponding weight and bias scalars per endpoint. It is a differentiable loss;
the ordinary AdamW decay remains active. It is not paired-mean AdamW decay.
Full Gaussian schedules use precomputed reflected convolution matrices and
retain live velocity gradients, including in captured training.

## Screening decisions

c=4 survived the broad c scan and 2/8 refinement. The c=8, moment=0.003 interaction
then narrowly led the balanced criterion; its advantage over the speed control
was too small to establish superiority. Constant PD-LK weight 0.1 survived the
weight/ramp comparisons. Constant registration weight 0.03 improved the screen;
the tested warmups did not improve it further. Sigma 0.4 cells beat the tested
broader schedules under the balanced criterion. GW-NCC 0.03 then improved the
matched box-NCC parent's mean normalized error from 0.980481 to 0.966655.

| Leading screened candidate | Mean normalized error | Selection utility | Actual seconds/case |
| --- | ---: | ---: | ---: |
| GW-NCC, L2 0.01 | 0.963110 | 0.964966 | 131.2 |
| GW-NCC, L2 100 | 0.966623 | 0.968019 | 130.2 |
| GW-NCC, L2 0 | 0.966655 | 0.970690 | 136.4 |

These cold arm means exceed the final time target. The three candidates are
promoted alongside the speed control specifically to test accuracy and actual
runtime on more cases. No winner is claimed from screening alone.

## Mechanism observations

NCC tolerates changes in predicted intensity mean and contrast, whereas the
derivative losses and fixed inference regularization respond to those changes.
Moment matching limits that freedom. Removing moments gave similar sampled NCC
but much smaller sampled output variances on OASIS and worse registration on
OASIS/abdomen, while IXI improved. This is consistent with a contrast-dependent
training/inference balance; the experiment does not separate those contributions.

Broader training smoothing removes spatial variation from solved velocities.
The narrower 0.4-cell kernel preserved more variation and improved the balanced
screen. A full box-fraction 0.2→0.1 schedule slightly improved the raw mean but
lost on the worst-cohort penalty. At an 8³ grid, that schedule means 1.6→0.8
cells, not the same support as the three-tap 0.4-cell kernel. These results do
not show that annealing is generally unhelpful, nor do they tune PCG smoothing.

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

Validation compares the speed control and GW-NCC with L2 0, 0.01 and 100 on
26 additional pairs each. All 104 fits succeeded.

| Configuration | DIR-Lab TRE mm | OASIS Dice | Abdomen Dice | IXI Dice | LPBA40 forward Dice | Seconds/case | Complete |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Speed control | 5.65500 | 0.80145 | 0.35735 | 0.75044 | 0.71432 | 96.5 | 26/26 |
| GW-NCC, L2 0.01 | 4.95930 | 0.79861 | 0.36236 | 0.77251 | 0.71886 | 91.9 | 26/26 |
| GW-NCC, L2 100 | 4.96835 | 0.79691 | 0.36211 | 0.76966 | 0.71866 | 92.4 | 26/26 |
| GW-NCC, L2 0 | 5.01263 | 0.79774 | 0.35906 | 0.77274 | 0.72019 | 92.0 | 26/26 |

Lambda0.01 has normalized mean error0.955822 and utility0.958681 versus1.0 for
the control. Four cohorts improve; OASIS Dice declines by0.00284. All208 PCG
interval solves converged for each configuration. Maximum abdomen folding is
0.2044% versus1.0107% for the control; folding remains present. Its OASIS HD95
is slightly worse, while abdomen and LPBA40 HD95 improve.

Lambda 100 has mean normalized error 0.960349 and utility 0.964915. The 0.01
arm is slightly better on every cohort and faster in this matched comparison.
The zero-L2 arm has mean normalized error 0.958499 and utility 0.962233.
The selected 0.01 arm improves three cohorts but loses slightly on IXI/LPBA40
against zero L2. Its normalized-error advantage is only 0.00268. This small,
single-seed comparison does not establish a robust benefit of positive L2;
most of the gain over the speed control comes from the rest of the recipe.

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

The repository default remains the separate 5k-step recipe. The selected
preset is opt-in. The recipe was locked before any final fit was dispatched;
all 234 final inputs passed preflight with no missing files.

## Final cohort results and runtime

All 234 pairs succeeded under the one recipe selected on expanded validation.
The table uses population SD across case means; these are descriptive case
variations, not training-seed confidence intervals.

| Cohort | Successful / intended | Primary mean ± case SD | Actual seconds/case |
| --- | ---: | ---: | ---: |
| DIR-Lab | 10/10 | 2.75830 ± 1.77278 mm | 112.5 |
| OASIS | 19/19 | 0.80815 ± 0.02104 Dice | 87.1 |
| AbdomenCTCT | 45/45 | 0.41819 ± 0.08736 Dice | 87.8 |
| IXI test | 115/115 | 0.76029 ± 0.02908 Dice | 89.3 |
| LPBA40 test, forward | 45/45 | 0.71845 ± 0.01621 Dice | 87.0 |

DIR-Lab reverse TRE is 2.89000 ± 1.88381 mm. All 80 PCG intervals converged.
Lung-ROI folding is zero, while whole-grid folding is nonzero in 5/10 cases;
its maximum is 0.000519%. No whole-grid nonfolding claim is made.

OASIS HD95 is 1.89779 ± 0.44069 voxels; official shifted SDlogJ is
0.06718 ± 0.00412. All 152 PCG intervals converged. Twelve cases have nonzero
interior folding; mean fraction is 9.63e-7 and maximum fraction is 3.56e-6.

Abdomen HD95 is 16.65343 ± 5.62628 voxels; official shifted SDlogJ is
0.17542 ± 0.08796. All 360 PCG intervals converged. Folding is nonzero in
42/45 cases, with mean 0.04063% and maximum 0.37989%.

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

All **1,872/1,872 PCG solves converged**. Final case invocation time is
89.39279 ± 6.63746 seconds, median 88.59463 and 95th percentile 91.08687 seconds.
The maximum is 136.41762 seconds; five cold invocations exceed 120 seconds.
The complete 234-case batch took 87.90 minutes through the four-GPU queue.
These totals include actual setup, training, checkpoint handling, recovery,
reconstruction diagnostics and evaluation. Setup is included in training time;
compiler caches are reused across compatible cases. No setup time was subtracted
to meet either target. All five complete-cohort means are below 120 seconds,
and the pair-weighted overall mean meets the preferred 90-second target.

The full run necessarily repeats 20 screened/validated public pairs under the
same seed and locked configuration. These are not extra experiments.

| Cohort | Required repeats | Mean absolute primary-score change | Maximum absolute change |
| --- | ---: | ---: | ---: |
| DIR-Lab | 4 | 0.04633 mm | 0.07859 mm |
| OASIS | 8 | 0.00195 Dice | 0.00587 Dice |
| AbdomenCTCT | 8 | 0.00500 Dice | 0.00903 Dice |

Execution is not bitwise reproducible across these fits. The exact numerical
source has not been isolated. This variation reinforces the qualification on
the small positive-L2 selection advantage. No extra seeds were launched.

The 54 public pairs not used to tune this campaign are also reported separately:

| Cohort | Pairs | Primary mean ± case SD | Seconds/case |
| --- | ---: | ---: | ---: |
| DIR-Lab | 6 | 2.04207 ± 0.75895 mm | 126.0 |
| OASIS | 11 | 0.81128 ± 0.02455 Dice | 87.3 |
| AbdomenCTCT | 37 | 0.42997 ± 0.08715 Dice | 87.9 |

These have a different case mix and are not a matched before/after comparison
or untouched historical test cohorts. The six DIR-Lab cases include previously
unused compilation geometries; their actual mean exceeds 120 seconds. Runtime
claims concern the complete-case mean with all incurred setup included, not a
guarantee that every cold invocation is under two minutes. Brain test subjects
remain separate from the validation subjects used for this campaign.

## Protocol details and limitations

DIR-Lab reports T00→T50 and T50→T00 rounded TRE separately; masks select crop
centers only. OASIS uses official labels1–35 and AbdomenCTCT labels1–13, with
official unit-voxel HD95 and shifted SDlogJ. True unshifted folding fractions
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

There is also an accuracy tradeoff against the [older three-cohort accuracy
recipe](archived_reports/LONG_RESEARCH_REPORT.md): that 5k-step RK2, orthogonal-query,
Sobol-time recipe recorded DIR-Lab 2.861/2.970 mm, OASIS Dice 0.81349 and abdomen
Dice 0.43685. The present recipe improves DIR-Lab TRE but has lower OASIS and
abdomen Dice. It is not a universal improvement over all previously measured
presets. The older recipe was not tested on IXI/LPBA40 in this campaign; this
comparison is historical context, not a new matched five-cohort experiment.

## Completion checks

All seven requested variable families completed their planned arms. Screening
used exactly two frozen pairs per cohort and excluded the old quick4; expanded
validation used the same additional 26 pairs for all four configurations. The
winning recipe was locked before the full run and was not changed after test
scores appeared. All 778 run records match their respective frozen candidate
configurations, with training seed 42 throughout.

The final audit inspected every input path, protocol metric count, logged final
training step and checkpoint. All 234 checkpoints loaded on CPU with finite
saved tensors, the expected paired-model sizes and Euler/captured/native metadata;
their configurations match the locked preset. Exact cohort membership, original
brain test splits, evaluation seeds, all four GPU IDs and the time budget passed.
Data files have no modification times after the goal began. Case directories
contain only checkpoints, metrics, run records and logs; recovery scratch is
clean. No data or machine-local settings were staged, and Git HEAD is unchanged.

The backend checks include analytical/all-parameter gradient comparisons and the
complete six-update CUDA Gaussian-schedule/L2 self-test, which passed in 30.292
seconds before the adaptive continuation. The focused 11 Euler/RK2 synthetic
checks also passed. No extra multiseed validation or saved-model recovery was
launched for the final audit.

## Evidence

[Frozen selection](../results/generalist_sweep_20261116/selection.json) ·
[Adaptive decisions](../results/generalist_sweep_20261116/campaign_decisions.json) ·
[All case records](../results/generalist_sweep_20261116/cases.csv) ·
[Structured scores and audit](../results/generalist_sweep_20261116/analysis.json) ·
[Detailed live analysis](../results/generalist_sweep_20261116/LIVE_REPORT.md) ·
[L2 checkpoint probe](../results/generalist_sweep_20261116/l2_checkpoint_probe.json).
Also retained: [required same-seed repeats](../results/generalist_sweep_20261116/repeatability.json),
[public non-tuning subsets](../results/generalist_sweep_20261116/public_non_tuning_subsets.json),
and the [standalone summary figure](../results/generalist_sweep_20261116/sweep_summary.pdf).
Final verification artifacts: [checkpoint hashes and state audit](../results/generalist_sweep_20261116/checkpoint_audit.json),
[timing and directional details](../results/generalist_sweep_20261116/final_details.json),
[requirement-by-requirement completion audit](../results/generalist_sweep_20261116/completion_audit.json).
The ignored result directory also retains resolved configs, source snapshots,
per-fit checkpoints, complete training logs and official metric details.
