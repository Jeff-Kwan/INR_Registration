# GS-Reg transfer review, 2026-11-15

This is an experimental review during the 90–120 minute research round. The neighboring checkout was read in place. Its code, inputs and outputs were left intact. No suggestion here changes the canonical INR method or presets.

## Mechanism and direct evidence

GS-Reg fits a source-specific, fixed-size Gaussian intensity representation before estimating its geometric movement. INR jointly fits paired SIREN endpoints and asks their parameter interpolation to explain motion. GS-Reg therefore separates representation capacity from the subsequent registration objective more directly. A low endpoint fitting error can still coexist with poor correspondence or field reconstruction in either method. The relevant transferable experiment holds the learned appearance fixed while varying only the matching scale or inference route.

Cross-project clinical scores therefore combine representation, normalization, optimization and recovery differences. The comparisons below use controlled changes within GS-Reg to motivate isolated INR tests.


GS-Reg also derives the local NCC translation information after projecting spatial gradients away from nuisance offset and gain. For centered target intensities v and centered spatial gradients Gc, use Gp = Gc − v(vᵀGc)/(vᵀv). Away from variance floors, the self-match translation Hessian is GpᵀGp/(vᵀv). A linear ramp has nonzero image gradient and zero translational information under NCC. Thus high patch variance, high gradient magnitude and high endpoint NCC each answer a different question. Matching may remain ambiguous along poorly observable axes even if the SIREN reproduces the image accurately. [GS method](/workspace/3DGS_Registration/docs/METHOD.md)

For target 0007 at sigma8 with radius3, only 39.77% of organ-weighted windows exceed its variance floor. Pancreas patches are almost all below it. Enlarging support restores some signal but a controlled matching experiment failed to improve registration. The Gaussian images, clipping policy and variance floor differ from INR, so these fractions cannot be assigned to INR. The transferable diagnostic measures the actual image/SIREN patch information and capture landscape under our normalization. [GS observation probe](/workspace/3DGS_Registration/results/marathon_20261115/medical/diagnostics/ncc_observability/probe.py) · [GS scalar evidence](/workspace/3DGS_Registration/results/marathon_20261115/medical/diagnostics/ncc_observability/summary.json)

## Transfer candidates and distinguishing observations

| Candidate | Mechanism | Observation that supports it | Competing explanation or qualification |
|---|---|---|---|
| Retain native finishing updates | End at full spatial detail after broad matching establishes approximate placement | Broad early scales help large motion while native finishing preserves small-label Dice | More total optimization time alone can explain a benefit without a time-matched control |

The GS broad schedule is paired with a linearly decreasing learning rate and coherence strength, followed by 200 native updates at fixed low LR. Any INR transfer must isolate blur/scale from optimizer and iteration count. That evidence supports a dataset-qualified ablation, and gives no reason to add an unconditional optimizer coupling to INR. [GS schedule and force findings](/workspace/3DGS_Registration/docs/METHOD.md) · [GS paired controls](/workspace/3DGS_Registration/docs/MEDICAL_RESULTS.md)

## Focused validation plan

It will use held fixed checkpoint weights, seed 42, original input geometry and no fitted or label-assisted deformation field. Locations and labels are evaluation-only. GPU execution will be coordinated with the shared research queue.

A positive landscape result would motivate an isolated, opt-in image-scale experiment. Existing GW-NCC smooths moments in velocity-grid bins and is distinct from blurring the observed images before matching. Its scale does not automatically reproduce the GS continuation effect. The paired-SIREN derivatives, shared-grid PD-LK and brightness PCG/RK2 recovery must retain their current definitions in that experiment.

## Review status

The initial read covered GS AGENTS.md, README.md, METHOD.md, MEDICAL_RESULTS.md, SYNTHETIC_RESULTS.md, the local-affine tracker, correlation implementation and the saved observability probe. No algorithm was imported or executed from the neighboring checkout. Validation results will be appended as they finish.

## Native anatomical information and the cost of blur

For each present label it samples 64 centers at seed 42 and jitters each by at most a quarter voxel to avoid systematic placement at interpolation knots. It uses 9³ samples over 8/16 mm physical patches, excludes outside-image sites, and obtains exact trilinear interpolation gradients through query autograd. The gradients are converted to derivatives per millimeter. Both the full neighborhood and its label-restricted subset are recorded. No image gradients or labels enter training.

The projected self-match information was independently checked against the exact float64 Hessian of signed NCC, with maximum absolute error 4.0e-15. A synthetic gradient made entirely of offset/gain directions gives zero information as required. The numeric information measures local precision around a correct self-match. They do not measure whether the optimizer can reach that basin or whether source and target appearances are homologous.

| Organ | Median native voxels | Native λmin (mm⁻²) | Sigma4 λmin (mm⁻²) | Native valid patch fraction | Sigma4 valid fraction |
|---|---:|---:|---:|---:|---:|
| liver | 210692 | 0.10030 | 0.00084 | 0.994 | 0.537 |
| right kidney | 21149 | 0.03312 | 0.00187 | 1.000 | 0.994 |
| left kidney | 21480.5 | 0.03173 | 0.00176 | 1.000 | 1.000 |
| pancreas | 11465.5 | 0.05210 | 0.00199 | 1.000 | 0.964 |
| left adrenal gland | 632 | 0.02960 | 0.00096 | 1.000 | 1.000 |
| right adrenal gland | 779.5 | 0.04007 | 0.00276 | 1.000 | 1.000 |

Each information value averages the within-image median over the ten images. All entries use the full 8 mm label-centered patch. Validity requires at least eight in-grid samples and normalized-image variance at least 1e-6. These native adrenal neighborhoods have comparable minimum information to kidney neighborhoods despite their much smaller labels. Native texture absence is therefore an incomplete explanation for poor adrenal Dice. A small organ can carry useful local evidence while contributing little to a global objective or while the optimizer remains outside its matching basin.

At sigma 4 mm the minimum information falls by roughly 31× for the left adrenal and 15× for the right adrenal. Liver valid-patch fraction falls from 99.4% to 53.7%. Larger 16 mm support restores valid liver neighborhoods to 78.6% and some minimum information, but does not restore native precision. The variance threshold is an evaluation validity rule, rather than a target-window gate in our GW-NCC training objective. The exact ratios are diagnostic, without a fitted uncertainty model or registration-accuracy guarantee.

[Per-image/per-label measurements](../results/research_20261115/gs_information/) · [Organ summary](../results/research_20261115/gs_information/summary.json) · [Independent Hessian validation](../results/research_20261115/gs_information/hessian_validation.json) · [Probe source](../.local/research_20261115/gs_transfer/information.py)

## Rejected global intensity-gauge explanation

GS-Reg keeps fitted amplitudes fixed while tracking geometry. This motivated checking whether our linearly interpolated SIREN parameters introduce a large global gain/offset change that brightness recovery could mistake for transport. It regressed the exact model temporal derivative q onto a constant and the centered intensity I. This isolates the global offset rate and logarithmic scale rate at each sampled time.

| Checkpoint | Mean middle-time q energy explained by gain/offset | Maximum middle-time fraction |
|---|---:|---:|
| oasis 1 | 0.871% | 1.141% |

No sampled path shows a middle-time global standard-deviation collapse. This is a diagnostic sparse-motion model, without an inferred full truth field.

The result rejects a simple global gain/offset explanation for these weak cases. Spatially local intensity/path errors remain possible. A global intensity correction is therefore a poor next experiment for this round. Projecting q alone would also define no guaranteed consistent image path, and can remove a legitimate geometric intensity change. No recovery equation or checkpoint was modified. [Gauge summary](../results/research_20261115/gs_gauge/summary.json) · [Probe source](../.local/research_20261115/gs_transfer/gauge.py)

## Comparison with GS field reconstruction

GS-Reg's controlled endpoint-image versus decoded-field audit found an actual gap even with identical fitted Gaussians. Source-fit and subvoxel interpolation controls are smaller than the gap. The audit does not uniquely identify chart approximation, midpoint interpolation, bidirectional blending or inverse iteration as its cause. A later single-forward replacement improved some internal consistency scores but worsened every checked official medical metric and increased folding 6–13 times. [Controlled GS image/field report](/workspace/3DGS_Registration/results/marathon_20261115/medical/diagnostics/image_field_consistency/CONSISTENCY_REPORT.md) · [GS synthetic/rejection record](/workspace/3DGS_Registration/docs/SYNTHETIC_RESULTS.md)

The transferable discipline is to compare the same appearance model under the training surrogate and the final scoring field, while retaining original-image patch fidelity and anatomy metrics as controls. In INR the estimators differ between shared-grid PD-LK training and global brightness PCG recovery. A discrepancy there should be localized before adding another regularizer or replacing the solver. That observation strengthens this diagnostic priority, and does not establish which part of the train-to-recovery pipeline causes it.

## Exact crop exposure and small-label weighting

Endpoint reconstruction samples 32,768 discrete crop voxels. Each registration direction draws 32,768 continuous points from the central 80% of the crop-center extent. For the latter, anatomical membership is modeled by nearest-label unit-voxel cells, and the audit integrates fractional intersections at each box edge exactly. Separable cumulative sums give the expected count conditional on every crop origin. Averaging the binomial zero-count probability across origins retains crop-level absence and finite-particle sampling separately. Constant-mask and direct fractional-cell controls pass. These quantities describe sampling exposure, not the full influence of a point on the nonlinear NCC objective or whether transported endpoints reach the correct target label.

| Dataset / label | Median voxels | Mean endpoint samples/update | Mean particles/direction/update | Probability of zero particles | Mean particles when crop exposes label |
|---|---:|---:|---:|---:|---:|
| oasis / 18 | 66 | 1.33 | 1.70 | 66.3% | 4.28 |
| oasis / 19 | 1148.5 | 25.17 | 31.26 | 18.4% | 36.92 |
| oasis / 34 | 51.5 | 1.17 | 1.50 | 67.0% | 3.80 |
| oasis / 35 | 1432.5 | 32.94 | 40.61 | 17.0% | 47.25 |

Values average unique images equally. The probability of zero particles averages over all crop origins, while the conditional count averages only crop boxes with positive label support. The small-label means are greater than full-volume uniform sampling would yield because their central positions receive more crop coverage. The central 80% rule nevertheless leaves more updates with no adrenal particles than the full-crop endpoint sampler. Merely replacing crops by whole-image uniform sampling could reduce their mean exposure; these results do not support that replacement.

The two smallest weak OASIS labels 18/34 receive roughly 1.5–1.7 particles per direction on average and none in two thirds of updates. Labels 19/35 are also weak, but receive 31–41 particles and are absent in only 17–18% of updates. Their poor overlap needs explanations beyond rare sampling. Small-label Dice sensitivity and endpoint volume imbalance are already supported by the weak-case analysis.

These are descriptive associations across repeated subjects, without causal attribution or independent-subject significance claims.

The most discriminating next sampling test holds expected spatial density and total particle count fixed while changing coverage variance. A label-free stratified or multi-center design can then be judged by small-structure exposure, anatomical NCC and official metrics at matched time. An information-weighted sampler also needs a uniform component, because weak native contrast and fine labels can coexist and a pure high-gradient policy can discard them. Both would be opt-in experiments. The exact exposure calculation alone does not establish that either change improves registration. [Exposure and protocol](../results/research_20261115/gs_exposure/summary.json) · [Case/label/volume controls](../results/research_20261115/gs_exposure/correlations.json) · [Paired scalar rows](../results/research_20261115/gs_exposure/pair_label_rows.csv) · [Probe source](../.local/research_20261115/gs_transfer/exposure.py)

## Independent review of the coarse-registration screen

I checked every wrapper hash against the saved manifest, verified all resolved configs equal the starting width-512 accuracy recipe, and checked every saved PCG interval converged. The wrapper substitutes only the moving/fixed observed images passed to the registration decoder. Endpoint targets, analytical SIREN derivatives, shared-grid estimation, mask sampling, regularization, optimizer and frozen recovery stay defined by the starting recipe. Affine-column norms are already in array-axis order at this adapter entry, so converting sigma millimeters to per-axis voxels is consistent. This is a causal fixed-bandwidth screen, distinct from GS-Reg's jointly scale-matched rendering/transport continuation.

| Registration-image sigma (mm) | Two-case mean forward / reverse (mm) |
|---|---:|
| 0 | 6.220 / 6.379 |
| 1.5 | 6.157 / 6.361 |
| 3 | 6.309 / 6.502 |
| 6 | 6.264 / 6.444 |


The experiment does not justify promoting observed-image blur or scheduling a large staged-blur search in this round. The frozen landscape showed a mechanism worth testing, and the training screen failed to produce a convincing method gain. A future scale continuation experiment would need an independent reason to expect that its return to native detail solves an identified failure, plus matched-time controls and an OASIS/small-organ qualification. The GS result alone supplies neither. [Eight-fit manifest and results](../results/research_20261115/coarse/)

## Orthogonal hidden initialization as a falsifiable conditioning idea

A narrower hidden layer might retain useful derivative geometry if its weight matrix avoids the small singular values of a square IID matrix. The proposed initialization keeps the first layer, scalar output, biases and paired-endpoint identity intact, while replacing each internal square weight by W=(sqrt(2)/omega)Q for orthogonal Q. Its mean squared linear gain matches ordinary SIREN initialization. This does not imply a well-conditioned nonlinear map: the layer Jacobian is sqrt(2)diag(cos(phase))Q, whose small cosine gates can remain weak. The spatial gradient/Hessian combinations used by registration also depend on compositions, output weights and local query geometry.

Before any optimizer experiment, I ran a seed 42 width 384/four-layer CPU fixture with 512 shared cells, two orthogonal symmetric query pairs per cell, unchanged first/output parameters and the native OASIS derivative scales. The falsification metric was the lower-decile smallest eigenvalue divided by trace of the actual unregularized shared-grid normal matrix mean(ggᵀ+4HᵀH), compared with the same IID initialized pair. Higher values represent a less singular local velocity estimate at the same relative scale.

| Quantity | IID hidden weights | Orthogonal hidden weights |
|---|---:|---:|
| PD-LK minimum/trace, p10 | 0.00919 | 0.00834 |
| PD-LK minimum/trace, median | 0.03280 | 0.02692 |
| Spatial gradient RMS | 0.13159 | 0.13729 |
| Spatial Hessian RMS | 0.03273 | 0.03288 |
| Last hidden activation effective rank | 26.69240 | 27.89079 |

The internal linear matrices become nearly perfectly conditioned, but the lower-decile PD-LK ratio worsens by 9.3% and the median by 17.9%. Gradient/Hessian RMS stays close, and the sampled last-layer activation effective rank rises slightly. Improved linear-matrix conditioning therefore fails its proposed derivative-conditioning prediction on this fixture. No orthogonal-initialization optimizer run was launched. This is one initialized network and one fixed query fixture; it does not prove all orthogonal SIRENs train worse. Any future benefit would require a more specific mechanism than treating weight orthogonality as dynamical isometry. [Initialization scalars](../results/research_20261115/gs_orthogonal/initialization.json) · [CPU probe source](../.local/research_20261115/gs_transfer/orthogonal_initialization.py)

## Physical support of the current training surrogate

The native grid-8 half-crop recipe has different physical support across datasets. Velocity cells use the crop-center extent divided by 8, while GW-NCC moment bins span the central 80% particle box divided by 8. These are separate grids even when both have 8 bins. Sigma 0.75 cells is therefore a broad physical context relative to several small anatomical structures. The three-tap kernel has interior weights [0.22561, 0.54878, 0.22561] and actual discrete RMS 0.67173 cells; sigma 0.75 is its nominal Gaussian parameter.

| Quantity, XYZ millimeters | OASIS |
|---|---|
| Velocity-cell pitch | 9.88 / 13.88 / 11.88 |
| Velocity nominal sigma | 7.41 / 10.41 / 8.91 |
| Velocity three-tap RMS | 6.63 / 9.32 / 7.98 |
| GW moment-bin pitch | 7.90 / 11.10 / 9.50 |
| GW nominal sigma | 5.93 / 8.33 / 7.13 |
| GW three-bin full span | 23.70 / 33.30 / 28.50 |

The median volume-equivalent radii of OASIS labels 18/34 are 2.51/2.31 mm. Even the smallest GW nominal bandwidth is 2.36/2.57 times those radii. The larger weak labels 19/35 have radii 6.50/6.99 mm, comparable to the same bandwidth. These are per-image geometry medians, without using labels to configure training.

This provides a concrete reason that high surrounding-patch NCC need not isolate a small label's placement: each local moment estimate combines nearby tissue, while the global half of GW-NCC covers the crop. The velocity field also pools neighboring cells before its training trajectory. It does not establish a hard lower bound on the final deformation's spatial resolution. Frozen PCG uses a denser recovery grid and different global brightness-flow equations, and may recover finer variation. The discrepancy is specifically between training-surrogate support and small anatomy. A proposed smaller-window or denser-grid experiment must measure both conditioning/occupancy and official small-label Dice; simply reducing sigma can weaken the shared-cell estimate. [Physical support scalars](../results/research_20261115/gs_support/summary.json) · [Per-image geometry](../results/research_20261115/gs_support/images.json)

This agrees with the absence of a robust coarse-training gain. [Matched-valid landscape groups](../results/research_20261115/weak_cases/matched_landscape_groups.json)

## Independent frozen-recovery audit and interpretation

The full 74-case recovery-smoothness control finished during the round. The only config difference is smoothness 0.01→0.003.

Mean HD95 changes 16.7678→16.8023 voxels and folding rises 0.1313%→0.3009%. These controls support a dataset-qualified inference-regularization lever with an explicit topology/boundary tradeoff. They reject a common lower-smoothness change across all datasets.

It reinforces separating representation fidelity, local matching information, learned temporal transport and final recovery when identifying the active limitation. Reported sweep times are recovery-only and must not be presented as complete fresh-fit timings. [Independent 74-case audit](../results/research_20261115/gs_frozen_recovery_audit.json)

## Prepared captured registration-weight transfer test

This motivates testing whether the direct alignment gradient is underweighted relative to endpoint fitting and path regularization. Increasing registration weight 0.01→0.03 changes the emphasis of the same loss; it does not alter its intensity normalization, particle sampling, velocity estimator or decoder. The ordinary independent-time result requires a separate transfer check in the captured Sobol recipe.

The ordinary two-case criterion was met, and the captured quick4 fits subsequently completed on explicitly released GPUs 0/1. The exact source/config and per-case checkpoint/image comparisons passed the independent audit. This is an experimental proposal with no canonical setting change. [Prepared config and source manifest](../results/research_20261115/registration_weight003/manifest.json) · [Standard-queue preview](../results/research_20261115/registration_weight003/quick4_preview.json)

## Captured quick4 registration-weight result

The captured transfer check finished all four frozen quick4 pairs with 32/32 converged recovery intervals. Registration weight 0.03 improves mean Dice and HD95 while lowering mean and maximum folding at the original recovery smoothness 0.01. Three of four pairs improve Dice and three improve HD95. Pair 2 folding rises slightly in absolute terms while the other three fall, so the per-case qualification is retained.

| Quick4 diagnostic | Starting reg 0.01 / recovery 0.01 | Reg 0.03 / recovery 0.01 | Same reg 0.03 checkpoints / recovery 0.003 |
|---|---:|---:|---:|
| Dice | 0.42880 | 0.43605 | 0.44835 |
| HD95 voxels | 19.16050 | 18.62528 | 18.90908 |
| Mean folding percent | 0.03804 | 0.01702 | 0.14741 |
| SDlogJ | 0.23344 | 0.19657 | 0.33271 |

The frozen 0.003 interaction reuses the four new checkpoint byte hashes and changes only recovery smoothness. It raises mean Dice further but worsens HD95 and increases mean folding 8.66 times. The decision to validate registration weight 0.03 on the remaining 41 pairs is based solely on the original-smoothness quick4 arm, before using the combined-recipe scores. This avoids selecting a new hybrid from the four prototype results. The canonical method remains the tagged reference pending the user's review.

The four-case result supports a stronger direct matching signal as an experimentally useful lever, despite the endpoint GW-NCC probes showing that higher local appearance fidelity alone can worsen registration. The full-cohort validation below supplies the final result. The new quick4 actual mean end-to-end time is 83.09 s versus 80.16 s for its starting-reference cases; these runs have different cache/concurrency history, so that difference is descriptive. [Paired quick4 audit](../results/research_20261115/registration_weight003/quick4_comparison.json) · [Captured fit queue](../results/research_20261115/registration_weight003/quick4/summary.json) · [Frozen interaction queue](../results/research_20261115/registration_weight003/quick4_s003/summary.json)

## Width and effective recovery regularization

A constant gain applied to the same interpolated image path multiplies both g and q by a. In brightness PCG, that scales the data term by a² while fixed smoothness and ridge remain absolute. Algebraically, the same velocity equation then has effective smoothness and ridge divided by a². This is distinct from the rejected time-varying global gain/offset explanation above. It makes amplitude calibration a potential confound when comparing different fitted representations, without supplying an automatic inference correction.

The weak-case workstream measured the same 4,096 uniform coordinates and eight recovery times for 15 paired width-512/384 checkpoints. The modest MRI/CT reductions do not establish a scalar calibration mechanism either: field orientation, spatial distribution and temporal coherence can change independently of global energy. The planned frozen width×smoothness comparison supplies an empirical control. No adaptive gain or solver change follows from these moments. [Matched energy probe](../results/research_20261115/weak_cases/global_energy.json)

The remaining 41-pair shared queue started at 13:19 HKT after all prior GPU work explicitly released its devices. At 13:25 HKT, 12/41 non-quick4 fits had completed without failures. Their provisional paired means were Dice −0.00603, HD95 +0.383 voxels and folding +0.01474 percentage points. This initial counterevidence shows why the positive quick4/development examples require the full official cohort. The source, config and evaluation stayed fixed throughout the queue. This entry records provisional mid-run evidence; the completed 45-pair union below supplies the final interpretation.

A variance-normalized follow-up qualifies the energy result. Dividing each model's gradient energy by its own mean frame variance gives median ratios 0.983, 0.994 and 1.068. Local anatomical fidelity and directional/path structure can still differ. This supports retaining effective-regularization calibration as a qualification of architecture comparisons, with the frozen 2×2 control supplying the actual clinical-score evidence. It does not establish scalar gain as the cause of any observed score difference.

## Independent smaller-model cohort audit

The width-384/four-layer 6k recipe completed all 74 official pairs, including 12 already-completed frozen quick4 fits reused without another optimization run. I independently verified every pair, checkpoint byte hash, finite state, source hash and same-image/inference configuration. The only recipe changes are width 512→384, updates 5000→6000 and OneCycle peak 1e-4→1.5e-4. All 592 recovery intervals converged. The resulting primary means are slightly worse on all three datasets, with lower actual per-case runtime and lower mean folding. This supports a measured economy/accuracy tradeoff, without proving equal-or-better accuracy.

Candidate per-case timings come from two persistent GPU workers, while the original cohort used four. Twelve quick4 fits are reused in the 74-case union, so the remaining-queue elapsed wall time is not an independently rerun 74-case wall-time measurement. The separate same-device balanced ABBA controls provide the cleaner runtime contrast. [Independent full-cohort audit](../results/research_20261115/gs_width_full_audit.json) · [Width-agent complete analysis](RESEARCH_WIDTH_20261115.md)

## Final independent review status

The recovery comparison reused the two trained width configurations and changed only fixed smoothness.

| Width / updates / recovery smoothness | OASIS Dice |
|---|---:|
| 512 / 5k / 0.01 | 0.81337 |
| 384 / 6k / 0.01 | 0.81205 |
| 512 / 5k / 0.003 | 0.81109 |
| 384 / 6k / 0.003 | 0.81046 |

Both widths incur higher folding at lower smoothness. This is an empirical operating-point interaction, with no demonstrated universal architecture or recovery improvement.

All GPU work owned by this review is terminal. The neighboring GS checkout was read only; this review introduced no canonical-source or default-preset change and no commit. Scalar evidence, experiment-only configs and local scripts are retained under ignored research output locations. The four live research documents are the visible review artifacts. The source remains the tagged starting version for the user's later methodology discussion.
