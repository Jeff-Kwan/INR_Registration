# GS-Reg transfer review, 2026-11-15

This is an experimental review during the 90–120 minute research round. The neighboring checkout was read in place. Its code, inputs and outputs were left intact. No suggestion here changes the canonical INR method or presets.

## Mechanism and direct evidence

GS-Reg fits a source-specific, fixed-size Gaussian intensity representation before estimating its geometric movement. INR jointly fits paired SIREN endpoints and asks their parameter interpolation to explain motion. GS-Reg therefore separates representation capacity from the subsequent registration objective more directly. A low endpoint fitting error can still coexist with poor correspondence or field reconstruction in either method. The relevant transferable experiment holds the learned appearance fixed while varying only the matching scale or inference route.

GS-Reg also uses one fixed-image intensity transform for both endpoints with moving-image clipping, whereas INR independently normalizes the endpoints and uses lung-mask crop centers for DIR-Lab. Cross-project clinical scores therefore combine representation, normalization, optimization and recovery differences. The comparisons below use controlled changes within GS-Reg to motivate isolated INR tests.

The strongest measured clue is spatial capture range. In GS-Reg, reusing the exact same fitted Gaussians and widening the common blur stack from 4/2/1/0 to 8/4/2/1/0 reduces mean DIR-Lab continuous TRE from 2.79000 to 2.13107 mm. Case 08 moves from 9.21757 to 5.89611 mm, while case 09 worsens from 1.61681 to 1.69987 mm. OASIS Dice changes only +0.000633, and abdomen +0.003698 with greater folding. This supports a broad-scale capture experiment on large-motion DIR-Lab, with explicit small-motion controls. It does not support a common scale increase across all datasets. [GS medical evidence](/workspace/3DGS_Registration/docs/MEDICAL_RESULTS.md)

GS-Reg also derives the local NCC translation information after projecting spatial gradients away from nuisance offset and gain. For centered target intensities v and centered spatial gradients Gc, use Gp = Gc − v(vᵀGc)/(vᵀv). Away from variance floors, the self-match translation Hessian is GpᵀGp/(vᵀv). A linear ramp has nonzero image gradient and zero translational information under NCC. Thus high patch variance, high gradient magnitude and high endpoint NCC each answer a different question. Matching may remain ambiguous along poorly observable axes even if the SIREN reproduces the image accurately. [GS method](/workspace/3DGS_Registration/docs/METHOD.md)

The GS abdomen audit demonstrates why scale needs a paired information check. For target 0007 at sigma8 with radius3, only 39.77% of organ-weighted windows exceed its variance floor. Pancreas patches are almost all below it. Enlarging support restores some signal but a controlled matching experiment failed to improve registration. The Gaussian images, clipping policy and variance floor differ from INR, so these fractions cannot be assigned to INR. The transferable diagnostic measures the actual image/SIREN patch information and capture landscape under our normalization. [GS observation probe](/workspace/3DGS_Registration/results/marathon_20261115/medical/diagnostics/ncc_observability/probe.py) · [GS scalar evidence](/workspace/3DGS_Registration/results/marathon_20261115/medical/diagnostics/ncc_observability/summary.json)

## Transfer candidates and distinguishing observations

| Candidate | Mechanism | Observation that supports it | Competing explanation or qualification |
|---|---|---|---|
| Coarse-to-fine observed-image matching | Broadening matched image features increases the basin for large displacement before native detail is restored | Landmark patch correlation has a useful coarse-scale descent direction outside the native basin, and weak DIR-Lab TRE improves at fixed fit/time | Excess smoothing removes small-label appearance or improves the surrogate while folding rises |
| NCC information diagnostics | Project out brightness nuisance directions before judging local spatial information | Weak labels or landmarks show a low smallest information eigenvalue despite adequate fitted NCC | A large information eigenvalue with wrong correspondence points to a capture, representation-path or inference problem |
| Appearance/path separation | Reuse an identical frozen checkpoint for paired inference or diagnostic controls | Strong endpoint fit and matching landscape coexist with bad map TRE | Fit capacity explains the error if reducing width degrades anatomical NCC and registration together |
| Retain native finishing updates | End at full spatial detail after broad matching establishes approximate placement | Broad early scales help large motion while native finishing preserves small-label Dice | More total optimization time alone can explain a benefit without a time-matched control |

The GS broad schedule is paired with a linearly decreasing learning rate and coherence strength, followed by 200 native updates at fixed low LR. Any INR transfer must isolate blur/scale from optimizer and iteration count. Its force-sharing candidate reduced DIR-Lab TRE by 4.90% but had mixed OASIS and abdomen effects. That evidence supports a dataset-qualified ablation, and gives no reason to add an unconditional optimizer coupling to INR. [GS schedule and force findings](/workspace/3DGS_Registration/docs/METHOD.md) · [GS paired controls](/workspace/3DGS_Registration/docs/MEDICAL_RESULTS.md)

## Focused validation plan

A scalar-only frozen probe will examine DIR-Lab cases 01 and 08 at known corresponding landmark centers. It will compare image patches and SIREN patches at native and wider support, record translation NCC curves around the true center and identity, and compare high/low landmark errors where available. This tests whether representation fidelity, motion magnitude or local ambiguity better predicts weak landmarks. It will use held fixed checkpoint weights, seed 42, original input geometry and no fitted or label-assisted deformation field. Locations and labels are evaluation-only. GPU execution will be coordinated with the shared research queue.

A positive landscape result would motivate an isolated, opt-in image-scale experiment. Existing GW-NCC smooths moments in velocity-grid bins and is distinct from blurring the observed images before matching. Its scale does not automatically reproduce the GS continuation effect. The paired-SIREN derivatives, shared-grid PD-LK and brightness PCG/RK2 recovery must retain their current definitions in that experiment.

## Review status

The initial read covered GS AGENTS.md, README.md, METHOD.md, MEDICAL_RESULTS.md, SYNTHETIC_RESULTS.md, the local-affine tracker, correlation implementation and the saved observability probe. No algorithm was imported or executed from the neighboring checkout. Validation results will be appended as they finish.

## Frozen landmark capture landscapes

The probe finished cases 01, 07, 08 using their exact starting-baseline checkpoints and original landmark pairs. It sampled observed and rasterized SIREN intensities on the same native grids, with 9³ physical patch sites, 8/16 mm support and Gaussian blur sigma 0/2/4 mm. Each landmark was queried along the line from identity placement to the known target correspondence, extended to 1.5 times the true displacement. Line points were t = 0, 0.125, …, 1.5. The diagnostic writes per-landmark scalar correlations, true motion magnitudes and axis-shift curves. The learned raster exists only in RAM. Out-of-image patch sites are excluded. Correlations require at least eight samples and population variance above 1e-6 in both patches. Known correspondences are evaluation-only.

| Case | Image | Blur mm | True-match NCC | First step toward truth improves | Best line position near truth |
|---|---|---:|---:|---:|---:|
| 01 | observed | 0 | 0.8114 | 85.0% | 77.0% |
| 01 | observed | 2 | 0.9245 | 78.3% | 67.0% |
| 01 | observed | 4 | 0.9755 | 51.0% | 35.0% |
| 01 | siren | 0 | 0.8689 | 84.7% | 72.3% |
| 01 | siren | 2 | 0.8964 | 85.3% | 70.3% |
| 01 | siren | 4 | 0.9353 | 83.0% | 48.0% |
| 07 | observed | 0 | 0.7513 | 63.3% | 96.7% |
| 07 | observed | 2 | 0.8560 | 68.3% | 90.7% |
| 07 | observed | 4 | 0.9015 | 77.3% | 73.3% |
| 07 | siren | 0 | 0.6988 | 69.3% | 78.7% |
| 07 | siren | 2 | 0.7554 | 74.7% | 73.0% |
| 07 | siren | 4 | 0.8363 | 81.7% | 62.3% |
| 08 | observed | 0 | 0.7568 | 60.0% | 97.7% |
| 08 | observed | 2 | 0.8679 | 66.7% | 96.0% |
| 08 | observed | 4 | 0.8928 | 65.3% | 85.7% |
| 08 | siren | 0 | 0.7424 | 68.3% | 87.0% |
| 08 | siren | 2 | 0.7755 | 67.0% | 86.7% |
| 08 | siren | 4 | 0.8245 | 69.7% | 76.0% |

“Near truth” means that the largest of the sampled line correlations occurs at t between 0.75 and 1.25. The first step is one eighth of the actual motion, so its physical length varies across landmarks. These are capture-landscape probes, not registration accuracy or a ground-truth neighborhood field.

Case 08's observed patches retain a strong true-correspondence optimum even at native scale: 97.7% peak near the true position along that particular line, but only 60.0% improve at the initial step. This separates local match quality from reaching its basin. Blur 2 raises the initial-direction fraction to 66.7% while preserving 96.0% near-truth peaks. Blur 4 raises mean true-match correlation further but reduces near-truth peaks to 85.7%, showing that a smoother surrogate can also make the optimum less distinctive. The SIREN native line peaks near truth for 87.0% of case 08 landmarks, versus 97.7% in the actual images, despite its relatively high endpoint patch fidelity. Thus both capture and learned local shape can contribute.

The clear prediction is that moderate broadening may help case 08 early motion while excessive fixed blur can underperform. Fixed blur is only a mechanism screen. A positive result should be followed by a native-finishing schedule and time-matched controls. [Per-landmark scalar evidence](../results/research_20261115/gs_landscapes/) · [Local probe source](../.local/research_20261115/gs_transfer/landscape.py)

## Native anatomical information and the cost of blur

The second probe evaluates all ten unique abdomen images, independently of their role as moving/fixed in the directed pairs. For each present label it samples 64 centers at seed 42 and jitters each by at most a quarter voxel to avoid systematic placement at interpolation knots. It uses 9³ samples over 8/16 mm physical patches, excludes outside-image sites, and obtains exact trilinear interpolation gradients through query autograd. The gradients are converted to derivatives per millimeter. Both the full neighborhood and its label-restricted subset are recorded. No image gradients or labels enter training.

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

At sigma 4 mm the minimum information falls by roughly 31× for the left adrenal and 15× for the right adrenal. Liver valid-patch fraction falls from 99.4% to 53.7%. Larger 16 mm support restores valid liver neighborhoods to 78.6% and some minimum information, but does not restore native precision. The variance threshold is an evaluation validity rule, rather than a target-window gate in our GW-NCC training objective. This predicts that a broad scale helpful for large DIR-Lab motion may hurt small-organ placement or final refinement unless native detail returns. The exact ratios are diagnostic, without a fitted uncertainty model or registration-accuracy guarantee.

[Per-image/per-label measurements](../results/research_20261115/gs_information/) · [Organ summary](../results/research_20261115/gs_information/summary.json) · [Independent Hessian validation](../results/research_20261115/gs_information/hessian_validation.json) · [Probe source](../.local/research_20261115/gs_transfer/information.py)

## Rejected global intensity-gauge explanation

GS-Reg keeps fitted amplitudes fixed while tracking geometry. This motivated checking whether our linearly interpolated SIREN parameters introduce a large global gain/offset change that brightness recovery could mistake for transport. On frozen DIR-Lab 01/07/08, OASIS 1 and abdomen 19 checkpoints, the diagnostic evaluated the same 8,192 uniform seed 42 coordinates at t=0,.125,.375,.625,.875,1. It regressed the exact model temporal derivative q onto a constant and the centered intensity I. This isolates the global offset rate and logarithmic scale rate at each sampled time.

| Checkpoint | Mean middle-time q energy explained by gain/offset | Maximum middle-time fraction |
|---|---:|---:|
| abdomenctct 19 | 0.924% | 1.033% |
| dirlab 1 | 0.180% | 0.274% |
| dirlab 7 | 0.115% | 0.302% |
| dirlab 8 | 0.143% | 0.336% |
| oasis 1 | 0.871% | 1.141% |

No sampled path shows a middle-time global standard-deviation collapse. On the three DIR-Lab cases, removing this fitted nuisance component from q increases the mean landmark transport residual q+g·D from 0.10589 to 0.10612, 0.16558 to 0.16907 and 0.22779 to 0.23042 respectively. Correlation with TRE is essentially unchanged. Here D is the known sparse moving-to-fixed landmark displacement, extended as a constant across each 27-site 8 mm patch and evaluated along that landmark's straight reference path. This is a diagnostic sparse-motion model, without an inferred full truth field.

The result rejects a simple global gain/offset explanation for these weak cases. Spatially local intensity/path errors remain possible. A global intensity correction is therefore a poor next experiment for this round. Projecting q alone would also define no guaranteed consistent image path, and can remove a legitimate geometric intensity change. No recovery equation or checkpoint was modified. [Gauge summary](../results/research_20261115/gs_gauge/summary.json) · [Probe source](../.local/research_20261115/gs_transfer/gauge.py)

## Comparison with GS field reconstruction

GS-Reg's controlled endpoint-image versus decoded-field audit found an actual gap even with identical fitted Gaussians. For the abdomen example, the selected endpoint image has native LNCC loss 0.209 while the fitted source pulled through its recovered field has loss 0.743. Source-fit and subvoxel interpolation controls are smaller than the gap. The audit does not uniquely identify chart approximation, midpoint interpolation, bidirectional blending or inverse iteration as its cause. A later single-forward replacement improved some internal consistency scores but worsened every checked official medical metric and increased folding 6–13 times. [Controlled GS image/field report](/workspace/3DGS_Registration/results/marathon_20261115/medical/diagnostics/image_field_consistency/CONSISTENCY_REPORT.md) · [GS synthetic/rejection record](/workspace/3DGS_Registration/docs/SYNTHETIC_RESULTS.md)

The transferable discipline is to compare the same appearance model under the training surrogate and the final scoring field, while retaining original-image patch fidelity and anatomy metrics as controls. In INR the estimators differ between shared-grid PD-LK training and global brightness PCG recovery. A discrepancy there should be localized before adding another regularizer or replacing the solver. The weak-case workstream already observes that learned warped-patch agreement can exceed original-image agreement precisely at high-TRE landmarks. That observation strengthens this diagnostic priority, and does not establish which part of the train-to-recovery pipeline causes it.

## Exact crop exposure and small-label weighting

The exposure audit enumerates the distribution over every possible integer half-crop origin in each of the 20 unique OASIS and 10 unique abdomen images. Endpoint reconstruction samples 32,768 discrete crop voxels. Each registration direction draws 32,768 continuous points from the central 80% of the crop-center extent. For the latter, anatomical membership is modeled by nearest-label unit-voxel cells, and the audit integrates fractional intersections at each box edge exactly. Separable cumulative sums give the expected count conditional on every crop origin. Averaging the binomial zero-count probability across origins retains crop-level absence and finite-particle sampling separately. Constant-mask and direct fractional-cell controls pass. These quantities describe sampling exposure, not the full influence of a point on the nonlinear NCC objective or whether transported endpoints reach the correct target label.

| Dataset / label | Median voxels | Mean endpoint samples/update | Mean particles/direction/update | Probability of zero particles | Mean particles when crop exposes label |
|---|---:|---:|---:|---:|---:|
| abdomenctct / liver | 210692 | 2169.29 | 2506.94 | 23.8% | 3212.88 |
| abdomenctct / pancreas | 11465.5 | 188.84 | 235.98 | 28.7% | 320.31 |
| abdomenctct / left adrenal gland | 632 | 11.13 | 13.88 | 55.9% | 29.95 |
| abdomenctct / right adrenal gland | 779.5 | 12.99 | 16.06 | 57.1% | 35.23 |
| oasis / 18 | 66 | 1.33 | 1.70 | 66.3% | 4.28 |
| oasis / 19 | 1148.5 | 25.17 | 31.26 | 18.4% | 36.92 |
| oasis / 34 | 51.5 | 1.17 | 1.50 | 67.0% | 3.80 |
| oasis / 35 | 1432.5 | 32.94 | 40.61 | 17.0% | 47.25 |

Values average unique images equally. The probability of zero particles averages over all crop origins, while the conditional count averages only crop boxes with positive label support. The small-label means are greater than full-volume uniform sampling would yield because their central positions receive more crop coverage. The central 80% rule nevertheless leaves more updates with no adrenal particles than the full-crop endpoint sampler. Merely replacing crops by whole-image uniform sampling could reduce their mean exposure; these results do not support that replacement.

The two smallest weak OASIS labels 18/34 receive roughly 1.5–1.7 particles per direction on average and none in two thirds of updates. Labels 19/35 are also weak, but receive 31–41 particles and are absent in only 17–18% of updates. Their poor overlap needs explanations beyond rare sampling. Small-label Dice sensitivity and endpoint volume imbalance are already supported by the weak-case analysis.

After controlling case, anatomical label identity, log mean label volume and endpoint volume imbalance, zero-particle probability versus Dice has rank correlation +0.043 for OASIS and −0.137 for abdomen. The pooled negative relationships therefore mostly reflect anatomy and size; a modest abdomen association remains. These are descriptive associations across repeated subjects, without causal attribution or independent-subject significance claims.

The most discriminating next sampling test holds expected spatial density and total particle count fixed while changing coverage variance. A label-free stratified or multi-center design can then be judged by small-structure exposure, anatomical NCC and official metrics at matched time. An information-weighted sampler also needs a uniform component, because weak native contrast and fine labels can coexist and a pure high-gradient policy can discard them. Both would be opt-in experiments. The exact exposure calculation alone does not establish that either change improves registration. [Exposure and protocol](../results/research_20261115/gs_exposure/summary.json) · [Case/label/volume controls](../results/research_20261115/gs_exposure/correlations.json) · [Paired scalar rows](../results/research_20261115/gs_exposure/pair_label_rows.csv) · [Probe source](../.local/research_20261115/gs_transfer/exposure.py)

## Independent review of the coarse-registration screen

The root workstream completed eight fresh fits on DIR-Lab 07/08 at fixed observed-registration-image blur sigma 0/1.5/3/6 mm. I checked every wrapper hash against the saved manifest, verified all resolved configs equal the starting width-512 accuracy recipe, and checked every saved PCG interval converged. The wrapper substitutes only the moving/fixed observed images passed to the registration decoder. Endpoint targets, analytical SIREN derivatives, shared-grid estimation, mask sampling, regularization, optimizer and frozen recovery stay defined by the starting recipe. Affine-column norms are already in array-axis order at this adapter entry, so converting sigma millimeters to per-axis voxels is consistent. This is a causal fixed-bandwidth screen, distinct from GS-Reg's jointly scale-matched rendering/transport continuation.

| Registration-image sigma (mm) | Case 07 forward / reverse TRE (mm) | Case 08 forward / reverse TRE (mm) | Two-case mean forward / reverse (mm) |
|---|---:|---:|---:|
| 0 | 4.379 / 4.517 | 8.060 / 8.240 | 6.220 / 6.379 |
| 1.5 | 4.303 / 4.456 | 8.010 / 8.266 | 6.157 / 6.361 |
| 3 | 4.448 / 4.625 | 8.169 / 8.378 | 6.309 / 6.502 |
| 6 | 4.439 / 4.499 | 8.088 / 8.390 | 6.264 / 6.444 |

Sigma 1.5 changes the two-case mean forward TRE by −0.063 mm and reverse by −0.017 mm. Case 08 reverse worsens slightly. Larger blur shows no consistent benefit and sigma 6 raises mean whole-grid folding from 0.00192% to 0.00304%; all lung folding fractions remain zero. The small 1.5 mm changes are insufficient to establish a reproducible improvement given single-seed fresh-fit CUDA variation and only two deliberately selected weak cases. First-arm timing includes different compilation history, so the 85–101 s queue-time range cannot establish a blur speedup.

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

| Quantity, XYZ millimeters | OASIS | AbdomenCTCT |
|---|---|---|
| Velocity-cell pitch | 9.88 / 13.88 / 11.88 | 23.75 / 19.75 / 31.75 |
| Velocity nominal sigma | 7.41 / 10.41 / 8.91 | 17.81 / 14.81 / 23.81 |
| Velocity three-tap RMS | 6.63 / 9.32 / 7.98 | 15.95 / 13.27 / 21.33 |
| GW moment-bin pitch | 7.90 / 11.10 / 9.50 | 19.00 / 15.80 / 25.40 |
| GW nominal sigma | 5.93 / 8.33 / 7.13 | 14.25 / 11.85 / 19.05 |
| GW three-bin full span | 23.70 / 33.30 / 28.50 | 57.00 / 47.40 / 76.20 |

The median volume-equivalent radii of OASIS labels 18/34 are 2.51/2.31 mm. Even the smallest GW nominal bandwidth is 2.36/2.57 times those radii. The larger weak labels 19/35 have radii 6.50/6.99 mm, comparable to the same bandwidth. Abdomen adrenal radii are 10.65/11.42 mm; the local GW context extends across substantially larger physical spans, and the actual thin/elongated anatomy can be narrower than an equivalent sphere. These are per-image geometry medians, without using labels to configure training.

This provides a concrete reason that high surrounding-patch NCC need not isolate a small label's placement: each local moment estimate combines nearby tissue, while the global half of GW-NCC covers the crop. The velocity field also pools neighboring cells before its training trajectory. It does not establish a hard lower bound on the final deformation's spatial resolution. Frozen PCG uses a denser recovery grid and different global brightness-flow equations, and may recover finer variation. The discrepancy is specifically between training-surrogate support and small anatomy. A proposed smaller-window or denser-grid experiment must measure both conditioning/occupancy and official small-label Dice; simply reducing sigma can weaken the shared-cell estimate. [Physical support scalars](../results/research_20261115/gs_support/summary.json) · [Per-image geometry](../results/research_20261115/gs_support/images.json)

A matched-landmark follow-up by the weak-case workstream further qualifies the broad-scale interpretation. For case 08's highest-TRE quartile, the observed 8 mm patch first-step improvement fraction decreases from 0.630 at native scale to 0.521 at sigma 2 and 0.384 at sigma 4, despite a pooled improvement at sigma 2. Therefore the average landscape benefit does not preferentially help the actual worst landmarks. This agrees with the absence of a robust coarse-training gain. [Matched-valid landscape groups](../results/research_20261115/weak_cases/matched_landscape_groups.json)

## Current training decoder versus frozen PCG on weak DIR-Lab cases

To test whether accurate local training alignment was being lost by global recovery, the final starting-version checkpoints for cases 07/08 were replayed on 64 native mask-centered crops at seed 42. Each draw consumes the ordinary 32k endpoint sampling stream, uses the same endpoint-mask choice, orthogonal queries and Sobol times, and computes the captured width-512 shared-cell velocity algebra, three-tap smoothing and two midpoint RK2 half-steps. Landmarks participate only when their start lies in the actual central 80% particle box. All 300 landmarks in each direction were covered, with 5–43 crops per landmark. No weights were updated, no recovery map was computed or saved, and the PCG comparison uses the previously validated per-landmark errors from the same checkpoint.

| Case / direction | Mean local decoder TRE | Crop-averaged endpoint TRE | PCG, same landmarks | Best-crop truth oracle TRE |
|---|---:|---:|---:|---:|
| 07 / forward T00→T50 | 6.549 | 6.499 | 4.413 | 5.465 |
| 07 / reverse T50→T00 | 6.802 | 6.752 | 4.612 | 5.667 |
| 08 / forward T00→T50 | 9.957 | 9.915 | 8.042 | 8.686 |
| 08 / reverse T50→T00 | 10.584 | 10.540 | 8.319 | 9.247 |

All errors are continuous millimeters and weight landmarks equally. The crop-averaged endpoint averages predicted coordinates before computing error. The best-crop oracle chooses the lowest true landmark error independently for each landmark, so it uses evaluation truth and cannot be an inference method. Even that oracle has a worse mean than PCG in all four comparisons. Across-crop endpoint variance accounts for only about 1.0% of local squared error in case 07 and 0.54–0.56% in case 08. Averaging crop noise therefore leaves nearly all of this local decoder deficit.

This rejects the hypothesis of a hidden, accurate two-field local training map that global PCG loses on these two frozen models. The local surrogate is already inaccurate, and PCG improves its landmark endpoints. The contrast does not isolate one estimator term: local 8³ crop geometry, two temporal samples and two RK2 half-steps differ jointly from the global eight-time PCG solve and its denser recovery grid. A future consistency experiment should therefore vary one of these at a time. The existing weak path-forcing and original-image patch analyses remain relevant to why the learned interpolation supplies insufficient motion. [Case 07 sparse evidence](../results/research_20261115/gs_local_decoder/case07.json) · [Case 08 sparse evidence](../results/research_20261115/gs_local_decoder/case08.json) · [Probe source](../.local/research_20261115/gs_transfer/local_decoder.py)

The local-decoder travel audit further identifies systematic undershoot. Mean projection onto the true movement is 0.455/0.428 for case 07 forward/reverse and 0.425/0.380 for case 08, versus 0.661/0.642 and 0.621/0.608 for PCG. A single truth-fitted gain applied to the crop-averaged local displacement improves its error, but the best gains 1.83–2.12 still leave mean errors 4.70/4.79 mm and 8.81/9.35 mm. Those remain worse than the corresponding PCG errors. This oracle uses landmarks as evaluation truth and supplies no deployable gain rule. It shows that amplitude underestimation accounts for part of the local error while directional/path discrepancies remain. [Sparse travel decomposition](../results/research_20261115/gs_local_decoder/travel_audit.json)

## Independent frozen-recovery audit and interpretation

The full 74-case recovery-smoothness control finished during the round. I independently checked the exact 19 OASIS / 45 abdomen / 10 DIR-Lab pair set, all source hashes, every original checkpoint byte hash, the original run configs and the evaluated configs. The only config difference is smoothness 0.01→0.003. All 592 PCG intervals converged, each case verified temporary-scratch cleanup and output inventories contain scalar metrics/provenance plus sparse landmark metrics only.

Abdomen Dice improves 0.438975→0.453706 with 41/45 pairwise gains despite identical endpoint models and anatomical patch fidelity. Mean HD95 changes 16.7678→16.8023 voxels and folding rises 0.1313%→0.3009%. OASIS Dice declines 0.813372→0.811087 with only 3/19 gains, and rounded DIR-Lab forward/reverse TRE changes 2.8471/2.9639→2.8689/2.9755 mm. These controls support a dataset-qualified inference-regularization lever with an explicit topology/boundary tradeoff. They reject a common lower-smoothness change across all datasets.

Because appearance is frozen, this result directly demonstrates that raising endpoint anatomical NCC is not a prerequisite for improving abdominal Dice. It reinforces separating representation fidelity, local matching information, learned temporal transport and final recovery when identifying the active limitation. Reported sweep times are recovery-only and must not be presented as complete fresh-fit timings. [Independent 74-case audit](../results/research_20261115/gs_frozen_recovery_audit.json)

## Prepared captured registration-weight transfer test

A matched ordinary-execution weak-case experiment found that stronger observed-image registration weighting can improve abdominal overlap while endpoint patch fidelity decreases. This motivates testing whether the direct alignment gradient is underweighted relative to endpoint fitting and path regularization. Increasing registration weight 0.01→0.03 changes the emphasis of the same loss; it does not alter its intensity normalization, particle sampling, velocity estimator or decoder. The ordinary independent-time result requires a separate transfer check in the captured Sobol recipe.

A plan is prepared for the frozen AbdomenCTCT quick4 selection 2/8/18/41 with the starting width-512 5k recipe and that one weight change. The config is frozen and validated, source/config hashes are recorded, and normal `medical.abdomen_experiments` planning confirms four jobs with no missing inputs. The ordinary two-case criterion was met, and the captured quick4 fits subsequently completed on explicitly released GPUs 0/1. The exact source/config and per-case checkpoint/image comparisons passed the independent audit. This is an experimental proposal with no canonical setting change. [Prepared config and source manifest](../results/research_20261115/registration_weight003/manifest.json) · [Standard-queue preview](../results/research_20261115/registration_weight003/quick4_preview.json)

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

The weak-case workstream measured the same 4,096 uniform coordinates and eight recovery times for 15 paired width-512/384 checkpoints. Mean 384/512 gradient-energy ratios are 0.908 for four OASIS cases, 0.879 for five abdomen cases and 1.060 for six DIR-Lab cases; corresponding q-energy ratios are 0.915, 0.915 and 1.038. DIR-Lab therefore contradicts a lower-gradient explanation for its narrower-model differences. The modest MRI/CT reductions do not establish a scalar calibration mechanism either: field orientation, spatial distribution and temporal coherence can change independently of global energy. The planned frozen width×smoothness comparison supplies an empirical control. No adaptive gain or solver change follows from these moments. [Matched energy probe](../results/research_20261115/weak_cases/global_energy.json)

The remaining 41-pair shared queue started at 13:19 HKT after all prior GPU work explicitly released its devices. At 13:25 HKT, 12/41 non-quick4 fits had completed without failures. Their provisional paired means were Dice −0.00603, HD95 +0.383 voxels and folding +0.01474 percentage points. This initial counterevidence shows why the positive quick4/development examples require the full official cohort. The source, config and evaluation stayed fixed throughout the queue. This entry records provisional mid-run evidence; the completed 45-pair union below supplies the final interpretation.

A variance-normalized follow-up qualifies the energy result. Median 384/512 frame-variance ratios are 0.930 OASIS, 0.872 abdomen and 1.005 DIR-Lab. Dividing each model's gradient energy by its own mean frame variance gives median ratios 0.983, 0.994 and 1.068. The MRI/abdomen global gradient-energy reduction is therefore largely amplitude-like in these sampled checkpoints, rather than evidence of a large global loss of spatial derivative energy per unit image variance. Local anatomical fidelity and directional/path structure can still differ. This supports retaining effective-regularization calibration as a qualification of architecture comparisons, with the frozen 2×2 control supplying the actual clinical-score evidence. It does not establish scalar gain as the cause of any observed score difference.

## Independent smaller-model cohort audit

The width-384/four-layer 6k recipe completed all 74 official pairs, including 12 already-completed frozen quick4 fits reused without another optimization run. I independently verified every pair, checkpoint byte hash, finite state, source hash and same-image/inference configuration. The only recipe changes are width 512→384, updates 5000→6000 and OneCycle peak 1e-4→1.5e-4. All 592 recovery intervals converged. The resulting primary means are slightly worse on all three datasets, with lower actual per-case runtime and lower mean folding. This supports a measured economy/accuracy tradeoff, without proving equal-or-better accuracy.

Candidate per-case timings come from two persistent GPU workers, while the original cohort used four. Twelve quick4 fits are reused in the 74-case union, so the remaining-queue elapsed wall time is not an independently rerun 74-case wall-time measurement. The separate same-device balanced ABBA controls provide the cleaner runtime contrast. [Independent full-cohort audit](../results/research_20261115/gs_width_full_audit.json) · [Width-agent complete analysis](RESEARCH_WIDTH_20261115.md)

## Full 45-pair captured registration-weight validation

The remaining 41 fits completed and the frozen quick4 union contains exactly 45 official directed pairs. The final independent audit checked every source/protocol/config hash, identical image geometry and evaluation conventions, finite checkpoint weights and the single intended training-config difference. All 360 recovery intervals converged. The normal queue exited successfully, all GPU workers stopped and no temporary recovery directories remained before the next frozen sweep began.

| Full AbdomenCTCT mean | Starting reg 0.01 | Captured reg 0.03 |
|---|---:|---:|
| Dice | 0.43897 | 0.44255 |
| HD95 voxels | 16.76780 | 16.79084 |
| Folding percent | 0.13131 | 0.12493 |
| SDlogJ | 0.27687 | 0.27444 |
| Actual end-to-end seconds | 79.75724 | 80.41216 |
| Training seconds | 68.98346 | 69.25430 |

Dice improves in 27/45 pairs, with median paired change +0.00310. Excluding the two ordinary-execution development pairs 2/19 leaves mean paired Dice change +0.00240 across 43 pairs. HD95 improves in only 20/45 pairs, and its median paired change is +0.06591voxels. The favorable quick4 HD95 result therefore fails to transfer to the full cohort.

Mean folding falls, but 19/45 pairs increase. The 90th-percentile folding fraction changes 0.53467%→0.57239%, and maximum folding 0.88834%→0.88915%. Thus the result is a modest Dice improvement with mixed boundary and folding-tail effects. It supports a qualified experimental alignment-weight candidate, without establishing a uniformly better abdominal method. The setting remains outside the canonical presets.

The primary 45 pairs all retain recovery smoothness 0.01. The four-checkpoint 0.003 interaction is separate and does not enter this denominator. Actual runtime aggregates include the initial two-GPU quick4 phase and the remaining four-GPU shared queue, with different cache history from the original baseline. No designed speedup follows from multiplying a loss coefficient. [Full 45-pair audit and paired metrics](../results/research_20261115/registration_weight003/full45_audit.json) · [Post-training cleanup verification](../results/research_20261115/registration_weight003/post_training_cleanup.json) · [Frozen source/config manifest](../results/research_20261115/registration_weight003/manifest.json)

The per-label follow-up sharpens the qualification. In fixed baseline-volume quartiles, the smallest anatomical-label group declines 0.22789→0.22321 Dice. Right/left kidney mean gains are +0.02087/+0.01786, while right/left adrenal changes are −0.00992/−0.00250. The lowest initial-centroid-mismatch quartile declines by 0.00947 Dice, whereas larger-mismatch groups gain. Centroid displacement remains a placement proxy, rather than deformation ground truth. Stronger global registration weighting therefore improves the overall mean mainly through selected organ/misalignment regimes and does not solve the small-adrenal weakness. That conclusion is consistent with adequate native local information coexisting with sparse exposure and broad training context. [Fixed-size-group and per-label comparison](../results/research_20261115/registration_weight003/label_size_comparison.json) · [Paired label rows](../results/research_20261115/registration_weight003/label_comparison.csv)

## Final independent review status

The last width-384/smoothness-0.003 cohort completed all 74 cases. Its independent audit verifies exact official pair coverage, checkpoint byte identity, stable source hashes, a smoothness-only config change relative to each width-384 model, 592 converged PCG intervals and per-case temporary-scratch cleanup. The complete fixed-capacity×smoothness comparison is:

| Width / updates / recovery smoothness | OASIS Dice | Abdomen Dice | DIR-Lab rounded forward / reverse TRE, mm |
|---|---:|---:|---:|
| 512 / 5k / 0.01 | 0.81337 | 0.43897 | 2.847 / 2.964 |
| 384 / 6k / 0.01 | 0.81205 | 0.43449 | 2.928 / 3.037 |
| 512 / 5k / 0.003 | 0.81109 | 0.45371 | 2.869 / 2.975 |
| 384 / 6k / 0.003 | 0.81046 | 0.45265 | 2.962 / 3.061 |

Lower smoothness narrows the smaller-model Dice gaps on brain and abdomen, while the DIR-Lab deficit remains. Both widths incur higher folding at lower smoothness. This is an empirical operating-point interaction, with no demonstrated universal architecture or recovery improvement. The lower-gradient/amplitude hypothesis alone cannot explain the six examined DIR-Lab models, whose gradient energy is higher at width 384. [Independent final frozen audit](../results/research_20261115/gs_frozen_width384_recovery_audit.json)

All GPU work owned by this review is terminal. The neighboring GS checkout was read only; this review introduced no canonical-source or default-preset change and no commit. Scalar evidence, experiment-only configs and local scripts are retained under ignored research output locations. The four live research documents are the visible review artifacts. The source remains the tagged starting version for the user's later methodology discussion.
