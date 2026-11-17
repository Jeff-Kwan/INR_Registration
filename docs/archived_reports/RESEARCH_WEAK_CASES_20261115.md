# Weak-case research, 15 November 2026

The investigation is complete. The reference is the tagged starting version and the completed 74-pair run in `results/full_baseline_20261115_run`. This analysis uses all 19 OASIS pairs, all 45 abdomen pairs and all ten DIR-Lab cases. Case selection based on the reference scores is development analysis.

## Mechanism and diagnostic design

A high local reconstruction NCC means the endpoint SIREN preserves the patch's intensity structure. A small label can nevertheless have low Dice after a small boundary displacement. Label-size and local-fidelity associations therefore need to be inspected separately. A pooled correlation can also reflect differences between anatomical labels rather than variation within one label.

The first pass joins per-label reference Dice and voxel HD95 with frozen endpoint patch NCC and mean endpoint label volume. Both endpoint NCC values are averaged. Each pair–label observation receives equal weight. Patches contain 9³ samples over 8 mm, with 64 centers per endpoint label. Full-patch NCC includes surrounding tissue; within-region NCC restricts the same patch to that label. Missing labels and undefined local NCC are excluded explicitly. Case and label controls are ordinary least-squares residualization, followed by descriptive rank correlations. The associations are descriptive because anatomical subjects recur across pairs.

## First observations

OASIS has 665 scored pair–label observations. The smallest volume quartile has mean Dice 0.631, compared with 0.883 in the largest quartile. Full-patch endpoint NCC is 0.975 and 0.959 in those quartiles respectively. Pooled full-patch NCC versus Dice has Spearman rho −0.236. After residualizing both quantities against case and label identity, rho is −0.004. Label-size and anatomical identity explain much of the pooled trend.

Mean endpoint label-volume imbalance retains rho −0.366 with Dice after the same case and label controls. This is consistent with small structures being sensitive to geometric mismatch. Labels remain evaluation-only, and the association informs diagnosis.

The lowest mean-Dice OASIS labels include 34, 19, 35 and 18. Their full-patch NCC spans approximately 0.952–0.991. High surrounding-tissue fidelity therefore coexists with substantial registration errors in these structures.

The next diagnostic computes individual DIR-Lab landmark errors, original motion, endpoint patch fidelity and true-correspondence patch correlation on the same frozen checkpoint. This distinguishes reconstruction weakness from large-motion or correspondence failure.

Scalar source evidence is under `results/research_20261115/weak_cases/label_rows.csv` and `label_summary.json`. The analysis script is `.local/weak_cases/analyze_baseline.py`. Persisted evidence consists of scalar diagnostics.

## Anatomical size and initial alignment

Abdomen has 568 pair–label observations with finite endpoint-local NCC in both endpoints. The smallest volume quartile has Dice 0.228 and full-patch NCC 0.798. The largest quartile has Dice 0.613 and full-patch NCC 0.588. The weakest mean-Dice labels are 4, 10, 13, 12 and 11. Pooled full-patch NCC versus Dice has rho −0.168; after case and label controls it is −0.006. Within-region NCC has controlled rho +0.038. Neither is a strong explanation of residual anatomical registration performance.

Initial label alignment gives a stronger description. Across abdomen pair–label observations, identity Dice versus registered Dice has rho +0.869 and initial centroid separation versus registered Dice has rho −0.389. Small-label Dice improves from 0.113 before registration to 0.228 afterward; the largest quartile improves from 0.427 to 0.613. For OASIS, small structures improve substantially despite low final Dice: the smallest size quartile improves from identity Dice 0.332 to registered Dice 0.631. The geometry table includes all 665 OASIS label observations. Within-label NCC is defined for 663 of them, with undefined patch statistics recorded explicitly.

These relationships support a distinction between image fidelity, initial correspondence difficulty and Dice sensitivity to boundary displacement. Centroid separation is only an initial-motion proxy. It describes initial alignment.

## Landmark-wise error and fidelity

Frozen recovery was explicitly approved by the user. All ten DIR-Lab checkpoints were recovered into temporary scratch. All 80 PCG intervals converged, every scratch directory was removed, and the forward/reverse mean continuous TRE reproduced the baseline within 1e-4 mm. Each direction provides 3,000 landmark observations. The local diagnostic uses 8-mm windows centered on the corresponding landmark and samples the recovered map over the full fixed patch when assessing post-registration image NCC.

| Forward-TRE quartile | Mean TRE (mm) | Original motion (mm) | Mean endpoint-fit NCC | Observed warped NCC | Learned warped NCC |
|---|---:|---:|---:|---:|---:|
| Lowest | 0.691 | 5.082 | 0.794 | 0.895 | 0.962 |
| Second | 1.291 | 6.248 | 0.761 | 0.816 | 0.941 |
| Third | 2.100 | 7.959 | 0.707 | 0.679 | 0.907 |
| Highest | 7.402 | 14.556 | 0.597 | 0.272 | 0.686 |

Endpoint-fit NCC versus forward TRE has pooled rho −0.418. After case identity and original motion are controlled, rho is −0.332. The reverse controlled association is −0.338. Unlike the segmentation-label analysis, local endpoint fidelity retains a moderate association with lung-landmark error.

The excess of learned warped-patch NCC over observed warped-patch NCC grows with TRE. The forward pooled rho is +0.644 and remains +0.225 after case, original motion and endpoint-fit NCC are controlled. This is consistent with smooth learned intensities accepting matches that align their coarse image structure while missing original local features. The statistic is descriptive, and the warped agreement also depends on the measured registration error.

## Case 08 localization

Case 08 has mean initial landmark motion 14.995 mm and forward continuous TRE 8.042 mm. Its learned warped-patch NCC is 0.712 while observed warped-patch NCC is 0.397. Dividing its landmarks into equal-count native-Z thirds gives mean TRE 4.877, 8.644 and 10.531 mm, with original motion 8.735, 14.021 and 22.015 mm. The largest-motion native-Z region is particularly weak. These are native array-Z regions, without assigning an anatomical superior/inferior interpretation to the preparation's arbitrary local frame.

The worst individual forward landmark has 29.305 mm original motion and 31.015 mm residual error. The recovered movement projected onto the true displacement is −0.054, so the failure can include an incorrect movement direction rather than a uniformly short trajectory. Its endpoint-fit NCC is 0.517.

## Focused endpoint-objective experiment

A matched experimental comparison was completed on abdomen pairs 19 and 2, and OASIS pair 10. The hypothesis is that global endpoint covariance leaves fine local CT structure underrepresented. Endpoint GW-NCC at 16³ or 8³ moment bins may increase local fit, but its benefit must be measured in registration as well as endpoint NCC. Finer bins also reduce moment occupancy, so a 16³ failure alone would not rule out local endpoint fitting.

Every arm uses width 512, 5,000 updates, seed 42 and the same AdamW/OneCycle settings, registration loss and recovery. Both NCC and GW-NCC arms use ordinary execution and independent time draws. This execution/sampling control differs from the captured/Sobol starting recipe, and comparison is within these matched arms. Presets and scalar results live under ignored `results/research_20261115/weak_cases/endpoint_experiments`; no canonical preset is modified.

Additional scalar evidence is `anatomy_geometry.csv`, `anatomy_geometry_summary.json`, `landmark_rows.csv`, `landmark_summary.json` and `landmark_followup.json`. The local analysis scripts are under `.local/weak_cases`. All persisted investigation outputs are scalar diagnostics.

## Crop exposure and path forcing

A diagnostic Monte Carlo with 100,000 crop origins per case samples the current crop law: equal mask endpoint selection, uniform selected-mask voxel, half-image crop centered then shifted to fit. It estimates expected inclusion, and does not replay the CUDA training draws. The highest-TRE forward quartile has full-crop/central-particle-support probabilities 0.442/0.289; the lowest-TRE quartile has 0.324/0.218. High-error landmarks receive higher average crop inclusion. After case, original motion and endpoint-fit controls, full-crop exposure versus TRE has rho −0.037 and central particle-start support has rho −0.048. The controlled exposure association is small. Results are in `crop_exposure.json`.

The frozen parameter path was also evaluated at 33 times along each observed straight landmark endpoint path. At each sample, temporal derivative q is compared with the brightness-transport target −g·delta, using the actual endpoint displacement in native voxels. The straight connection is a diagnostic path. Intermediate-time landmark ground truth is unavailable.

| Forward-TRE quartile | Energy-weighted q-on-target slope | Normalized transport-residual RMS |
|---|---:|---:|
| Lowest | 0.977 | 0.162 |
| Second | 0.980 | 0.205 |
| Third | 0.956 | 0.239 |
| Highest | 0.666 | 0.586 |

After case, original motion and endpoint-fit controls, per-landmark q-on-target slope versus TRE has rho −0.241 and unnormalized transport-residual RMS versus TRE has rho +0.310. Larger errors therefore coincide with less coherent temporal forcing beyond their endpoint-fidelity association. The energy-weighted summaries avoid allowing weak-gradient paths to dominate ratios. Per-point quarter/three-quarter derivatives and all-path summaries are in `path_forcing.json`.

## Narrower/deeper failure probe

The width agent's 256-neuron, six-layer, 5k-update DIR-Lab case 01 candidate had forward rounded TRE 4.210 mm, compared with 0.987 mm in the starting baseline. A frozen CPU probe finds 8-mm landmark endpoint-fit NCC 0.152/0.141 versus 0.800/0.787 for moving/fixed reference endpoints. Required transport-target RMS is 0.132 versus 0.441, while temporal q RMS is 0.541 versus 0.432. Normalized transport-residual RMS rises from 0.269 to 3.894, and sign coherence falls from 0.781 to 0.475. Large global endpoint NCC can conceal severe loss of local feature fidelity. Together with the width agent's large pre-clipping gradient norms, this is consistent with a poor optimization operating point; a capacity comparison requires a stable optimization operating point. Evidence is `deep256_case01.json`.

## Interpretation under test

The segmentation datasets show small-structure sensitivity and initial-alignment effects even when local endpoint NCC is high. Lung landmarks additionally show an association with local reconstruction fidelity and temporal-path forcing. Training's registration objective samples the observed images, whereas frozen recovery obtains its brightness-flow forcing from the SIRENs. These pathways can disagree locally. Raising local endpoint NCC is therefore worth testing on weak CT cases, but an endpoint-NCC gain needs a corresponding registration gain to support the proposed mechanism.

## Weak abdomen anatomy

Anatomical names below come from the unchanged official dataset metadata. Counts are pairs with finite scores and local diagnostics in both endpoint labels.

| Label | Anatomy | Pairs | Mean volume (voxels) | Dice | HD95 (voxels) | Full-patch NCC | Within-label NCC |
|---|---|---:|---:|---:|---:|---:|---:|
| 4 | gall bladder | 28 | 2309 | 0.072 | 21.37 | 0.722 | 0.586 |
| 10 | portal and splenic vein | 45 | 4161 | 0.157 | 18.64 | 0.805 | 0.727 |
| 13 | right adrenal gland | 45 | 800 | 0.178 | 12.39 | 0.787 | 0.674 |
| 12 | left adrenal gland | 45 | 653 | 0.209 | 10.92 | 0.825 | 0.700 |
| 11 | pancreas | 45 | 11436 | 0.222 | 21.24 | 0.710 | 0.624 |
| 7 | stomach | 45 | 62167 | 0.334 | 28.15 | 0.705 | 0.679 |

The independent GS-transfer probe found appreciable native 8-mm translation information around adrenal patches despite their low Dice. Combined with the small-label and initial-alignment effects, this favors investigation of correspondence capture, weighting and physical support over assuming that all weak small structures lack an image signal. Its global gain/offset probe explained only approximately 0.1–0.2% of temporal-derivative energy in DIR-Lab cases 01/07/08 and slightly worsened the known-path forcing residual after subtracting that component. The result directs attention to spatially varying path forcing. [GS transfer analysis](RESEARCH_GS_TRANSFER_20261115.md).

## Additional controlled checks

The preferred all-observation anatomical table is `label_summary_all.json`. It retains all 665 OASIS score observations and uses 663 only for within-label NCC, where two tiny-label patches have undefined local variance/support. Full-patch NCC versus Dice after case and label controls is −0.004 on all 665 observations. The smallest/largest OASIS size quartiles have Dice 0.631/0.883 and full-patch NCC 0.975/0.959. The primary `label_rows_all.csv` and `label_rows_all.json` retain all 1,233 scored anatomical observations with feature-specific validity.

Joining the independent capture landscapes to measured errors qualifies the blur idea. On 294 case-08 landmarks valid under every tested blur, observed 8-mm patch NCC improves in the first 12.5% step toward truth for 59.9% of points without blur and 67.3% with 2-mm blur. In the highest-TRE quartile's 73 matched-valid points, that fraction instead decreases from 63.0% to 52.1%; 4-mm blur gives 38.4%. Its near-truth peak fraction decreases from 98.6% to 95.9% and then 80.8%. The actual worst landmarks require their own capture comparison. `matched_landscape_groups.json` records the identical-point comparison.

A parameter-group decomposition uses the existing exact `parameter_contributions` helper and verifies that its contributions sum to q. In case 08, low/high-TRE quartiles have weighted transport-target slopes 0.987/0.161. The first hidden weight matrix contributes 0.704/0.143 to those slopes. First-layer weight/bias contributions are smaller, and the output bias contribution is approximately zero. Case 07's low/high-TRE slopes are 0.932/0.537. The decomposition directs attention to spatially varying transport contributions. A phase-adaptation learning-rate probe would test a specific optimization hypothesis, and would require matched controls before drawing a methodology conclusion. `path_components.json` stores the decomposition.

## Completed endpoint-objective screen

All nine matched ordinary-execution arms completed. Each uses one seed and 5,000 fresh updates. The local endpoint objectives substantially improve the chosen anatomical patch diagnostic on both abdomen cases, while their registration Dice decreases. This is direct evidence that increasing endpoint local NCC alone is insufficient for these weak CT registrations. OASIS pair 10 gives a small positive contrast. These single-case comparisons are development evidence.

| Dataset/pair | Endpoint loss | Dice | HD95 (voxels) | Full-patch NCC | Within-label NCC | Training (s) |
|---|---|---:|---:|---:|---:|---:|
| abdomenctct 2 | ncc | 0.3094 | 22.621 | 0.721 | 0.637 | 125.2 |
| abdomenctct 2 | gw8 | 0.3030 | 22.432 | 0.808 | 0.748 | 128.9 |
| abdomenctct 2 | gw16 | 0.2916 | 22.557 | 0.820 | 0.762 | 128.6 |
| abdomenctct 19 | ncc | 0.2249 | 19.609 | 0.714 | 0.625 | 151.1 |
| abdomenctct 19 | gw8 | 0.2144 | 19.495 | 0.804 | 0.741 | 155.1 |
| abdomenctct 19 | gw16 | 0.2018 | 20.306 | 0.813 | 0.751 | 142.6 |
| oasis 10 | ncc | 0.7823 | 2.551 | 0.967 | 0.900 | 128.5 |
| oasis 10 | gw8 | 0.7849 | 2.553 | 0.971 | 0.908 | 133.5 |
| oasis 10 | gw16 | 0.7908 | 2.494 | 0.978 | 0.928 | 134.7 |

The patch columns average defined endpoint-label diagnostics; an absent endpoint label contributes no patch statistic. Late-training global endpoint NCC also improves in the abdomen GW arms, while observed-image registration loss becomes larger. On weak pair 19, mean last-500-update registration loss is 0.376 with endpoint NCC and 0.420 with GW16; pre-clipping gradient norm is 1.29 and 2.56 respectively. The local objective changes optimization balance and the resulting parameter path, so endpoint fidelity and registration must be jointly assessed. Training summaries and exact min–max PSNR are in `endpoint_experiments/training_fit_summary.json` and `diagnostic_summary.json`.

## Weak-pair recovery geometry

For the same frozen starting-version abdomen-19 model, recovery smoothness 0.003 raises Dice from 0.2446 to 0.2837. Folding fraction rises from 0.00892% to 0.07284%. Smoothness 0.03 lowers Dice to 0.2118 and folding to 0.000568%. This fixed-checkpoint comparison changes only the PCG smoothness coefficient, preserving all eight temporal evaluations, RK2 settings and convergence tolerances.

The initial right/left kidney centroid distances are 54.2/48.4 mm. Default recovery leaves 45.6/41.9 mm centroid error. Right-kidney GW16 recovery has a nearly matched warped/fixed volume ratio of 1.05 but a 57.2-mm centroid error and Dice 0.0005. This is a positional failure despite comparable volume. Lower default-model smoothness raises right-kidney Dice slightly while increasing its warped/fixed volume ratio from 2.02 to 2.88 and its centroid error from 45.6 to 49.7 mm. Its Dice gain combines overlap and volume-distortion effects. The inferior vena cava gives a clearer alignment improvement, with centroid error falling from 14.3 to 7.5 mm.

The OASIS-10 contrast at smoothness 0.01/0.003/0.03 has Dice 0.78347/0.78164/0.78276 and folding fractions 0.00463%/0.02887%/0.000232%. A lower PCG smoothness is therefore a dataset-dependent accuracy/regularity tradeoff. All nine focused recovery probes converged and all owned temporary scratch was removed. `recovery_geometry/` stores scalar metrics and label geometry.

A full 74-checkpoint frozen comparison at smoothness 0.003 completed on GPU 1 to measure the cohort-wide tradeoff. It writes scalar scores, exact checkpoint/source/config provenance and per-landmark scalar errors; returned checkpoint payloads are discarded. This queue evaluates frozen models. Anatomical size and landmark motion groups are fixed from the original baseline.

## Full-cohort frozen recovery comparison

All 74 original checkpoints completed the 0.003 smoothness comparison, all 592 PCG intervals converged, and all 74 per-case provenance records verified temporary-scratch removal. Each evaluated configuration equals its recorded checkpoint configuration with only `recovery.smoothness` changed from 0.01 to 0.003. Checkpoint hashes and numerical-source hashes are recorded per case.

| Dataset | Main score, 0.01 → 0.003 | Cases improving | Mean folding percentage, 0.01 → 0.003 |
|---|---|---:|---:|
| OASIS, 19 pairs | Dice 0.81337 → 0.81109 | 3/19 | 0.00338% → 0.01971% |
| Abdomen, 45 pairs | Dice 0.43897 → 0.45371 | 41/45 | 0.13131% → 0.30094% |
| DIR-Lab, 10 cases | Forward rounded TRE 2.8471 → 2.8689 mm | 5/10 | Lung 0% → 0.000207% |

DIR-Lab reverse rounded TRE changes from 2.9639 to 2.9755 mm. Abdomen HD95 changes from 16.7678 to 16.8023 voxels. Its maximum case folding fraction at 0.003 is 1.3633%. The smallest through largest abdomen label-volume quartiles gain Dice 0.0120, 0.0165, 0.0144 and 0.0162. The overlap gain spans organ sizes. The focused weak-pair volume audit shows why the cohort overlap gain must be considered alongside geometric distortion.

For DIR-Lab, the highest original-motion quartile has mean forward continuous TRE 6.194 → 6.236 mm; the lowest-motion quartile has 1.134 → 1.221 mm. The highest baseline-TRE quartile changes from 7.402 to 7.408 mm. Reduced PCG smoothing therefore gives a weak-case-specific abdomen overlap gain while leaving the principal lung large-motion errors unresolved.

Mean adapter recovery times were 8.10 → 7.68 seconds for OASIS, 9.14 → 8.86 seconds for abdomen and 8.87 → 8.42 seconds for DIR-Lab. These are frozen-recovery stage measurements with different worker/cache histories from the original training run, so they describe the measured evaluation cost without establishing a general speedup.

The exact paired evidence is `results/research_20261115/recovery_s003_full/comparison.json`, with `label_comparison.csv` and `landmark_comparison.csv`. Source for the scalar-only queue and aggregation is `.local/weak_cases/frozen_recovery_queue.py` and `compare_full_recovery.py`.

## Local-fidelity and recovery interaction

The same ordinary-trained weak-pair checkpoints were recovered at both smoothness values. This matches the endpoint-objective experiment's execution and independent-time sampling controls.

| Endpoint objective | Dice at smoothness 0.01 | Dice at 0.003 | Mean full-patch NCC |
|---|---:|---:|---:|
| NCC | 0.22493 | 0.24820 | 0.714 |
| GW8 | 0.21443 | 0.23697 | 0.804 |
| GW16 | 0.20179 | 0.22495 | 0.813 |

Lower smoothness adds approximately 0.023 Dice in every arm. The relative endpoint-objective penalty remains. This directs the next test toward training-task balance and temporal transport, rather than assuming that PCG smoothing alone prevents the higher local fidelity from improving registration. A focused follow-up increased observed-image registration weight from 0.01 to 0.03 in matched NCC and GW16 fits, followed by the complete captured abdomen cohort.

## Observed-image registration balance

Four additional matched ordinary fits increased registration weight from 0.01 to 0.03, retaining endpoint NCC or GW16, seed 42, 5,000 updates and the same independent-time control. Global-NCC endpoint fits improve Dice on both chosen abdomen pairs despite slightly lower anatomical endpoint-fit NCC. The result motivates the separate captured quick4 transfer check before any cohort-wide recommendation.

| Pair | Endpoint | Registration weight | Dice | Anatomical fit NCC | Late registration loss | Late pre-clip norm | Logged clipped fraction |
|---|---|---:|---:|---:|---:|---:|---:|
| 19 | ncc | 0.01 | 0.2249 | 0.714 | 0.376 | 1.29 | 65.4% |
| 19 | ncc | 0.03 | 0.2951 | 0.695 | 0.337 | 1.36 | 76.9% |
| 19 | gw16 | 0.01 | 0.2018 | 0.813 | 0.420 | 2.56 | 100.0% |
| 19 | gw16 | 0.03 | 0.2188 | 0.812 | 0.376 | 2.61 | 100.0% |
| 2 | ncc | 0.01 | 0.3094 | 0.721 | 0.449 | 1.22 | 61.5% |
| 2 | ncc | 0.03 | 0.3177 | 0.700 | 0.399 | 1.34 | 65.4% |
| 2 | gw16 | 0.01 | 0.2916 | 0.820 | 0.497 | 2.46 | 100.0% |
| 2 | gw16 | 0.03 | 0.3095 | 0.814 | 0.448 | 2.54 | 100.0% |

Late statistics average 26 logged updates from step 4500 through 5000. The clipping fraction describes those logged samples. Global-NCC endpoint local fidelity falls by approximately 0.02 under stronger observed-image registration pressure while Dice rises. GW16 retains a roughly 0.10 local-fidelity advantage and incurs a larger gradient norm, with all late logged gradients clipped. The observed-image registration loss responds to the higher coefficient, and the GW16 endpoint variant continues to trail its matched global-NCC counterpart in Dice. The comparison distinguishes the value of a diagnostic from the effect of directly optimizing it.

The exact table, including HD95, folding and complete training time, is `results/research_20261115/registration_balance/comparison.json`. The scalar endpoint diagnostics use the same 8-mm protocol as the first screen.

## Paired width-384 landmark validation

All ten completed width-384/6k DIR-Lab checkpoints were recovered with their original 0.01 smoothness and compared on the same 6,000 landmark IDs as the reference. All 80 intervals converged and all temporary scratch was removed. Per-case checkpoint SHA-256, complete configuration and numerical-source hashes are included in `width384_landmarks/landmarks_NN.json`.

Mean continuous forward/reverse TRE changes from 2.8710/2.9759 mm at width 512 to 2.9405/3.0509 mm for the width-384 recipe. Mean landmark endpoint-fit NCC changes from 0.71466 to 0.71599. The table below uses original-motion quartiles fixed from the reference landmarks.

| Original motion quartile | Mean original motion (mm) | Forward TRE, 512 → 384 (mm) | Endpoint-fit NCC, 512 → 384 |
|---|---:|---:|---:|
| Lowest | 1.994 | 1.134 → 1.146 | 0.731 → 0.726 |
| Second | 4.742 | 1.766 → 1.820 | 0.695 → 0.697 |
| Third | 9.174 | 2.370 → 2.455 | 0.735 → 0.739 |
| Highest | 17.872 | 6.194 → 6.320 | 0.698 → 0.702 |

The large-motion quartile shows a larger absolute TRE change while local endpoint fidelity is retained. Groups chosen by the original model's residual TRE also include fit-to-fit regression toward the mean, so original-motion groups give the clearer capacity comparison. Both recipes use seed 42; this remains a single-seed recipe comparison with changed width, update budget and learning-rate operating point.

Across the known straight landmark paths, mean native spatial-gradient energy rises from 0.05493 to 0.06888 and temporal-derivative energy rises from 0.5022 to 0.6154. Energy-weighted q-on-transport-target slope is 0.9342/0.9294 and normalized transport-residual RMS is 0.2797/0.2754. The width-384 result retains coherent forcing and local features, in contrast with the unstable deeper-256 screen. The paired evidence is `width384_landmarks/comparison.json` and `landmark_comparison.csv`.

## Static amplitude and fixed recovery regularity

The global temporal-gauge probe measures the component of q explained by offset and gain directions within a model. Static intensity amplitude across experimental recipes scales both g and q. With a constant transformation `I' = a I + b`, spatial and temporal derivatives become `g' = a g` and `q' = a q`. Dividing the brightness-flow objective by `a²` gives effective smoothness `mu/a²` and ridge `ridge/a²`. Static amplitude can therefore change the relative prior strength while the temporal nuisance projection remains small.

A paired diagnostic queried 4,096 identical uniformly sampled normalized coordinates at all eight original recovery midpoint times. It used the frozen quick4 cases from every dataset, plus previously selected weak DIR-Lab cases 07/08 and abdomen pair 19, totaling 15 paired cases. Values below are median width-384/width-512 ratios across those case groups. Spatial derivatives retain native voxel units.

| Dataset probe | Cases | Spatial-gradient energy ratio | Temporal-derivative energy ratio | Image-variance ratio | Variance-normalized gradient-energy ratio |
|---|---:|---:|---:|---:|---:|
| OASIS | 4 | 0.918 | 0.915 | 0.930 | 0.983 |
| Abdomen | 5 | 0.844 | 0.895 | 0.872 | 0.994 |
| DIR-Lab | 6 | 1.058 | 1.046 | 1.005 | 1.068 |

The MR/CT global gradient-energy changes largely track image variance. DIR-Lab retains a small global increase and a larger increase at landmark paths. This qualifies an interpretation of narrower models as automatically smoother or gradient-poor. The fixed 0.01/0.003 capacity-by-recovery comparison is the empirical interaction test. These measurements support later reporting and calibration of effective regularity per experimental preset; the current experiment retains fixed coefficients. The exact samples, times and checkpoint hashes are stored in `global_energy.json`.

## Absolute and size-relative anatomical errors

The complete anatomical rows are retained as `label_rows_all.csv` and `label_rows_all.json`, including both OASIS observations with undefined within-label NCC. All 665 OASIS and 568 abdomen observations have defined voxel HD95. Volume quartiles use every scored observation.

| Dataset | Size quartile | Dice | HD95 (voxels) | Mean HD95 / equivalent radius |
|---|---|---:|---:|---:|
| OASIS | Smallest | 0.631 | 3.038 | 0.697 |
| OASIS | Largest | 0.883 | 1.448 | 0.059 |
| Abdomen | Smallest | 0.228 | 11.816 | 1.939 |
| Abdomen | Largest | 0.613 | 20.316 | 0.813 |

The equivalent-sphere radius is `(3V/(4pi))^(1/3)` using the mean endpoint label volume in voxels. The last column averages each observation's HD95/radius ratio. It is a characteristic-size diagnostic for irregular structures. Abdomen's small structures have smaller absolute boundary error and larger relative error; OASIS's small structures have larger absolute and relative error. These measurements qualify how much low small-label Dice can reflect geometric scale sensitivity. The exact quartiles and validity counts are in `size_normalized_hd95.json`.

## Full-45 captured registration-weight transfer

The captured width-512 recipe with registration weight 0.03 completed the frozen quick4 selection and the remaining 41 official abdomen pairs. The full 45-pair mean Dice changes from 0.43897 to 0.44255, with 27 pairs improving. HD95 changes from 16.7678 to 16.7908 voxels, and mean folding changes from 0.13131% to 0.12493%. The overall gain is smaller than the development-pair and quick4 gains.

The complete 568 paired label observations reveal where the change acts.

| Fixed reference label-size quartile | Dice at weight 0.01 | Dice at weight 0.03 | Change |
|---|---:|---:|---:|
| Smallest | 0.22789 | 0.22321 | −0.00468 |
| Second | 0.33576 | 0.34271 | +0.00695 |
| Third | 0.57856 | 0.58963 | +0.01107 |
| Largest | 0.61280 | 0.61352 | +0.00071 |

Right and left kidneys gain mean Dice 0.02087 and 0.01786. Right and left adrenal glands change by −0.00992 and −0.00250, while stomach changes by −0.00879. Initial-centroid-separation quartiles have Dice gains −0.00947, +0.00168, +0.01011 and +0.01175. Stronger observed-image registration pressure predominantly helps larger initial mismatches and medium-size organs, while the weak small-adrenal problem remains.

The scalar evidence is `results/research_20261115/registration_weight003/label_size_comparison.json` and `label_comparison.csv`. Every case matches the reference input paths and complete captured configuration with only the registration-weight change. Labels remain evaluation-only. The comparison supports a modest, qualified abdomen preset proposal for review, alongside its per-organ regressions.

## Completed capacity-by-smoothness comparison

The final fixed-smoothness interaction completed all 74 width-384 checkpoints. Every one of its 592 PCG intervals converged, and every per-case provenance record verified scratch cleanup. The two trained recipe families were evaluated at the same fixed coefficients, original recovery geometries, eight temporal evaluations, RK2 settings and true-residual tolerances.

| Recipe | Fixed smoothness | OASIS Dice | Abdomen Dice | DIR-Lab forward/reverse rounded TRE (mm) | OASIS mean folding | Abdomen mean folding |
|---|---:|---:|---:|---:|---:|---:|
| 512 neurons | 0.010 | 0.81337 | 0.43897 | 2.8471 / 2.9639 | 0.00338% | 0.13131% |
| 512 neurons | 0.003 | 0.81109 | 0.45371 | 2.8689 / 2.9755 | 0.01971% | 0.30094% |
| 384 neurons | 0.010 | 0.81205 | 0.43449 | 2.9281 / 3.0373 | 0.00274% | 0.09097% |
| 384 neurons | 0.003 | 0.81046 | 0.45265 | 2.9616 / 3.0615 | 0.01739% | 0.23477% |

The abdomen Dice response to lower smoothness is +0.01473 for width 512 and +0.01816 for width 384. Its capacity gap narrows at the lower coefficient, with higher folding and slightly worse HD95 in both models. Brain and lung scores show the same slight degradation under lower smoothness in both recipes. The interaction supports joint reporting of capacity, fixed regularity and geometric quality. It leaves the choice of an operating point for review.

Exact paired records are in `results/research_20261115/recovery_w384_s003_full/comparison.json`, `label_comparison.csv` and `landmark_comparison.csv`. All 6,000 width-384 landmark IDs match the starting reference; the fixed recovery comparison uses the width-384 original-map errors as its paired reference.

## Completion and scope

All investigation GPU processes are terminal, and the final checked process snapshot has zero compute PIDs. All owned frozen-recovery temporary directories were removed, with cleanup assertions recorded per case. Persisted research outputs comprise scalar diagnostics, provenance and the fresh-fit endpoint checkpoints in ignored result directories. The methodology proposals await review.

The evidence uses one training seed, official public-validation pairs and development probes chosen from baseline weaknesses. Anatomical subjects recur across directed pairs, so the correlation analysis is descriptive. Local NCC has feature-specific validity counts, with every scored anatomical observation retained in the primary tables. Straight landmark paths provide a model-based transport diagnostic; intermediate-time trajectories and appearance constancy are assumptions. Larger registration weight gives a modest full-abdomen mean gain with small-adrenal regressions, while lower PCG smoothness gives an overlap/regularity tradeoff. Neither result supports treating endpoint patch NCC as a sufficient target for registration quality.
