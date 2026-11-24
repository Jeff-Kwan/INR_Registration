# Weak-case research, 15 November 2026

The investigation is complete. The reference is the tagged starting version and the completed 74-pair run in `results/full_baseline_20261115_run`. Case selection based on the reference scores is development analysis.

## Mechanism and diagnostic design

A high local reconstruction NCC means the endpoint SIREN preserves the patch's intensity structure. A small label can nevertheless have low Dice after a small boundary displacement. Label-size and local-fidelity associations therefore need to be inspected separately. A pooled correlation can also reflect differences between anatomical labels rather than variation within one label.

The first pass joins per-label reference Dice and voxel HD95 with frozen endpoint patch NCC and mean endpoint label volume. Both endpoint NCC values are averaged. Each pair–label observation receives equal weight. Patches contain 9³ samples over 8 mm, with 64 centers per endpoint label. Full-patch NCC includes surrounding tissue; within-region NCC restricts the same patch to that label. Missing labels and undefined local NCC are excluded explicitly. Case and label controls are ordinary least-squares residualization, followed by descriptive rank correlations. The associations are descriptive because anatomical subjects recur across pairs.

## First observations

OASIS has 665 scored pair–label observations. The smallest volume quartile has mean Dice 0.631, compared with 0.883 in the largest quartile. Full-patch endpoint NCC is 0.975 and 0.959 in those quartiles respectively. Pooled full-patch NCC versus Dice has Spearman rho −0.236. After residualizing both quantities against case and label identity, rho is −0.004. Label-size and anatomical identity explain much of the pooled trend.

Mean endpoint label-volume imbalance retains rho −0.366 with Dice after the same case and label controls. This is consistent with small structures being sensitive to geometric mismatch. Labels remain evaluation-only, and the association informs diagnosis.

The lowest mean-Dice OASIS labels include 34, 19, 35 and 18. Their full-patch NCC spans approximately 0.952–0.991. High surrounding-tissue fidelity therefore coexists with substantial registration errors in these structures.

This distinguishes reconstruction weakness from large-motion or correspondence failure.

Scalar source evidence is under `results/research_20261115/weak_cases/label_rows.csv` and `label_summary.json`. The analysis script is `.local/weak_cases/analyze_baseline.py`. Persisted evidence consists of scalar diagnostics.

## Anatomical size and initial alignment

The smallest volume quartile has Dice 0.228 and full-patch NCC 0.798. The largest quartile has Dice 0.613 and full-patch NCC 0.588. The weakest mean-Dice labels are 4, 10, 13, 12 and 11. Pooled full-patch NCC versus Dice has rho −0.168; after case and label controls it is −0.006. Within-region NCC has controlled rho +0.038. Neither is a strong explanation of residual anatomical registration performance.

Initial label alignment gives a stronger description. Small-label Dice improves from 0.113 before registration to 0.228 afterward; the largest quartile improves from 0.427 to 0.613. For OASIS, small structures improve substantially despite low final Dice: the smallest size quartile improves from identity Dice 0.332 to registered Dice 0.631. The geometry table includes all 665 OASIS label observations. Within-label NCC is defined for 663 of them, with undefined patch statistics recorded explicitly.

These relationships support a distinction between image fidelity, initial correspondence difficulty and Dice sensitivity to boundary displacement. Centroid separation is only an initial-motion proxy. It describes initial alignment.

## Focused endpoint-objective experiment

The hypothesis is that global endpoint covariance leaves fine local CT structure underrepresented. Endpoint GW-NCC at 16³ or 8³ moment bins may increase local fit, but its benefit must be measured in registration as well as endpoint NCC. Finer bins also reduce moment occupancy, so a 16³ failure alone would not rule out local endpoint fitting.

Every arm uses width 512, 5,000 updates, seed 42 and the same AdamW/OneCycle settings, registration loss and recovery. Both NCC and GW-NCC arms use ordinary execution and independent time draws. This execution/sampling control differs from the captured/Sobol starting recipe, and comparison is within these matched arms. Presets and scalar results live under ignored `results/research_20261115/weak_cases/endpoint_experiments`; no canonical preset is modified.

The local analysis scripts are under `.local/weak_cases`. All persisted investigation outputs are scalar diagnostics.

## Narrower/deeper failure probe


## Interpretation under test

The segmentation datasets show small-structure sensitivity and initial-alignment effects even when local endpoint NCC is high. Training's registration objective samples the observed images, whereas frozen recovery obtains its brightness-flow forcing from the SIRENs. These pathways can disagree locally. Raising local endpoint NCC is therefore worth testing on weak CT cases, but an endpoint-NCC gain needs a corresponding registration gain to support the proposed mechanism.

## Additional controlled checks

The preferred all-observation anatomical table is `label_summary_all.json`. It retains all 665 OASIS score observations and uses 663 only for within-label NCC, where two tiny-label patches have undefined local variance/support. Full-patch NCC versus Dice after case and label controls is −0.004 on all 665 observations. The smallest/largest OASIS size quartiles have Dice 0.631/0.883 and full-patch NCC 0.975/0.959. The primary `label_rows_all.csv` and `label_rows_all.json` retain all 1,233 scored anatomical observations with feature-specific validity.

Joining the independent capture landscapes to measured errors qualifies the blur idea. Its near-truth peak fraction decreases from 98.6% to 95.9% and then 80.8%.

A parameter-group decomposition uses the existing exact `parameter_contributions` helper and verifies that its contributions sum to q. The first hidden weight matrix contributes 0.704/0.143 to those slopes. First-layer weight/bias contributions are smaller, and the output bias contribution is approximately zero. The decomposition directs attention to spatially varying transport contributions. A phase-adaptation learning-rate probe would test a specific optimization hypothesis, and would require matched controls before drawing a methodology conclusion. `path_components.json` stores the decomposition.

## Completed endpoint-objective screen

All nine matched ordinary-execution arms completed. Each uses one seed and 5,000 fresh updates. This is direct evidence that increasing endpoint local NCC alone is insufficient for these weak CT registrations. OASIS pair 10 gives a small positive contrast. These single-case comparisons are development evidence.

| Dataset/pair | Endpoint loss | Dice | HD95 (voxels) | Full-patch NCC | Within-label NCC | Training (s) |
|---|---|---:|---:|---:|---:|---:|
| oasis 10 | ncc | 0.7823 | 2.551 | 0.967 | 0.900 | 128.5 |
| oasis 10 | gw8 | 0.7849 | 2.553 | 0.971 | 0.908 | 133.5 |
| oasis 10 | gw16 | 0.7908 | 2.494 | 0.978 | 0.928 | 134.7 |

The patch columns average defined endpoint-label diagnostics; an absent endpoint label contributes no patch statistic. On weak pair 19, mean last-500-update registration loss is 0.376 with endpoint NCC and 0.420 with GW16; pre-clipping gradient norm is 1.29 and 2.56 respectively. The local objective changes optimization balance and the resulting parameter path, so endpoint fidelity and registration must be jointly assessed. Training summaries and exact min–max PSNR are in `endpoint_experiments/training_fit_summary.json` and `diagnostic_summary.json`.

## Weak-pair recovery geometry

Folding fraction rises from 0.00892% to 0.07284%. Smoothness 0.03 lowers Dice to 0.2118 and folding to 0.000568%. This fixed-checkpoint comparison changes only the PCG smoothness coefficient, preserving all eight temporal evaluations, RK2 settings and convergence tolerances.

The initial right/left kidney centroid distances are 54.2/48.4 mm. Default recovery leaves 45.6/41.9 mm centroid error. Right-kidney GW16 recovery has a nearly matched warped/fixed volume ratio of 1.05 but a 57.2-mm centroid error and Dice 0.0005. This is a positional failure despite comparable volume. Lower default-model smoothness raises right-kidney Dice slightly while increasing its warped/fixed volume ratio from 2.02 to 2.88 and its centroid error from 45.6 to 49.7 mm. Its Dice gain combines overlap and volume-distortion effects. The inferior vena cava gives a clearer alignment improvement, with centroid error falling from 14.3 to 7.5 mm.

The OASIS-10 contrast at smoothness 0.01/0.003/0.03 has Dice 0.78347/0.78164/0.78276 and folding fractions 0.00463%/0.02887%/0.000232%. A lower PCG smoothness is therefore a dataset-dependent accuracy/regularity tradeoff. All nine focused recovery probes converged and all owned temporary scratch was removed. `recovery_geometry/` stores scalar metrics and label geometry.

A full 74-checkpoint frozen comparison at smoothness 0.003 completed on GPU 1 to measure the cohort-wide tradeoff. This queue evaluates frozen models.

## Full-cohort frozen recovery comparison

All 74 original checkpoints completed the 0.003 smoothness comparison, all 592 PCG intervals converged, and all 74 per-case provenance records verified temporary-scratch removal. Each evaluated configuration equals its recorded checkpoint configuration with only `recovery.smoothness` changed from 0.01 to 0.003. Checkpoint hashes and numerical-source hashes are recorded per case.

| Dataset | Main score, 0.01 → 0.003 | Cases improving | Mean folding percentage, 0.01 → 0.003 |
|---|---|---:|---:|
| OASIS, 19 pairs | Dice 0.81337 → 0.81109 | 3/19 | 0.00338% → 0.01971% |

Its maximum case folding fraction at 0.003 is 1.3633%. The overlap gain spans organ sizes. The focused weak-pair volume audit shows why the cohort overlap gain must be considered alongside geometric distortion.

These are frozen-recovery stage measurements with different worker/cache histories from the original training run, so they describe the measured evaluation cost without establishing a general speedup.

Source for the scalar-only queue and aggregation is `.local/weak_cases/frozen_recovery_queue.py` and `compare_full_recovery.py`.

## Local-fidelity and recovery interaction

The same ordinary-trained weak-pair checkpoints were recovered at both smoothness values. This matches the endpoint-objective experiment's execution and independent-time sampling controls.

| Endpoint objective | Dice at smoothness 0.01 | Dice at 0.003 | Mean full-patch NCC |
|---|---:|---:|---:|
| NCC | 0.22493 | 0.24820 | 0.714 |
| GW8 | 0.21443 | 0.23697 | 0.804 |
| GW16 | 0.20179 | 0.22495 | 0.813 |

Lower smoothness adds approximately 0.023 Dice in every arm. The relative endpoint-objective penalty remains. This directs the next test toward training-task balance and temporal transport, rather than assuming that PCG smoothing alone prevents the higher local fidelity from improving registration.

## Observed-image registration balance

Four additional matched ordinary fits increased registration weight from 0.01 to 0.03, retaining endpoint NCC or GW16, seed 42, 5,000 updates and the same independent-time control. The result motivates the separate captured quick4 transfer check before any cohort-wide recommendation.

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

## Static amplitude and fixed recovery regularity

The global temporal-gauge probe measures the component of q explained by offset and gain directions within a model. Static intensity amplitude across experimental recipes scales both g and q. With a constant transformation `I' = a I + b`, spatial and temporal derivatives become `g' = a g` and `q' = a q`. Dividing the brightness-flow objective by `a²` gives effective smoothness `mu/a²` and ridge `ridge/a²`. Static amplitude can therefore change the relative prior strength while the temporal nuisance projection remains small.

A paired diagnostic queried 4,096 identical uniformly sampled normalized coordinates at all eight original recovery midpoint times. Values below are median width-384/width-512 ratios across those case groups. Spatial derivatives retain native voxel units.

| Dataset probe | Cases | Spatial-gradient energy ratio | Temporal-derivative energy ratio | Image-variance ratio | Variance-normalized gradient-energy ratio |
|---|---:|---:|---:|---:|---:|
| OASIS | 4 | 0.918 | 0.915 | 0.930 | 0.983 |

The MR/CT global gradient-energy changes largely track image variance. This qualifies an interpretation of narrower models as automatically smoother or gradient-poor. The fixed 0.01/0.003 capacity-by-recovery comparison is the empirical interaction test. These measurements support later reporting and calibration of effective regularity per experimental preset; the current experiment retains fixed coefficients. The exact samples, times and checkpoint hashes are stored in `global_energy.json`.

## Absolute and size-relative anatomical errors

The complete anatomical rows are retained as `label_rows_all.csv` and `label_rows_all.json`, including both OASIS observations with undefined within-label NCC. Volume quartiles use every scored observation.

| Dataset | Size quartile | Dice | HD95 (voxels) | Mean HD95 / equivalent radius |
|---|---|---:|---:|---:|
| OASIS | Smallest | 0.631 | 3.038 | 0.697 |
| OASIS | Largest | 0.883 | 1.448 | 0.059 |

The equivalent-sphere radius is `(3V/(4pi))^(1/3)` using the mean endpoint label volume in voxels. The last column averages each observation's HD95/radius ratio. It is a characteristic-size diagnostic for irregular structures. These measurements qualify how much low small-label Dice can reflect geometric scale sensitivity. The exact quartiles and validity counts are in `size_normalized_hd95.json`.

## Completed capacity-by-smoothness comparison

The final fixed-smoothness interaction completed all 74 width-384 checkpoints. Every one of its 592 PCG intervals converged, and every per-case provenance record verified scratch cleanup. The two trained recipe families were evaluated at the same fixed coefficients, original recovery geometries, eight temporal evaluations, RK2 settings and true-residual tolerances.

| Recipe | Fixed smoothness | OASIS Dice | OASIS mean folding |
|---|---:|---:|---:|
| 512 neurons | 0.010 | 0.81337 | 0.00338% |
| 512 neurons | 0.003 | 0.81109 | 0.01971% |
| 384 neurons | 0.010 | 0.81205 | 0.00274% |
| 384 neurons | 0.003 | 0.81046 | 0.01739% |

On OASIS, the capacity gap narrows at lower smoothness, with higher folding and slightly worse HD95 in both models.

## Completion and scope

All investigation GPU processes are terminal, and the final checked process snapshot has zero compute PIDs. All owned frozen-recovery temporary directories were removed, with cleanup assertions recorded per case. Persisted research outputs comprise scalar diagnostics, provenance and the fresh-fit endpoint checkpoints in ignored result directories. The methodology proposals await review.

The evidence uses one training seed, official public-validation pairs and development probes chosen from baseline weaknesses. Anatomical subjects recur across directed pairs, so the correlation analysis is descriptive. Local NCC has feature-specific validity counts, with every scored anatomical observation retained in the primary tables. Neither result supports treating endpoint patch NCC as a sufficient target for registration quality.
