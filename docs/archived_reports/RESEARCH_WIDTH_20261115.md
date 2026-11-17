# Smaller SIREN research, 15 November 2026

Workstream complete. All experimental fits and frozen probes are terminal. Canonical tracked code and presets match `starting-version`.

This is an experimental research record. The canonical presets remain the tagged starting version `993567e`. All fits use seed 42 and frozen recovery settings.

## Full-cohort result

The width-384/four-layer, 6k, peak-1.5e-4 recipe completes all 74 official pairs with 890,882 parameters per pair versus 1,581,058 for width 512. All checkpoints are finite, all 592 recovery intervals converge and the original execution source matches `starting-version` byte for byte. It reduces runtime while slightly reducing mean registration accuracy. This is a candidate tradeoff for review, not a canonical selection or an accuracy-equivalence result.

| Dataset | Width-512 main / second score | Width-384 main / second score | Width-512 mean s | Width-384 mean s |
|---|---:|---:|---:|---:|
| OASIS, 19 | Dice 0.81337 / HD95 1.79587 | Dice 0.81205 / HD95 1.81373 | 79.86 | 73.09 |
| Abdomen, 45 | Dice 0.43897 / HD95 16.76780 | Dice 0.43449 / HD95 16.72349 | 79.76 | 72.33 |
| DIR-Lab, 10 | Forward / reverse TRE 2.84712 / 2.96386 | Forward / reverse TRE 2.92812 / 3.03728 | 110.30 | 91.44 |

HD95 is in voxels and TRE is in millimetres. Mean folding is lower with width 384 on OASIS (2.74e-5 versus 3.38e-5) and abdomen (0.000910 versus 0.001313); DIR-Lab lung folding is zero for both. The balanced same-GPU full-fit check confirms about 10.1% lower warm end-to-end time on OASIS pair 1 and abdomen pair 2, with 7.9–8.1% lower warm training time. Full-cohort timings retain two-GPU versus historical four-GPU queue/cache differences and should be read alongside that controlled verification.

The complete candidate denominator reuses 12 exact-config, exact-implementation quick4 fits and trains the other 62. Its incremental two-GPU wall time was 2,321.5 seconds (38 minutes 41 seconds). Scalar evidence and case/source manifests are in `results/research_20261115/width/full_aggregate.json` and `full_audit.json`. The 256-width recipes tested did not meet the joint accuracy/time target. The six-layer failure was substantially repaired by lowering the rate, while its OASIS accuracy still trailed the reference. The 11.4% width-384 numerical execution saving is a separate validated prototype; the mixed 7k pilot was closed after a warm repeat exceeded the target time.

## Question and mechanism

A smaller network reduces dense FP32 matrix multiplication cost, but it may need a different learning rate or more updates to represent small anatomical structures and the spatial derivatives used by registration. Extra depth can recover capacity while increasing repeated derivative cost and optimization difficulty. We compare complete adapter time and registration accuracy together. Endpoint fidelity alone cannot select a model because its derivative geometry also determines the recovered motion.

The starting width-512 model has four sine layers, omega 30, output initialization scale 30, AdamW with decay 0.001, a OneCycle peak of 1e-4 and 5,000 updates. Native captured training uses endpoint NCC with moments 0.01, PD-LK 0.1, GW registration NCC 0.01, two orthogonal query pairs and scrambled Sobol times. Recovery uses the unchanged eight-time, cap-256 PCG and RK2 recipe.

## Initial screen

At 12:10 Hong Kong time, GPUs 2 and 3 started a shared persistent queue. The seven presets below each run OASIS pair 1, AbdomenCTCT pair 2 and DIR-Lab case 01. These are members of the respective frozen quick4 subsets. They form an initial operating-point screen. Subsequent comparisons extend promising settings to the full frozen quick4 cases.

| Preset | Width | Layers | Updates | OneCycle peak |
|---|---:|---:|---:|---:|
| w256_5k_lr1 | 256 | 4 | 5,000 | 1e-4 |
| w256_5k_lr2 | 256 | 4 | 5,000 | 2e-4 |
| w384_5k_lr1 | 384 | 4 | 5,000 | 1e-4 |
| w384_5k_lr15 | 384 | 4 | 5,000 | 1.5e-4 |
| w256_8k_lr15 | 256 | 4 | 8,000 | 1.5e-4 |
| w256_l6_5k_lr15 | 256 | 6 | 5,000 | 1.5e-4 |
| w384_l5_5k_lr15 | 384 | 5 | 5,000 | 1.5e-4 |

The production implementation applies endpoint affine fusion and scalar-curvature derivative accumulation specifically at width 512. Narrower models currently use the original full-gradient derivative chain. This affects timing interpretation and will be assessed separately if it determines the comparison. The first case for each specialization can include compilation. Per-case training, recovery, scoring and total queue timings are retained.

The initial reference is the completed 74-case starting-version run. Its corresponding pair outputs, exact configs and timings are recorded in each experiment job. Fit-to-fit CUDA variation at seed 42 remains possible. Development selection on these public validation pairs is reported explicitly.

## Live evidence

The screening, matched timing and original-execution 74-pair cohort are complete. The frozen capacity-by-recovery interaction is also complete. Configs and plans are in `results/research_20261115/width`. The ignored harness is `.local/research_20261115/width/run.py`. No checkpoint is resumed between variants, and all result paths are unique.
## Capacity and equal-time follow-up

Paired parameter counts are 1,581,058 at width 512/four layers; 890,882 at width 384/four layers; 397,314 at width 256/four layers; 660,482 at width 256/six layers; and 1,186,562 at width 384/five layers.

The first warm width-256 runs take about 35 seconds end to end and 31 seconds in training, compared with approximately 80 seconds for the width-512 Learn2Reg baseline. The quality loss on the first OASIS pair persists at both 1e-4 and 2e-4 peak rates, so raising the learning rate alone does not recover the starting model's result. This motivates spending the compute saving on additional updates.

A parent-authorized follow-up is queued behind the initial screen. It runs a 10,000-update width-256 variant at peak rate 1e-4, plus 6,000-update width-384 variants at peak 1.5e-4 with input-layer factors 1 and 2/3. The latter keeps input-layer peak updates at 1e-4 while allowing faster hidden-layer movement. The 10,000-update width-256 peak-1.5e-4 branch was pruned because its 8,000-update version regressed all three screened pairs. The six-layer width-256 branch was not extended because its abdomen Dice fell to 0.2585 versus 0.2951. The initial 7,500-update width-384 plan was pruned before execution because measured throughput predicted that it would exceed the width-512 time budget. The time target remains the measured width-512 5,000-update end-to-end cost.

## Six-layer operating-point failure

The transferred width-256, six-layer, 5k, peak-1.5e-4 recipe fails on DIR-Lab case 01, reaching 4.210 mm forward TRE versus 0.987 mm for the reference. All eight PCG intervals converge, while lung folding reaches 0.0305%. In its final 500 updates, mean pre-clipping gradient norm is 81.43 against a clipping threshold of 1. Endpoint NCC is 0.953/0.954, PD-LK is 0.300 and registration loss is 0.183. The four-layer width-384 peak-1.5e-4 control has gradient norm 0.56, endpoint NCC 0.994/0.993, PD-LK 0.0157 and registration loss 0.0258.

This shows an unstable transferred operating point with severe gradient scaling. Both rates exceed the clipping threshold on all 250 logged samples; the lower rate reduces final mean pre-clipping norm to 1.78 and raises crop endpoint NCC to 0.991/0.991. The intervention changes the severity of the gradient/step interaction while preserving architecture and sampled inputs. It does not isolate a depth-capacity limit or demonstrate that additional depth itself must fail. Additional sine compositions multiply coordinate and parameter derivative factors, which can make the derivative-based objective sensitive to the learning-rate operating point even when image-value variance is stable. The fixed-architecture rate rescue distinguishes this explanation from treating the failed recipe as a sufficient test of representational capacity. The weak-case workstream completed a frozen landmark probe. At case 01 landmarks, moving/fixed 8-mm patch NCC fell to 0.152/0.141 versus 0.800/0.787 for the width-512 reference. The RMS brightness-path mismatch normalized by the required change rose to 3.894 versus 0.269, with sign coherence 0.475 versus 0.781. The required spatial forcing also weakened (RMS 0.132 versus 0.441), while the model time derivative stayed large (0.541 versus 0.432). The high crop-global training NCC therefore hides poor landmark-neighborhood representation and a noisy parameter path. A lower-rate rescue at peak 5e-5 repaired the major failure on the same three pairs. DIR-Lab forward TRE improved from 4.210 to 1.113 mm, abdomen Dice from 0.2585 to 0.2954, while OASIS Dice fell from 0.8283 to 0.8218. Warm training took about 45 seconds. This supports an operating-point failure and leaves a remaining time/capacity tradeoff. Frozen evidence is `results/research_20261115/weak_cases/deep256_case01.json`. The scalar evidence is `results/research_20261115/width/depth_operating_point.json`.

## Broad comparison and optional execution prototype

The selected development candidate is width 384, four layers, 6,000 updates and OneCycle peak 1.5e-4, with all first-layer rates at the same factor as hidden layers. On frozen quick4 it yields OASIS Dice 0.82403 versus 0.82440 for the fresh unchanged width-512 control. Mean OASIS end-to-end time is 76.79 versus 80.75 seconds. Against the original full baseline, abdomen quick4 Dice is 0.42609 versus 0.42880, HD95 18.8768 versus 19.1605 voxels, and DIR-Lab rounded TRE is 1.6730/1.7363 versus 1.5921/1.6561 mm. These are near-miss development results, not an accuracy-equivalence claim.

At 12:37 Hong Kong time the two-GPU queue started completing the remaining 62 official pairs. Its full denominator will contain all 74 pairs, reusing only the 12 verified exact-config, exact-implementation completed fits. Reused fits retain their own timing records, and incremental queue wall time will be distinguished from a fresh whole-cohort benchmark. The full comparison uses the original execution path for width 384.

A separate frozen OASIS fitted-checkpoint prototype applies the existing scalar-curvature and endpoint-affine implementation to width 384 through an ignored local function override. Counterbalanced full objective and all-parameter backward replay time falls from 10.234 to 9.071 ms (11.4%). The maximum component-loss difference is 5.96e-8, total-loss difference 9.31e-10, concatenated gradient relative L2 is 1.21e-6, and all 20 parameter tensors are finite. The check took 42.4 seconds including compilation. The numerical prototype is not used in the 74-pair cohort. Evidence is `results/research_20261115/width/fused384_gpu_parity.json`.

## Frozen endpoint fidelity

These values use the existing min–max output calibration and the same 8-mm NCC windows as the baseline endpoint diagnostics. All quick4 pairs are included for the width-384 candidate and fresh width-512 control. Values are moving / fixed.

| Dataset | Recipe | PSNR dB | Anatomical patch NCC | Within-label NCC |
|---|---|---:|---:|---:|
| oasis | w512_control | 25.70 / 25.30 | 0.9689 / 0.9713 | 0.9034 / 0.9135 |
| oasis | w384_6k_lr15 | 26.45 / 25.87 | 0.9675 / 0.9693 | 0.8999 / 0.9089 |
| abdomenctct | w512_control | 19.39 / 18.65 | 0.7458 / 0.7300 | 0.6845 / 0.6458 |
| abdomenctct | w384_6k_lr15 | 17.77 / 17.89 | 0.7298 / 0.7167 | 0.6648 / 0.6317 |
| dirlab | w512_control | 13.75 / 13.37 | 0.8186 / 0.8086 | — |
| dirlab | w384_6k_lr15 | 12.82 / 13.13 | 0.8163 / 0.8062 | — |

Width 384 has higher OASIS min–max PSNR while anatomical NCC is slightly lower. Abdomen anatomical NCC also falls despite a higher quick4 Dice than the fresh width-512 control. These observations motivate using local fidelity together with motion-path and correspondence diagnostics. A single reconstruction scalar cannot select the registration model.

On DIR-Lab case 01, reducing the six-layer width-256 peak rate from 1.5e-4 to 5e-5 raises landmark NCC from 0.152/0.141 to 0.744/0.730 and min–max PSNR from 8.49/8.90 to 14.69/14.80 dB. This directly confirms that the earlier deep-model failure includes lost local anatomy and that the lower rate repairs much of it. The matched frozen scalar evidence is `results/research_20261115/width/fidelity_comparison.json`.
## Fused 7k follow-up and stopping decision

The numerical speed saving motivated a bounded 7,000-update width-384 pilot at the same peak rate. It kept a separate experimental handler and output tree. OASIS pair 1 Dice improved from 0.8268 at original-execution 6k to 0.8280 at fused 7k; abdomen pair 2 fell from 0.3065 to 0.3012; DIR-Lab case 01 improved from 1.0054 to 0.9855 mm rounded TRE. The fresh width-512 case-01 control was 0.9428 mm, so the apparent match to the older 0.9871-mm baseline does not establish superiority.

Cold fused abdomen timing was 112.15 seconds, including 104.29 seconds in training. One same-config warm-cache repeat measured 82.55 seconds end to end and 72.51 seconds in training, with nearly identical Dice 0.3011. The starting width-512 pair took 81.48 seconds, and fresh width-512 abdomen quick4 averaged 80.21 seconds. The remaining small overrun and mixed accuracy do not justify extending this combined 7k recipe to quick4 or a new full cohort during this round. The 11.4% replay saving remains a validated execution prototype. Spending that saving on 17% additional updates overshoots the measured budget. No further fused-7k optimizer trials are queued.

Cold/warm measurements have different process/cache histories: the cold abdomen fit was the second case in a persistent worker with a new fused image geometry; the repeat used the populated compiler cache in a fresh worker. These are measured end-to-end runs, while the exact 11.4% execution comparison uses counterbalanced graph replays on one GPU with fixed fitted weights and inputs.

## Completed OASIS cohort and label-size check

All 19 original-execution width-384 OASIS pairs are complete. Mean Dice is 0.812048 versus the starting width-512 0.813372. Mean recorded queue time is 73.09 versus 79.86 seconds. The baseline used four persistent GPU workers and the current full comparison uses two; the completed same-GPU A/B/B/A full-fit verification qualifies timing claims.

Within each pair, labels were ranked by fixed-label voxel volume and the smallest/largest quarters were averaged separately. Mean small-label Dice changes from 0.62636 to 0.62161 (−0.00475); large-label Dice changes from 0.88232 to 0.88030 (−0.00202). The largest label-level losses include ID 34 (mean 60.3 voxels, Dice −0.01798), ID 23 (748 voxels, −0.01383) and ID 18 (69.2 voxels, −0.01049). Labels 11 and 35 improve. Pooled size-versus-Dice-change Spearman correlation is −0.086 and correlation across the 35 label means is −0.065. Size groups therefore expose some vulnerable small structures, but size alone does not explain the recipe difference. Anatomy, location and derivative geometry remain confounded. This is a complete recipe comparison with changed width, peak rate and update schedule, not a width-only causal estimate. Evidence is `results/research_20261115/width/oasis_capacity_by_label_size.json`.

## Same-GPU time-to-quality verification

A balanced A/B/B/A complete-fit comparison started at 13:05 Hong Kong time after both GPUs were explicitly released. GPU 0 runs OASIS pair 1, and GPU 1 runs abdomen pair 2. A is the unchanged width-512, 5,000-update reference; B is the original-execution width-384, 6,000-update recipe. Each GPU keeps one persistent worker but constructs fresh models and optimizers and resets seed 42 for every fit. Configs, source/cache location and dataset geometry stay fixed within each A or B recipe. Per-case adapter stages distinguish training/recovery/evaluation from process startup and I/O. The eight fits target the small measured speed gain and same-seed score variation without adding a hyperparameter grid. Evidence is `results/research_20261115/width/abba_analysis.json`.

## Balanced timing outcome

All eight balanced fits completed successfully. The warm final A repetition is compared with the mean of B1 and B2 below, which avoids attributing the first A worker initialization to the architecture. A is width 512/5k; B is width 384/6k.

| Pair / GPU | Warm A end-to-end s | B mean end-to-end s | Reduction | Warm A training s | B mean training s | Reduction |
|---|---:|---:|---:|---:|---:|---:|
| oasis / GPU 0 | 80.998 | 72.817 | 10.10% | 71.390 | 65.589 | 8.13% |
| abdomenctct / GPU 1 | 81.699 | 73.422 | 10.13% | 70.930 | 65.352 | 7.86% |

Adapter-only timing yields 10.16% less time on OASIS and 10.20% on abdomen. Including both A runs gives 12.36% and 12.22%, but those means include first-worker overhead. Outside-adapter time was 2.20/2.35 seconds for the two initial A runs and 0.35/0.54 seconds for their final A repetitions. Training itself also warmed from 73.17 to 71.39 seconds on OASIS and 72.70 to 70.93 seconds on abdomen. Recovery contributes to the smaller recipe’s end-to-end gain: mean recovery is 6.04 versus 8.23 seconds on OASIS and 6.81 versus 9.25 on abdomen.

The two-fit mean Dice is 0.82866 for A versus 0.82605 for B on OASIS pair 1, and 0.29200 versus 0.30056 on abdomen pair 2. This verifies speed on two matched geometries and illustrates fit variation. Full-cohort accuracy remains the scientific denominator. Exact source/config/order and all stages are retained in `results/research_20261115/width/abba_analysis.json`.
## Completed abdomen cohort

All 45 original-execution width-384 abdomen pairs are complete. Dice is 0.434486 versus 0.438975 for the starting width-512 baseline, HD95 is 16.72349 versus 16.76780 voxels, folding fraction is 0.0009097 versus 0.0013131 and SDlogJ is 0.24000 versus 0.27687. Mean recorded queue time is 72.33 versus 79.76 seconds. The recipe trades a 0.00449 Dice reduction for lower folding and similar HD95. The matched same-GPU ABBA above verifies a warm runtime gain on one abdomen geometry, while these cohort times retain their queue/cache histories.

The smallest within-pair label quarter changes from Dice 0.22121 to 0.21417 (−0.00704), and the largest quarter changes from 0.61462 to 0.60931 (−0.00530). The evidence file `results/research_20261115/width/abdomenctct_capacity_by_label_size.json` retains all pair and label denominators and avoids treating absent labels as zero. The smaller recipe’s lower folding and SDlogJ are consistent with a more conservative recovered deformation, but these summary metrics do not identify whether representation, optimization or the time path causes it. The completed 0.003-recovery interaction below measures how that tradeoff changes with recovery regularization.

## DIR-Lab motion-stratified frozen comparison

The independent weak-case workstream recovered all ten candidate checkpoints and compared all 6,000 directional landmark observations with the starting reference. Continuous forward TRE changes from 2.87105 to 2.94051 mm and reverse TRE from 2.97592 to 3.05088 mm. Mean endpoint landmark-patch NCC is similar, 0.71466 versus 0.71599. All 80 candidate recovery intervals converge and scratch is cleaned.

Using the original true-motion quartiles, forward TRE changes from 1.134 to 1.146 mm in the lowest-motion group, 1.766 to 1.820 in the next, 2.370 to 2.455 in the third and 6.194 to 6.320 in the largest-motion group. The largest-motion loss is 0.126 mm versus 0.012 mm in the smallest-motion group. The all-path scalar forcing check has similar slopes (0.934 versus 0.929) and normalized residual (0.280 versus 0.275). Endpoint local correlation and this scalar forcing summary therefore do not account for the recovered-motion gap. Directional information, aperture ambiguity and spatial propagation still need consideration. This is distinct from the aggressive six-layer failure, where local anatomy and forcing both collapsed.

The scalar/per-landmark evidence is `results/research_20261115/weak_cases/width384_landmarks/comparison.json` and `landmark_comparison.csv`. These diagnostics qualify the complete 74-pair scores; they do not select new cases or use labels/landmarks for training.

## Completed capacity-by-recovery interaction

Both widths were evaluated on all 74 official pairs with the same frozen checkpoints at PCG smoothness 0.01 and 0.003. This changes the global recovery regularization, separately from the unchanged 0.75-cell training velocity smoothing. All 592 width-384 recovery intervals converge and all recovery scratch is cleaned. The experiment is evaluation-only and remains opt-in.

| Width / PCG smoothness | OASIS Dice | Abdomen Dice | Abdomen folding fraction | DIR-Lab forward / reverse rounded TRE mm |
|---|---:|---:|---:|---:|
| 512 / 0.01 | 0.813372 | 0.438975 | 0.0013131 | 2.84712 / 2.96386 |
| 512 / 0.003 | 0.811087 | 0.453706 | 0.0030094 | 2.86886 / 2.97550 |
| 384 / 0.01 | 0.812048 | 0.434486 | 0.0009097 | 2.92812 / 3.03728 |
| 384 / 0.003 | 0.810456 | 0.452649 | 0.0023477 | 2.96161 / 3.06145 |

At width 384, lower smoothness improves abdomen Dice on 42/45 pairs by 0.01816 on average, while mean folding rises 2.58× and HD95 worsens from 16.7235 to 16.7860 voxels. OASIS Dice falls by 0.00159 and mean folding rises 6.34×; DIR-Lab rounded TRE worsens in both directions. Abdomen reaches a similar Dice level at both widths with smoothness 0.003, but the topology and brain/lung tradeoffs remain.

The abdomen Dice response to the lower regularization is 0.00343 larger at width 384 than at width 512. This supports a coupling between representation and recovery regularization. It does not establish a universally better smaller model or a fixed correction based only on neuron count. The complete paired data are `results/research_20261115/recovery_w384_s003_full/comparison.json` and `results/research_20261115/recovery_s003_full/comparison.json`.
<!-- screen-results -->
Updated 13:40 Hong Kong time.

## Screen results (complete)

| Recipe | Dataset / pair | Main score | Width-512 baseline | Second score | Width-512 baseline second | End-to-end s | Training s |
|---|---|---:|---:|---:|---:|---:|---:|
| w256_5k_lr1 | oasis 1 | 0.81794 | 0.83368 | 1.68209 | 1.62170 | 72.65 | 64.86 |
| w256_5k_lr1 | abdomenctct 2 | 0.29407 | 0.29507 | 22.83957 | 23.19066 | 72.86 | 64.58 |
| w256_5k_lr2 | oasis 1 | 0.82175 | 0.83368 | 1.69441 | 1.62170 | 34.84 | 30.67 |
| w256_5k_lr1 | dirlab 1 | 1.15937 | 0.98710 | 1.21686 | 1.02643 | 59.76 | 56.20 |
| w256_5k_lr2 | abdomenctct 2 | 0.28688 | 0.29507 | 22.83255 | 23.19066 | 35.59 | 30.77 |
| w256_5k_lr2 | dirlab 1 | 1.16984 | 0.98710 | 1.21800 | 1.02643 | 34.79 | 31.18 |
| w384_5k_lr1 | oasis 1 | 0.82706 | 0.83368 | 1.68538 | 1.62170 | 93.40 | 83.15 |
| w384_5k_lr1 | abdomenctct 2 | 0.29056 | 0.29507 | 23.24321 | 23.19066 | 89.79 | 81.24 |
| w384_5k_lr15 | oasis 1 | 0.82648 | 0.83368 | 1.64605 | 1.62170 | 62.32 | 55.30 |
| w384_5k_lr1 | dirlab 1 | 1.00019 | 0.98710 | 1.01250 | 1.02643 | 85.79 | 80.00 |
| w384_5k_lr15 | abdomenctct 2 | 0.29856 | 0.29507 | 22.66841 | 23.19066 | 63.21 | 55.32 |
| w384_5k_lr15 | dirlab 1 | 1.00531 | 0.98710 | 1.00994 | 1.02643 | 61.07 | 55.52 |
| w256_8k_lr15 | oasis 1 | 0.82050 | 0.83368 | 1.66794 | 1.62170 | 53.33 | 49.00 |
| w256_8k_lr15 | abdomenctct 2 | 0.28516 | 0.29507 | 23.07027 | 23.19066 | 53.63 | 48.83 |
| w256_8k_lr15 | dirlab 1 | 1.21476 | 0.98710 | 1.26387 | 1.02643 | 53.18 | 49.43 |
| w256_l6_5k_lr15 | oasis 1 | 0.82833 | 0.83368 | 1.62426 | 1.62170 | 93.63 | 81.70 |
| w256_l6_5k_lr15 | abdomenctct 2 | 0.25849 | 0.29507 | 23.17179 | 23.19066 | 84.45 | 76.05 |
| w256_l6_5k_lr15 | dirlab 1 | 4.21015 | 0.98710 | 4.53163 | 1.02643 | 82.94 | 76.84 |
| w384_l5_5k_lr15 | oasis 1 | 0.80369 | 0.83368 | 1.85799 | 1.62170 | 112.97 | 100.51 |
| w384_l5_5k_lr15 | abdomenctct 2 | 0.26722 | 0.29507 | 23.31718 | 23.19066 | 110.85 | 99.87 |
| w384_l5_5k_lr15 | dirlab 1 | 1.78360 | 0.98710 | 1.80184 | 1.02643 | 106.43 | 98.25 |

## Budget results (complete)

| Recipe | Dataset / pair | Main score | Width-512 baseline | Second score | Width-512 baseline second | End-to-end s | Training s |
|---|---|---:|---:|---:|---:|---:|---:|
| w256_10k_lr1 | abdomenctct 2 | 0.29593 | 0.29507 | 22.95270 | 23.19066 | 70.16 | 63.50 |
| w256_10k_lr1 | oasis 1 | 0.82436 | 0.83368 | 1.68916 | 1.62170 | 70.81 | 64.36 |
| w256_10k_lr1 | dirlab 1 | 1.14642 | 0.98710 | 1.19570 | 1.02643 | 63.96 | 60.92 |
| w384_6k_lr15 | oasis 1 | 0.82675 | 0.83368 | 1.70554 | 1.62170 | 74.04 | 66.30 |
| w384_6k_lr15 | abdomenctct 2 | 0.30648 | 0.29507 | 22.16912 | 23.19066 | 74.56 | 66.14 |
| w384_6k_lr15 | dirlab 1 | 1.00538 | 0.98710 | 1.01782 | 1.02643 | 71.69 | 66.39 |
| w384_6k_lr15_first1 | oasis 1 | 0.82193 | 0.83368 | 1.70580 | 1.62170 | 72.88 | 66.09 |
| w384_6k_lr15_first1 | abdomenctct 2 | 0.30156 | 0.29507 | 22.80186 | 23.19066 | 73.91 | 66.19 |
| w384_6k_lr15_first1 | dirlab 1 | 1.04267 | 0.98710 | 1.02209 | 1.02643 | 71.53 | 66.35 |

## Rescue results (complete)

| Recipe | Dataset / pair | Main score | Width-512 baseline | Second score | Width-512 baseline second | End-to-end s | Training s |
|---|---|---:|---:|---:|---:|---:|---:|
| w256_l6_5k_lr05 | oasis 1 | 0.82180 | 0.83368 | 1.68510 | 1.62170 | 57.83 | 48.25 |
| w256_l6_5k_lr05 | abdomenctct 2 | 0.29544 | 0.29507 | 23.20288 | 23.19066 | 50.81 | 44.77 |
| w256_l6_5k_lr05 | dirlab 1 | 1.11296 | 0.98710 | 1.19233 | 1.02643 | 50.09 | 45.04 |

## Control results (complete)

| Recipe | Dataset / pair | Main score | Width-512 baseline | Second score | Width-512 baseline second | End-to-end s | Training s |
|---|---|---:|---:|---:|---:|---:|---:|
| w512_control | oasis 1 | 0.83057 | 0.83368 | 1.64943 | 1.62170 | 85.98 | 74.10 |
| w512_control | oasis 4 | 0.79656 | 0.79582 | 2.78680 | 2.79721 | 78.89 | 69.10 |
| w512_control | oasis 8 | 0.83162 | 0.83361 | 1.51715 | 1.50950 | 79.09 | 69.30 |
| w512_control | oasis 9 | 0.83887 | 0.83792 | 1.57857 | 1.60078 | 79.03 | 69.33 |
| w512_control | abdomenctct 2 | 0.28303 | 0.29507 | 23.49661 | 23.19066 | 81.65 | 70.95 |
| w512_control | abdomenctct 8 | 0.57045 | 0.58022 | 11.21434 | 10.87789 | 79.95 | 69.01 |
| w512_control | abdomenctct 18 | 0.47192 | 0.47485 | 12.05102 | 12.20756 | 79.60 | 68.99 |
| w512_control | abdomenctct 41 | 0.36278 | 0.36507 | 30.11737 | 30.36588 | 79.66 | 68.89 |
| w512_control | dirlab 1 | 0.94281 | 0.98710 | 1.01411 | 1.02643 | 79.10 | 71.30 |
| w512_control | dirlab 3 | 1.65333 | 1.65002 | 1.65573 | 1.77037 | 79.51 | 71.18 |
| w512_control | dirlab 4 | 1.72356 | 1.68668 | 1.75113 | 1.68435 | 79.05 | 71.11 |
| w512_control | dirlab 5 | 2.18328 | 2.04471 | 2.28030 | 2.14330 | 80.02 | 71.08 |

## Quick4 results (complete)

| Recipe | Dataset / pair | Main score | Width-512 baseline | Second score | Width-512 baseline second | End-to-end s | Training s |
|---|---|---:|---:|---:|---:|---:|---:|
| w384_6k_lr15 | oasis 4 | 0.79615 | 0.79582 | 2.89600 | 2.79721 | 80.72 | 68.97 |
| w384_6k_lr15 | oasis 8 | 0.83509 | 0.83361 | 1.38423 | 1.50950 | 81.23 | 69.88 |
| w384_6k_lr15 | oasis 9 | 0.83812 | 0.83792 | 1.58137 | 1.60078 | 71.19 | 64.44 |
| w384_6k_lr15 | abdomenctct 8 | 0.57448 | 0.58022 | 10.90427 | 10.87789 | 73.90 | 66.17 |
| w384_6k_lr15 | abdomenctct 18 | 0.47618 | 0.47485 | 11.81081 | 12.20756 | 73.69 | 66.11 |
| w384_6k_lr15 | abdomenctct 41 | 0.34723 | 0.36507 | 30.62306 | 30.36588 | 72.36 | 64.59 |
| w384_6k_lr15 | dirlab 3 | 1.72566 | 1.65002 | 1.80308 | 1.77037 | 97.99 | 92.00 |
| w384_6k_lr15 | dirlab 4 | 1.77818 | 1.68668 | 1.89998 | 1.68435 | 98.05 | 92.39 |
| w384_6k_lr15 | dirlab 5 | 2.18278 | 2.04471 | 2.22426 | 2.14330 | 96.90 | 90.78 |

## Full results (complete)

| Recipe | Dataset / pair | Main score | Width-512 baseline | Second score | Width-512 baseline second | End-to-end s | Training s |
|---|---|---:|---:|---:|---:|---:|---:|
| w384_6k_lr15 | oasis 4 | 0.79615 | 0.79582 | 2.89600 | 2.79721 | 80.72 | 68.97 |
| w384_6k_lr15 | oasis 1 | 0.82675 | 0.83368 | 1.70554 | 1.62170 | 74.04 | 66.30 |
| w384_6k_lr15 | oasis 8 | 0.83509 | 0.83361 | 1.38423 | 1.50950 | 81.23 | 69.88 |
| w384_6k_lr15 | oasis 9 | 0.83812 | 0.83792 | 1.58137 | 1.60078 | 71.19 | 64.44 |
| w384_6k_lr15 | abdomenctct 2 | 0.30648 | 0.29507 | 22.16912 | 23.19066 | 74.56 | 66.14 |
| w384_6k_lr15 | abdomenctct 8 | 0.57448 | 0.58022 | 10.90427 | 10.87789 | 73.90 | 66.17 |
| w384_6k_lr15 | abdomenctct 18 | 0.47618 | 0.47485 | 11.81081 | 12.20756 | 73.69 | 66.11 |
| w384_6k_lr15 | abdomenctct 41 | 0.34723 | 0.36507 | 30.62306 | 30.36588 | 72.36 | 64.59 |
| w384_6k_lr15 | dirlab 1 | 1.00538 | 0.98710 | 1.01782 | 1.02643 | 71.69 | 66.39 |
| w384_6k_lr15 | dirlab 3 | 1.72566 | 1.65002 | 1.80308 | 1.77037 | 97.99 | 92.00 |
| w384_6k_lr15 | dirlab 4 | 1.77818 | 1.68668 | 1.89998 | 1.68435 | 98.05 | 92.39 |
| w384_6k_lr15 | dirlab 5 | 2.18278 | 2.04471 | 2.22426 | 2.14330 | 96.90 | 90.78 |
| w384_6k_lr15 | oasis 2 | 0.76914 | 0.77134 | 2.71830 | 2.67178 | 77.41 | 67.70 |
| w384_6k_lr15 | oasis 3 | 0.79899 | 0.79836 | 1.96160 | 1.92064 | 78.76 | 69.17 |
| w384_6k_lr15 | oasis 5 | 0.80917 | 0.80882 | 1.59045 | 1.51686 | 71.15 | 64.38 |
| w384_6k_lr15 | oasis 7 | 0.80132 | 0.80374 | 1.95643 | 1.90297 | 71.31 | 64.49 |
| w384_6k_lr15 | oasis 6 | 0.82702 | 0.83169 | 1.43426 | 1.38413 | 71.37 | 64.52 |
| w384_6k_lr15 | oasis 11 | 0.80145 | 0.80182 | 1.90865 | 1.88764 | 71.17 | 64.36 |
| w384_6k_lr15 | oasis 10 | 0.78253 | 0.78347 | 2.56625 | 2.52013 | 71.35 | 64.48 |
| w384_6k_lr15 | oasis 12 | 0.81589 | 0.81898 | 1.59615 | 1.60611 | 71.06 | 64.33 |
| w384_6k_lr15 | oasis 13 | 0.77737 | 0.77200 | 2.04840 | 2.10930 | 71.34 | 64.47 |
| w384_6k_lr15 | oasis 14 | 0.84549 | 0.84565 | 1.38181 | 1.41130 | 70.98 | 64.34 |
| w384_6k_lr15 | oasis 15 | 0.82388 | 0.82418 | 1.51024 | 1.48910 | 71.17 | 64.46 |
| w384_6k_lr15 | oasis 16 | 0.82209 | 0.82370 | 1.40807 | 1.34996 | 71.00 | 64.33 |
| w384_6k_lr15 | oasis 17 | 0.81414 | 0.81832 | 1.63856 | 1.63673 | 71.26 | 64.43 |
| w384_6k_lr15 | oasis 18 | 0.80591 | 0.81176 | 1.73601 | 1.74804 | 70.92 | 64.31 |
| w384_6k_lr15 | oasis 19 | 0.83841 | 0.83921 | 1.43850 | 1.43776 | 71.25 | 64.45 |
| w384_6k_lr15 | abdomenctct 1 | 0.43060 | 0.43069 | 15.33500 | 15.70263 | 73.69 | 66.04 |
| w384_6k_lr15 | abdomenctct 3 | 0.50599 | 0.50901 | 14.17629 | 14.27592 | 74.07 | 66.24 |
| w384_6k_lr15 | abdomenctct 4 | 0.52413 | 0.54454 | 14.52904 | 14.08933 | 71.93 | 64.36 |
| w384_6k_lr15 | abdomenctct 5 | 0.50274 | 0.52182 | 16.91998 | 16.92129 | 72.21 | 64.51 |
| w384_6k_lr15 | abdomenctct 6 | 0.36054 | 0.36510 | 25.94371 | 26.34933 | 71.85 | 64.35 |
| w384_6k_lr15 | abdomenctct 9 | 0.51364 | 0.53188 | 12.92310 | 12.90655 | 71.91 | 64.41 |
| w384_6k_lr15 | abdomenctct 7 | 0.55673 | 0.56662 | 9.24781 | 9.55705 | 72.17 | 64.46 |
| w384_6k_lr15 | abdomenctct 10 | 0.28103 | 0.28400 | 24.52442 | 24.14990 | 72.09 | 64.41 |
| w384_6k_lr15 | abdomenctct 11 | 0.33867 | 0.34372 | 19.96409 | 20.16679 | 72.58 | 64.50 |
| w384_6k_lr15 | abdomenctct 12 | 0.35897 | 0.35242 | 17.45905 | 17.16040 | 72.05 | 64.39 |
| w384_6k_lr15 | abdomenctct 13 | 0.42983 | 0.43489 | 15.73747 | 15.32539 | 72.41 | 64.48 |
| w384_6k_lr15 | abdomenctct 14 | 0.30321 | 0.29865 | 31.64781 | 31.53231 | 71.92 | 64.32 |
| w384_6k_lr15 | abdomenctct 15 | 0.42945 | 0.42413 | 13.76713 | 13.78285 | 72.39 | 64.46 |
| w384_6k_lr15 | abdomenctct 16 | 0.48994 | 0.47143 | 12.64047 | 13.84207 | 72.00 | 64.29 |
| w384_6k_lr15 | abdomenctct 17 | 0.39894 | 0.41333 | 18.76389 | 18.39584 | 72.27 | 64.45 |
| w384_6k_lr15 | abdomenctct 19 | 0.24692 | 0.24462 | 18.35871 | 18.63000 | 71.86 | 64.36 |
| w384_6k_lr15 | abdomenctct 20 | 0.45442 | 0.45555 | 13.91956 | 14.43504 | 72.13 | 64.44 |
| w384_6k_lr15 | abdomenctct 21 | 0.43211 | 0.44299 | 19.19731 | 19.27656 | 72.04 | 64.39 |
| w384_6k_lr15 | abdomenctct 22 | 0.30893 | 0.32827 | 19.59129 | 19.14035 | 72.38 | 64.48 |
| w384_6k_lr15 | abdomenctct 23 | 0.32792 | 0.32802 | 21.11763 | 21.59600 | 71.97 | 64.36 |
| w384_6k_lr15 | abdomenctct 24 | 0.27943 | 0.28726 | 20.12132 | 19.69840 | 72.40 | 64.52 |
| w384_6k_lr15 | abdomenctct 25 | 0.41466 | 0.41898 | 14.06293 | 14.28782 | 71.95 | 64.39 |
| w384_6k_lr15 | abdomenctct 27 | 0.38324 | 0.40015 | 20.42641 | 19.66879 | 72.24 | 64.50 |
| w384_6k_lr15 | abdomenctct 26 | 0.53831 | 0.55003 | 10.87369 | 11.00802 | 72.07 | 64.40 |
| w384_6k_lr15 | abdomenctct 28 | 0.44354 | 0.42746 | 14.00231 | 14.39521 | 72.19 | 64.51 |
| w384_6k_lr15 | abdomenctct 29 | 0.40993 | 0.41215 | 20.51248 | 20.70230 | 72.01 | 64.37 |
| w384_6k_lr15 | abdomenctct 30 | 0.49399 | 0.50756 | 14.51575 | 14.06530 | 72.11 | 64.49 |
| w384_6k_lr15 | abdomenctct 31 | 0.42947 | 0.42080 | 13.71388 | 14.48439 | 72.05 | 64.38 |
| w384_6k_lr15 | abdomenctct 32 | 0.34282 | 0.34201 | 28.80379 | 28.73599 | 72.22 | 64.51 |
| w384_6k_lr15 | abdomenctct 33 | 0.44948 | 0.45414 | 12.28876 | 12.49065 | 72.07 | 64.35 |
| w384_6k_lr15 | abdomenctct 34 | 0.45047 | 0.43199 | 15.48431 | 15.19396 | 72.10 | 64.49 |
| w384_6k_lr15 | abdomenctct 35 | 0.53894 | 0.54031 | 10.41564 | 10.08964 | 71.86 | 64.37 |
| w384_6k_lr15 | abdomenctct 36 | 0.46604 | 0.47375 | 14.72126 | 14.94598 | 72.19 | 64.49 |
| w384_6k_lr15 | abdomenctct 37 | 0.54767 | 0.54089 | 9.66731 | 10.08136 | 72.11 | 64.37 |
| w384_6k_lr15 | abdomenctct 38 | 0.45847 | 0.46590 | 17.15686 | 17.10605 | 72.28 | 64.55 |
| w384_6k_lr15 | abdomenctct 39 | 0.50076 | 0.55083 | 12.90250 | 11.67861 | 71.88 | 64.37 |
| w384_6k_lr15 | abdomenctct 40 | 0.33315 | 0.33946 | 26.58056 | 26.17883 | 72.29 | 64.51 |
| w384_6k_lr15 | abdomenctct 42 | 0.37103 | 0.37601 | 20.42156 | 19.08562 | 71.92 | 64.39 |
| w384_6k_lr15 | abdomenctct 43 | 0.60819 | 0.61541 | 8.81978 | 8.68440 | 72.24 | 64.53 |
| w384_6k_lr15 | abdomenctct 44 | 0.59470 | 0.59078 | 7.90589 | 8.46559 | 71.89 | 64.38 |
| w384_6k_lr15 | abdomenctct 45 | 0.59849 | 0.60112 | 7.88892 | 9.62627 | 72.18 | 64.50 |
| w384_6k_lr15 | dirlab 2 | 1.07412 | 0.96173 | 1.11039 | 0.98396 | 99.43 | 92.96 |
| w384_6k_lr15 | dirlab 6 | 3.38154 | 3.32143 | 3.57407 | 3.45249 | 101.66 | 92.18 |
| w384_6k_lr15 | dirlab 8 | 8.18070 | 8.05019 | 8.49126 | 8.38086 | 73.94 | 64.85 |
| w384_6k_lr15 | dirlab 7 | 4.48965 | 4.49684 | 4.62505 | 4.62695 | 100.75 | 91.16 |
| w384_6k_lr15 | dirlab 9 | 2.70701 | 2.63574 | 2.62601 | 2.66772 | 73.88 | 64.84 |
| w384_6k_lr15 | dirlab 10 | 2.75620 | 2.63670 | 3.00086 | 2.90219 | 100.12 | 91.81 |

## Fused_pilot results (complete)

| Recipe | Dataset / pair | Main score | Width-512 baseline | Second score | Width-512 baseline second | End-to-end s | Training s |
|---|---|---:|---:|---:|---:|---:|---:|
| w384_fused_7k_lr15 | oasis 1 | 0.82803 | 0.83368 | 1.65666 | 1.62170 | 80.98 | 72.09 |
| w384_fused_7k_lr15 | abdomenctct 2 | 0.30119 | 0.29507 | 22.74341 | 23.19066 | 112.15 | 104.29 |
| w384_fused_7k_lr15 | dirlab 1 | 0.98553 | 0.98710 | 1.01837 | 1.02643 | 105.32 | 99.94 |

## Fused_warm results (complete)

| Recipe | Dataset / pair | Main score | Width-512 baseline | Second score | Width-512 baseline second | End-to-end s | Training s |
|---|---|---:|---:|---:|---:|---:|---:|
| w384_fused_7k_lr15_warm | abdomenctct 2 | 0.30108 | 0.29507 | 22.89330 | 23.19066 | 82.55 | 72.51 |

The main/second scores are rounded forward/reverse TRE (mm) for DIR-Lab and Dice/HD95 (voxels) for OASIS and abdomen. The first new width/depth specialization can include compilation. Every scalar baseline comes from the corresponding full-baseline pair. All completed jobs retain full per-case metrics and folding diagnostics. Learning-rate changes are never credited with execution speedups when their compile/cache history differs.

The fresh unchanged width-512 quick4 control completed on GPU 1 after the depth rescue. This gives a same-source, same-queue comparison and a reference for single-seed CUDA fit variation. Extending the update count also stretches the OneCycle schedule, so these are complete recipe comparisons.

The local experimental scalar-curvature/affine path passed six independent float64 full-objective and all-parameter VJP checks, covering four to six layers and box/GW registration. Maximum loss difference was 8.9e-16 and maximum relative gradient L2 was 1.3e-15. It is not enabled in production or in the initial screen.
