# Smaller SIREN research, 15 November 2026

Workstream complete. All experimental fits and frozen probes are terminal. Canonical tracked code and presets match `starting-version`.

This is an experimental research record. The canonical presets remain the tagged starting version `993567e`. All fits use seed 42 and frozen recovery settings.

## Full-cohort result

The width-384/four-layer, 6k, peak-1.5e-4 recipe completes all 74 official pairs with 890,882 parameters per pair versus 1,581,058 for width 512. All checkpoints are finite, all 592 recovery intervals converge and the original execution source matches `starting-version` byte for byte. It reduces runtime while slightly reducing mean registration accuracy. This is a candidate tradeoff for review, not a canonical selection or an accuracy-equivalence result.

| Dataset | Width-512 main / second score | Width-384 main / second score | Width-512 mean s | Width-384 mean s |
|---|---:|---:|---:|---:|
| OASIS, 19 | Dice 0.81337 / HD95 1.79587 | Dice 0.81205 / HD95 1.81373 | 79.86 | 73.09 |

Full-cohort timings retain two-GPU versus historical four-GPU queue/cache differences and should be read alongside that controlled verification.

The complete candidate denominator reuses 12 exact-config, exact-implementation quick4 fits and trains the other 62. Its incremental two-GPU wall time was 2,321.5 seconds (38 minutes 41 seconds). Scalar evidence and case/source manifests are in `results/research_20261115/width/full_aggregate.json` and `full_audit.json`. The 256-width recipes tested did not meet the joint accuracy/time target. The six-layer failure was substantially repaired by lowering the rate, while its OASIS accuracy still trailed the reference. The 11.4% width-384 numerical execution saving is a separate validated prototype; the mixed 7k pilot was closed after a warm repeat exceeded the target time.

## Question and mechanism

A smaller network reduces dense FP32 matrix multiplication cost, but it may need a different learning rate or more updates to represent small anatomical structures and the spatial derivatives used by registration. Extra depth can recover capacity while increasing repeated derivative cost and optimization difficulty. We compare complete adapter time and registration accuracy together. Endpoint fidelity alone cannot select a model because its derivative geometry also determines the recovered motion.

The starting width-512 model has four sine layers, omega 30, output initialization scale 30, AdamW with decay 0.001, a OneCycle peak of 1e-4 and 5,000 updates. Native captured training uses endpoint NCC with moments 0.01, PD-LK 0.1, GW registration NCC 0.01, two orthogonal query pairs and scrambled Sobol times. Recovery uses the unchanged eight-time, cap-256 PCG and RK2 recipe.

## Initial screen

At 12:10 Hong Kong time, GPUs 2 and 3 started a shared persistent queue. These are members of the respective frozen quick4 subsets. They form an initial operating-point screen. Subsequent comparisons extend promising settings to the full frozen quick4 cases.

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

A parent-authorized follow-up is queued behind the initial screen. It runs a 10,000-update width-256 variant at peak rate 1e-4, plus 6,000-update width-384 variants at peak 1.5e-4 with input-layer factors 1 and 2/3. The latter keeps input-layer peak updates at 1e-4 while allowing faster hidden-layer movement. The 10,000-update width-256 peak-1.5e-4 branch was pruned because its 8,000-update version regressed all three screened pairs. The initial 7,500-update width-384 plan was pruned before execution because measured throughput predicted that it would exceed the width-512 time budget. The time target remains the measured width-512 5,000-update end-to-end cost.

## Six-layer operating-point failure

In its final 500 updates, mean pre-clipping gradient norm is 81.43 against a clipping threshold of 1. Endpoint NCC is 0.953/0.954, PD-LK is 0.300 and registration loss is 0.183. The four-layer width-384 peak-1.5e-4 control has gradient norm 0.56, endpoint NCC 0.994/0.993, PD-LK 0.0157 and registration loss 0.0258.

The lower learning rate reduced the final mean pre-clipping gradient norm to 1.78 and raised crop endpoint NCC to 0.991/0.991. See the [operating-point evidence](../../results/research_20261115/width/depth_operating_point.json).

## Broad comparison and optional execution prototype

The selected development candidate is width 384, four layers, 6,000 updates and OneCycle peak 1.5e-4, with all first-layer rates at the same factor as hidden layers. On frozen quick4 it yields OASIS Dice 0.82403 versus 0.82440 for the fresh unchanged width-512 control. Mean OASIS end-to-end time is 76.79 versus 80.75 seconds. These are near-miss development results, not an accuracy-equivalence claim.


A separate frozen OASIS fitted-checkpoint prototype applies the existing scalar-curvature and endpoint-affine implementation to width 384 through an ignored local function override. Counterbalanced full objective and all-parameter backward replay time falls from 10.234 to 9.071 ms (11.4%). The maximum component-loss difference is 5.96e-8, total-loss difference 9.31e-10, concatenated gradient relative L2 is 1.21e-6, and all 20 parameter tensors are finite. The check took 42.4 seconds including compilation. The numerical prototype is not used in the 74-pair cohort. Evidence is `results/research_20261115/width/fused384_gpu_parity.json`.

## Frozen endpoint fidelity

These values use the existing min–max output calibration and the same 8-mm NCC windows as the baseline endpoint diagnostics. All quick4 pairs are included for the width-384 candidate and fresh width-512 control. Values are moving / fixed.

| Dataset | Recipe | PSNR dB | Anatomical patch NCC | Within-label NCC |
|---|---|---:|---:|---:|
| oasis | w512_control | 25.70 / 25.30 | 0.9689 / 0.9713 | 0.9034 / 0.9135 |
| oasis | w384_6k_lr15 | 26.45 / 25.87 | 0.9675 / 0.9693 | 0.8999 / 0.9089 |

Width 384 has higher OASIS min–max PSNR while anatomical NCC is slightly lower. These observations motivate using local fidelity together with motion-path and correspondence diagnostics. A single reconstruction scalar cannot select the registration model.

This directly confirms that the earlier deep-model failure includes lost local anatomy and that the lower rate repairs much of it. The matched frozen scalar evidence is `results/research_20261115/width/fidelity_comparison.json`.
## Fused 7k follow-up and stopping decision

The numerical speed saving motivated a bounded 7,000-update width-384 pilot at the same peak rate.

One same-config warm-cache repeat measured 82.55 seconds end to end and 72.51 seconds in training, with nearly identical Dice 0.3011. The remaining small overrun and mixed accuracy do not justify extending this combined 7k recipe to quick4 or a new full cohort during this round. The 11.4% replay saving remains a validated execution prototype. Spending that saving on 17% additional updates overshoots the measured budget. No further fused-7k optimizer trials are queued.

These are measured end-to-end runs, while the exact 11.4% execution comparison uses counterbalanced graph replays on one GPU with fixed fitted weights and inputs.

## Completed OASIS cohort and label-size check

All 19 original-execution width-384 OASIS pairs are complete. Mean Dice is 0.812048 versus the starting width-512 0.813372. Mean recorded queue time is 73.09 versus 79.86 seconds. The baseline used four persistent GPU workers and the current full comparison uses two; the completed same-GPU A/B/B/A full-fit verification qualifies timing claims.

Within each pair, labels were ranked by fixed-label voxel volume and the smallest/largest quarters were averaged separately. Mean small-label Dice changes from 0.62636 to 0.62161 (−0.00475); large-label Dice changes from 0.88232 to 0.88030 (−0.00202). The largest label-level losses include ID 34 (mean 60.3 voxels, Dice −0.01798), ID 23 (748 voxels, −0.01383) and ID 18 (69.2 voxels, −0.01049). Labels 11 and 35 improve. Pooled size-versus-Dice-change Spearman correlation is −0.086 and correlation across the 35 label means is −0.065. Size groups therefore expose some vulnerable small structures, but size alone does not explain the recipe difference. Anatomy, location and derivative geometry remain confounded. This is a complete recipe comparison with changed width, peak rate and update schedule, not a width-only causal estimate. Evidence is `results/research_20261115/width/oasis_capacity_by_label_size.json`.

## Same-GPU time-to-quality verification

A balanced A/B/B/A complete-fit comparison started at 13:05 Hong Kong time after both GPUs were explicitly released. A is the unchanged width-512, 5,000-update reference; B is the original-execution width-384, 6,000-update recipe. Each GPU keeps one persistent worker but constructs fresh models and optimizers and resets seed 42 for every fit. Configs, source/cache location and dataset geometry stay fixed within each A or B recipe. Per-case adapter stages distinguish training/recovery/evaluation from process startup and I/O. The eight fits target the small measured speed gain and same-seed score variation without adding a hyperparameter grid. Evidence is `results/research_20261115/width/abba_analysis.json`.

## Balanced timing outcome

All eight balanced fits completed successfully. The warm final A repetition is compared with the mean of B1 and B2 below, which avoids attributing the first A worker initialization to the architecture. A is width 512/5k; B is width 384/6k.

| Pair / GPU | Warm A end-to-end s | B mean end-to-end s | Reduction | Warm A training s | B mean training s | Reduction |
|---|---:|---:|---:|---:|---:|---:|
| oasis / GPU 0 | 80.998 | 72.817 | 10.10% | 71.390 | 65.589 | 8.13% |

Including both A runs gives 12.36% and 12.22%, but those means include first-worker overhead. Outside-adapter time was 2.20/2.35 seconds for the two initial A runs and 0.35/0.54 seconds for their final A repetitions.

This verifies speed on two matched geometries and illustrates fit variation. Full-cohort accuracy remains the scientific denominator. Exact source/config/order and all stages are retained in `results/research_20261115/width/abba_analysis.json`.
## Completed capacity-by-recovery interaction

Both widths were evaluated on all 74 official pairs with the same frozen checkpoints at PCG smoothness 0.01 and 0.003. This changes the global recovery regularization, separately from the unchanged 0.75-cell training velocity smoothing. All 592 width-384 recovery intervals converge and all recovery scratch is cleaned. The experiment is evaluation-only and remains opt-in.

| Width / PCG smoothness | OASIS Dice |
|---|---:|
| 512 / 0.01 | 0.813372 |
| 512 / 0.003 | 0.811087 |
| 384 / 0.01 | 0.812048 |
| 384 / 0.003 | 0.810456 |

This supports a coupling between representation and recovery regularization. It does not establish a universally better smaller model or a fixed correction based only on neuron count. The complete paired data are `results/research_20261115/recovery_w384_s003_full/comparison.json` and `results/research_20261115/recovery_s003_full/comparison.json`.
<!-- screen-results -->
Updated 13:40 Hong Kong time.

## Screen results (complete)

| Recipe | Dataset / pair | Main score | Width-512 baseline | Second score | Width-512 baseline second | End-to-end s | Training s |
|---|---|---:|---:|---:|---:|---:|---:|
| w256_5k_lr1 | oasis 1 | 0.81794 | 0.83368 | 1.68209 | 1.62170 | 72.65 | 64.86 |
| w256_5k_lr2 | oasis 1 | 0.82175 | 0.83368 | 1.69441 | 1.62170 | 34.84 | 30.67 |
| w384_5k_lr1 | oasis 1 | 0.82706 | 0.83368 | 1.68538 | 1.62170 | 93.40 | 83.15 |
| w384_5k_lr15 | oasis 1 | 0.82648 | 0.83368 | 1.64605 | 1.62170 | 62.32 | 55.30 |
| w256_8k_lr15 | oasis 1 | 0.82050 | 0.83368 | 1.66794 | 1.62170 | 53.33 | 49.00 |
| w256_l6_5k_lr15 | oasis 1 | 0.82833 | 0.83368 | 1.62426 | 1.62170 | 93.63 | 81.70 |
| w384_l5_5k_lr15 | oasis 1 | 0.80369 | 0.83368 | 1.85799 | 1.62170 | 112.97 | 100.51 |

## Budget results (complete)

| Recipe | Dataset / pair | Main score | Width-512 baseline | Second score | Width-512 baseline second | End-to-end s | Training s |
|---|---|---:|---:|---:|---:|---:|---:|
| w256_10k_lr1 | oasis 1 | 0.82436 | 0.83368 | 1.68916 | 1.62170 | 70.81 | 64.36 |
| w384_6k_lr15 | oasis 1 | 0.82675 | 0.83368 | 1.70554 | 1.62170 | 74.04 | 66.30 |
| w384_6k_lr15_first1 | oasis 1 | 0.82193 | 0.83368 | 1.70580 | 1.62170 | 72.88 | 66.09 |

## Rescue results (complete)

| Recipe | Dataset / pair | Main score | Width-512 baseline | Second score | Width-512 baseline second | End-to-end s | Training s |
|---|---|---:|---:|---:|---:|---:|---:|
| w256_l6_5k_lr05 | oasis 1 | 0.82180 | 0.83368 | 1.68510 | 1.62170 | 57.83 | 48.25 |

## Control results (complete)

| Recipe | Dataset / pair | Main score | Width-512 baseline | Second score | Width-512 baseline second | End-to-end s | Training s |
|---|---|---:|---:|---:|---:|---:|---:|
| w512_control | oasis 1 | 0.83057 | 0.83368 | 1.64943 | 1.62170 | 85.98 | 74.10 |
| w512_control | oasis 4 | 0.79656 | 0.79582 | 2.78680 | 2.79721 | 78.89 | 69.10 |
| w512_control | oasis 8 | 0.83162 | 0.83361 | 1.51715 | 1.50950 | 79.09 | 69.30 |
| w512_control | oasis 9 | 0.83887 | 0.83792 | 1.57857 | 1.60078 | 79.03 | 69.33 |

## Quick4 results (complete)

| Recipe | Dataset / pair | Main score | Width-512 baseline | Second score | Width-512 baseline second | End-to-end s | Training s |
|---|---|---:|---:|---:|---:|---:|---:|
| w384_6k_lr15 | oasis 4 | 0.79615 | 0.79582 | 2.89600 | 2.79721 | 80.72 | 68.97 |
| w384_6k_lr15 | oasis 8 | 0.83509 | 0.83361 | 1.38423 | 1.50950 | 81.23 | 69.88 |
| w384_6k_lr15 | oasis 9 | 0.83812 | 0.83792 | 1.58137 | 1.60078 | 71.19 | 64.44 |

## Full results (complete)

| Recipe | Dataset / pair | Main score | Width-512 baseline | Second score | Width-512 baseline second | End-to-end s | Training s |
|---|---|---:|---:|---:|---:|---:|---:|
| w384_6k_lr15 | oasis 4 | 0.79615 | 0.79582 | 2.89600 | 2.79721 | 80.72 | 68.97 |
| w384_6k_lr15 | oasis 1 | 0.82675 | 0.83368 | 1.70554 | 1.62170 | 74.04 | 66.30 |
| w384_6k_lr15 | oasis 8 | 0.83509 | 0.83361 | 1.38423 | 1.50950 | 81.23 | 69.88 |
| w384_6k_lr15 | oasis 9 | 0.83812 | 0.83792 | 1.58137 | 1.60078 | 71.19 | 64.44 |
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

## Fused_pilot results (complete)

| Recipe | Dataset / pair | Main score | Width-512 baseline | Second score | Width-512 baseline second | End-to-end s | Training s |
|---|---|---:|---:|---:|---:|---:|---:|
| w384_fused_7k_lr15 | oasis 1 | 0.82803 | 0.83368 | 1.65666 | 1.62170 | 80.98 | 72.09 |

## Fused_warm results (complete)

The first new width/depth specialization can include compilation. Every scalar baseline comes from the corresponding full-baseline pair. All completed jobs retain full per-case metrics and folding diagnostics. Learning-rate changes are never credited with execution speedups when their compile/cache history differs.

The fresh unchanged width-512 quick4 control completed on GPU 1 after the depth rescue. This gives a same-source, same-queue comparison and a reference for single-seed CUDA fit variation. Extending the update count also stretches the OneCycle schedule, so these are complete recipe comparisons.

The local experimental scalar-curvature/affine path passed six independent float64 full-objective and all-parameter VJP checks, covering four to six layers and box/GW registration. Maximum loss difference was 8.9e-16 and maximum relative gradient L2 was 1.3e-15. It is not enabled in production or in the initial screen.
