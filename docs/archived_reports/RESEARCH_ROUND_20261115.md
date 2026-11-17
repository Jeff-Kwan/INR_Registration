# SIREN registration research round, 15 November 2026

## Scope and timing

The research round completed from approximately 12:08 to 13:40 HKT, about 92 minutes, followed by the user's request to wrap up and discuss. All GPU experiments have finished. The reference source is Git tag `starting-version`, commit `993567e`. The full baseline is `results/full_baseline_20261115_run`, with 19 OASIS, 45 AbdomenCTCT and 10 DIR-Lab pairs. Each checkpoint used width 512, four sine layers, 5,000 updates, seed 42, captured GW-NCC/orthogonal/Sobol training and eight-time PCG+RK2 recovery. The canonical method and presets stay under review separately from these experiments.

## Questions and experimental controls

1. Determine whether label size, local endpoint fidelity and motion explain registration failures. Compare within-label versus surrounding-patch NCC and distinguish pooled correlations from within-case/within-label associations. Use all baseline pairs for descriptive analysis. Chosen weak cases are development probes.
2. Compare narrower width-384/256 models under matched time and update budgets, allowing schedule and learning-rate tuning. Frozen quick4 selections provide the initial screen. Report registration, folding, endpoint fidelity and complete adapter time. Keep compile costs visible.
3. Inspect neighboring GS-Reg read-only for mechanisms that transfer, then validate focused hypotheses without promoting them to defaults.

Every experiment uses the single configured seed. Labels and landmarks remain evaluation-only. Recovery maps stay in memory or temporary scratch. Checkpoints and scalar experiment records live in ignored results directories.

## Live allocation

| Worker | Allocation | Evidence and notes |
|---|---|---|
| Weak-case agent | GPU 0 and CPU | [Weak-case live notes](RESEARCH_WEAK_CASES_20261115.md) |
| Width agent | GPUs 2–3 | [Width live notes](RESEARCH_WIDTH_20261115.md) |
| GS transfer agent | CPU, then a coordinated GPU probe | [GS transfer live notes](RESEARCH_GS_TRANSFER_20261115.md) |
| Main agent | GPU 1, synthesis and follow-up experiments | This document |

## What the diagnostics distinguish

Endpoint fidelity, local spatial information, capture range, temporal forcing and recovery regularity answer different questions. An accurate endpoint image can still lead to an incorrect correspondence. Brightness flow constrains the velocity projection along an image gradient; scalar fidelity or straight-path forcing metrics alone do not establish the full recovered displacement.

| Question | Observation | Interpretation supported by this round |
|---|---|---|
| Do small labels reconstruct badly? | OASIS smallest/largest volume quartiles have Dice about 0.631/0.883, yet surrounding endpoint NCC about 0.975/0.959. Abdomen shows the same reversal. | Small-label overlap errors coexist with strong surrounding-image fidelity. Anatomical identity and boundary sensitivity confound pooled NCC–Dice correlations. |
| Does local fidelity predict error within anatomy? | After case/label controls, full-patch NCC–Dice correlation is near zero in both label cohorts. Within DIR-Lab, endpoint NCC retains a negative association with TRE after case/motion controls. | Local reconstruction is useful for locating lung failures, and insufficient as a general registration selection criterion. |
| Is small anatomy devoid of useful image information? | Native adrenal NCC translation information is comparable to kidney patches, while broad blur reduces adrenal information roughly 15–31-fold. | Broad filtering can destroy usable organ-specific evidence. High correlation at a smoothed true match does not establish a larger useful basin. |
| Is crop starvation the main weak-case mechanism? | Tiny labels often receive no particles in an update, but current crops increase their expected counts relative to whole-volume uniform sampling. High-TRE lung points are covered more often, and controlled exposure–Dice associations are weak. | Coverage is a diagnostic concern, without evidence that a blind sampling replacement fixes the failures. |
| Is global brightness drift responsible? | Mean middle-time offset/gain fractions are below 0.2% of q energy per examined lung; diagnostic subtraction slightly worsens forcing residuals. | A global brightness-gauge correction lacks support. Local appearance/path defects remain possible. |
| Does PCG discard an accurate training map? | On cases 07/08, the local two-field training decoder is less accurate than PCG, even after crop averaging and an evaluation-only travel-gain oracle. | A systematic local estimator/path deficit precedes recovery. PCG regularization still affects the final overlap/regularity tradeoff. |

Absolute and relative boundary errors also differ by label size. In abdomen, smallest/largest volume quartiles have mean HD95 11.82/20.32 voxels, while HD95 divided by equivalent-volume sphere radius is 1.94/0.81. Small organs have smaller absolute boundary errors but larger errors relative to their size. OASIS small structures have both worse absolute HD95 (3.04 versus 1.45 voxels) and worse relative error (0.697 versus 0.059). The sphere radius is a volume-based scale proxy, not an anatomical radius. [Size-normalized boundary evidence](../results/research_20261115/weak_cases/size_normalized_hd95.json)

The weak-label analysis uses all 665 scored OASIS pair–label observations, with metric-specific exclusions for undefined NCC, and 568 abdomen observations where the required labels are available. The 6,000 directional lung landmarks are paired by case, direction and landmark ID. Associations are descriptive. Repeated subjects and anatomical labels do not constitute independent clinical samples. [Weak-case evidence](RESEARCH_WEAK_CASES_20261115.md) · [GS transfer and information diagnostics](RESEARCH_GS_TRANSFER_20261115.md)

## Smaller models and complete runtime

The selected narrow experiment uses width 384, four sine layers, 6,000 updates, OneCycle peak 1.5e-4, seed 42 and the original execution math. The width-512 reference uses 5,000 updates and peak 1e-4. Other retained objective and recovery settings agree. Parameter count falls from 1,581,058 to 890,882, a 43.65% reduction.

| Dataset | Width 512 | Width 384 | Recorded full-case time, 512 → 384 |
|---|---:|---:|---:|
| OASIS, 19 pairs, Dice | 0.81337 | 0.81205 | 79.86 → 73.09 s |
| Abdomen, 45 pairs, Dice | 0.43897 | 0.43449 | 79.76 → 72.33 s |
| DIR-Lab, 10 cases, forward/reverse rounded TRE | 2.847/2.964 mm | 2.928/3.037 mm | 110.30 → 91.44 s |

The full-case timing columns have different queue/cache histories. Same-GPU A/B/B/A checks resolve a real speed effect on two representative pairs. Warm complete time falls 81.00→72.82 s on OASIS pair 1 and 81.70→73.42 s on abdomen pair 2, approximately 10.1% each. Training alone falls about 8%. First-worker startup is kept separate. The full narrow cohort reduces folding, but its strict shared accuracy target is unmet. No noninferiority margin was prespecified, so small score differences are reported directly.

At width 256, extra updates at the examined rates did not close the representative brain gap. Adding depth at a transferred high rate produced a severe fine-detail and transport failure. A lower-rate rescue repaired it, establishing an operating-point failure rather than a proof against depth. A width-384 fused-math prototype reduces frozen objective/backward time by 11.4% with verified parameter gradients. Its 7k-update fit pilot has mixed quality and exceeds the representative abdomen wall-time budget slightly even when warm, so that recipe is closed. [Full architecture study and controls](RESEARCH_WIDTH_20261115.md)

## Controlled improvement candidates

Improving local endpoint NCC on two abdomen development pairs did not improve their registration. Increasing observed-image registration pressure did, despite slightly lower endpoint patch fidelity. These interventions distinguish reconstruction quality from the model's ability to support useful motion.

| Matched ordinary-execution development pair | Endpoint NCC, registration weight 0.01 | Endpoint GW16, registration weight 0.01 | Endpoint NCC, registration weight 0.03 | Endpoint GW16, registration weight 0.03 |
|---|---:|---:|---:|---:|
| Abdomen 19, Dice | 0.22493 | 0.20179 | 0.29514 | 0.21877 |
| Abdomen 2, Dice | 0.30938 | 0.29162 | 0.31766 | 0.30946 |

These arms share ordinary execution and independent times. The captured Sobol transfer is evaluated separately. Its frozen abdomen quick4 result raises Dice 0.42880→0.43605, improves HD95 by about 0.535 voxels, and reduces mean/max folding. Full 45-pair validation is complete at original recovery smoothness 0.01. Mean Dice rises 0.438975→0.442553, with 27/45 pairs improving. HD95 worsens slightly, 16.76780→16.79084 voxels. Mean folding falls 0.13131%→0.12493%, while its 90th percentile rises 0.53467%→0.57239% and maximum is essentially unchanged (0.88834%→0.88915%). Actual mean complete time is 79.76→80.41 s. This is a small, qualified overlap improvement, without a full-cohort boundary or folding-tail improvement.

The smallest label-volume quartile loses Dice, 0.22789→0.22321, while kidneys improve approximately 0.018–0.021. Larger initial centroid-displacement groups benefit, while the smallest-displacement group regresses. The setting helps some initially misaligned, mid-size anatomy and does not solve the small-adrenal problem. [Full 45-pair audit](../results/research_20261115/registration_weight003/full45_audit.json) · [Label and initial-motion groups](../results/research_20261115/registration_weight003/label_size_comparison.json)

A second lever changes recovery smoothness on exactly the same frozen models, with no optimizer updates. Its all-74 result is complete.

| Fixed-checkpoint recovery change, 0.01 → 0.003 | Accuracy change | Mean folding change |
|---|---|---|
| OASIS, 19 pairs | Dice 0.81337 → 0.81109 | 0.00338% → 0.01971% |
| Abdomen, 45 pairs | Dice 0.43897 → 0.45371, 41/45 improving | 0.1313% → 0.3009% |
| DIR-Lab, 10 cases | Rounded TRE 2.847/2.964 → 2.869/2.975 mm | Lung folding 0 → 0.000207% |

Abdomen HD95 also worsens slightly at lower smoothness. Geometry checks show both genuine alignment changes and organ expansion. The extra overlap is a regularity tradeoff, not an unconditional improvement. Combining lower recovery smoothness with the stronger-training-weight quick4 checkpoints raises Dice again but multiplies folding about 8.7-fold and worsens HD95. That combined recipe is not selected.

## Capacity and recovery interact

The final scalar-only recovery comparison reused every width-384 checkpoint and changed only fixed smoothness 0.01→0.003. All 74 cases and 592 solves passed provenance, convergence and scratch-cleanup audits.

| Width-384 frozen recovery | Smoothness 0.01 | Smoothness 0.003 |
|---|---:|---:|
| OASIS Dice | 0.81205 | 0.81046 |
| Abdomen Dice | 0.43449 | 0.45265 |
| Abdomen mean folding | 0.09097% | 0.23477% |
| DIR-Lab forward/reverse rounded TRE | 2.928/3.037 mm | 2.962/3.061 mm |

At the same lower smoothness, the narrow abdomen model comes close to the wide model's Dice (0.45265 versus 0.45371) with less folding (0.23477% versus 0.30094%). Both carry more folding than the starting wide/smoothness-0.01 reference. Brain and lung accuracy remain lower. This is a measured tradeoff rather than a common winning replacement.

Matched global probes also qualify the mechanism. Narrow MRI/CT gradient energy is modestly lower, largely tracking output intensity variance, while lung gradient energy is higher. Constant image gain scales the brightness data term quadratically and therefore changes the effective strength of a fixed PCG penalty. This is distinct from temporal gain/offset drift, which was a negative diagnostic. Neither static amplitude nor a single scalar forcing score explains every architecture difference. The study retains fixed recovery coefficients throughout. [Capacity-by-recovery evidence](../results/research_20261115/recovery_w384_s003_full/comparison.json) · [Independent audit](../results/research_20261115/gs_frozen_width384_recovery_audit.json)

## Suggestions reserved for review

- Consider width 384/6k as a measured economy option, with its accuracy loss and lower folding visible. Its approximately 10% speed saving and 44% parameter reduction do not establish accuracy equivalence.
- Prioritize observed-image alignment and temporal-path quality over raising reconstruction NCC alone. The captured registration-weight-0.03 candidate has a modest full-abdomen mean Dice gain, with small-label and boundary regressions that qualify it for discussion.
- Treat fixed recovery smoothness as a dataset/model operating point with explicit overlap, boundary and folding measurements. The fixed capacity-by-smoothness comparison is complete; no adaptive recovery is introduced.
- Transfer GS-Reg's diagnostic separation of capture range and local information. Its blur benefit does not transfer reliably to this SIREN training route. Keep native detail, matched controls and the worst-point analysis in any future scale experiment.

All findings are development evidence from public validation cases used for diagnostics or tuning. Labels and landmarks remain evaluation-only. The canonical method, presets and `starting-version` tag remain unchanged.

## Live log

- 12:08 HKT — confirmed baseline source and all four GPUs idle. Started parallel analysis, architecture screens and the GS-Reg review. `/workspace/AGENTS.md` is absent; repository AGENTS.md and current consolidated documentation supply local instructions.
- 12:11 HKT — baseline metadata identifies strong OASIS reconstruction (approximately 0.968 label-surrounding patch NCC), weaker abdomen (0.72–0.74) and weaker DIR-Lab landmark fidelity (0.71–0.72 on 8-mm patches). These cross-dataset averages motivate examination of individual anatomy and motion rather than a causal interpretation.

- 12:15 HKT — all ten frozen DIR-Lab recoveries completed after explicit user authorization. Scalar landmark diagnostics reproduce aggregate TRE, all 80 PCG intervals converge, and all temporary scratch directories are verified removed. Across 6,000 directional landmark observations, the endpoint fidelity–error association persists after case/motion controls. This differs from the near-zero anatomy-controlled label NCC–Dice association in OASIS and abdomen.
- 12:16 HKT — GPU 1 is running a two-case DIR-Lab bandwidth ablation (cases 07/08, registration-image Gaussian sigma 0/1.5/3/6 mm), with original endpoint targets, original estimator derivatives and original brightness recovery. A sigma-zero wrapper control checks the experimental dispatch route. GPU 0 starts matched ordinary-execution endpoint NCC/GW8/GW16 probes on abdomen pairs 19/2 and OASIS pair 10. These change endpoint fitting evidence without using evaluation labels for training. GPUs 2/3 continue width, depth and learning-rate screens, followed by narrower equal-time extra-update trials.
- 12:16 HKT — the GS review motivates distinct tests of capture range and local observability. Observed DIR-Lab patches can correlate more strongly under broad filtering, while broad filtering sharply reduces abdominal NCC translation information. A shared blur increase therefore has a dataset-dependent mechanism and needs explicit small-structure controls.
- 12:24 HKT — a focused global brightness-gauge probe rejected one possible explanation. Projection of frozen q onto global intensity offset/gain explained below 0.2% of q energy on lung cases 01/07/08 and below 1% on the two brain/abdomen controls. Removing that component worsened the straight-path transport residual. Global gauge correction therefore receives no GPU validation this round.
- 12:24 HKT — depth-6/width-256 at the transferred high learning rate fails locally despite global NCC near 0.95. Case-01 landmark endpoint NCC falls to 0.15/0.14, versus 0.80/0.79 in the reference, with severe clipping and much poorer temporal forcing. A lower-LR rescue is prepared as a specific operating-point test. Unpromising architecture/LR arms are being pruned before broader validation.
- 12:32 HKT — completed the eight-fit registration-image bandwidth experiment. Mean forward/reverse continuous TRE on DIR-Lab 07/08 is 6.2195/6.3786 mm at sigma 0; 6.1568/6.3611 at 1.5 mm; 6.3088/6.5017 at 3 mm; and 6.2636/6.4444 at 6 mm. Moderate-blur changes are small relative to observed fit variability; broad blur fails to improve both directions. All 64 recovery solves converge. The matched highest-TRE landmark analysis also shows the average coarse capture effect does not help the worst points. No staged-blur training grid is planned.
- 12:32 HKT — width/depth learning-rate screen completed. The lower-LR six-layer width-256 rescue repairs the severe high-LR failure, but retains an OASIS fidelity gap. Width 384, four layers, 6k updates and OneCycle peak 1.5e-4 proceeds to the frozen quick4 comparison. A 10k width-256 four-layer fit is fast enough but does not close the representative brain gap. New width-512 quick4 controls measure contemporary runtime and same-seed fit variation.
- 12:36 HKT — a proposed orthogonal hidden initialization was tested algebraically before spending optimizer runs. Although its weight singular values are equal by construction, the actual initialized PD-LK information matrix became less isotropic on the fixed width-384 fixture (median minimum-eigenvalue/trace 0.0328 → 0.0269). This falsifies the proposed initial conditioning benefit. The unused optimizer wrapper will be removed; scalar evidence remains in the GS transfer review.
- 12:38 HKT — the width-384 fused-math prototype reduces a fitted full-objective/backward replay from 10.234 to 9.071 ms (11.4%), with full-gradient relative L2 1.21e-6 and maximum component loss difference 5.96e-8. This is an experimental execution handler, with no change to canonical guards. The full 74-case validation retains original width-384 math for a clean denominator. A separate three-case 7k-update fused pilot is prepared to test whether the measured saving can buy accuracy at the width-512 wall-time budget.
- 12:41 HKT — matched endpoint GW-NCC experiments demonstrate a fidelity/registration disconnect on two abdomen pairs. On weak pair 19, endpoint NCC→GW16 raises surrounding-patch NCC from 0.714 to 0.813 and within-label NCC from 0.625 to 0.751, while Dice falls from 0.2249 to 0.2018. Pair 2 shows the same direction; OASIS pair 10 instead improves. These are development probes with matched ordinary execution and independent time draws, distinct from the captured Sobol baseline.
- 12:41 HKT — frozen recovery smoothness is a more promising controlled lever. On the same weak pair-19 checkpoint, 0.01→0.003 raises Dice 0.2446→0.2837 and folding 0.0089%→0.0728%; 0.03 lowers Dice to 0.2118. A broader scalar-only frozen-checkpoint comparison is being prepared. Every map remains temporary and every setting is experimental.
- 12:54 HKT — full OASIS validation completed for width 384/four layers/6k updates. Dice is 0.812048 versus 0.813372 for the starting width-512 recipe, HD95 1.81373 versus 1.79587 voxels. Historical per-pair wall time is 73.09 versus 79.86 s, with different GPU/cache histories; same-GPU A/B/B/A checks are prepared to resolve the timing claim. Accuracy equivalence is not established.
- 12:54 HKT — the fused 7k-update pilot is closed after its specified timing check. The same warm abdomen fit takes 82.55 s versus 112.15 s cold, but is still slightly over its approximately 80-s width-512 comparison and has mixed three-case quality. The 11.4% frozen step saving remains a prototype result, not a selected training recipe.
- 12:54 HKT — weak pair-19 recovery smoothness has an additive effect across endpoint losses: ordinary NCC/GW16/GW8 each gains about 0.023 Dice at smoothness 0.003, preserving GW's relative deficit. A four-fit ordinary-execution 2×2 objective-balance check now compares endpoint NCC/GW16 at registration weight 0.03 against the existing weight-0.01 controls on pairs 19 and 2. This tests the measured clipping/registration-loss competition directly. It is the final planned optimizer hypothesis beyond the width cohort and its timing verification.
- 13:02 HKT — all 74 frozen smoothness-0.003 evaluations completed, all 592 PCG intervals converged and all recorded scratch directories were removed. OASIS Dice changes 0.813372→0.811087, abdomen 0.438975→0.453706, and DIR-Lab rounded forward/reverse TRE 2.8471/2.9639→2.8689/2.9755 mm. Abdomen improves on 41/45 pairs but mean folding rises 0.1313%→0.3009%, maximum 1.3633%, with slightly worse mean HD95. This is a dataset-specific overlap/regularity tradeoff. An independent provenance/config/cleanup audit passed.
- 13:02 HKT — direct observed-image registration pressure improves the weak pair-19 ordinary NCC fit: weight 0.01→0.03 raises Dice 0.22493→0.29514 while surrounding endpoint NCC falls 0.714→0.695. GW16 at the same stronger pressure remains lower, Dice 0.21877, despite higher local fidelity. The second matched development pair is still pending. A captured quick4 transfer test is prepared conditionally; no broader new optimizer cohort will be started without that evidence.
- 13:16 HKT — same-GPU A/B/B/A verification confirms a real width time saving. Warm width-512 control versus mean width-384 runs is 81.00→72.82 s on OASIS pair 1 and 81.70→73.42 s on abdomen pair 2, approximately 10.1% less complete time on each. Training alone falls 8.1%/7.9%. First-worker startup is reported separately rather than credited as an architecture gain.
- 13:16 HKT — captured registration-weight-0.03 quick4 transfer passes its prespecified development criterion. Mean Dice 0.42880→0.43605, HD95 19.1605→18.6253 voxels, mean folding 0.03804%→0.01703%, maximum folding 0.10967%→0.04457%. Three of four pairs improve Dice and three improve HD95. The remaining 41 official abdomen pairs are approved for a single four-GPU queue after the current architecture cohort and its scalar landmark probe release the devices. Union with quick4 will give the full 45-pair denominator. Recovery smoothness stays 0.01 in this candidate.

- 13:25 HKT — the first 12 completed non-quick4 registration-weight-0.03 pairs reverse the pilot trend, with mean Dice lower and HD95/folding higher. The fixed 41-pair continuation is completing unchanged; no case is removed and no coefficient is adjusted during the run. This is precisely the qualification the broad validation is intended to resolve.

## Discussion priorities

The most targeted next experiment would separate registration NCC window scale from velocity-grid scale while holding particles and optimization budget fixed. Current training support is broad relative to the smallest structures. A smaller appearance window could give their observed texture a more local influence, while keeping the derivative estimator cost fixed. The competing explanation is noisy sparse moments or an insufficiently local velocity estimator. Small-label gains with preserved large-organ alignment would support the window hypothesis; increased variance or folding would argue for a different intervention. This experiment has not been run or added to the canonical interface.

The shared starting recipe remains the reference for accuracy. Width 384/6k is an economy option with measured accuracy costs. Higher registration pressure and lower recovery smoothness are dataset- and anatomy-qualified candidates for review. Endpoint patch NCC, PSNR, forcing diagnostics, Dice/TRE and folding are kept together when judging them.

## Completion and cleanup audit

- The round exceeded 90 wall-clock minutes and wrapped well before the two-hour limit. Three subagents and all four GPUs performed parallel work, with explicit device handoffs.
- The width study completed all 74 cases, same-GPU timing controls and 6,000 paired directional landmark diagnostics. The full 45-pair registration-weight candidate and both full 74-case frozen-recovery comparisons completed. Each broad cohort passed its independent provenance and true-residual convergence audit.
- All 194 optimizer-run checkpoints are retained under ignored research results. Scalar diagnostics, configs, source hashes and logs are retained for review. No persistent dense deformation, warped-image, velocity or reconstruction volumes were produced by this round.
- Every owned recovery scratch directory was checked after exit. All GPU workers have exited. Two unused preparation wrappers were removed after their hypotheses were closed.
- Tracked method, configuration, launcher and packaging files match `starting-version`. The work adds four research documents and ignored experimental artifacts. No research candidate was made canonical and no new commit or tag was created.
- The results use one configured seed (42), including explicit same-seed repeats for timing/variation checks. The public validation cohorts were used for development and selection. Repeated subjects and post-selection comparisons prevent a held-out or statistical-equivalence claim.
