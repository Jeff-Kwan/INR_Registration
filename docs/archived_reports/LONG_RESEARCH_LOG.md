# Live ten-hour research log

Launch 2026-11-14T00:41:37+08:00. After the user's extension, minimum research finish is **2026-11-14T09:41:37+08:00** and final deadline is **2026-11-14T10:41:37+08:00**. All times use Hong Kong time. Earlier entries retain the original clock for historical context; the extended clock governs current decisions.

Source snapshot and per-case evidence are under `results/long_research_20261114/`. The snapshot includes current uncommitted method changes, hashes, and the repository diff.

## T+7h40: a flow/map distinction and temporal quadrature

The original OASIS Dice is **0.812787** and HD95 is **1.80517 voxels**. The quick4 gain largely fails to transfer. Ten of 19 pairs improve and nine worsen, with little change in mean Dice. Worst folding improves **0.00007594→0.00006199**.

Those trajectories reproduce the saved map's positions within about **8.2e-5 / 5.1e-5 voxels** at the original substeps. Increasing per-interval substeps to 16 and 64 keeps these determinants positive. This supports severe compression and spatially sampled map differences as the source of these observed folds, rather than simply unstable RK2 stepping. It does not excuse negative official map Jacobians or establish positivity everywhere. Evidence: `fold_jacobian_probe/summary.json`.


On OASIS pair 1, the auxiliary-gradient deviation relative to its mean was 0.66 with independent times and 0.86 with antithetic times. Fixed midpoint times differed from the 4×4 temporal reference by 0.49 in relative gradient norm.

The 512-wide, four-layer OASIS quick4 control gives Dice 0.81844 at 3k updates, versus about 0.827 at 5k.

**Audit fixes and scheduling evidence.** The no-argument configuration API now reads the current default preset rather than silently returning obsolete MSE/4k settings; all experiments used explicit configs and are unaffected. Forty presets validate, and 42 focused synthetic/mocked tests pass (two CUDA checks skipped in that CPU invocation). Both adapters now synchronize the frozen recovery boundary for stage timing. Full workers 1/3 were unaffected. Evidence: `default_api_audit_checks.log`, `matched_lr_queue_correction.json`, `width512_3k_timing_retry` when complete.

The final reservation is revised from the opening conservative freeze after observing about **100 s/pair** and an already completed fallback full cohort. Four GPUs need roughly 31 minutes of ideal work, with additional cold-start/tail allowance. Freeze any last shared candidate by **00:55 UTC (T+8h13)**, leaving **106 minutes** to the hard deadline for a complete run, retries, score audit and documentation. Minimum work remains through **01:41:37 UTC**. Unvalidated late ideas remain research findings.

## T+7h: completed accelerated cohort, stronger candidate and failure audit

The accelerated reference completed all 19 OASIS pairs and averaged **104.89 s** per pair. The original averaged **163.93 s**. Bulk sampling changed RNG order.

Worst folding fraction on OASIS was 0.00005037.

**Cold-worker failure and correction.** Newly added peak-memory telemetry called `reset_peak_memory_stats` before a fresh worker initialized CUDA. The first candidate full launch and the new GW 40k launch failed before fitting; the queued size experiment inherited the failure. Those output roots are preserved. Initialize CUDA explicitly before resetting counters. A synthetic test through the actual fresh `GPUWorker` now passes, reports positive peak memory and zero retained allocations. Restart the full and long experiments from fresh immutable snapshots. Do not patch a running snapshot. Evidence: `cold_worker_telemetry_validation.json`, failed `retained_full`/`long_gw_40k`, restarted `retained_full_v2`/`long_gw_40k_v2`.

Tetrahedral queries give OASIS quick4 Dice 0.825361.

The two-stage 2k-coarse/3k-raw control gives OASIS Dice **0.812678**, below the joint reference.

On the two examined OASIS pairs, **all** folds are central and **38–47%** lie inside labels. The Gaussian cascade accumulates central folds across stages, including labeled anatomy. This rejects outer-shell exposure as the leading explanation in these examples. Whole-grid folds remain reported even where labels exclude them. A frozen sparse-coordinate Jacobian probe is queued to distinguish continuous-flow compression from sampled-map finite differences. Evidence: `fold_localization_probe/summary.json`.

The 8k→16k→32k endpoint curriculum gives OASIS quick4 Dice 0.825772. Changing point count adds compilation and graph-transition costs.

GPU 0 now runs the current GW+orthogonal recipe continuously to 40k on cases 03 and 08, preserving the exact first-5k schedule and optimizer/RNG state; checkpoint replay follows. GPU 2 finishes the sampling curriculum, then tests width 768/three layers/3k steps. A width-512/four-layer/3k control is queued after GPU 0's frozen probes. The wider candidate has about 50% more parameters but roughly 10% less total dense arithmetic than 512/four-layer/5k; only measured time-to-quality can decide. Current full validation remains running while these controls execute.

## Opening allocation

GPU0 runs the complete shared-recipe reference. GPU1 profiles the hot training path. GPUs2 and3 run the 40k convergence quick4 comparison, preserving the 5k learning-rate trajectory before continuing at its final LR. Reassign workers after measured results.

## Decisions and observations

- Launch. The shared recipe used width 512, uniform omega 30, output scale 30, OneCycle and AdamW decay 0.001.

- Initial profile. Width512, head30, NCC and32k particles give endpoint loss/backward7.53ms and full loss/backward21.84ms. The trace is dominated by SIMT FP32 matrix multiplies. GPU1 now compares the high matmul-precision mode and will check the resulting analytical outputs and parameter gradients. GPUs0/2/3 remain on reference and long training.

- Matmul tradeoff. The high mode reduces the combined step from21.84 to18.07ms and frozen derivative evaluation from29.90 to20.72ms per262k sites. Matched real-case gradient error is0.98% at1k and6.32% at5k. GPU1 now runs allthree quick4 cohorts under this explicit numerical ablation to measure speed and well-roundedness.
- Recovery compiler check. The bounded mixed-shape PCG reproductions, including a version avoiding prior static compilation, show about0.13ms/step in all three variants and relative output errors below8e-8. The historical135ms slowdown has not appeared in these checks. Continue monitoring real mixed-shape reference timings.

- Exact first-5k schedule verification passed at every step of the40k control. The first logged values agree to about1e-7 across devices; trajectories diverge slightly by5k under ordinary CUDA reductions. Interpret the continuation within each saved trajectory, and use the full reference as the matched recipe rather than requiring bitwise checkpoint identity.

- Shared-representation test prepared. GPU1 next compares a1000-step prefix fitting one SIREN to an equally drawn endpoint, followed by copying its weights and full Adam state into the second endpoint. The independent control gives both endpoint models1000 reconstruction-only steps before the same joint phase. Both retain the5k LR schedule and charge the prefix to training time. A synthetic live fit verifies the split and independent optimizer-state copies. Saved milestones bracket the transition so blur and path formation can be assessed.

## Evidence around the second hour

- Full OASIS reference completes all19 official pairs at0.81279Dice; quick4 is easier at about0.826. Keep the full cohort as the final denominator.
- Captured execution scored 0.82552 Dice on OASIS quick4. Native `solve_ex` with asynchronous singularity checking supported capture with a matched VJP difference of about 5e-7.
- On OASIS quick4, stride-2 recovery reduced recovery time from 19.36 to 2.14 seconds and Dice from 0.82602 to 0.81595.
- Width affects update geometry. At initial peakLR1e-4, first-hidden direct phase RMS is0.072,0.102,0.134,0.191rad for widths256,384,512,768; first-layer phase RMS stays near0.0055rad. Test a width384 learning-rate adjustment from these measurements, keeping the first-layer rate fixed, before attributing all width gains to capacity. Initial Gaussian-moment gradients are below1% of the NCC gradient in the measured crop, so global head-moment calibration is not a leading explanation.
- Same-cost orthogonal queries give OASIS quick4 Dice 0.82570.
This is not the main explanation for the large-volume gap.

## T+2h45: execution, model width and a targeted curriculum

The 40k OASIS quick4 run finished at mean Dice 0.82796.

The complete-step graph v1 retained1successfulcase and11failures. Root cause: TorchDynamo's automatic dynamic-shape generalization introduced an unpinned CPU shape tensor during capture on latercases. v2 requests static shapes; the firstsix cases have completed withoutfailure. The graph retains highestFP32, nativechecked LU, live full derivatives/solves/trajectories and fusedAdamW. Bulk sampling changes RNG draw order while retaining marginal distributions. This remains an explicit sampling/execution trial.


The image-capture probe predicts a concrete tradeoff: on severe large motionσ8mm makes the initialNCC direction more reliable but broadens the localization peak. The corresponding curriculum changes registration images only; endpointtruth, crop coverage, points and regularizer remainfixed. The nonlinear-path repair changes noendpoint predictions, isolating a different possible bottleneck.

## T+5h: coarse representation, frozen integration and complete reference

These remain visible denominators, including the difficult cases outside quick4. Evidence: `reference_full/summary.json`.

The next 64 jobs failed before training because persistent-worker TorchDynamo specializations exceeded the eight-geometry cache limit. The failures are retained in `captured_full/`. The correction retires the joint objective's old shape specializations before compiling each pair; it changes neither samples nor loss equations. `captured_retained_v2` now runs the 19+45 remainder, with the first five OASIS jobs successful. Its launch cwd and explicit import-source provenance are recorded separately. Do not call this full-cohort validation complete yet.


On OASIS, scaling the Gaussian to maximum physical FOV/64 gives Dice 0.812953 and lowers small-label Dice by about 0.019. The continuous Gaussian curriculum gives Dice 0.824287 using FOV/64 for 1k steps, FOV/128 for 1k, then raw images for 3k.

**Numerical RK2 accuracy is not the main source of the observed folds.** Reusing identical frozen velocity fields, increasing substeps per interval from 4 to 8 and 16 changes Dice by only a few parts in 1e5. No training or velocity-estimation settings changed in this diagnostic. Invest in the learned path and field regularity rather than simply adding integration work. Evidence: `integration_convergence_probe/summary.json`.

The analytical temporal derivative is `q = J_theta f · (theta_target - theta_source)`, and its spatial derivative enters PD-LK.


The translation-tangent quadrature feature bank also fails to improve the shared result. Constraining anatomy to a common representation is too restrictive in these forms; it does not invalidate all shared-representation strategies.

Current allocation: GPU0 completes the corrected full-cohort accelerated reference; GPU1 tests the coherent affine optimizer; GPU2 follows the continuous pyramid with the essential uniform-16k control for importance sampling; GPU3 follows exact-native-RNG orthogonal queries with the refinement-budget cascade. CPU probes inspect complete three-direction frames and global endpoint gauges. All runs retain the frozen quick4 selections and single seed.

## T+5h50: useful controls and the next allocation

The corrected accelerated run completed all 19 OASIS pairs with Dice 0.813295 and mean pipeline time 104.89 s. The reference gave Dice 0.812787 and 163.93 s.

Its first three OASIS results regress, including a folding increase on pair 4. Keep this as a large-motion mechanism lead. The queued 2k+3k Gaussian control distinguishes the middle scale from simply allocating one more thousand sharp updates.

The importance proposal has a real CT benefit at the same point count, with a small brain tradeoff. It is retained for further sampling-quality work, rather than discarded for missing the shared threshold.

Frozen temporal quadrature finishes **28 recoveries on 14 cases**. Maximum absolute changes are 0.01855 mm / 0.000750 / 0.001350. Recovery is about **5.9 s**. Eight evaluations (eight substeps each) give still smaller deviations and slightly positive mean Dice changes. Preserve 64 total RK2 substeps when comparing these settings.

The first three-frame wave mixed an old loaded recovery function with a newly imported adapter. OASIS failed after training, and the fitted weights were not saved. The corrected launches used consistent source snapshots.


The exact affine transport check confirms the current differentiated-brightness identity to about **1e-16**. Thus the omitted spatial-velocity term is not supported as the dominant practical error in this diagnostic. No alternative strain/covariant objective was added.

Global gauge interventions finish negatively: endpoint NCC stays invariant while registration often worsens substantially. Preserve learned endpoint calibration. More coherent global first-layer movement alone is not the missing ingredient.

GPU1 is testing retained GW-NCC registration in the complete-step graph, then the early-small/late-large endpoint sample budget. GPU2 completes the three-frame transfer retry, then four-site tetrahedral sampling. GPU3 completes the three-scale transfer test, then the equal-total-step two-stage control. GPU0 continues the accelerated full cohort. The final full-run freeze remains no later than T+8h; continue research through T+9h.

## T+9h20 — complete controls and speed-frontier validation

All intervals converge and HD95 means remain finite. The Sobol full pass is finishing separately.

It trades a small quick4 brain decrease against the original recipe, so it is not promoted. A complete 74-pair speed/quality frontier pass uses newly released GPUs 0/1, then 2/3 after the queued frozen fold/oracle probes and matched additive controls. Conservative completion estimate is 02:28 UTC, leaving deadline contingency. This is an additional validation opportunity, not a reason to abandon complete denominators or to select on quick4.

The public optional Sobol sampler passes all six CUDA synthetic checks, including full loss/parameter VJPs, native masked/unmasked replay, configuration guards and exact temporal-table/spatial-stream preservation. The CPU suite contains 123 tests: 110 pass and 13 CUDA-dependent tests are skipped in that invocation. Default temporal sampling remains independent.

## T+9h35 — queue repair


A finished frozen-probe child remained a zombie because its parent postponed `wait()` until the parallel training controller returned. That controller's GPU2 thread waited on `/proc/PID` disappearance, preventing the controller from returning after its two successful phase-A results. Inspection confirmed no live research GPU child remained. The idle controller was stopped, both completed records were preserved and its summary rebuilt, and the exact matched zero-head control was queued after the full fast run. This freed GPUs2/3 to join the full comparison. The orchestration failure and retry are explicit; no training result was discarded. Final future PID waits must treat zombie processes as finished, or reap children concurrently.


## Final comparison and mechanism decision

All six full fitting cohorts finish: 444 protocol entries, plus the 74-checkpoint fixed-prior comparison. The fast 4k/high-matmul/four-time prototype reaches 55.01 s with 2.96198/3.09201 mm, 0.811434 and 0.435252. It meets raw pooled timing, but brain Dice/HD95 lose against the original, so the joint accuracy-preservation goal is not claimed. The default configuration is unchanged.

The late matched additive control gives OASIS pair 4 Dice of 0.79730 for the zero head and 0.79028 for the opposite-head initialization. Both use the same frozen coarse checkpoint.


The completion audit checked case identities, seed 42, PCG convergence and finite pair metrics. See the [research report](LONG_RESEARCH_REPORT.md).

### Closing source cleanup

The final audit removed unused gradient-augmented recovery code and its two tests. Brightness-only recovery remained in use. The CPU suite passed 108 checks and skipped 13 CUDA checks.
