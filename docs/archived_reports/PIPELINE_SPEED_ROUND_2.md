# Second four-GPU SIREN optimization round

The four registrations finished in **89.1 seconds**, down from **95.2 seconds** with the version already optimized in the first round. Batch elapsed time fell 6.4%.

| Case | End to end, before → after (s) | Training, before → after (s) | End-to-end reduction |
|---|---:|---:|---:|
| DIR-Lab 01 | 81.00 → 73.06 | 73.33 → 64.90 | 9.8% |
| DIR-Lab 03 | 79.17 → 76.56 | 70.54 → 67.86 | 3.3% |
| OASIS pair 1 | 92.59 → 85.44 | 80.50 → 73.41 | 7.7% |
| AbdomenCTCT pair 2 | 93.14 → 87.14 | 80.09 → 73.97 | 6.4% |

Core benchmark order was control, candidate, candidate, control. Batch times in that order were 93.30, 89.29, 88.99 and 97.18 seconds. Each run started a fresh interpreter on each RTX 5090, with warmed compiler and filesystem caches. Elapsed time covers loading, 5,000 updates, applicable endpoint reconstruction, frozen recovery, scoring and output serialization. It also includes process startup and exit. These measurements cover four representative pairs.

The exact final source completed an additional four-case batch in 88.53 seconds. A final guard change restores immediate LU errors for standalone diagnostic calls and retains deferred checking for joint training. This confirmation is outside the counterbalanced means.

Each registration used the same case, seed 42, FP32 precision, configuration, GPU assignment and launch concurrency in each version. DIR-Lab used the default native width-256 model. The two Learn2Reg cases used the native width-512 captured GW-NCC, orthogonal-query, Sobol-time recipe. All runs used /venv/main. [First-round benchmark](PIPELINE_SPEED_ROUND.md) · [Local measurements and source hashes](../results/round2/comparison.json) · [Four-GPU harness](../.local/round2/batch.py)

## Width-512 training

On OASIS pair 1 the initial real training trace measured 14.33 ms per update, with 14.23 ms in device activity. Three FP32 matrix-multiply kernel families accounted for 8.76 ms. The spatial Hessian and mixed space-time derivative previously propagated multiple reverse tangent lanes through the hidden weights. The new calculation follows one reverse adjoint, accumulates each sine layer’s curvature, and uses spatial symmetry to compute each independent entry once. It also accumulates the mixed terms from time-varying affine weights. The endpoint pass fuses the narrow three-coordinate input projection and scalar output reduction. Autograd retains all parameter, coordinate and time dependencies. These specializations apply to the captured 3D CUDA FP32 width-512 path.

With fixed inputs and all 20 parameter gradients, the combined objective and backward fell from 14.24 to 12.86 ms (9.7%). Each time is the median of five groups of 20 CUDA graph replays. Pairing endpoint matrix multiplies into a batch increased this workload to 18.86 ms. Splitting the endpoint sample rows and enabling library-wide GEMM tuning also increased time. [Original kernel profile](../results/round2/width512/profile.json) · [Combined microbenchmark](../results/round2/width512/combined.json)

On the same GPU and real OASIS pair, a separate complete training-step check measured 14.83 → 13.33 ms per update (10.1% less). The windows include sampling, backward, gradient clipping, logging conversion and the optimizer. Each median covers three 80-step windows, followed by a ten-update device trace. [Control](../results/round2/width512/full_step_before.json) · [Final](../results/round2/width512/full_step_final.json)

## DIR-Lab host waits

Two independent three-update traces of case 01 each showed 27 → 3 CUDA stream synchronizations, 21 → 3 scalar device-to-host copies and 6 → 0 host-to-device copies. Accumulated stream wait fell from 0.767–0.781 ms to 0.016–0.025 ms per trace. The single remaining wait per ordinary update checks trajectory finiteness, loss and the pre-clipping gradient norm before the optimizer. The compiled shared-cell LU solve marks failed cells; all cells contribute to the checked PD-LK energy. On-device endpoint-mask selection preserves the two native random draws, and a case-local graph replays coordinate conversion and intensity gathers. The median across four 80-update windows fell from 15.24 to 13.82 ms. [Control trace counts](../results/round2/dirlab/before.json) · [Optimized trace counts](../results/round2/dirlab/graphed.json)

## Gradient and score checks

For the combined width-512 objective on identical synthetic inputs, total loss matched, the largest component loss difference was 1.19e-7, relative L2 across all unclipped gradients was 5.27e-7 and the largest relative L2 for any parameter tensor was 8.87e-7. Independent float64 nested-autograd tests cover both endpoint models, coordinates and time. For the DIR-Lab path, 128 masked crop draws gave identical origins, indices and final Philox state. On three full objectives the loss matched exactly and the largest relative L2 difference across the concatenated parameter gradients was 1.80e-8. [DIR-Lab gradient checks](../results/round2/dirlab/training_parity.log)

On the fitted control checkpoints for OASIS and abdomen, both source versions received identical images, model parameters and sampler outputs. Three seed-42 draws per case covered steps 501, 2501 and 5000. Across six full GW-NCC objectives, the largest individual loss-term difference was 2.65e-06, the largest relative L2 across concatenated gradients was 1.24e-04 and the largest per-tensor relative L2 was 1.87e-03. [OASIS fitted gradients](../results/round2/width512/oas1-fitted-gradient.json) · [Abdomen fitted gradients](../results/round2/width512/abd2-fitted-gradient.json)

On OASIS the largest fitted full-gradient relative L2 was 2.21e-6. Abdomen produced the largest difference, 1.24e-4. Its worst per-tensor relative error was 1.87e-3 on a scalar output bias; the absolute bias-gradient difference was 2.76e-7. Maximum absolute total-loss difference across both cases was 1.31e-7.

Rerunning the starting version twice on the same three abdomen inputs gave a maximum full-gradient relative L2 difference of 4.30e-06. [Control repeat](../results/round2/width512/control_repeat/abd2-fitted-gradient.json)

CPU reference and production CUDA scoring on identical real recovered maps agreed exactly for label-wise Dice, voxel HD95, shifted folding counts and unshifted folding fractions. The SDlogJ difference was zero on OASIS and 2.78e-17 on abdomen. DIR-Lab whole-grid and lung folding matched exactly. The largest difference across both directions’ 300-landmark statistics was 4.45e-15. The benchmark source hashes cover both scorers. [DIR-Lab](../results/round2/scoring/same_map.json) · [OASIS](../results/round2/scoring/oasis.json) · [Abdomen](../results/round2/scoring/abdomenctct.json)

The same-map check was repeated on the exact final-source checkpoints for all three protocols. All label scores and folding counts again matched. Maximum SDlogJ error was 0.00e+00, and the largest DIR-Lab statistic difference was 1.78e-15. [Final DIR-Lab](../results/round2/scoring_final/same_map.json) · [Final OASIS](../results/round2/scoring_final/oasis.json) · [Final abdomen](../results/round2/scoring_final/abdomenctct.json)

Fresh 5,000-step CUDA fits vary even with seed 42. The scores below average the two full fits per version. DIR-Lab reports rounded TRE separately in each direction. Learn2Reg HD95 is in voxels.

| Case | Metric | Before | After |
|---|---|---:|---:|
| DIR-Lab 01 | Forward TRE (mm) | 1.18795 | 1.19173 |
| DIR-Lab 01 | Reverse TRE (mm) | 1.22558 | 1.23497 |
| DIR-Lab 03 | Forward TRE (mm) | 2.66690 | 2.72555 |
| DIR-Lab 03 | Reverse TRE (mm) | 2.68464 | 2.75763 |
| OASIS pair 1 | Dice | 0.82863 | 0.83049 |
| OASIS pair 1 | HD95 | 1.67145 | 1.64493 |
| OASIS pair 1 | SDlogJ | 0.07587 | 0.07575 |
| AbdomenCTCT pair 2 | Dice | 0.30824 | 0.29821 |
| AbdomenCTCT pair 2 | HD95 | 22.60804 | 23.14412 |
| AbdomenCTCT pair 2 | SDlogJ | 0.16281 | 0.16315 |

Abdomen pair 2 had lower mean Dice and higher HD95 after retraining. DIR-Lab 03 also had higher TRE in both directions, while OASIS pair 1 had higher Dice and lower HD95. Two fits per version at one seed on these four pairs establish timing and same-input numerical agreement. Full-cohort accuracy equivalence remains unverified.

All 192 PCG intervals in the timed comparison converged. Maximum measured peak PyTorch allocation across these cases was 2.52 GiB before and 2.52 GiB after. Each case produced scalar metrics and an endpoint checkpoint. Dense recovery maps stayed in memory or temporary recovery scratch.

35 focused synthetic tests passed on the final source with CUDA enabled. Coverage includes masked crop RNG, immediate and deferred LU fault handling, full SIREN derivatives and gradients, captured sampling, native RK2 and official-reference scoring. [Test output](../results/round2/final_tests.log)
