# Second four-GPU SIREN optimization round

The four registrations finished in **89.1 seconds**, down from **95.2 seconds** with the version already optimized in the first round. Batch elapsed time fell 6.4%.

| Case | End to end, before → after (s) | Training, before → after (s) | End-to-end reduction |
|---|---:|---:|---:|
| OASIS pair 1 | 92.59 → 85.44 | 80.50 → 73.41 | 7.7% |

Core benchmark order was control, candidate, candidate, control. Batch times in that order were 93.30, 89.29, 88.99 and 97.18 seconds. Each run started a fresh interpreter on each RTX 5090, with warmed compiler and filesystem caches. Elapsed time covers loading, 5,000 updates, applicable endpoint reconstruction, frozen recovery, scoring and output serialization. It also includes process startup and exit. These measurements cover four representative pairs.

The exact final source completed an additional four-case batch in 88.53 seconds. A final guard change restores immediate LU errors for standalone diagnostic calls and retains deferred checking for joint training. This confirmation is outside the counterbalanced means.

Each registration used the same case, seed 42, FP32 precision, configuration, GPU assignment and launch concurrency in each version. The two Learn2Reg cases used the native width-512 captured GW-NCC, orthogonal-query, Sobol-time recipe. All runs used /venv/main. [First-round benchmark](PIPELINE_SPEED_ROUND.md) · [Local measurements and source hashes](../results/round2/comparison.json) · [Four-GPU harness](../.local/round2/batch.py)

## Width-512 training

On OASIS pair 1 the initial real training trace measured 14.33 ms per update, with 14.23 ms in device activity. Three FP32 matrix-multiply kernel families accounted for 8.76 ms. The spatial Hessian and mixed space-time derivative previously propagated multiple reverse tangent lanes through the hidden weights. The new calculation follows one reverse adjoint, accumulates each sine layer’s curvature, and uses spatial symmetry to compute each independent entry once. It also accumulates the mixed terms from time-varying affine weights. The endpoint pass fuses the narrow three-coordinate input projection and scalar output reduction. Autograd retains all parameter, coordinate and time dependencies. These specializations apply to the captured 3D CUDA FP32 width-512 path.

With fixed inputs and all 20 parameter gradients, the combined objective and backward fell from 14.24 to 12.86 ms (9.7%). Each time is the median of five groups of 20 CUDA graph replays. Pairing endpoint matrix multiplies into a batch increased this workload to 18.86 ms. Splitting the endpoint sample rows and enabling library-wide GEMM tuning also increased time. [Original kernel profile](../results/round2/width512/profile.json) · [Combined microbenchmark](../results/round2/width512/combined.json)

On the same GPU and real OASIS pair, a separate complete training-step check measured 14.83 → 13.33 ms per update (10.1% less). The windows include sampling, backward, gradient clipping, logging conversion and the optimizer. Each median covers three 80-step windows, followed by a ten-update device trace. [Control](../results/round2/width512/full_step_before.json) · [Final](../results/round2/width512/full_step_final.json)

## Gradient and score checks

For the combined width-512 objective on identical synthetic inputs, total loss matched, the largest component loss difference was 1.19e-7, relative L2 across all unclipped gradients was 5.27e-7 and the largest relative L2 for any parameter tensor was 8.87e-7. Independent float64 nested-autograd tests cover both endpoint models, coordinates and time.

Three seed-42 draws per case covered steps 501, 2501 and 5000.

On OASIS the largest fitted full-gradient relative L2 was 2.21e-6. Its worst per-tensor relative error was 1.87e-3 on a scalar output bias; the absolute bias-gradient difference was 2.76e-7. Maximum absolute total-loss difference across both cases was 1.31e-7.

CPU reference and production CUDA scoring on identical real recovered maps agreed exactly for label-wise Dice, voxel HD95, shifted folding counts and unshifted folding fractions.

The same-map check was repeated on the exact final-source checkpoints for all three protocols. All label scores and folding counts again matched.

Fresh 5,000-step CUDA fits vary even with seed 42. The scores below average the two full fits per version. Learn2Reg HD95 is in voxels.

| Case | Metric | Before | After |
|---|---|---:|---:|
| OASIS pair 1 | Dice | 0.82863 | 0.83049 |
| OASIS pair 1 | HD95 | 1.67145 | 1.64493 |
| OASIS pair 1 | SDlogJ | 0.07587 | 0.07575 |

Two fits per version at one seed on these four pairs establish timing and same-input numerical agreement. Full-cohort accuracy equivalence remains unverified.

All 192 PCG intervals in the timed comparison converged. Maximum measured peak PyTorch allocation across these cases was 2.52 GiB before and 2.52 GiB after. Each case produced scalar metrics and an endpoint checkpoint. Dense recovery maps stayed in memory or temporary recovery scratch.

35 focused synthetic tests passed on the final source with CUDA enabled. Coverage includes masked crop RNG, immediate and deferred LU fault handling, full SIREN derivatives and gradients, captured sampling, native RK2 and official-reference scoring. [Test output](../results/round2/final_tests.log)
