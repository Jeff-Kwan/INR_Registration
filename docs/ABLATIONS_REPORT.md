# SIRENPath ablation aggregate report

Completed 16 November 2026: **133/133 evaluations succeeded**, zero process failures. All seven runs cover the same **19 official directed L2R-OASIS validation pairs**, seed42. Dataset preparation was frozen. There are95 trained fits and38 no-training controls. Every requested metric has19/19 case coverage.

Values are **mean ± population SD**. PSNR and Patch NCC first average the fixed/moving endpoint scores within each pair.

| Run | PSNR (dB) ↑ | Patch NCC ↑ | DSC ↑ | HD95 (voxels) ↓ | NPJ (%) ↓ | SDlogJ ↓ | Runtime (s/case) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1. Reconstruction only | 35.86 ± 0.47 | 0.9268 ± 0.0049 | 0.5368 ± 0.0419 | 4.7110 ± 0.8761 | 0.000047 ± 0.000057 | 0.4976 ± 0.0367 | 51.23 ± 6.44 |
| 2. Transport only | 28.87 ± 0.63 | 0.7995 ± 0.0204 | 0.6979 ± 0.0378 | 2.8212 ± 0.5548 | 0.000000 ± 0.000000 | 0.2564 ± 0.0264 | 79.06 ± 9.80 |
| 3. Registration only | 32.20 ± 0.32 | 0.8702 ± 0.0103 | 0.7015 ± 0.0426 | 2.8571 ± 0.6799 | 0.000335 ± 0.000324 | 0.4255 ± 0.0447 | 81.70 ± 8.13 |
| 4. Both losses | 29.28 ± 0.91 | 0.8044 ± 0.0209 | 0.7356 ± 0.0319 | 2.4232 ± 0.5111 | 0.000041 ± 0.000173 | 0.2518 ± 0.0262 | 78.39 ± 1.20 |
| 5. SIRENPath | 6.72 ± 0.59 | 0.9429 ± 0.0043 | 0.8147 ± 0.0210 | 1.8155 ± 0.4313 | 0.001073 ± 0.000647 | 0.3425 ± 0.0258 | 80.50 ± 7.50 |
| 6. Raw intensity crossfade | ∞ (exact endpoints) | 1.0000 ± 0.0000 | 0.7083 ± 0.0434 | 2.8228 ± 0.5720 | 0.000000 ± 0.000000 | 0.1720 ± 0.0175 | 3.39 ± 0.09 |
| 7. Canonical reconstructions crossfade | 6.72 ± 0.59 | 0.9429 ± 0.0043 | 0.7667 ± 0.0360 | 2.4542 ± 0.6023 | 0.005071 ± 0.003101 | 0.3913 ± 0.0374 | 5.71 ± 3.19 |

Runs1–4 use MSE endpoint fitting: reconstruction only, PD-LK only, registration only, and both losses, respectively. Run5 is canonical SIRENPath with NCC endpoints and both losses. All trained runs keep the same AdamW decay and inter-SIREN L2; MSE runs disable the NCC-specific moment penalty. Run6 crossfades raw normalized images; run7 crossfades the exact run5 endpoint reconstructions without rescaling or retraining. All use the same stage-3 PCG/RK2 recovery.

- **Both losses help:** run4 DSC0.7356 exceeds transport-only0.6979 and registration-only0.7015.
- **Canonical SIRENPath has the highest DSC and lowest HD95:**0.8147 and1.8155 voxels.
- **Same fitted endpoints, different paths:** run5 exceeds reconstructed-image crossfade (run7) by0.0480 DSC, with HD95 lower by0.6386 voxels. Runs5/7 have exactly equal endpoint PSNR/Patch NCC. This contrast changes the interpolation path and analytical versus finite-difference derivative estimation together.

PSNR is whole-grid, peak1, without affine intensity calibration. NCC-trained output scale/offset need not match the input intensity scale, so the low uncalibrated PSNR in runs5/7 should be read alongside their high Patch NCC. Raw-control PSNR∞ and Patch NCC1 are guaranteed by endpoint identity, not fitted reconstruction quality.

Patch NCC is signed Pearson correlation over all valid9³ windows, stride1; constant target windows are omitted. HD95 retains the official L2R voxel convention (numerically mm at1mm spacing). NPJ is100×mean(detJ≤0). SDlogJ is unshifted std(log(max(detJ,1e-9))); both Jacobian metrics use the two-voxel interior. Official shifted L2R SDlogJ is additionally retained in case files.

Runtime is per-case worker wall time including loading, fitting/setup, endpoint diagnostics, recovery, scoring and checkpoint serialization, on RTX5090 GPUs. **Run7 runtime is incremental and excludes the shared run5 training cost.** Its reused training time and method runtime including shared training are retained separately. No training occurs in runs6/7.

[Study design and exact metric definitions](SIRENPATH_ABLATIONS.md) · [Machine-readable aggregates](../results/sirenpath_l2r_oasis_ablations_20261116/summary.json) · [Frozen plan/configurations](../results/sirenpath_l2r_oasis_ablations_20261116/plan.json) · [Generated run report](../results/sirenpath_l2r_oasis_ablations_20261116/ABLATIONS_REPORT.md)
