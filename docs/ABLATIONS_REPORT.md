# SIRENPath ablation aggregate report

All **285 evaluations succeeded** on the same 19 directed Learn2Reg OASIS validation pairs at seed 42. The original seven conditions finished on 16 November 2026 (133 evaluations). Eight supplementary training conditions finished on 22 November (152 fits). Together these comprise 247 trained fits and 38 crossfade evaluations. Every metric covers all 19 pairs in each condition.

Values are **mean ± population SD**. PSNR and Patch NCC first average the fixed/moving endpoint scores within each pair.

## Original study

| Run | PSNR (dB) ↑ | Patch NCC ↑ | DSC ↑ | HD95 (voxels) ↓ | NPJ (%) ↓ | SDlogJ ↓ | Runtime (s/case) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1. Reconstruction only (endpoint parameter penalty on, 0.01) | 35.86 ± 0.47 | 0.9268 ± 0.0049 | 0.5368 ± 0.0419 | 4.7110 ± 0.8761 | 0.000047 ± 0.000057 | 0.4976 ± 0.0367 | 51.23 ± 6.44 |
| 2. Local transport loss (LT) only (endpoint parameter penalty on, 0.01) | 28.87 ± 0.63 | 0.7995 ± 0.0204 | 0.6979 ± 0.0378 | 2.8212 ± 0.5548 | 0.000000 ± 0.000000 | 0.2564 ± 0.0264 | 79.06 ± 9.80 |
| 3. Trajectory-based registration only (endpoint parameter penalty on, 0.01) | 32.20 ± 0.32 | 0.8702 ± 0.0103 | 0.7015 ± 0.0426 | 2.8571 ± 0.6799 | 0.000335 ± 0.000324 | 0.4255 ± 0.0447 | 81.70 ± 8.13 |
| 4. Both losses (endpoint parameter penalty on, 0.01) | 29.28 ± 0.91 | 0.8044 ± 0.0209 | 0.7356 ± 0.0319 | 2.4232 ± 0.5111 | 0.000041 ± 0.000173 | 0.2518 ± 0.0262 | 78.39 ± 1.20 |
| 5. SIRENPath (endpoint parameter penalty on, 0.01) | 6.72 ± 0.59 | 0.9429 ± 0.0043 | 0.8147 ± 0.0210 | 1.8155 ± 0.4313 | 0.001073 ± 0.000647 | 0.3425 ± 0.0258 | 80.50 ± 7.50 |
| 6. Raw crossfade | ∞ (exact endpoints) | 1.0000 ± 0.0000 | 0.7083 ± 0.0434 | 2.8228 ± 0.5720 | 0.000000 ± 0.000000 | 0.1720 ± 0.0175 | 3.39 ± 0.09 |
| 7. Fitted crossfade | 6.72 ± 0.59 | 0.9429 ± 0.0043 | 0.7667 ± 0.0360 | 2.4542 ± 0.6023 | 0.005071 ± 0.003101 | 0.3913 ± 0.0374 | 5.71 ± 3.19 |

## Supplementary study

LT denotes the local transport loss. Trajectory-based registration combines GW-NCC with a boundary penalty. NCC + moments uses moment weight 0.005. AdamW weight decay is 0.001 throughout.

| Run | Endpoint parameter penalty weight | PSNR (dB) ↑ | Patch NCC ↑ | DSC ↑ | HD95 (voxels) ↓ | NPJ (%) ↓ | SDlogJ ↓ | Runtime (s/case) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8. NCC + moments, reconstruction | 0.01 | -0.32 ± 0.22 | 0.9564 ± 0.0042 | 0.6879 ± 0.0353 | 3.4202 ± 0.7587 | 0.821331 ± 0.129365 | 2.1704 ± 0.1454 | 46.87 ± 1.03 |
| 9. NCC + moments, LT | 0.01 | 6.72 ± 0.60 | 0.9441 ± 0.0043 | 0.8169 ± 0.0208 | 1.7947 ± 0.4216 | 0.001168 ± 0.000733 | 0.3455 ± 0.0281 | 75.52 ± 7.01 |
| 10. NCC + moments, trajectory-based registration | 0.01 | -0.29 ± 0.22 | 0.9518 ± 0.0044 | 0.7323 ± 0.0379 | 3.0574 ± 0.7993 | 0.414335 ± 0.090813 | 1.6633 ± 0.1512 | 78.42 ± 7.02 |
| 11. MSE, both losses | 0 | 29.24 ± 0.57 | 0.8044 ± 0.0188 | 0.7341 ± 0.0322 | 2.4225 ± 0.4893 | 0.000000 ± 0.000000 | 0.2526 ± 0.0221 | 79.65 ± 8.52 |
| 12. NCC + moments, both losses | 0 | 6.72 ± 0.59 | 0.9428 ± 0.0044 | 0.8146 ± 0.0213 | 1.8082 ± 0.4370 | 0.000940 ± 0.000680 | 0.3418 ± 0.0261 | 79.47 ± 6.51 |
| 13. MSE reconstruction alone | 0 | 35.83 ± 0.46 | 0.9268 ± 0.0052 | 0.5313 ± 0.0370 | 4.8415 ± 0.8579 | 0.000078 ± 0.000109 | 0.5015 ± 0.0310 | 49.58 ± 6.76 |
| 16. NCC + moments, reconstruction | 0 | -0.32 ± 0.22 | 0.9565 ± 0.0042 | 0.6873 ± 0.0345 | 3.4228 ± 0.6449 | 0.815792 ± 0.124282 | 2.1648 ± 0.1401 | 46.03 ± 0.73 |
| 19. NCC without moments, both losses | 0.01 | 16.51 ± 0.45 | 0.9482 ± 0.0047 | 0.7843 ± 0.0221 | 2.0791 ± 0.4288 | 0.000000 ± 0.000000 | 0.1859 ± 0.0123 | 79.72 ± 5.55 |

## Findings across both studies

Runs1–4 use MSE endpoint fitting: reconstruction only, LT only, registration only, and both losses, respectively. Run5 is canonical SIRENPath with NCC endpoints and both losses. Original trained runs 1–5 keep the endpoint parameter penalty weight at 0.01; MSE runs disable the NCC-specific moment penalty. Run6 crossfades raw normalized images; run7 crossfades the exact run5 endpoint reconstructions without rescaling or retraining. All use the same stage-3 PCG/RK2 recovery.

- **With MSE endpoints and the endpoint parameter penalty on,** run4 DSC0.7356 exceeds LT-only 0.6979 and registration-only0.7015.
- **Run 9 had the highest mean Dice (0.8169) and lowest mean HD95 (1.7947 voxels).** It used NCC + moments, LT and the endpoint parameter penalty, with the trajectory-based registration loss off. Canonical run 5 scored 0.8147 and 1.8155, respectively. These are descriptive differences on one seed and 19 pairs.
- **Endpoint fitting mattered across all six matched comparisons.** NCC + moments had higher mean Dice than MSE with the same LT, registration and endpoint parameter penalty settings. The matched run IDs (MSE, NCC) were (1, 8), (2, 9), (3, 10), (4, 5), (13, 16) and (11, 12).
- **The endpoint parameter penalty had little effect on mean Dice with both losses active.** Changing its weight from 0.01 to zero changed MSE Dice from 0.7356 to 0.7341 and NCC + moments Dice from 0.8147 to 0.8146. For reconstruction alone, the corresponding changes were 0.5368 to 0.5313 and 0.6879 to 0.6873.
- **Removing moment matching from the canonical recipe** lowered mean Dice from 0.8147 to 0.7843 and raised HD95 from 1.8155 to 2.0791 voxels. Run 19 also had a lower mean SDlogJ (0.1859 versus 0.3425).
- **Same fitted endpoints, different paths:** run5 exceeds Fitted crossfade (run7) by0.0480 DSC, with HD95 lower by0.6386 voxels. Runs5/7 have exactly equal endpoint PSNR/Patch NCC. This contrast changes the interpolation path and analytical versus finite-difference derivative estimation together.

PSNR is whole-grid, peak1, without affine intensity calibration. NCC-trained output scale/offset need not match the input intensity scale, so the low uncalibrated PSNR in runs5/7 should be read alongside their high Patch NCC. Raw-control PSNR∞ and Patch NCC1 are guaranteed by endpoint identity, not fitted reconstruction quality.

Patch NCC is signed Pearson correlation over all valid9³ windows, stride1; constant target windows are omitted. HD95 retains the official L2R voxel convention (numerically mm at1mm spacing). NPJ is100×mean(detJ≤0). SDlogJ is unshifted std(log(max(detJ,1e-9))); both Jacobian metrics use the two-voxel interior. Official shifted L2R SDlogJ is additionally retained in case files.

Runtime is per-case worker wall time including loading, fitting/setup, endpoint diagnostics, recovery, scoring and checkpoint serialization, on RTX5090 GPUs. **Run7 runtime is incremental and excludes the shared run5 training cost.** Its reused training time and method runtime including shared training are retained separately. No training occurs in runs6/7.

[Study design and exact metric definitions](SIRENPATH_ABLATIONS.md) · [Supplementary aggregates](../results/sirenpath_l2r_oasis_supplement/summary.json) · [Original aggregates](../results/sirenpath_l2r_oasis_ablations_20261116/summary.json) · [Frozen plan/configurations](../results/sirenpath_l2r_oasis_ablations_20261116/plan.json) · [Generated run report](../results/sirenpath_l2r_oasis_ablations_20261116/ABLATIONS_REPORT.md)
