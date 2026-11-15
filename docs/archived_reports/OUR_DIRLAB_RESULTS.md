# Our method: DIR-Lab results

Collected 2026-11-12. Single seed 42; 5000 steps; width 256, four hidden layers, omega 30. Endpoint MSE + shared-grid full-gradient PD-LK weight 0.1 + box NCC weight 0.01; PCG+RK2 recovery. Separate T00/T50 R231 masks remain active.

## Uncropped run

Source: `results/dirlab_ours_nocrop_20261112_121125/`. All **10/10** cases completed successfully. Frozen configuration is `config.json`; case scores/checkpoints, per-case logs and elapsed times are retained in ignored results. Primary point map is T00→T50; reverse TRE is reported separately. Each case contains 300 landmarks.

The only configuration change from the earlier run is `crop_fraction: null` instead of `0.5`. This selects the existing full-volume mode, which also enables its existing out-of-bounds boundary term; it is not an otherwise identical spatial-sampling ablation. Masks, seed, loss weights and model/recovery settings are unchanged.

TRE below is in mm. Rounded and continuous conventions stay distinct. Aggregate means average case means; SD below is across cases, not pooled landmark spread.

| Case | Initial TRE | Forward rounded | Forward continuous | Reverse rounded | Time (s) | Lung folding | PCG converged |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 01 | 3.892 | 2.008 | 1.980 | 2.002 | 134.4 | 0.000000 | 0/16 |
| 02 | 4.338 | 2.731 | 2.557 | 2.746 | 142.0 | 0.000000 | 0/16 |
| 03 | 6.943 | 3.414 | 3.367 | 3.494 | 138.3 | 0.000000 | 0/16 |
| 04 | 9.830 | 5.533 | 5.536 | 5.876 | 134.5 | 0.000000 | 0/16 |
| 05 | 7.477 | 4.388 | 4.436 | 4.502 | 139.0 | 0.000000 | 0/16 |
| 06 | 10.891 | 12.405 | 12.361 | 13.382 | 281.2 | 0.000000 | 0/16 |
| 07 | 11.026 | 6.626 | 6.675 | 6.741 | 292.9 | 0.000000 | 0/16 |
| 08 | 14.995 | 13.514 | 13.503 | 13.803 | 278.4 | 0.000000 | 0/16 |
| 09 | 7.918 | 6.425 | 6.444 | 6.991 | 279.4 | 0.000000 | 0/16 |
| 10 | 7.301 | 11.652 | 11.605 | 12.313 | 268.6 | 0.000000 | 0/16 |

## Aggregate and external comparison

| Method | Mean rounded forward TRE (mm) | Scored cases |
| --- | ---: | ---: |
| convexadam | 1.402 | 10/10 |
| nir | 1.561 | 10/10 |
| fireants | 4.040 | 10/10 |
| Ours, uncropped | 6.870 | 10/10 |

- Initial TRE: **8.461 ± 3.160 mm**, mean ± case SD.

- Forward rounded TRE: **6.870 ± 3.982 mm**, mean ± case SD.

- Forward continuous TRE: **6.846 ± 3.991 mm**, mean ± case SD.

- Reverse rounded TRE: **7.185 ± 4.217 mm**, mean ± case SD.

- Cases with mean rounded forward TRE >2 mm: **10/10**.
- Recovery PCG convergence: **0/160 intervals**. Completion does not imply PCG convergence.
- Mean end-to-end case time: **3.48 min**; cases 01–05: **2.29 min**, 06–10: **4.67 min**.
- Serial-equivalent case time: **34.81 min**. The three-worker static assignment had a critical-path sum of **13.84 min**, excluding launcher overhead.

## Original cropped run

Source: `results/dirlab_ours_20261112_120047/`. Cases 05 and 09 failed because a sampled crop had no moving-mask particle sites; **8/10** completed. Successful-case averages are not full-cohort averages.

| Case | Cropped forward TRE | Uncropped forward TRE |
| --- | ---: | ---: |
| 01 | 1.933 | 2.008 |
| 02 | 2.768 | 2.731 |
| 03 | 3.432 | 3.414 |
| 04 | 4.423 | 5.533 |
| 05 | Failed crop sampling | 4.388 |
| 06 | 4.893 | 12.405 |
| 07 | 5.540 | 6.626 |
| 08 | 8.362 | 13.514 |
| 09 | Failed crop sampling | 6.425 |
| 10 | 4.520 | 11.652 |

On the same eight completed cropped cases: cropped mean **4.484 mm**, uncropped mean **7.235 mm**. Uncropping resolves the observed sampling crashes; these measurements alone do not demonstrate an accuracy improvement.

External preset limitations and full results: [external baselines](EXTERNAL_BASELINES.md). No training or metric settings were tuned during this collection.
