# Our method: OASIS results

Collected 2026-11-12. All **19/19 official public-validation directed pairs** completed successfully, with all metrics finite. One seed (42), 5000 training steps, paired SIRENs width 256/depth 4/omega 30; MSE + full-gradient shared-grid PD-LK weight 0.1 + box NCC weight 0.01; default crop fraction 0.5; PCG+RK2 recovery. Labels 1–35 are evaluation-only.

## Provenance

- Run directory: `results/oasis_ours_20261112_123205/` (ignored generated artifacts).
- `plan.json` records a shared queue across GPUs 0–3; `configs/default.json` freezes configuration. Per-pair directories contain scores, permitted checkpoints and run provenance; adjacent logs contain training reports.
- Summary SHA256: `a97805d8119de5f705bfd61076e584fd3225a0bc08d2e3e650d3de10b8b0c794`.
- Evaluation follows [OASIS protocol](OASIS.md). HD95 is in official voxel units, numerically mm at this release’s 1 mm spacing. SDlogJ uses the official shifted/clipped definition. Means below average pair-level scores; ± values are population SD across the 19 pairs.

## Registration and external comparison

| Method | DSC ↑ | HD95 ↓ | SDlogJ ↓ | Scored |
| --- | ---: | ---: | ---: | ---: |
| Ours | 0.7599 ± 0.0309 | 2.2422 ± 0.4932 | 0.0504 ± 0.0056 | 19/19 |
| convexadam | 0.7633 ± 0.0304 | 2.3260 ± 0.5419 | 0.0596 ± 0.0028 | 19/19 |
| fireants | 0.7437 ± 0.0416 | 2.4643 ± 0.6291 | 0.0294 ± 0.0015 | 19/19 |
| nir | 0.7089 ± 0.0615 | 3.1032 ± 0.9100 | 0.1864 ± 0.0150 | 19/19 |

External presets and limitations: [external baseline report](EXTERNAL_BASELINES.md). This single-seed comparison does not establish statistical significance.

## Reconstruction and runtime

PSNR is computed exhaustively from the **final fitted endpoint models** over the whole native volume, using peak 1 after independent min/max normalization. These are not training-minibatch estimates, lung-only scores or warped-image similarity. Reported PSNR is the mean of pair-level dB scores.

- Moving endpoint PSNR (dB): **34.15 ± 0.53**.
- Fixed endpoint PSNR (dB): **34.06 ± 0.53**.
- End-to-end time (s): **127.00 ± 2.87**.
- Mean Training (including first-use compilation): **93.66 s**.
- Mean Whole-volume reconstruction scoring: **0.19 s**.
- Mean Recovery plus evaluation: **29.05 s**.

Total serial-equivalent job time: **40.22 min**. Shared-queue GPU work critical path: **10.61 min**, excluding launcher overhead.

## Recovery diagnostics

- PCG intervals meeting configured tolerance: **57/304**. Successful execution is distinct from convergence.
- Mean unshifted interior folding fraction: **0.00000000**; maximum pair fraction **0.00000000**.

## Per-pair results

| Fixed → moving | DSC | HD95 | SDlogJ | Moving PSNR | Fixed PSNR | Time (s) | PCG converged |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0395_to_0396 | 0.7873 | 1.930 | 0.0429 | 34.44 | 34.46 | 129.8 | 1/16 |
| 0396_to_0397 | 0.7221 | 2.832 | 0.0543 | 34.45 | 34.23 | 130.7 | 0/16 |
| 0397_to_0398 | 0.7618 | 2.377 | 0.0460 | 35.01 | 34.63 | 130.0 | 0/16 |
| 0398_to_0399 | 0.7098 | 3.455 | 0.0575 | 34.09 | 34.66 | 132.4 | 2/16 |
| 0399_to_0400 | 0.7484 | 1.993 | 0.0474 | 34.58 | 34.22 | 126.0 | 4/16 |
| 0400_to_0401 | 0.7619 | 2.029 | 0.0471 | 34.65 | 34.66 | 124.9 | 2/16 |
| 0401_to_0402 | 0.7053 | 3.131 | 0.0655 | 34.49 | 34.08 | 129.5 | 2/16 |
| 0402_to_0403 | 0.7777 | 2.028 | 0.0577 | 34.17 | 34.73 | 124.3 | 4/16 |
| 0403_to_0404 | 0.8071 | 2.031 | 0.0474 | 34.26 | 34.15 | 122.2 | 3/16 |
| 0404_to_0405 | 0.7365 | 2.878 | 0.0517 | 34.17 | 33.83 | 125.5 | 1/16 |
| 0405_to_0406 | 0.7525 | 2.200 | 0.0456 | 34.15 | 34.27 | 121.8 | 4/16 |
| 0406_to_0407 | 0.7629 | 2.105 | 0.0466 | 33.53 | 34.12 | 127.5 | 1/16 |
| 0407_to_0408 | 0.7068 | 2.641 | 0.0583 | 33.02 | 33.27 | 125.7 | 4/16 |
| 0408_to_0409 | 0.8017 | 1.705 | 0.0462 | 33.37 | 33.19 | 126.6 | 6/16 |
| 0409_to_0410 | 0.7921 | 1.718 | 0.0464 | 34.49 | 33.31 | 130.2 | 2/16 |
| 0410_to_0411 | 0.7820 | 1.728 | 0.0493 | 33.73 | 34.53 | 129.2 | 4/16 |
| 0411_to_0412 | 0.7852 | 1.909 | 0.0489 | 34.53 | 33.58 | 126.4 | 3/16 |
| 0412_to_0413 | 0.7799 | 2.036 | 0.0485 | 33.13 | 34.22 | 125.4 | 9/16 |
| 0413_to_0414 | 0.7580 | 1.876 | 0.0497 | 34.52 | 33.03 | 124.8 | 5/16 |

Recovery stage breakdown and operator bottlenecks: [OASIS recovery profile](OASIS_RECOVERY_PROFILE.md).
