# Datasets and classical comparisons

## Available datasets

All existing dataset paths and bytes are preserved. Dataset setup remains under the datasets package and refuses to overwrite existing destinations. [Dataset setup and acquisition documentation](../medical/datasets/README.md) describes the supported recipes.

| Preset | Pairing | Runtime intensity mapping | Role |
|---|---|---|---|
| lpba | S01 fixed; configured subjects or discovered S02–S10 | MRI nonzero-foreground percentiles 1/99 | First application focus |
| oasis | Explicit fixed record and moving list/count | MRI nonzero-foreground percentiles 1/99 | Available probe |
| hippocampus | Explicit fixed record and moving list/count | MRI nonzero-foreground percentiles 1/99 | Available probe |
| abdomenct | Explicit fixed record and moving list/count | Configured HU window | Available probe |
| lungct | Each case's own expiration/inspiration channels | Configured HU window | Available paired loader; historical failed cohort remains qualified |

Released probe geometry is verified rather than reapplied. Mismatched grids raise. LPBA prepared-data provenance is unresolved; neither setup nor loading invents raw-release preprocessing. Annotation loading is separate from image loading. Lung masks and the subset of available manual landmarks remain evaluation-only.

`pairs` lists exact identifiers; `pairs --verify-data` validates selected image grids, mappings, and annotations without fitting. No dataset scan is part of the synthetic validation suite.

## Maintained baselines

ANTs SyN is the primary reference. Elastix B-spline and SimpleITK diffeomorphic Demons are also supported. All methods receive exactly the same selected image arrays and runtime intensity mappings and use the same metric implementation.

The packaged settings preserve the existing protocol: ANTs SyNOnly with identity initialization, CC radius 2, iterations [50,50,30], flow sigma 3, total sigma 0, and step .2; elastix's library default affine+B-spline parameter maps with normalized correlation; Demons with 100 iterations and standard deviations 1. Elastix includes an affine stage whereas SyNOnly does not. Record that distinction and toolkit versions when interpreting comparisons; these are named local protocols, not universal optimal settings.

Toolkits receive a zero-origin, axis-aligned canonical grid with the original orthogonal voxel spacing. This common change of physical frame does not resample images. Adapters convert complete transforms to fixed-grid inverse displacements in (W,H,D) voxel units; ANTs composes its full returned image-warp chain. Evaluation uses the original NIfTI affine. Toolkit transforms live only in temporary directories.

The forward map for classical diagnostics is obtained by numerical inversion of the inverse displacement. Its approximate status and convergence are recorded. A failed inversion must not be presented as an exact inverse or hidden behind a successful image-similarity score.

The configured runtime seed controls SIREN fitting. Classical adapters retain toolkit randomness and threading defaults; deterministic classical reproduction is not claimed. Versions and explicit algorithm settings are recorded so this limitation remains visible.

## Current packaged training objective (2026-11-11)

All five dataset configs now default to MSE endpoint reconstruction, strict shared-grid PD-LK (`objective.weight=0.1`, `rho=0`), and bidirectional local NCC registration (`gaussian_registration.weight=0.05`). The local surrogate uses the selected 8³/two-pair, half-size-crop configuration with random field times and three-tap smoothing. Because it activates on the first optimizer update, the packaged configs explicitly skip the early PSNR gate; PSNR is still reported diagnostically and the gate status remains null. This default was selected from existing LPBA S02/S03 comparisons, not from new training in this preset update. The prior MSE+PD-LK-only configs remain available as `medical/configs/<dataset>_mse_pdlk.json` and retain the PSNR gate.

## Fresh benchmark settings (2026-11-10)

The fresh cohorts use the maintained SIREN architecture, shared initialization, linear parameter interpolation, endpoint reconstruction, strict standard PD-LK (`rho=0`), and the separate 16-interval optical-flow/RK2 recovery. Every pair runs at seed 42. Only the dataset-dependent PD-LK weight is pinned per cohort, from retained configs and historical reports:

| Cohort | `objective.weight` | Retained provenance |
|---|---:|---|
| LPBA40 (S02–S10 vs S01) | 0.1 | Maintained preset; archived full-cohort reference Dice 0.698714 / HD95 6.3175 mm at seed 42 (`docs/archive/methods/BEST_RESULTS.md`), reproduced as 0.6982 / 6.327 (`docs/archive/methods/ARCHITECTURES.md`) |
| OASIS (0002–0011 vs 0001) | 0.1 | Archived calibrated full cohort: SIREN Dice 0.7937 / HD95 2.568 mm, 10/10 gate pass; the probe-benchmark 0.01 setting gave 0.766 / 2.87 (`docs/archive/methods/ARCHITECTURES.md`, `docs/archive/medical/probe_benchmark_analysis.md`) |
| Hippocampus (0002–0021 vs 0001) | 0.01 | Probe benchmark Dice 0.752 / HD95 3.20 mm, 20/20; the weight-0.1 calibration pair collapsed to ~5.4 dB and failed the gate (`docs/archive/medical/probe_benchmark_analysis.md`) |
| Abdomen CT (0002–0011 vs 0001) | 0.01 | Probe benchmark Dice 0.3545 / HD95 43.35 mm, 10/10; the architecture harness reported 0.3555 / 43.68 at the same weight. Weight 0.1 fails the endpoint gate on this cohort (`docs/archive/medical/probe_benchmark.json`, `docs/archive/methods/ARCHITECTURES.md`) |

Common settings are the packaged presets: SIREN width 256, four sine layers, `omega=30`; 5,000 Adam steps, batch 32,768, learning rate 2e-4 with final factor 0.25, gradient clip 1.0; `c=4`, `eta=1e-6`, `rho=0`, 4,096 PD-LK points; recovery with 16 intervals, smoothness 0.01, ridge 1e-6, 500 PCG iterations, relative tolerance 2e-5, and four midpoint-RK2 substeps; endpoint gate 25 dB PSNR; MONAI macro Dice and symmetric HD95 in millimetres.

These weights are the prior MSE+PD-LK-only baseline settings, inherited from different grids and harnesses; they are not a cross-dataset optimum. The fresh runs used those values before the packaged default was changed. They remain selectable with the corresponding `*_mse_pdlk.json` configs, and are recorded here and in each run's saved `config.json` and `manifest.json`. Historical numbers in this document and in `docs/RESULTS.md` are not pooled with the fresh measurements.

## Five-cohort default benchmark and shared-configuration sweep (2026-11-12)

The packaged shared-grid default was rerun on all five cohorts at revision `4055b07` (dirty
worktree), seed 42, with the five packaged JSON configs unchanged. New SIREN means (Dice / HD95
mm) are LPBA40 0.7112 / 6.14, OASIS 0.7660 / 2.87, Hippocampus 0.7711 / 3.11, Abdomen CT
0.4104 / 40.26 and Lung CT 0.8123 / 48.52; Lung CT remains a failed cohort and is not pooled
into a headline average. Reused classical rows come from the 2026-11-10 run and are marked as
such. A pre-declared 27-candidate sweep over `c`, local box fraction and recovery smoothness
retained the packaged `(4, 0.5, .01)` configuration; the [report](reports/default_method_baselines_20261112.md)
records the tuning subset, cohort-balanced criterion, per-pair tables and the decision.

## Reporting

Record pair membership, preprocessing, config, package versions, initialization stages, seed, failures, image similarity, Dice/HD95, folding, outside sampling, composition error, and elapsed time. SIREN reports additionally include endpoint gates and interval-wise flow convergence. Endpoint PSNR is inapplicable to classical methods.

Reports retain denominators and scored pair IDs. Compare the same cohorts and show gate failures instead of ranking favorable subsets. These released training-label probes are not hidden-test Learn2Reg leaderboard evaluations. No superiority claim follows from the existing single-seed results.

Metric version 2 corrects the old classical outside-sampling check to use absolute coordinates and computes composition distances using the full affine basis. Historical metrics are preserved without relabeling or recomputation. No new benchmark was run as part of the 2026-11-10 overhaul; a subsequent fresh full-cohort benchmark at revision `44f1ca2` is recorded in the [centralized report](reports/benchmark_20261110.md) and summarized in [qualified evidence](RESULTS.md).
