# Fresh medical-registration benchmark: LPBA40, OASIS, Hippocampus MR, Abdomen CT

**Report date:** 2026-11-10 (Asia/Hong_Kong)
**Code revision:** `44f1ca2bda8eddec4023c35d1a3a8270f328746c` (worktree clean at launch)
**Environment:** torch 2.14.1+cu132, numpy 2.3.5, nibabel 5.4.2, MONAI 1.6.1, antspyx 0.6.3, itk-elastix 0.25.4, SimpleITK 2.5.7; one NVIDIA GeForce RTX 5090 (32 GB), 32 CPU cores.
**Cohorts:** LPBA40 (9 pairs, S02–S10 vs S01), OASIS (10 pairs, 0002–0011 vs 0001), Hippocampus MR (20 pairs, 0002–0021 vs 0001), Abdomen CT (10 pairs, 0002–0011 vs 0001). Lung is excluded.
**Seed:** 42 for every SIREN fit.

## Evidence status

This is the centralized record for the fresh benchmark. Every SIREN and classical number in the cohort and per-pair tables was produced by the fresh runs under `results/benchmark_20261110/` on 2026-11-09/10 UTC. Historical values appear only in the explicitly labelled context table and are not pooled with the fresh measurements.

The runs use the maintained pipeline unchanged: paired spatial SIRENs with shared initialization and linear parameter interpolation, endpoint MSE reconstruction, strict standard pointwise PD-LK (`rho=0`), then a separate recovery that solves the globally regularized brightness-flow equation at 16 interval midpoints with PCG and integrates with midpoint RK2 (four substeps) forwards and backwards. Classical methods receive exactly the same selected image arrays, preprocessing, pair membership, affine, and evaluation code.

Before the cohort runs, the implementation was validated end-to-end on one pair per dataset and the three classical adapters on one LPBA pair. Those preliminary runs, originally under `results/validation/`, are preserved locally under `.local_archive/benchmark_cleanup_20261110/validation/` with an original-path/hash manifest. They are not part of the reported denominators.

## Protocol

### SIREN PD-LK

| Setting | Value |
|---|---|
| Model | SIREN width 256, four sine layers, `omega=30`, scalar linear readout; two endpoints with shared initialization |
| Trajectory | `theta(t) = (1-t) theta_moving + t theta_fixed` |
| Endpoint loss | Sum of moving and fixed intensity MSE on a shared 32,768-voxel minibatch |
| PD-LK | Pointwise independent 3×3 local solve at 4,096 queries and one shared uniform time; `c=4`, `eta=1e-6`, `rho=0` |
| Optimization | 5,000 Adam steps, learning rate 2e-4 with final factor 0.25, gradient clip 1.0, seed 42 |
| Recovery | 16 intervals, analytic SIREN `g,q`, PCG (`s=0.01`, `eta_r=1e-6`, max 500 iterations, relative tolerance 2e-5), midpoint RK2 with 4 substeps, inverse by reverse-time integration |
| Endpoint gate | Both full-volume reconstructions must reach 25 dB PSNR; a gate failure skips registration |
| Evaluation | MONAI macro Dice over the union of nonzero labels (`ignore_empty=False`); symmetric HD95 in mm (max of directed 95th-percentile surface distances); folding is the fraction of voxels with Jacobian determinant ≤ 0 |

### Pinned per-dataset coupling

Only the PD-LK weight differs by cohort, recovered from retained configs and reports. The values are recorded in each run's saved `config.json` and `manifest.json`; the provenance is also in `docs/BENCHMARKS.md`.

| Cohort | `objective.weight` | Provenance |
|---|---:|---|
| LPBA40 | 0.1 | Maintained preset; archived full-cohort standard-PD-LK reference 0.698714 Dice / 6.3175 mm at seed 42 (`docs/archive/methods/BEST_RESULTS.md`), reproduced at 0.6982 / 6.327 (`docs/archive/methods/ARCHITECTURES.md`) |
| OASIS | 0.1 | Archived calibrated full OASIS cohort: SIREN 0.7937 Dice / 2.568 mm, 10/10 gate pass; the probe-benchmark 0.01 setting gave 0.766 / 2.87 (`docs/archive/methods/ARCHITECTURES.md`, `docs/archive/medical/probe_benchmark_analysis.md`) |
| Hippocampus MR | 0.01 | Probe benchmark 0.752 Dice / 3.20 mm, 20/20; weight 0.1 collapsed the endpoint reconstruction on the calibration pair (`docs/archive/medical/probe_benchmark_analysis.md`) |
| Abdomen CT | 0.01 | Probe benchmark 0.3545 Dice / 43.35 mm, 10/10; the architecture harness reported 0.3555 / 43.68 at the same weight; weight 0.1 fails the endpoint gate on this cohort (`docs/archive/medical/probe_benchmark.json`, `docs/archive/methods/ARCHITECTURES.md`) |

These are single-seed historical choices inherited from different grids and harnesses, not a cross-dataset optimum.

### Classical protocols

| Method | Settings |
|---|---|
| ANTs SyNOnly | Identity initialization, cross-correlation metric, radius 2, iterations [50, 50, 30], flow sigma 3, total sigma 0, gradient step 0.2, SyN sampling 2 |
| elastix affine + B-spline | Library default affine and B-spline parameter maps, `AdvancedNormalizedCorrelation`, tuned multi-resolution defaults (10 mm final grid) |
| SimpleITK Demons | Diffeomorphic Demons, 100 iterations, standard deviations 1 |

All three run on the same runtime-normalized arrays and the same zero-origin, axis-aligned canonical lattice with the original voxel spacing. Adapters return a fixed-to-moving displacement in voxel units; evaluation uses the original NIfTI affine. ANTs composes its full returned transform chain. Classical toolkit threading was limited to 8 ITK/OMP threads; deterministic classical reproduction is not claimed. Elastix includes an affine stage whereas SyNOnly does not.

The classical forward map is obtained by numerical inversion of the inverse displacement (30 fixed-point iterations, tolerance 1e-4 voxels). Its approximate status and convergence are recorded per pair; a failed inversion is not treated as an exact inverse.

## Artifact policy

`Data/` was not written to. A SHA-256 manifest of all 2,343 dataset files (about 3.17 GiB) was taken before the runs and compared afterwards; the comparison is recorded at the end of this report. Run directories are ignored by Git and contain only final endpoint weights and compact scalar evidence: per-pair `endpoints.pt`, `metrics.json`, `per_label_metrics.json`, `flow_diagnostics.json`, `training.csv`, plus per-run `config.json`, `manifest.json`, and summaries. No deformation maps, warped volumes, dense trajectories, or persistent velocity arrays are saved. Recovery velocity scratch lives under the ignored `results/.scratch/` and is removed after each pair.

## Historical context (not pooled)

| Cohort | Historical observation | Source |
|---|---|---|
| LPBA40 | Standard PD-LK seed 42, S02–S10: Dice 0.698714, HD95 6.3175 mm, zero folding, gates passed | `docs/archive/methods/BEST_RESULTS.md` |
| OASIS | Probe recipe (`lambda=0.01`): 0.766 / 2.87; calibrated `lambda=0.1`: 0.7937 / 2.568 | `docs/archive/medical/probe_benchmark_analysis.md`, `docs/archive/methods/ARCHITECTURES.md` |
| Hippocampus | Probe recipe: 0.752 / 3.20, 20 pairs | `docs/archive/medical/probe_benchmark_analysis.md` |
| Abdomen CT | Probe recipe: 0.3545 / 43.35, 10 pairs (architecture harness: 0.3555 / 43.68) | `docs/archive/medical/probe_benchmark.json`, `docs/archive/methods/ARCHITECTURES.md` |
| Classical (3 pairs per cohort) | OASIS: ANTs 0.8055, Demons 0.8062, elastix 0.7317; LPBA: ANTs 0.7271, elastix 0.6946, Demons 0.6803; Abdomen: elastix 0.3944, ANTs 0.3581, Demons 0.3379 | `docs/archive/methods/ARCHITECTURES.md` |

The historical classical table used only the first three pairs per dataset under a different harness, so it is not numerically comparable to the fresh full-cohort means.

## Cohort comparison

| Cohort | Method | Scored / attempted | Dice mean ± SD | HD95 mean (mm) | Folding mean | Warp NCC | Time (s/pair) | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| LPBA40 | Identity (no deformation) | 9 / 9 | 0.5479 | 7.94 | 0 | — | — | Reference only |
| LPBA40 | SIREN PD-LK | 9 / 9 | 0.6985 ± 0.0200 | 6.33 | 0 | 0.9886 | 66.4 | gate fail 0, skipped 0; flow all-converged 6/9 pairs |
| LPBA40 | ANTs SyNOnly | 9 / 9 | 0.7204 ± 0.0116 | 6.21 | 0 | 0.9905 | 284.8 | inv. unconverged 9 |
| LPBA40 | elastix affine+B-spline | 9 / 9 | 0.6900 ± 0.0108 | 6.36 | 8.82e-07 | 0.9778 | 30.0 | inv. unconverged 9 |
| LPBA40 | SimpleITK Demons | 9 / 9 | 0.6648 ± 0.0284 | 6.87 | 0 | 0.9840 | 38.3 | inv. unconverged 9 |
| OASIS | Identity (no deformation) | 10 / 10 | 0.4925 | 5.28 | 0 | — | — | Reference only |
| OASIS | SIREN PD-LK | 10 / 10 | 0.7915 ± 0.0219 | 2.57 | 4.48e-06 | 0.9789 | 75.6 | gate fail 0, skipped 0; flow all-converged 5/10 pairs |
| OASIS | ANTs SyNOnly | 10 / 10 | 0.7968 ± 0.0170 | 2.64 | 0 | 0.9768 | 380.0 | inv. unconverged 10 |
| OASIS | elastix affine+B-spline | 10 / 10 | 0.7097 ± 0.0275 | 3.53 | 1.34e-03 | 0.9540 | 37.6 | inv. unconverged 10 |
| OASIS | SimpleITK Demons | 10 / 10 | 0.7896 ± 0.0238 | 2.86 | 0 | 0.9676 | 48.8 | inv. unconverged 10 |
| Hippocampus | Identity (no deformation) | 20 / 20 | 0.6083 | 3.68 | 0 | — | — | Reference only |
| Hippocampus | SIREN PD-LK | 20 / 20 | 0.7525 ± 0.0544 | 3.20 | 2.38e-04 | 0.8408 | 36.6 | gate fail 0, skipped 0; flow all-converged 20/20 pairs |
| Hippocampus | ANTs SyNOnly | 20 / 20 | 0.7447 ± 0.0474 | 3.29 | 0 | 0.6647 | 17.9 | inv. unconverged 20 |
| Hippocampus | elastix affine+B-spline | 20 / 20 | 0.6770 ± 0.0665 | 4.59 | 4.23e-02 | 0.8731 | 3.4 | inv. unconverged 20 |
| Hippocampus | SimpleITK Demons | 20 / 20 | 0.7450 ± 0.0544 | 3.30 | 0 | 0.7906 | 2.4 | inv. unconverged 20 |
| Abdomen CT | Identity (no deformation) | 10 / 10 | 0.2316 | 46.70 | 0 | — | — | Reference only |
| Abdomen CT | SIREN PD-LK | 10 / 10 | 0.3545 ± 0.1238 | 43.35 | 2.70e-04 | 0.8814 | 63.1 | gate fail 0, skipped 0; flow all-converged 8/10 pairs |
| Abdomen CT | ANTs SyNOnly | 10 / 10 | 0.3537 ± 0.1356 | 44.23 | 0 | 0.7776 | 411.6 | inv. unconverged 10 |
| Abdomen CT | elastix affine+B-spline | 10 / 10 | 0.3364 ± 0.1239 | 54.01 | 5.56e-02 | 0.9080 | 108.6 | inv. unconverged 10 |
| Abdomen CT | SimpleITK Demons | 10 / 10 | 0.3049 ± 0.1143 | 45.06 | 0 | 0.7916 | 46.2 | inv. unconverged 10 |

## Denominators

| Cohort | Attempted pairs | SIREN completed / failed / skipped (gate) | Classical completed / failed |
|---|---:|---|---|
| LPBA40 | 9 | 9 / 0 / 0 (0) | 27 / 0 |
| OASIS | 10 | 10 / 0 / 0 (0) | 30 / 0 |
| Hippocampus | 20 | 20 / 0 / 0 (0) | 60 / 0 |
| Abdomen CT | 10 | 10 / 0 / 0 (0) | 30 / 0 |

## Failures and skipped evaluations

No failed pairs were recorded in any cohort.

## What the fresh results show

### SIREN reproduction of the retained references

The fresh SIREN means reproduce the retained historical references closely. The historical values are single-seed evidence from earlier harnesses; they are shown here as a consistency check, not pooled into the fresh statistics. OASIS is the calibrated weight-0.1 run; Hippocampus and Abdomen CT use the exact probe-benchmark aggregates; LPBA40 uses the archived sweep reference.

| Cohort | Fresh Dice | Historical Dice | Fresh HD95 (mm) | Historical HD95 (mm) |
|---|---:|---:|---:|---:|
| LPBA40 | 0.6985 | 0.698714 | 6.33 | 6.3175 |
| OASIS (weight 0.1) | 0.7915 | 0.7937 | 2.57 | 2.568 |
| Hippocampus MR | 0.7525 | 0.7525 | 3.20 | 3.195 |
| Abdomen CT | 0.3545 | 0.3545 | 43.35 | 43.35 |

All 49 SIREN pairs passed the 25 dB endpoint gate. The smallest endpoint margin was 25.06 dB on Abdomen CT; LPBA40 and Hippocampus cleared the gate comfortably, and OASIS stayed near 30 dB. Recovered-map folding was exactly zero on all nine LPBA40 pairs, with small nonzero fractions elsewhere (maximum 9.4e-4 on Hippocampus, 7.1e-4 on Abdomen CT, 1.2e-5 on OASIS).

Flow convergence was recorded per interval, not silently repaired. Across the four cohorts, 766 of 784 interval solves met the configured `2e-5` relative tolerance; 39 of 49 pairs had every interval converge. The misses were near-misses at about `2.00–2.02e-5` and are listed per pair. LPBA40 (6/9 pairs) and OASIS (5/10) carry most of the warnings; Hippocampus converged on every interval of every pair; Abdomen CT missed on 2 of its 10 pairs.

### SIREN versus the classical protocols

- **LPBA40.** ANTs SyNOnly is the strongest classical method on this cohort: Dice 0.7204 versus SIREN 0.6985 (+0.0219), HD95 6.21 versus 6.33 mm. Elastix is 0.6900 / 6.36 mm and Demons 0.6648 / 6.87 mm. The ordering agrees with the historical three-pair observation.
- **OASIS.** ANTs reaches the highest Dice (0.7968) just above SIREN (0.7915), while SIREN has the lower HD95 (2.57 versus 2.64 mm) and slightly higher warped NCC (0.9789 versus 0.9768). Demons is close on Dice (0.7896) but worse on HD95 (2.86 mm); elastix is clearly behind at 0.7097 / 3.53 mm.
- **Hippocampus MR.** SIREN has the best Dice and HD95 (0.7525 / 3.20 mm), ahead of Demons (0.7450 / 3.30) and ANTs (0.7447 / 3.29). ANTs improves labels but its warped-image NCC is only 0.665, so this protocol under-registers intensity on the 64³ crops even where label overlap improves. Elastix is weakest (0.6770 / 4.59 mm), folds 4.2% of voxels, and its cohort HD95 is worse than identity (4.59 versus 3.68 mm) even though its Dice improves; it worsened HD95 on 14 of 20 pairs and Dice on 4 of 20. SIREN improved Dice on all 20 pairs but worsened HD95 on 4; ANTs worsened Dice on one pair and HD95 on 6.
- **Abdomen CT.** SIREN (0.3545 / 43.35 mm) and ANTs (0.3537 / 44.23 mm) are effectively tied on Dice, with SIREN slightly better on HD95 and NCC (0.8814 versus 0.7776). Elastix reaches 0.3364 but folds 5.6% of voxels, has HD95 54.0 mm (worse than identity's 46.70 mm), and worsened HD95 on 8 of 10 pairs; Demons trails at 0.3049 / 45.06 mm. SIREN improved Dice on all 10 pairs, but HD95 worsened slightly on 0005 (86.20 → 86.37 mm) and 0007 (45.47 → 46.37 mm), and pairs 0005 and 0009 remain above 86 mm.

The image-similarity column can disagree with label overlap: ANTs on Hippocampus has respectable Dice but low NCC, and elastix on Abdomen CT and Hippocampus has high NCC with substantial folding. Neither metric alone characterises map quality, and the per-pair tables are the primary evidence.

### Runtime

SIREN took 66, 76, 37 and 63 s per pair (LPBA40, OASIS, Hippocampus, Abdomen CT) including the 5,000-step endpoint fit, recovery, and evaluation. Classical means were 285/380/18/412 s for ANTs, 30/38/3/109 s for elastix, and 38/49/2/46 s for Demons. The classical jobs ran concurrently with each other and with SIREN, with ITK/OMP limited to 8 threads each; these are wall-clock observations under load, not a controlled speed comparison, and effective CPU contention was not controlled.

Saved per-pair timing fields separate fitting from recovery. Means in seconds:

| Cohort | All 5,000 training steps | Post-training recovery | Total, including loading/evaluation |
|---|---:|---:|---:|
| LPBA40 | 36.8 | 17.9 | 66.4 |
| OASIS | 36.6 | 29.4 | 75.6 |
| Hippocampus | 35.6 | 0.6 | 36.6 |
| Abdomen CT | 36.6 | 21.2 | 63.1 |

The baseline trainer uses small independent local PD-LK systems, not global PCG at each step. Its 16 global flow solves belong to the recovery column.

### Registration-aware training did not complete

The proposed original-image registration-aware objective is not represented in these tables. Removal commit `f593cd8` records about 62 seconds per active LPBA optimizer step when differentiating through the global flow/RK2 map. It required primal and adjoint PCG solves over 16 intervals, plus checkpoint recomputation. That recorded feasibility observation is not a controlled timing comparison with the baseline benchmark. The branch was removed before full-cohort training on LPBA40, OASIS, and Abdomen CT completed. See the [historical design and retirement record](../archive/methods/REGISTRATION_AWARE_LOSS.md); synthetic gradient correctness does not establish registration efficacy.

### What is not claimed

This is one seed per pair. None of the differences above is converted into a general superiority claim: on LPBA40 the classical reference leads on Dice, on OASIS the two leading methods differ by 0.005 Dice with opposite HD95 order, and on Abdomen CT the leading pair is within its observed spread. Elastix's large folding and worse-than-identity HD95 on Hippocampus and Abdomen CT are recorded as failures of that protocol on those cohorts, not as an endorsement of the SIREN maps. The classical methods include an affine stage in the elastix case (and not in SyNOnly), use their named local settings, and have an approximate forward-inversion diagnostic that failed to converge on every pair. The SIREN flow warnings above qualify its maps where they occur.

## LPBA40 — per-pair evidence

### SIREN PD-LK

| Pair | Endpoint PSNR min (dB) | Dice pre → post | HD95 pre → post (mm) | NCC pre → post | Folding | Flow conv. | Max residual | Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S02 | 32.20 | 0.5630 → 0.7025 | 7.84 → 6.60 | 0.8890 → 0.9887 | 0 | 14/16 | 2.01e-05 | 64.64 |
| S03 | 32.56 | 0.5518 → 0.6859 | 8.09 → 6.93 | 0.9191 → 0.9893 | 0 | 16/16 | 2.00e-05 | 62.92 |
| S04 | 32.61 | 0.6163 → 0.7307 | 6.24 → 5.07 | 0.9212 → 0.9880 | 0 | 14/16 | 2.02e-05 | 58.07 |
| S05 | 31.24 | 0.5449 → 0.6951 | 7.51 → 6.12 | 0.8734 → 0.9878 | 0 | 16/16 | 2.00e-05 | 66.18 |
| S06 | 31.95 | 0.4740 → 0.6585 | 9.56 → 6.75 | 0.8600 → 0.9880 | 0 | 16/16 | 1.95e-05 | 65.63 |
| S07 | 32.65 | 0.5837 → 0.7124 | 8.10 → 6.28 | 0.9180 → 0.9896 | 0 | 16/16 | 1.99e-05 | 71.56 |
| S08 | 31.67 | 0.4974 → 0.6886 | 8.21 → 6.67 | 0.8766 → 0.9885 | 0 | 15/16 | 2.01e-05 | 73.04 |
| S09 | 32.35 | 0.5689 → 0.6935 | 7.66 → 6.74 | 0.9235 → 0.9891 | 0 | 16/16 | 1.95e-05 | 67.23 |
| S10 | 31.67 | 0.5313 → 0.7197 | 8.25 → 5.81 | 0.8612 → 0.9889 | 0 | 16/16 | 1.97e-05 | 68.30 |

### Classical methods

| Pair | Method | Dice pre → post | HD95 pre → post (mm) | NCC pre → post | Folding | Outside | Composition (mm) | Inv. conv. | Time (s) |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| S02 | ANTs SyNOnly | 0.5630 → 0.7202 | 7.84 → 6.38 | 0.8890 → 0.9897 | 0 | 0.000 | 0.03 | no | 314.77 |
| S02 | elastix affine+B-spline | 0.5630 → 0.6869 | 7.84 → 6.53 | 0.8890 → 0.9772 | 0 | 0.167 | 0.58 | no | 39.87 |
| S02 | SimpleITK Demons | 0.5630 → 0.6650 | 7.84 → 7.01 | 0.8890 → 0.9853 | 0 | 0.027 | 0.03 | no | 46.35 |
| S03 | ANTs SyNOnly | 0.5518 → 0.7181 | 8.09 → 6.63 | 0.9191 → 0.9922 | 0 | 0.000 | 0.03 | no | 267.44 |
| S03 | elastix affine+B-spline | 0.5518 → 0.7091 | 8.09 → 6.31 | 0.9191 → 0.9796 | 0 | 0.092 | 0.31 | no | 31.13 |
| S03 | SimpleITK Demons | 0.5518 → 0.6618 | 8.09 → 7.24 | 0.9191 → 0.9876 | 0 | 0.016 | 0.02 | no | 40.09 |
| S04 | ANTs SyNOnly | 0.6163 → 0.7430 | 6.24 → 5.22 | 0.9212 → 0.9884 | 0 | 0.000 | 0.03 | no | 287.18 |
| S04 | elastix affine+B-spline | 0.6163 → 0.6879 | 6.24 → 6.07 | 0.9212 → 0.9789 | 0 | 0.048 | 0.17 | no | 28.03 |
| S04 | SimpleITK Demons | 0.6163 → 0.7139 | 6.24 → 5.45 | 0.9212 → 0.9874 | 0 | 0.009 | 0.02 | no | 34.87 |
| S05 | ANTs SyNOnly | 0.5449 → 0.7213 | 7.51 → 5.86 | 0.8734 → 0.9890 | 0 | 0.000 | 0.04 | no | 277.73 |
| S05 | elastix affine+B-spline | 0.5449 → 0.6729 | 7.51 → 6.44 | 0.8734 → 0.9734 | 7.93e-06 | 0.173 | 0.75 | no | 25.28 |
| S05 | SimpleITK Demons | 0.5449 → 0.6606 | 7.51 → 6.58 | 0.8734 → 0.9812 | 0 | 0.022 | 0.03 | no | 34.30 |
| S06 | ANTs SyNOnly | 0.4740 → 0.7051 | 9.56 → 6.43 | 0.8600 → 0.9903 | 0 | 0.000 | 0.05 | no | 287.40 |
| S06 | elastix affine+B-spline | 0.4740 → 0.6909 | 9.56 → 6.40 | 0.8600 → 0.9768 | 0 | 0.235 | 1.26 | no | 25.11 |
| S06 | SimpleITK Demons | 0.4740 → 0.6012 | 9.56 → 7.95 | 0.8600 → 0.9756 | 0 | 0.028 | 0.03 | no | 36.94 |
| S07 | ANTs SyNOnly | 0.5837 → 0.7172 | 8.10 → 6.46 | 0.9180 → 0.9917 | 0 | 0.000 | 0.03 | no | 276.71 |
| S07 | elastix affine+B-spline | 0.5837 → 0.7033 | 8.10 → 5.99 | 0.9180 → 0.9798 | 0 | 0.125 | 0.44 | no | 29.82 |
| S07 | SimpleITK Demons | 0.5837 → 0.6847 | 8.10 → 6.96 | 0.9180 → 0.9881 | 0 | 0.022 | 0.02 | no | 37.02 |
| S08 | ANTs SyNOnly | 0.4974 → 0.7088 | 8.21 → 6.45 | 0.8766 → 0.9905 | 0 | 0.000 | 0.03 | no | 284.92 |
| S08 | elastix affine+B-spline | 0.4974 → 0.6805 | 8.21 → 6.58 | 0.8766 → 0.9764 | 0 | 0.147 | 0.61 | no | 32.59 |
| S08 | SimpleITK Demons | 0.4974 → 0.6495 | 8.21 → 7.07 | 0.8766 → 0.9835 | 0 | 0.019 | 0.02 | no | 36.93 |
| S09 | ANTs SyNOnly | 0.5689 → 0.7130 | 7.66 → 6.63 | 0.9235 → 0.9915 | 0 | 0.000 | 0.03 | no | 291.14 |
| S09 | elastix affine+B-spline | 0.5689 → 0.6817 | 7.66 → 6.95 | 0.9235 → 0.9803 | 0 | 0.108 | 0.33 | no | 28.98 |
| S09 | SimpleITK Demons | 0.5689 → 0.6759 | 7.66 → 6.97 | 0.9235 → 0.9874 | 0 | 0.021 | 0.02 | no | 39.14 |
| S10 | ANTs SyNOnly | 0.5313 → 0.7363 | 8.25 → 5.79 | 0.8612 → 0.9911 | 0 | 0.000 | 0.05 | no | 276.31 |
| S10 | elastix affine+B-spline | 0.5313 → 0.6966 | 8.25 → 5.97 | 0.8612 → 0.9777 | 0 | 0.221 | 1.04 | no | 29.34 |
| S10 | SimpleITK Demons | 0.5313 → 0.6704 | 8.25 → 6.59 | 0.8612 → 0.9795 | 0 | 0.027 | 0.03 | no | 39.45 |

## OASIS — per-pair evidence

### SIREN PD-LK

| Pair | Endpoint PSNR min (dB) | Dice pre → post | HD95 pre → post (mm) | NCC pre → post | Folding | Flow conv. | Max residual | Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OASIS_0002_0000 | 30.39 | 0.4587 → 0.7761 | 6.17 → 3.17 | 0.8641 → 0.9801 | 1.18e-05 | 11/16 | 2.02e-05 | 81.33 |
| OASIS_0003_0000 | 30.20 | 0.5456 → 0.8273 | 4.15 → 1.78 | 0.8712 → 0.9803 | 4.80e-06 | 16/16 | 2.00e-05 | 74.70 |
| OASIS_0004_0000 | 30.40 | 0.4983 → 0.8037 | 5.14 → 2.46 | 0.8727 → 0.9809 | 4.36e-07 | 16/16 | 1.99e-05 | 72.23 |
| OASIS_0005_0000 | 30.16 | 0.4304 → 0.7811 | 6.48 → 3.26 | 0.8571 → 0.9768 | 2.47e-06 | 16/16 | 2.00e-05 | 75.26 |
| OASIS_0006_0000 | 30.21 | 0.4268 → 0.8009 | 5.86 → 2.45 | 0.8459 → 0.9778 | 7.27e-06 | 14/16 | 2.00e-05 | 78.26 |
| OASIS_0007_0000 | 30.07 | 0.5045 → 0.7940 | 5.69 → 2.94 | 0.8589 → 0.9780 | 4.21e-06 | 15/16 | 2.00e-05 | 73.51 |
| OASIS_0008_0000 | 30.23 | 0.5009 → 0.7502 | 4.93 → 2.46 | 0.8631 → 0.9767 | 0 | 14/16 | 2.02e-05 | 77.50 |
| OASIS_0009_0000 | 29.90 | 0.6093 → 0.8079 | 3.84 → 2.11 | 0.8544 → 0.9786 | 7.41e-06 | 15/16 | 2.01e-05 | 76.31 |
| OASIS_0010_0000 | 30.28 | 0.4819 → 0.7654 | 5.11 → 2.43 | 0.8616 → 0.9797 | 4.94e-06 | 16/16 | 1.98e-05 | 72.95 |
| OASIS_0011_0000 | 30.16 | 0.4684 → 0.8081 | 5.46 → 2.68 | 0.8661 → 0.9802 | 1.45e-06 | 16/16 | 2.00e-05 | 73.79 |

### Classical methods

| Pair | Method | Dice pre → post | HD95 pre → post (mm) | NCC pre → post | Folding | Outside | Composition (mm) | Inv. conv. | Time (s) |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| OASIS_0002_0000 | ANTs SyNOnly | 0.4587 → 0.7872 | 6.17 → 3.22 | 0.8641 → 0.9775 | 0 | 0.000 | 0.05 | no | 441.80 |
| OASIS_0002_0000 | elastix affine+B-spline | 0.4587 → 0.7023 | 6.17 → 3.53 | 0.8641 → 0.9569 | 2.37e-03 | 0.060 | 0.20 | no | 47.65 |
| OASIS_0002_0000 | SimpleITK Demons | 0.4587 → 0.7802 | 6.17 → 3.58 | 0.8641 → 0.9682 | 0 | 0.018 | 0.04 | no | 54.33 |
| OASIS_0003_0000 | ANTs SyNOnly | 0.5456 → 0.8343 | 4.15 → 1.77 | 0.8712 → 0.9767 | 0 | 0.000 | 0.05 | no | 390.17 |
| OASIS_0003_0000 | elastix affine+B-spline | 0.5456 → 0.7586 | 4.15 → 2.42 | 0.8712 → 0.9545 | 8.21e-04 | 0.049 | 0.10 | no | 37.38 |
| OASIS_0003_0000 | SimpleITK Demons | 0.5456 → 0.8321 | 4.15 → 2.08 | 0.8712 → 0.9683 | 0 | 0.015 | 0.03 | no | 53.58 |
| OASIS_0004_0000 | ANTs SyNOnly | 0.4983 → 0.7949 | 5.14 → 2.59 | 0.8727 → 0.9809 | 0 | 0.000 | 0.05 | no | 382.31 |
| OASIS_0004_0000 | elastix affine+B-spline | 0.4983 → 0.7340 | 5.14 → 3.14 | 0.8727 → 0.9574 | 1.31e-04 | 0.030 | 0.04 | no | 35.03 |
| OASIS_0004_0000 | SimpleITK Demons | 0.4983 → 0.8062 | 5.14 → 2.60 | 0.8727 → 0.9717 | 0 | 0.016 | 0.03 | no | 50.17 |
| OASIS_0005_0000 | ANTs SyNOnly | 0.4304 → 0.7742 | 6.48 → 3.52 | 0.8571 → 0.9732 | 0 | 0.000 | 0.06 | no | 385.79 |
| OASIS_0005_0000 | elastix affine+B-spline | 0.4304 → 0.6710 | 6.48 → 3.79 | 0.8571 → 0.9510 | 2.00e-03 | 0.056 | 0.16 | no | 34.57 |
| OASIS_0005_0000 | SimpleITK Demons | 0.4304 → 0.7705 | 6.48 → 3.55 | 0.8571 → 0.9671 | 0 | 0.018 | 0.04 | no | 51.47 |
| OASIS_0006_0000 | ANTs SyNOnly | 0.4268 → 0.7953 | 5.86 → 2.72 | 0.8459 → 0.9745 | 0 | 0.000 | 0.06 | no | 398.99 |
| OASIS_0006_0000 | elastix affine+B-spline | 0.4268 → 0.6848 | 5.86 → 4.06 | 0.8459 → 0.9545 | 2.30e-03 | 0.126 | 0.43 | no | 40.07 |
| OASIS_0006_0000 | SimpleITK Demons | 0.4268 → 0.7862 | 5.86 → 2.97 | 0.8459 → 0.9651 | 0 | 0.024 | 0.06 | no | 52.03 |
| OASIS_0007_0000 | ANTs SyNOnly | 0.5045 → 0.7935 | 5.69 → 3.06 | 0.8589 → 0.9764 | 0 | 0.000 | 0.05 | no | 389.60 |
| OASIS_0007_0000 | elastix affine+B-spline | 0.5045 → 0.6785 | 5.69 → 6.27 | 0.8589 → 0.9486 | 2.98e-03 | 0.032 | 0.15 | no | 37.22 |
| OASIS_0007_0000 | SimpleITK Demons | 0.5045 → 0.7892 | 5.69 → 3.15 | 0.8589 → 0.9676 | 0 | 0.016 | 0.04 | no | 48.35 |
| OASIS_0008_0000 | ANTs SyNOnly | 0.5009 → 0.8006 | 4.93 → 2.23 | 0.8631 → 0.9757 | 0 | 0.000 | 0.06 | no | 361.14 |
| OASIS_0008_0000 | elastix affine+B-spline | 0.5009 → 0.7059 | 4.93 → 3.10 | 0.8631 → 0.9542 | 4.46e-04 | 0.093 | 0.25 | no | 33.69 |
| OASIS_0008_0000 | SimpleITK Demons | 0.5009 → 0.7455 | 4.93 → 2.79 | 0.8631 → 0.9664 | 0 | 0.019 | 0.05 | no | 43.33 |
| OASIS_0009_0000 | ANTs SyNOnly | 0.6093 → 0.8102 | 3.84 → 2.04 | 0.8544 → 0.9753 | 0 | 0.000 | 0.05 | no | 351.33 |
| OASIS_0009_0000 | elastix affine+B-spline | 0.6093 → 0.7388 | 3.84 → 2.50 | 0.8544 → 0.9513 | 1.44e-03 | 0.054 | 0.17 | no | 36.33 |
| OASIS_0009_0000 | SimpleITK Demons | 0.6093 → 0.8124 | 3.84 → 2.35 | 0.8544 → 0.9638 | 0 | 0.012 | 0.03 | no | 45.27 |
| OASIS_0010_0000 | ANTs SyNOnly | 0.4819 → 0.7726 | 5.11 → 2.53 | 0.8616 → 0.9758 | 0 | 0.000 | 0.06 | no | 347.50 |
| OASIS_0010_0000 | elastix affine+B-spline | 0.4819 → 0.6947 | 5.11 → 3.21 | 0.8616 → 0.9547 | 7.03e-04 | 0.060 | 0.15 | no | 35.40 |
| OASIS_0010_0000 | SimpleITK Demons | 0.4819 → 0.7688 | 5.11 → 2.59 | 0.8616 → 0.9669 | 0 | 0.019 | 0.04 | no | 43.35 |
| OASIS_0011_0000 | ANTs SyNOnly | 0.4684 → 0.8051 | 5.46 → 2.77 | 0.8661 → 0.9817 | 0 | 0.000 | 0.06 | no | 350.90 |
| OASIS_0011_0000 | elastix affine+B-spline | 0.4684 → 0.7283 | 5.46 → 3.32 | 0.8661 → 0.9570 | 1.67e-04 | 0.065 | 0.14 | no | 38.63 |
| OASIS_0011_0000 | SimpleITK Demons | 0.4684 → 0.8047 | 5.46 → 2.96 | 0.8661 → 0.9706 | 0 | 0.019 | 0.04 | no | 45.66 |

## Hippocampus — per-pair evidence

### SIREN PD-LK

| Pair | Endpoint PSNR min (dB) | Dice pre → post | HD95 pre → post (mm) | NCC pre → post | Folding | Flow conv. | Max residual | Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HippocampusMR_0002_0000 | 32.94 | 0.7893 → 0.7963 | 2.22 → 2.50 | 0.5732 → 0.9159 | 1.37e-04 | 16/16 | 1.99e-05 | 39.96 |
| HippocampusMR_0003_0000 | 33.40 | 0.7138 → 0.8042 | 2.45 → 2.37 | 0.6534 → 0.8780 | 7.63e-06 | 16/16 | 1.83e-05 | 37.02 |
| HippocampusMR_0004_0000 | 33.80 | 0.6052 → 0.7950 | 3.50 → 2.70 | 0.4989 → 0.8168 | 6.10e-05 | 16/16 | 1.92e-05 | 36.86 |
| HippocampusMR_0005_0000 | 33.12 | 0.5527 → 0.7519 | 4.36 → 3.77 | 0.4596 → 0.8667 | 1.53e-05 | 16/16 | 1.94e-05 | 36.23 |
| HippocampusMR_0006_0000 | 32.67 | 0.6067 → 0.7676 | 3.61 → 2.72 | 0.5745 → 0.8372 | 5.61e-04 | 16/16 | 1.84e-05 | 36.18 |
| HippocampusMR_0007_0000 | 34.61 | 0.5898 → 0.7526 | 3.37 → 3.80 | 0.3351 → 0.7315 | 8.05e-04 | 16/16 | 1.90e-05 | 35.82 |
| HippocampusMR_0008_0000 | 32.48 | 0.6689 → 0.7911 | 3.58 → 2.00 | 0.5585 → 0.8819 | 1.37e-04 | 16/16 | 1.93e-05 | 36.08 |
| HippocampusMR_0009_0000 | 32.51 | 0.3495 → 0.6053 | 6.88 → 4.79 | 0.2753 → 0.6922 | 2.82e-04 | 16/16 | 1.91e-05 | 36.79 |
| HippocampusMR_0010_0000 | 33.97 | 0.5723 → 0.8102 | 3.70 → 3.00 | 0.4511 → 0.8236 | 2.21e-04 | 16/16 | 1.68e-05 | 36.66 |
| HippocampusMR_0011_0000 | 33.39 | 0.5961 → 0.7810 | 3.37 → 2.53 | 0.5235 → 0.8584 | 5.15e-04 | 16/16 | 1.85e-05 | 36.23 |
| HippocampusMR_0012_0000 | 32.65 | 0.4509 → 0.6690 | 4.55 → 3.50 | 0.5397 → 0.8708 | 1.53e-04 | 16/16 | 1.97e-05 | 36.32 |
| HippocampusMR_0013_0000 | 33.64 | 0.6678 → 0.7826 | 2.70 → 2.50 | 0.4417 → 0.8359 | 9.54e-05 | 16/16 | 1.91e-05 | 35.99 |
| HippocampusMR_0014_0000 | 32.98 | 0.6554 → 0.7472 | 3.39 → 3.50 | 0.4207 → 0.8307 | 1.22e-04 | 16/16 | 1.94e-05 | 36.30 |
| HippocampusMR_0015_0000 | 33.82 | 0.6633 → 0.7471 | 4.18 → 4.06 | 0.5355 → 0.8219 | 0 | 16/16 | 1.84e-05 | 36.21 |
| HippocampusMR_0016_0000 | 33.31 | 0.6923 → 0.7765 | 3.08 → 3.00 | 0.5553 → 0.8659 | 3.43e-05 | 16/16 | 1.91e-05 | 35.94 |
| HippocampusMR_0017_0000 | 37.97 | 0.5921 → 0.6905 | 3.30 → 2.62 | 0.5922 → 0.8791 | 5.72e-05 | 16/16 | 1.93e-05 | 36.41 |
| HippocampusMR_0018_0000 | 36.44 | 0.5801 → 0.6629 | 4.00 → 5.46 | 0.5459 → 0.8596 | 2.67e-04 | 16/16 | 1.97e-05 | 36.51 |
| HippocampusMR_0019_0000 | 34.27 | 0.6498 → 0.8118 | 2.72 → 2.47 | 0.5565 → 0.8818 | 9.42e-04 | 16/16 | 1.94e-05 | 36.92 |
| HippocampusMR_0020_0000 | 33.38 | 0.6127 → 0.7801 | 3.56 → 2.62 | 0.4778 → 0.8241 | 3.36e-04 | 16/16 | 1.94e-05 | 37.01 |
| HippocampusMR_0021_0000 | 33.77 | 0.5577 → 0.7264 | 5.00 → 4.00 | 0.4459 → 0.8438 | 1.53e-05 | 16/16 | 1.85e-05 | 36.15 |

### Classical methods

| Pair | Method | Dice pre → post | HD95 pre → post (mm) | NCC pre → post | Folding | Outside | Composition (mm) | Inv. conv. | Time (s) |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| HippocampusMR_0002_0000 | ANTs SyNOnly | 0.7893 → 0.7649 | 2.22 → 2.62 | 0.5732 → 0.7123 | 0 | 0.000 | 0.03 | no | 20.30 |
| HippocampusMR_0002_0000 | elastix affine+B-spline | 0.7893 → 0.7070 | 2.22 → 2.70 | 0.5732 → 0.8714 | 5.79e-03 | 0.166 | 0.62 | no | 11.29 |
| HippocampusMR_0002_0000 | SimpleITK Demons | 0.7893 → 0.7953 | 2.22 → 2.50 | 0.5732 → 0.8761 | 0 | 0.055 | 0.10 | no | 2.38 |
| HippocampusMR_0003_0000 | ANTs SyNOnly | 0.7138 → 0.7964 | 2.45 → 2.50 | 0.6534 → 0.7718 | 0 | 0.000 | 0.03 | no | 16.95 |
| HippocampusMR_0003_0000 | elastix affine+B-spline | 0.7138 → 0.7563 | 2.45 → 2.62 | 0.6534 → 0.8951 | 3.01e-02 | 0.137 | 1.12 | no | 2.83 |
| HippocampusMR_0003_0000 | SimpleITK Demons | 0.7138 → 0.7969 | 2.45 → 2.41 | 0.6534 → 0.8419 | 0 | 0.047 | 0.07 | no | 2.50 |
| HippocampusMR_0004_0000 | ANTs SyNOnly | 0.6052 → 0.8032 | 3.50 → 2.62 | 0.4989 → 0.6880 | 0 | 0.000 | 0.03 | no | 17.06 |
| HippocampusMR_0004_0000 | elastix affine+B-spline | 0.6052 → 0.7466 | 3.50 → 3.50 | 0.4989 → 0.8811 | 1.45e-02 | 0.230 | 1.30 | no | 3.88 |
| HippocampusMR_0004_0000 | SimpleITK Demons | 0.6052 → 0.7954 | 3.50 → 3.00 | 0.4989 → 0.7935 | 0 | 0.072 | 0.13 | no | 5.14 |
| HippocampusMR_0005_0000 | ANTs SyNOnly | 0.5527 → 0.7232 | 4.36 → 3.80 | 0.4596 → 0.6279 | 0 | 0.000 | 0.04 | no | 19.21 |
| HippocampusMR_0005_0000 | elastix affine+B-spline | 0.5527 → 0.6540 | 4.36 → 3.56 | 0.4596 → 0.8750 | 1.87e-02 | 0.186 | 1.28 | no | 2.84 |
| HippocampusMR_0005_0000 | SimpleITK Demons | 0.5527 → 0.7270 | 4.36 → 3.66 | 0.4596 → 0.8431 | 0 | 0.055 | 0.12 | no | 2.68 |
| HippocampusMR_0006_0000 | ANTs SyNOnly | 0.6067 → 0.7505 | 3.61 → 3.00 | 0.5745 → 0.7232 | 0 | 0.000 | 0.04 | no | 18.24 |
| HippocampusMR_0006_0000 | elastix affine+B-spline | 0.6067 → 0.6939 | 3.61 → 3.16 | 0.5745 → 0.9007 | 4.56e-02 | 0.065 | 0.93 | no | 3.09 |
| HippocampusMR_0006_0000 | SimpleITK Demons | 0.6067 → 0.7500 | 3.61 → 3.00 | 0.5745 → 0.8116 | 0 | 0.053 | 0.10 | no | 2.70 |
| HippocampusMR_0007_0000 | ANTs SyNOnly | 0.5898 → 0.7130 | 3.37 → 3.98 | 0.3351 → 0.5101 | 0 | 0.000 | 0.03 | no | 21.31 |
| HippocampusMR_0007_0000 | elastix affine+B-spline | 0.5898 → 0.6533 | 3.37 → 4.50 | 0.3351 → 0.8429 | 1.43e-01 | 0.230 | 2.68 | no | 4.77 |
| HippocampusMR_0007_0000 | SimpleITK Demons | 0.5898 → 0.7487 | 3.37 → 3.82 | 0.3351 → 0.6885 | 0 | 0.061 | 0.14 | no | 2.36 |
| HippocampusMR_0008_0000 | ANTs SyNOnly | 0.6689 → 0.7958 | 3.58 → 2.00 | 0.5585 → 0.7019 | 0 | 0.000 | 0.03 | no | 17.49 |
| HippocampusMR_0008_0000 | elastix affine+B-spline | 0.6689 → 0.7086 | 3.58 → 4.06 | 0.5585 → 0.8889 | 3.00e-02 | 0.155 | 0.93 | no | 3.00 |
| HippocampusMR_0008_0000 | SimpleITK Demons | 0.6689 → 0.7872 | 3.58 → 2.18 | 0.5585 → 0.8265 | 0 | 0.050 | 0.09 | no | 2.44 |
| HippocampusMR_0009_0000 | ANTs SyNOnly | 0.3495 → 0.6573 | 6.88 → 4.54 | 0.2753 → 0.3983 | 0 | 0.000 | 0.04 | no | 18.61 |
| HippocampusMR_0009_0000 | elastix affine+B-spline | 0.3495 → 0.6395 | 6.88 → 7.76 | 0.2753 → 0.7613 | 1.22e-01 | 0.329 | 3.22 | no | 2.54 |
| HippocampusMR_0009_0000 | SimpleITK Demons | 0.3495 → 0.6329 | 6.88 → 4.90 | 0.2753 → 0.5704 | 0 | 0.064 | 0.13 | no | 2.40 |
| HippocampusMR_0010_0000 | ANTs SyNOnly | 0.5723 → 0.7861 | 3.70 → 3.16 | 0.4511 → 0.6257 | 0 | 0.000 | 0.03 | no | 24.17 |
| HippocampusMR_0010_0000 | elastix affine+B-spline | 0.5723 → 0.7337 | 3.70 → 3.80 | 0.4511 → 0.8345 | 4.52e-02 | 0.245 | 1.77 | no | 3.18 |
| HippocampusMR_0010_0000 | SimpleITK Demons | 0.5723 → 0.8073 | 3.70 → 2.91 | 0.4511 → 0.7550 | 0 | 0.073 | 0.12 | no | 2.59 |
| HippocampusMR_0011_0000 | ANTs SyNOnly | 0.5961 → 0.7468 | 3.37 → 2.43 | 0.5235 → 0.6865 | 0 | 0.000 | 0.03 | no | 17.42 |
| HippocampusMR_0011_0000 | elastix affine+B-spline | 0.5961 → 0.6637 | 3.37 → 3.80 | 0.5235 → 0.8850 | 4.61e-02 | 0.157 | 1.26 | no | 2.73 |
| HippocampusMR_0011_0000 | SimpleITK Demons | 0.5961 → 0.7444 | 3.37 → 3.50 | 0.5235 → 0.8046 | 0 | 0.061 | 0.10 | no | 2.90 |
| HippocampusMR_0012_0000 | ANTs SyNOnly | 0.4509 → 0.7293 | 4.55 → 3.56 | 0.5397 → 0.7283 | 0 | 0.000 | 0.03 | no | 18.19 |
| HippocampusMR_0012_0000 | elastix affine+B-spline | 0.4509 → 0.6162 | 4.55 → 4.55 | 0.5397 → 0.9081 | 4.38e-02 | 0.117 | 1.23 | no | 3.13 |
| HippocampusMR_0012_0000 | SimpleITK Demons | 0.4509 → 0.6859 | 4.55 → 3.82 | 0.5397 → 0.8098 | 0 | 0.056 | 0.09 | no | 2.50 |
| HippocampusMR_0013_0000 | ANTs SyNOnly | 0.6678 → 0.7686 | 2.70 → 2.87 | 0.4417 → 0.6567 | 0 | 0.000 | 0.04 | no | 18.66 |
| HippocampusMR_0013_0000 | elastix affine+B-spline | 0.6678 → 0.6901 | 2.70 → 3.62 | 0.4417 → 0.8638 | 3.97e-02 | 0.211 | 1.25 | no | 2.63 |
| HippocampusMR_0013_0000 | SimpleITK Demons | 0.6678 → 0.7653 | 2.70 → 3.06 | 0.4417 → 0.8006 | 0 | 0.061 | 0.12 | no | 1.89 |
| HippocampusMR_0014_0000 | ANTs SyNOnly | 0.6554 → 0.7889 | 3.39 → 2.80 | 0.4207 → 0.6521 | 0 | 0.000 | 0.04 | no | 15.40 |
| HippocampusMR_0014_0000 | elastix affine+B-spline | 0.6554 → 0.7143 | 3.39 → 3.50 | 0.4207 → 0.8701 | 8.67e-02 | 0.191 | 1.49 | no | 2.32 |
| HippocampusMR_0014_0000 | SimpleITK Demons | 0.6554 → 0.7698 | 3.39 → 3.18 | 0.4207 → 0.7753 | 0 | 0.067 | 0.12 | no | 2.07 |
| HippocampusMR_0015_0000 | ANTs SyNOnly | 0.6633 → 0.7271 | 4.18 → 3.87 | 0.5355 → 0.6920 | 0 | 0.000 | 0.03 | no | 15.94 |
| HippocampusMR_0015_0000 | elastix affine+B-spline | 0.6633 → 0.7063 | 4.18 → 3.68 | 0.5355 → 0.8729 | 4.31e-03 | 0.185 | 0.66 | no | 2.54 |
| HippocampusMR_0015_0000 | SimpleITK Demons | 0.6633 → 0.7445 | 4.18 → 3.87 | 0.5355 → 0.8004 | 0 | 0.074 | 0.14 | no | 1.90 |
| HippocampusMR_0016_0000 | ANTs SyNOnly | 0.6923 → 0.7588 | 3.08 → 3.08 | 0.5553 → 0.7359 | 0 | 0.000 | 0.04 | no | 19.81 |
| HippocampusMR_0016_0000 | elastix affine+B-spline | 0.6923 → 0.6740 | 3.08 → 4.56 | 0.5553 → 0.8814 | 7.91e-03 | 0.154 | 0.53 | no | 3.11 |
| HippocampusMR_0016_0000 | SimpleITK Demons | 0.6923 → 0.7561 | 3.08 → 3.00 | 0.5553 → 0.8192 | 0 | 0.063 | 0.09 | no | 2.77 |
| HippocampusMR_0017_0000 | ANTs SyNOnly | 0.5921 → 0.6478 | 3.30 → 4.50 | 0.5922 → 0.7644 | 0 | 0.000 | 0.03 | no | 16.45 |
| HippocampusMR_0017_0000 | elastix affine+B-spline | 0.5921 → 0.5266 | 3.30 → 12.02 | 0.5922 → 0.8983 | 4.56e-03 | 0.139 | 0.48 | no | 2.43 |
| HippocampusMR_0017_0000 | SimpleITK Demons | 0.5921 → 0.6010 | 3.30 → 3.64 | 0.5922 → 0.7950 | 0 | 0.051 | 0.16 | no | 1.92 |
| HippocampusMR_0018_0000 | ANTs SyNOnly | 0.5801 → 0.6545 | 4.00 → 4.77 | 0.5459 → 0.7061 | 0 | 0.000 | 0.03 | no | 15.89 |
| HippocampusMR_0018_0000 | elastix affine+B-spline | 0.5801 → 0.5129 | 4.00 → 8.51 | 0.5459 → 0.9009 | 4.94e-02 | 0.077 | 0.87 | no | 2.80 |
| HippocampusMR_0018_0000 | SimpleITK Demons | 0.5801 → 0.7011 | 4.00 → 3.66 | 0.5459 → 0.7896 | 0 | 0.067 | 0.11 | no | 2.02 |
| HippocampusMR_0019_0000 | ANTs SyNOnly | 0.6498 → 0.8046 | 2.72 → 2.58 | 0.5565 → 0.7364 | 0 | 0.000 | 0.03 | no | 15.22 |
| HippocampusMR_0019_0000 | elastix affine+B-spline | 0.6498 → 0.7667 | 2.72 → 3.62 | 0.5565 → 0.8955 | 1.69e-02 | 0.135 | 0.80 | no | 2.42 |
| HippocampusMR_0019_0000 | SimpleITK Demons | 0.6498 → 0.8128 | 2.72 → 2.50 | 0.5565 → 0.8504 | 0 | 0.054 | 0.10 | no | 1.94 |
| HippocampusMR_0020_0000 | ANTs SyNOnly | 0.6127 → 0.7542 | 3.56 → 3.07 | 0.4778 → 0.6479 | 0 | 0.000 | 0.03 | no | 16.34 |
| HippocampusMR_0020_0000 | elastix affine+B-spline | 0.6127 → 0.7400 | 3.56 → 3.00 | 0.4778 → 0.9011 | 5.70e-02 | 0.163 | 1.23 | no | 4.20 |
| HippocampusMR_0020_0000 | SimpleITK Demons | 0.6127 → 0.7591 | 3.56 → 3.41 | 0.4778 → 0.7674 | 0 | 0.048 | 0.08 | no | 2.02 |
| HippocampusMR_0021_0000 | ANTs SyNOnly | 0.5577 → 0.7223 | 5.00 → 4.00 | 0.4459 → 0.5279 | 0 | 0.000 | 0.03 | no | 14.46 |
| HippocampusMR_0021_0000 | elastix affine+B-spline | 0.5577 → 0.6368 | 5.00 → 5.34 | 0.4459 → 0.8346 | 3.49e-02 | 0.148 | 0.93 | no | 2.21 |
| HippocampusMR_0021_0000 | SimpleITK Demons | 0.5577 → 0.7196 | 5.00 → 4.00 | 0.4459 → 0.7939 | 0 | 0.054 | 0.10 | no | 1.70 |

## Abdomen CT — per-pair evidence

### SIREN PD-LK

| Pair | Endpoint PSNR min (dB) | Dice pre → post | HD95 pre → post (mm) | NCC pre → post | Folding | Flow conv. | Max residual | Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AbdomenCTCT_0002_0000 | 25.85 | 0.2920 → 0.5254 | 26.35 → 21.93 | 0.7039 → 0.9132 | 2.33e-04 | 16/16 | 1.99e-05 | 66.14 |
| AbdomenCTCT_0003_0000 | 25.06 | 0.2134 → 0.3499 | 54.68 → 45.36 | 0.5617 → 0.8726 | 6.70e-04 | 16/16 | 1.96e-05 | 63.73 |
| AbdomenCTCT_0004_0000 | 25.86 | 0.2047 → 0.3173 | 36.07 → 33.23 | 0.6542 → 0.8911 | 1.79e-04 | 16/16 | 1.98e-05 | 62.51 |
| AbdomenCTCT_0005_0000 | 25.78 | 0.0572 → 0.1007 | 86.20 → 86.37 | 0.5181 → 0.8349 | 7.14e-04 | 15/16 | 2.01e-05 | 64.67 |
| AbdomenCTCT_0006_0000 | 25.73 | 0.3307 → 0.4605 | 27.99 → 22.40 | 0.7116 → 0.9099 | 1.46e-04 | 16/16 | 2.00e-05 | 62.43 |
| AbdomenCTCT_0007_0000 | 25.11 | 0.1887 → 0.3022 | 45.47 → 46.37 | 0.6481 → 0.8752 | 2.13e-04 | 15/16 | 2.02e-05 | 63.26 |
| AbdomenCTCT_0008_0000 | 25.95 | 0.2834 → 0.4020 | 31.58 → 30.15 | 0.7315 → 0.9159 | 2.09e-04 | 16/16 | 2.00e-05 | 61.90 |
| AbdomenCTCT_0009_0000 | 25.16 | 0.1399 → 0.1953 | 89.09 → 87.08 | 0.5567 → 0.8107 | 9.44e-05 | 16/16 | 1.97e-05 | 61.23 |
| AbdomenCTCT_0010_0000 | 25.43 | 0.3386 → 0.4435 | 35.28 → 32.55 | 0.6606 → 0.8741 | 5.05e-05 | 16/16 | 1.98e-05 | 63.34 |
| AbdomenCTCT_0011_0000 | 25.88 | 0.2669 → 0.4481 | 34.28 → 28.07 | 0.6931 → 0.9164 | 1.95e-04 | 16/16 | 1.97e-05 | 62.24 |

### Classical methods

| Pair | Method | Dice pre → post | HD95 pre → post (mm) | NCC pre → post | Folding | Outside | Composition (mm) | Inv. conv. | Time (s) |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| AbdomenCTCT_0002_0000 | ANTs SyNOnly | 0.2920 → 0.4704 | 26.35 → 22.72 | 0.7039 → 0.8301 | 0 | 0.000 | 0.06 | no | 484.62 |
| AbdomenCTCT_0002_0000 | elastix affine+B-spline | 0.2920 → 0.5193 | 26.35 → 34.16 | 0.7039 → 0.9333 | 3.80e-02 | 0.017 | 2.49 | no | 120.03 |
| AbdomenCTCT_0002_0000 | SimpleITK Demons | 0.2920 → 0.4448 | 26.35 → 22.76 | 0.7039 → 0.8491 | 0 | 0.013 | 0.08 | no | 49.79 |
| AbdomenCTCT_0003_0000 | ANTs SyNOnly | 0.2134 → 0.3276 | 54.68 → 51.24 | 0.5617 → 0.7061 | 0 | 0.000 | 0.09 | no | 432.66 |
| AbdomenCTCT_0003_0000 | elastix affine+B-spline | 0.2134 → 0.2746 | 54.68 → 60.46 | 0.5617 → 0.9021 | 5.49e-02 | 0.313 | 11.08 | no | 109.82 |
| AbdomenCTCT_0003_0000 | SimpleITK Demons | 0.2134 → 0.3006 | 54.68 → 50.72 | 0.5617 → 0.7377 | 0 | 0.023 | 0.12 | no | 51.76 |
| AbdomenCTCT_0004_0000 | ANTs SyNOnly | 0.2047 → 0.2764 | 36.07 → 35.57 | 0.6542 → 0.7632 | 0 | 0.000 | 0.04 | no | 432.22 |
| AbdomenCTCT_0004_0000 | elastix affine+B-spline | 0.2047 → 0.3894 | 36.07 → 36.86 | 0.6542 → 0.9279 | 4.54e-02 | 0.078 | 3.51 | no | 109.08 |
| AbdomenCTCT_0004_0000 | SimpleITK Demons | 0.2047 → 0.2681 | 36.07 → 34.93 | 0.6542 → 0.8036 | 0 | 0.016 | 0.09 | no | 50.07 |
| AbdomenCTCT_0005_0000 | ANTs SyNOnly | 0.0572 → 0.0855 | 86.20 → 86.26 | 0.5181 → 0.6581 | 0 | 0.000 | 0.10 | no | 434.23 |
| AbdomenCTCT_0005_0000 | elastix affine+B-spline | 0.0572 → 0.1177 | 86.20 → 103.92 | 0.5181 → 0.8794 | 9.47e-02 | 0.307 | 13.90 | no | 110.17 |
| AbdomenCTCT_0005_0000 | SimpleITK Demons | 0.0572 → 0.0729 | 86.20 → 86.20 | 0.5181 → 0.6880 | 0 | 0.017 | 0.10 | no | 46.91 |
| AbdomenCTCT_0006_0000 | ANTs SyNOnly | 0.3307 → 0.5666 | 27.99 → 23.24 | 0.7116 → 0.8637 | 0 | 0.000 | 0.15 | no | 430.90 |
| AbdomenCTCT_0006_0000 | elastix affine+B-spline | 0.3307 → 0.4732 | 27.99 → 27.06 | 0.7116 → 0.9266 | 3.48e-02 | 0.191 | 5.02 | no | 110.83 |
| AbdomenCTCT_0006_0000 | SimpleITK Demons | 0.3307 → 0.4352 | 27.99 → 26.18 | 0.7116 → 0.8522 | 0 | 0.023 | 0.12 | no | 50.75 |
| AbdomenCTCT_0007_0000 | ANTs SyNOnly | 0.1887 → 0.2470 | 45.47 → 44.00 | 0.6481 → 0.8023 | 0 | 0.000 | 0.14 | no | 399.04 |
| AbdomenCTCT_0007_0000 | elastix affine+B-spline | 0.1887 → 0.2404 | 45.47 → 56.22 | 0.6481 → 0.9003 | 5.65e-02 | 0.226 | 8.45 | no | 103.62 |
| AbdomenCTCT_0007_0000 | SimpleITK Demons | 0.1887 → 0.2345 | 45.47 → 45.79 | 0.6481 → 0.7973 | 0 | 0.023 | 0.10 | no | 43.40 |
| AbdomenCTCT_0008_0000 | ANTs SyNOnly | 0.2834 → 0.3975 | 31.58 → 28.49 | 0.7315 → 0.8501 | 0 | 0.000 | 0.09 | no | 381.74 |
| AbdomenCTCT_0008_0000 | elastix affine+B-spline | 0.2834 → 0.3415 | 31.58 → 38.08 | 0.7315 → 0.9293 | 3.81e-02 | 0.041 | 2.69 | no | 105.05 |
| AbdomenCTCT_0008_0000 | SimpleITK Demons | 0.2834 → 0.3601 | 31.58 → 29.71 | 0.7315 → 0.8593 | 0 | 0.018 | 0.09 | no | 42.05 |
| AbdomenCTCT_0009_0000 | ANTs SyNOnly | 0.1399 → 0.2489 | 89.09 → 85.87 | 0.5567 → 0.6931 | 0 | 0.000 | 0.15 | no | 369.98 |
| AbdomenCTCT_0009_0000 | elastix affine+B-spline | 0.1399 → 0.1801 | 89.09 → 109.35 | 0.5567 → 0.8585 | 7.82e-02 | 0.366 | 16.43 | no | 106.64 |
| AbdomenCTCT_0009_0000 | SimpleITK Demons | 0.1399 → 0.1687 | 89.09 → 88.43 | 0.5567 → 0.6942 | 0 | 0.027 | 0.11 | no | 44.11 |
| AbdomenCTCT_0010_0000 | ANTs SyNOnly | 0.3386 → 0.4518 | 35.28 → 32.18 | 0.6606 → 0.7893 | 0 | 0.000 | 0.07 | no | 374.71 |
| AbdomenCTCT_0010_0000 | elastix affine+B-spline | 0.3386 → 0.3877 | 35.28 → 40.48 | 0.6606 → 0.8931 | 6.50e-02 | 0.190 | 7.62 | no | 107.45 |
| AbdomenCTCT_0010_0000 | SimpleITK Demons | 0.3386 → 0.3969 | 35.28 → 33.99 | 0.6606 → 0.8043 | 0 | 0.032 | 0.13 | no | 41.56 |
| AbdomenCTCT_0011_0000 | ANTs SyNOnly | 0.2669 → 0.4655 | 34.28 → 32.68 | 0.6931 → 0.8201 | 0 | 0.000 | 0.12 | no | 376.32 |
| AbdomenCTCT_0011_0000 | elastix affine+B-spline | 0.2669 → 0.4400 | 34.28 → 33.52 | 0.6931 → 0.9295 | 4.99e-02 | 0.047 | 3.56 | no | 103.47 |
| AbdomenCTCT_0011_0000 | SimpleITK Demons | 0.2669 → 0.3669 | 34.28 → 31.87 | 0.6931 → 0.8302 | 0 | 0.018 | 0.09 | no | 41.38 |

## Qualifications and interpretation limits

- These are single-seed (42) point estimates. The ± values in the cohort table are population standard deviations over pairs, not confidence intervals, and do not quantify seed variability.
- The identity row applies no deformation; it is the released affine pre-alignment baseline, not a random-overlap baseline.
- SIREN flow convergence is reported per interval. The configured recovery tolerance is `2e-5`; a pair with one or more intervals above it is still scored, and the near-miss is shown rather than silently rerun at a looser tolerance. Flow non-convergence does not by itself invalidate the recovered map, but it qualifies it.
- OASIS uses the archived calibrated weight 0.1. That calibration was single-seed and came from a different harness; it is the best previously supported value, not a cross-dataset optimum.
- Classical toolkits retain their own randomness and threading. Their maps are not claimed to be deterministically reproducible. Elastix includes an affine stage whereas SyNOnly does not, so the comparison is between named local protocols, not between universally optimal configurations.
- The classical forward map is a numerical inversion of the returned inverse displacement. Every classical pair records an unconverged inversion at the 30-iteration tolerance, so the classical composition column is an approximate diagnostic only; it is not evidence about invertibility, and the inverse maps used for warping and label scores are not affected by that diagnostic. A failed inversion is not presented as an exact inverse.
- LPBA40 has no ground-truth deformation in the prepared tree. Its evidence is label overlap and image similarity only; no target-registration error is reported.
- The LPBA prepared-data preprocessing provenance remains unresolved. This benchmark reads the existing prepared bytes and does not resample, renormalize, or rewrite them.
- All label scores use the released training-split labels. They are not hidden-test Learn2Reg leaderboard evaluations and are not comparable to published test-set numbers.
- No method is declared superior from this single-seed comparison. Differences smaller than the observed spread, or arising from one cohort or a gate-failing subset, are not treated as evidence of a general advantage.

## Reproduction

```bash
# Use an environment with the project and its classical extras installed.
# Keep the evidence above intact: every reproduction needs fresh directories.
benchmark_root="results/benchmark_$(date -u +%Y%m%dT%H%M%SZ)_$$"

# SIREN baselines (GPU), one cohort at a time
for d in lpba oasis hippocampus abdomenct; do
  python -m medical register --dataset "$d" \
    --output "$benchmark_root/siren/$d"
done

# Classical comparisons (CPU); each cohort can run independently
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=8 OMP_NUM_THREADS=8
for d in lpba oasis hippocampus abdomenct; do
  python -m medical baseline --dataset "$d" \
    --methods ants_syn elastix_bspline sitk_demons \
    --output "$benchmark_root/classical/$d"
done
```

`register` fits the two endpoint SIRENs, applies the 25 dB endpoint gate, recovers the 16-interval flow/RK2 map, and evaluates labels. `baseline` runs the three toolkits on the same images and evaluation. Both write only compact evidence.

## Artifact and dataset preservation

A SHA-256 manifest of all 2,343 files under `Data/` (3,405,406,449 bytes, about 3.17 GiB) was recorded before the benchmark and recomputed after it. The two manifests are byte-identical; no dataset path, file, or byte was written, renamed, or regenerated. The scratch area used for temporary recovery velocities is under the ignored results tree and was removed after each pair.

The benchmark tree `results/benchmark_20261110/` contains 83,266,043 bytes (about 79.4 MiB): 49 final endpoint checkpoints (`endpoints.pt`), 718 JSON evidence files (per-pair metrics, per-label metrics, flow diagnostics, status, configs, manifests, summaries), 57 CSV logs, and 6 run logs. There are no `.nii`, `.nii.gz`, `.npy`, `.npz`, or image artifacts in the tree. The tree is ignored by Git and is not part of the commit.

The subsequent handover cleanup preserves that entire benchmark tree byte-for-byte. Preliminary validation files and their five endpoint checkpoints are archived intact under the ignored `.local_archive/benchmark_cleanup_20261110/validation/`; a separate manifest retains their paths relative to the original `results/validation/` root and SHA-256 hashes. Dataset and benchmark preservation checks are recorded in that local cleanup receipt. Only disposable package bytecode is removed.

Dense maps, warped volumes, trajectories, and persistent velocity arrays are not saved anywhere by the pipeline.
