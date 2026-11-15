# Default-method baselines across five prepared cohorts and a bounded shared-configuration sweep

**Report date:** 2026-11-12 (Asia/Hong_Kong)
**Benchmark code revision:** `4055b071fbf861e3ee46b68937e864a91bbad23d`, worktree dirty (the uncommitted
objective/configuration changes are part of this study; every run records `code.dirty=true`).
**Environment:** torch 2.14.1+cu132, numpy 2.3.5, nibabel 5.4.2, MONAI 1.6.1, antspyx 0.6.3,
itk-elastix 0.25.4, SimpleITK 2.5.7; one NVIDIA GeForce RTX 5090 (32 GB), 32 CPU cores.
**Seed:** 42 for every SIREN fit.

This report covers the repository's current packaged default method on all five prepared
cohorts — LPBA40, OASIS, Hippocampus MR, Abdomen CT and Lung CT — followed by a
pre-declared, bounded sweep over the PD-LK coefficient \(c\), the local box fraction and
the recovery smoothness \(s\).

**Summary.** The packaged default was rerun end-to-end on all 69 pairs of the five prepared
cohorts. Mean Dice / HD95 are LPBA40 0.7112 / 6.14 mm, OASIS 0.7660 / 2.87 mm, Hippocampus
0.7711 / 3.11 mm, Abdomen CT 0.4104 / 40.26 mm and Lung CT 0.8123 / 48.52 mm. The method leads
the reused classical rows on Hippocampus and Abdomen CT, trails ANTs SyNOnly on LPBA40 Dice
(though with slightly better HD95) and on OASIS, and still fails on Lung CT, where post-recovery
Dice is below identity and HD95 roughly doubles. A
pre-declared 27-candidate cohort-balanced sweep over `(c, box fraction, recovery smoothness)`
did not produce a configuration that met every promotion condition: the best-Dice eligible
candidate gained 0.0038 Dice but worsened cohort-balanced HD95 by 0.61 mm, while a markedly
more converged candidate (13/15 versus 7/15 fully converged pairs) gained only 0.0004 Dice. The packaged `(4, 0.5, .01)` default is therefore retained,
the packaged JSON files are untouched, and no tuning image is presented as independent
validation.

## Evidence status

* **All SIREN numbers in this report are new runs** made on 2026-11-11/12 UTC under
  `results/default_method_benchmark_20261112/`. Earlier SIREN benchmark numbers are **not
  pooled**: small implementation details of the objective/training code changed after
  revision `44f1ca2`, and the current packaged objective is the shared-grid PD-LK plus
  local box NCC default rather than the earlier pointwise PD-LK baseline.
* **Classical numbers are reused** from `results/benchmark_20261110/classical/`, produced at
  clean revision `44f1ca2bda8eddec4023c35d1a3a8270f328746c`. The reuse is matched on pair
  membership, preprocessing, canonical-grid handling, baseline settings and metric version 2;
  the shared `evaluation.py` change since that revision only adds the explicit PSNR-gate skip
  and a non-finite reconstruction guard, neither of which touches classical registration or
  label metrics. Every reused row is marked `reused`. No classical run was made for Lung CT,
  so that cell is explicitly unavailable rather than imputed.
* **Tuning-subset results are kept separate from held-out results.** The three central pairs
  per cohort used for tuning are a deterministic subset. No candidate met the pre-declared
  promotion rule, so no configuration was frozen or rerun; had one been promoted, it would have
  been validated only by a fresh full-cohort rerun, never by the tuning numbers. No independent
  validation is claimed on tuning images.

## Method actually executed

The packaged configuration is MSE endpoint reconstruction plus **shared-grid strict PD-LK**
(`pdlk_sampling="paired_grid"`) plus **bidirectional box NCC** registration, followed by the
standard separate **global brightness-flow PCG + midpoint RK2** recovery. The full equations and
coordinate conventions are in [METHOD.md](../METHOD.md); the executed settings are:

| Setting | Value |
|---|---|
| Endpoint model | paired scalar SIREN, width 256, four sine layers, `omega=30`, shared initialization |
| Trajectory | linear weight interpolation, \(I(x,t)=f_{(1-t)\theta_M+t\theta_F}(x)\) |
| Endpoint loss | moving + fixed intensity MSE on a shared 32,768-voxel minibatch |
| PD-LK objective | shared-grid strict PD-LK, `weight=0.1`, `rho=0`, `c=4`, `eta=1e-6`, 8³ cells, two symmetric pairs per cell, `pair_offset_fraction=0.2` |
| Local registration | bidirectional box NCC, `weight=0.05`, half-size crop (`crop_fraction=0.5`), 4,096 particles per direction, central-80% starts, two Euler steps, three-tap `sigma_cells=0.75` Gaussian on the cell grid |
| Reconstruction samples | separate from registration particles (`reconstruction_at_particles=false`) |
| Optimizer | 5,000 Adam steps, batch 32,768, learning rate 2e-4 with final factor 0.25, global gradient-norm clip 1.0, seed 42 |
| Gradient handling | one summed backward pass, no derivative checkpoint recomputation, no separate term cap |
| Recovery | 16 interval midpoints, `smoothness=0.01`, `ridge=1e-6`, PCG diagonal preconditioning, max 500 iterations, relative tolerance 2e-5, four midpoint-RK2 substeps, first interval from zero, later intervals from the preceding solution |
| Endpoint gate | explicitly skipped in the packaged presets (`skip_psnr_gate=true`); `endpoint_quality_pass` is `null`, never `passed` |
| Label metric (version 2) | MONAI macro Dice over the union of nonzero labels (`ignore_empty=False`); symmetric HD95 in mm, maximum of the two directed 95th-percentile surface distances |
| Folding | fraction of voxels with deformation Jacobian determinant \(\le 0\) |

The five packaged JSON configs (`medical/configs/{lpba,oasis,hippocampus,abdomenct,lungct}.json`)
were used unchanged. They share the values above and differ only in dataset selection,
preprocessing (MR versus CT window), expected geometry and expected shape/spacing.

### Cohorts and membership

| Cohort | Fixed | Moving | Pairs | Preprocessing |
|---|---|---|---:|---|
| LPBA40 | S01 | S02–S10 | 9 | MR nonzero-foreground 1st/99th percentile to [0,1], zero background |
| OASIS | `OASIS_0001_0000` | `OASIS_0002_0000`–`OASIS_0011_0000` | 10 | same MR normalization; 160×224×192, 1 mm |
| Hippocampus MR | `HippocampusMR_0001_0000` | `HippocampusMR_0002_0000`–`HippocampusMR_0021_0000` | 20 | same MR normalization; 64³, 1 mm |
| Abdomen CT | `AbdomenCTCT_0001_0000` | `AbdomenCTCT_0002_0000`–`AbdomenCTCT_0011_0000` | 10 | CT window [-160, 240] HU, no background zeroing; 192×160×256, 2 mm |
| Lung CT | each case's `_0000` (expiration) | same case's `_0001` (inspiration) | 20 | CT window [-1000, 400] HU, no background zeroing; 192×192×208, 1.75/1.25/1.75 mm |

Lung CT labels are the released lung masks (`masksTr`), not a dense anatomical segmentation;
they are used only for evaluation and are the same arrays the historical lung cohort used.

### Execution and artifacts

Commands (one fresh ignored output directory per cohort):

```bash
python -m medical register --dataset lpba        --output results/default_method_benchmark_20261112/lpba_register
python -m medical register --dataset oasis       --output results/default_method_benchmark_20261112/oasis_register
python -m medical register --dataset hippocampus --output results/default_method_benchmark_20261112/hippocampus_register
python -m medical register --dataset abdomenct   --output results/default_method_benchmark_20261112/abdomenct_register
python -m medical register --dataset lungct      --output results/default_method_benchmark_20261112/lungct_register
```

Run directories are ignored by Git and contain only final endpoint weights and compact scalar
evidence: per-pair `endpoints.pt`, `metrics.json`, `per_label_metrics.json`,
`flow_diagnostics.json`, `training.csv`, plus per-run `config.json`, `manifest.json` and
`summary.json`/`summary.csv`. No deformation maps, warped volumes or persistent velocity arrays
are written. Recovery velocity scratch lives under ignored `results/.scratch/` and is removed
after each pair. `Data/` was not written to (see the preservation check at the end).

## Current default: aggregate results

All 69 attempted SIREN pairs completed; there were no gate failures because the gate was
explicitly skipped by the packaged presets. Values are means over scored pairs; Dice SD is
across pairs. Endpoint PSNR is the diagnostic minimum of the two endpoint reconstructions in
that cohort.

| Cohort | Scored / attempted | Dice mean ± SD | HD95 mean (mm) | Folding mean (max) | Min endpoint PSNR src/tgt (dB) | Train s/pair | Recovery s/pair | Total s/pair | PCG pairs all-converged |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LPBA40 | 9 / 9 | 0.7112 ± 0.0108 | 6.14 | 0 (0) | 29.68 / 30.52 | 58.1 | 11.1 | 73.7 | 6 / 9 |
| OASIS | 10 / 10 | 0.7660 ± 0.0298 | 2.87 | 1.89e-06 (8.28e-06) | 27.64 / 27.64 | 58.6 | 20.0 | 83.6 | 3 / 10 |
| Hippocampus MR | 20 / 20 | 0.7711 ± 0.0476 | 3.11 | 1.93e-05 (1.72e-04) | 24.10 / 26.89 | 57.9 | 0.5 | 58.6 | 19 / 20 |
| Abdomen CT | 10 / 10 | 0.4104 ± 0.1285 | 40.26 | 3.35e-05 (1.02e-04) | 20.68 / 21.31 | 58.6 | 25.1 | 87.3 | 2 / 10 |
| Lung CT | 20 / 20 | 0.8123 ± 0.1892 | 48.52 | 1.99e-03 (4.75e-03) | 20.81 / 22.21 | 57.9 | 22.3 | 83.2 | 14 / 20 |

Identity (no-deformation) macro Dice / HD95 for the same pairs: LPBA40 0.5479 / 7.94 mm, OASIS 0.4925 / 5.28 mm, Hippocampus 0.6083 / 3.68 mm, Abdomen CT 0.2316 / 46.70 mm, Lung CT 0.8522 / 26.55 mm.

### Reused classical comparison

Classical rows are the previously completed, matched full-cohort runs at revision `44f1ca2`;
they are shown for context and are not new measurements. Toolkits received the same selected
arrays, preprocessing and evaluation code, with ANTs SyNOnly, elastix affine+B-spline and
SimpleITK diffeomorphic Demons at the packaged settings. Deterministic classical reproduction
was never claimed.

| Cohort | Method | n | Dice | HD95 (mm) | Folding | Evidence |
|---|---|---:|---:|---:|---:|---|
| LPBA40 | ANTs SyNOnly | 9 | 0.7204 | 6.21 | 0 | reused |
| LPBA40 | elastix affine+B-spline | 9 | 0.6900 | 6.36 | 8.82e-07 | reused |
| LPBA40 | SimpleITK Demons | 9 | 0.6648 | 6.87 | 0 | reused |
| OASIS | ANTs SyNOnly | 10 | 0.7968 | 2.65 | 0 | reused |
| OASIS | elastix affine+B-spline | 10 | 0.7097 | 3.53 | 1.34e-03 | reused |
| OASIS | SimpleITK Demons | 10 | 0.7896 | 2.86 | 0 | reused |
| Hippocampus MR | ANTs SyNOnly | 20 | 0.7447 | 3.29 | 0 | reused |
| Hippocampus MR | elastix affine+B-spline | 20 | 0.6770 | 4.59 | 4.23e-02 | reused |
| Hippocampus MR | SimpleITK Demons | 20 | 0.7450 | 3.30 | 0 | reused |
| Abdomen CT | ANTs SyNOnly | 10 | 0.3537 | 44.23 | 0 | reused |
| Abdomen CT | elastix affine+B-spline | 10 | 0.3364 | 54.01 | 5.56e-02 | reused |
| Abdomen CT | SimpleITK Demons | 10 | 0.3049 | 45.06 | 0 | reused |
| Lung CT | ANTs / elastix / Demons | — | — | — | — | **unavailable** |

The classical forward map is an approximate numerical inversion of the inverse displacement;
the reused runs recorded that this inversion did not converge on any pair. That diagnostic
qualifies the classical rows but does not affect their Dice/HD95.

### Pair-level outcomes and interpretation

| Cohort | Dice improved vs identity | HD95 improved vs identity | Identity → post Dice | Identity → post HD95 (mm) |
|---|---:|---:|---:|---:|
| LPBA40 | 9 / 9 | 9 / 9 | 0.5479 → 0.7112 | 7.94 → 6.14 |
| OASIS | 10 / 10 | 10 / 10 | 0.4925 → 0.7660 | 5.28 → 2.87 |
| Hippocampus MR | 20 / 20 | 16 / 20 | 0.6083 → 0.7711 | 3.68 → 3.11 |
| Abdomen CT | 10 / 10 | 9 / 10 | 0.2316 → 0.4104 | 46.70 → 40.26 |
| Lung CT | 12 / 20 | 9 / 20 | 0.8522 → 0.8123 | 26.55 → 48.52 |

* **LPBA40.** The default reaches 0.7112 / 6.14 mm. It remains below the reused ANTs SyNOnly
  row on Dice (0.7204) but is slightly better on HD95 (6.14 versus 6.21 mm), and it is above
  elastix (0.6900 / 6.36) and Demons (0.6648 / 6.87). All nine pairs improve on both metrics
  relative to identity.
* **OASIS.** 0.7660 / 2.87 mm is a clear improvement over identity, but it trails the reused
  ANTs (0.7968 / 2.65) and Demons (0.7896 / 2.86) rows on Dice; this cohort is the main
  regression relative to the earlier pointwise-PD-LK baseline (0.7915 / 2.57).
* **Hippocampus MR.** 0.7711 / 3.11 mm leads all three reused classical rows on both metrics
  (ANTS 0.7447 / 3.29, Demons 0.7450 / 3.30), but HD95 worsened on 4 of 20 pairs even though
  Dice improved on all 20.
* **Abdomen CT.** 0.4104 / 40.26 mm leads the reused classical rows (ANTS 0.3537 / 44.23,
  elastix 0.3364 / 54.01, Demons 0.3049 / 45.06) and improves over identity on every pair's
  Dice, though HD95 worsened on one pair. Two pairs remain very poor (0005 and 0009 above
  73 mm), and PCG fully converges on only 2 of 10 pairs.
* **Lung CT.** This cohort still fails as a registration benchmark. Post-recovery Dice
  (0.8123) is **below** the identity value (0.8522), and HD95 rises from 26.55 to 48.52 mm.
  Dice improves on only 12 of 20 pairs and HD95 on only 9. The worst cases (0001, 0003, 0015,
  0017, 0020) collapse to 0.46–0.60 Dice and 109–131 mm HD95; 0008 and 0019 are also poor
  (0.69 and 0.64 Dice). Folding reaches 0.47% on the worst pair. No classical lung row exists
  for comparison. This repeats the historical lung failure and is
  reported as a failure, not as a success masked by the other cohorts.

### What changed relative to the earlier pointwise-PD-LK baseline

The table below is **historical context, not pooled**. It compares the new default with the
2026-11-10 fresh benchmark, which used the same model/recovery but the earlier pointwise
independent PD-LK objective. The objective and several small implementation details differ, so
the deltas are descriptive, not a controlled causal attribution.

| Cohort | Dice old → new | Δ Dice | HD95 old → new (mm) | Δ HD95 (mm) | Folding old → new | PCG pairs old → new | Total s/pair old → new |
|---|---:|---:|---:|---:|---:|---:|---:|
| LPBA40 | 0.6985 → 0.7112 | +0.0127 | 6.33 → 6.14 | −0.19 | 0 → 0 | 6/9 → 6/9 | 66.4 → 73.7 |
| OASIS | 0.7915 → 0.7660 | −0.0255 | 2.57 → 2.87 | +0.29 | 4.48e-06 → 1.89e-06 | 5/10 → 3/10 | 75.6 → 83.6 |
| Hippocampus MR | 0.7525 → 0.7711 | +0.0186 | 3.20 → 3.11 | −0.08 | 2.38e-04 → 1.93e-05 | 20/20 → 19/20 | 36.6 → 58.6 |
| Abdomen CT | 0.3545 → 0.4104 | +0.0559 | 43.35 → 40.26 | −3.09 | 2.70e-04 → 3.35e-05 | 8/10 → 2/10 | 63.1 → 87.3 |
| Lung CT† | 0.8566 → 0.8123 | −0.0443 | 35.90 → 48.52 | +12.62 | 5.08e-03 → 1.99e-03 | — → 14/20 | 81.1 → 83.2 |

† The old Lung CT value is the 2026-11-07 probe recipe (`lambda=.01`, pointwise PD-LK), not the
2026-11-10 fresh benchmark, which excluded Lung CT. That probe passed all 20 endpoint gates but
made HD95 worse than identity; its convergence was not logged in the archived aggregate. All
other "old" values are the 2026-11-10 fresh benchmark at revision `44f1ca2`. "New" values are the
2026-11-12 rerun above. Folding is the mean fraction; PCG counts pairs whose 16 intervals all
converged; total time includes loading and evaluation. The objective and small implementation
details differ, so the deltas are descriptive, not a controlled ablation. Classical rows are
unchanged/reused and are therefore not shown as an old/new pair.

The current default is therefore **not uniformly better** than the earlier objective: it
improves LPBA40, Hippocampus and Abdomen CT, but degrades OASIS and Lung CT. This is a
cross-cohort transfer risk of a configuration selected from LPBA screens, and it is the reason
the tuning study below uses a cohort-balanced criterion rather than LPBA alone.

### Reconstruction side of the regressions

The new objective lowered endpoint reconstruction in every cohort (mean of the minimum of the
two endpoint PSNRs; endpoint MSE is the mean over both endpoints; post-warp NCC is the recovered
image similarity):

| Cohort | Endpoint PSNR min old → new (dB) | Endpoint MSE mean old → new | Post-warp NCC old → new | r(ΔPSNR, ΔDice) | r(ΔPSNR, ΔHD95) |
|---|---:|---:|---:|---:|---:|
| LPBA40 | 32.10 → 30.51 | 5.70e-04 → 8.33e-04 | 0.9886 → 0.9879 | −0.40 | +0.59 |
| OASIS | 30.20 → 28.06 | 9.06e-04 → 1.52e-03 | 0.9789 → 0.9751 | +0.82 | −0.91 |
| Hippocampus MR | 33.76 → 25.49 | 3.17e-04 → 2.23e-03 | 0.8408 → 0.8687 | +0.31 | −0.50 |
| Abdomen CT | 25.58 → 21.47 | 2.60e-03 → 6.79e-03 | 0.8814 → 0.8861 | −0.32 | +0.36 |
| Lung CT† | 27.41 → 22.22 | — | 0.8176 → 0.8494 | +0.35 | −0.30 |

† Lung's old values are the 2026-11-07 probe recipe, not the 2026-11-10 fresh benchmark; its
archived aggregate does not contain an endpoint MSE. The correlations are Pearson across that
cohort's pairs; with 9–20 pairs they are descriptive only. At first sight OASIS looks like a reconstruction-linked regression: all 10 pairs lost Dice,
the PSNR drop correlates positively with the Dice drop (+0.82) and negatively with the HD95
change (−0.91), and post-warp NCC fell on every pair. The 27-candidate sweep, however, shows the
opposite relationship **across configurations**:

| Cohort | r(mean PSNR, mean Dice) across 27 candidates | r(mean PSNR, mean HD95) across 27 candidates |
|---|---:|---:|
| LPBA40 | +0.59 | +0.08 |
| OASIS | **−0.70** | **+0.77** |
| Hippocampus MR | +0.23 | −0.04 |
| Abdomen CT | +0.53 | +0.03 |
| Lung CT | **−0.27** | **+0.66** |

On OASIS, larger boxes raise endpoint PSNR (box 0.4 ≈ 27.3–27.4 dB, 0.5 ≈ 27.8 dB,
0.6 ≈ 28.2–28.3 dB) but lower Dice; the best OASIS candidate (c=6, box=0.4, s=0.01; Dice
0.7606) is among the lowest-PSNR candidates (27.39 dB on the same three pairs, versus 30.17 dB
for the old run on those pairs). So the reconstruction drop and the OASIS regression are
co-symptoms of the changed objective/box, not evidence that lower reconstruction caused the
regression. Lung shows the same warning: endpoint PSNR fell 5.2 dB, yet 9/20 pairs improved
Dice, post-warp NCC rose from 0.818 to 0.849, and across candidates higher PSNR went with worse
HD95 (+0.66). Hippocampus and Abdomen CT lost 8.3 and 4.1 dB of endpoint PSNR but improved Dice,
and their few regressed pairs had essentially the same PSNR drop as the improved pairs (Hippocampus
−8.48 versus −8.22 dB; Abdomen −4.10 versus −4.11 dB).

The honest summary is: the OASIS regression **is accompanied by** lower endpoint reconstruction
and lower post-warp NCC, but the tuning evidence does not support blaming reconstruction. Lower
reconstruction is neither necessary nor sufficient for a registration regression here, and the
skipped PSNR gate would not have caught the OASIS decline (its 27.6–28.9 dB endpoints are above
the nominal 25 dB threshold). These are correlational observations on one seed, not a causal
decomposition.

## Per-case results

Dice and HD95 are post-recovery; "pre" is the identity (no-deformation) label metric for the
same pair, computed evaluation-only. PCG convergence counts intervals meeting the configured
relative tolerance; the maximum residual is the largest `relative_linear_residual` over the 16
intervals. Times separate the 5,000-step fit from post-training recovery; total includes
loading and evaluation. Pair identifiers are abbreviated to the subject stem; the per-run
manifest retains the full `_0000`/`_0001` channel identifiers.

### LPBA40

| Pair | PSNR src/tgt (dB) | Endpoint MSE src/tgt | Dice (pre → post) | HD95 mm (pre → post) | Folding | PCG conv. | Max resid. | Train s | Recover s | Total s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S02 | 30.42 / 30.96 | 0.00091 / 0.00080 | 0.5630 → 0.7092 | 7.84 → 6.29 | 0 | 16/16 | 2.00e-05 | 61.9 | 11.6 | 78.2 |
| S03 | 31.27 / 31.05 | 0.00075 / 0.00079 | 0.5518 → 0.7160 | 8.09 → 6.38 | 0 | 16/16 | 2.00e-05 | 57.8 | 11.5 | 74.2 |
| S04 | 32.08 / 31.10 | 0.00062 / 0.00078 | 0.6163 → 0.7264 | 6.24 → 5.31 | 0 | 14/16 | 2.01e-05 | 57.4 | 12.2 | 74.4 |
| S05 | 29.68 / 30.52 | 0.00108 / 0.00089 | 0.5449 → 0.7025 | 7.51 → 6.06 | 0 | 14/16 | 2.00e-05 | 58.0 | 11.3 | 73.8 |
| S06 | 30.10 / 30.91 | 0.00098 / 0.00081 | 0.4740 → 0.7046 | 9.56 → 6.19 | 0 | 16/16 | 1.93e-05 | 57.5 | 10.4 | 71.6 |
| S07 | 31.62 / 30.96 | 0.00069 / 0.00080 | 0.5837 → 0.7169 | 8.10 → 5.89 | 0 | 15/16 | 2.00e-05 | 58.7 | 11.3 | 74.1 |
| S08 | 30.07 / 30.86 | 0.00098 / 0.00082 | 0.4974 → 0.6984 | 8.21 → 6.62 | 0 | 16/16 | 1.98e-05 | 57.2 | 10.9 | 72.3 |
| S09 | 31.53 / 31.13 | 0.00070 / 0.00077 | 0.5689 → 0.6999 | 7.66 → 6.81 | 0 | 16/16 | 1.99e-05 | 57.6 | 10.6 | 72.4 |
| S10 | 30.06 / 30.77 | 0.00099 / 0.00084 | 0.5313 → 0.7266 | 8.25 → 5.75 | 0 | 16/16 | 2.00e-05 | 57.2 | 10.5 | 71.9 |

### OASIS

| Pair | PSNR src/tgt (dB) | Endpoint MSE src/tgt | Dice (pre → post) | HD95 mm (pre → post) | Folding | PCG conv. | Max resid. | Train s | Recover s | Total s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OASIS_0002_0000 | 28.87 / 28.00 | 0.00130 / 0.00159 | 0.4587 → 0.7348 | 6.17 → 3.57 | 4.80e-06 | 13/16 | 2.02e-05 | 64.4 | 20.5 | 90.4 |
| OASIS_0003_0000 | 28.53 / 28.54 | 0.00140 / 0.00140 | 0.5456 → 0.8210 | 4.15 → 1.71 | 5.81e-07 | 14/16 | 2.03e-05 | 58.4 | 19.0 | 82.0 |
| OASIS_0004_0000 | 28.40 / 28.38 | 0.00145 / 0.00145 | 0.4983 → 0.7772 | 5.14 → 2.78 | 0 | 16/16 | 1.98e-05 | 57.8 | 19.6 | 82.2 |
| OASIS_0005_0000 | 27.88 / 27.99 | 0.00163 / 0.00159 | 0.4304 → 0.7542 | 6.48 → 3.64 | 2.47e-06 | 14/16 | 2.03e-05 | 57.9 | 20.4 | 83.0 |
| OASIS_0006_0000 | 28.33 / 27.64 | 0.00147 / 0.00172 | 0.4268 → 0.7481 | 5.86 → 3.14 | 8.28e-06 | 13/16 | 2.05e-05 | 58.5 | 20.9 | 84.7 |
| OASIS_0007_0000 | 27.64 / 27.80 | 0.00172 / 0.00166 | 0.5045 → 0.7684 | 5.69 → 3.27 | 1.60e-06 | 15/16 | 2.01e-05 | 58.3 | 20.3 | 83.7 |
| OASIS_0008_0000 | 28.13 / 27.87 | 0.00154 / 0.00163 | 0.5009 → 0.7237 | 4.93 → 2.79 | 0 | 16/16 | 1.97e-05 | 58.0 | 20.2 | 83.4 |
| OASIS_0009_0000 | 28.36 / 28.40 | 0.00146 / 0.00145 | 0.6093 → 0.8014 | 3.84 → 2.04 | 1.02e-06 | 16/16 | 1.95e-05 | 57.1 | 19.4 | 81.2 |
| OASIS_0010_0000 | 28.36 / 28.14 | 0.00146 / 0.00154 | 0.4819 → 0.7519 | 5.11 → 2.73 | 1.45e-07 | 14/16 | 2.03e-05 | 57.7 | 19.9 | 82.3 |
| OASIS_0011_0000 | 28.39 / 28.17 | 0.00145 / 0.00152 | 0.4684 → 0.7796 | 5.46 → 3.00 | 0 | 15/16 | 2.01e-05 | 57.8 | 20.0 | 83.1 |

### Hippocampus MR

| Pair | PSNR src/tgt (dB) | Endpoint MSE src/tgt | Dice (pre → post) | HD95 mm (pre → post) | Folding | PCG conv. | Max resid. | Train s | Recover s | Total s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HippocampusMR_0002 | 26.08 / 28.49 | 0.00247 / 0.00142 | 0.7893 → 0.7946 | 2.22 → 2.58 | 0 | 16/16 | 1.94e-05 | 64.1 | 1.0 | 65.5 |
| HippocampusMR_0003 | 25.65 / 28.46 | 0.00272 / 0.00143 | 0.7138 → 0.8216 | 2.45 → 2.21 | 0 | 16/16 | 1.94e-05 | 57.1 | 0.5 | 57.8 |
| HippocampusMR_0004 | 25.55 / 28.35 | 0.00279 / 0.00146 | 0.6052 → 0.8191 | 3.50 → 2.24 | 1.91e-05 | 16/16 | 1.66e-05 | 58.1 | 0.5 | 58.8 |
| HippocampusMR_0005 | 25.29 / 27.93 | 0.00296 / 0.00161 | 0.5527 → 0.7539 | 4.36 → 3.08 | 0 | 16/16 | 1.97e-05 | 58.2 | 0.5 | 58.9 |
| HippocampusMR_0006 | 24.95 / 28.16 | 0.00320 / 0.00153 | 0.6067 → 0.7693 | 3.61 → 3.00 | 3.43e-05 | 16/16 | 1.85e-05 | 57.4 | 0.5 | 58.0 |
| HippocampusMR_0007 | 25.76 / 27.14 | 0.00265 / 0.00193 | 0.5898 → 0.7723 | 3.37 → 3.38 | 3.81e-05 | 16/16 | 1.93e-05 | 57.7 | 0.5 | 58.4 |
| HippocampusMR_0008 | 24.92 / 27.94 | 0.00322 / 0.00161 | 0.6689 → 0.8062 | 3.58 → 2.00 | 0 | 16/16 | 1.81e-05 | 56.9 | 0.5 | 57.6 |
| HippocampusMR_0009 | 24.10 / 27.14 | 0.00389 / 0.00193 | 0.3495 → 0.7620 | 6.88 → 2.91 | 2.29e-05 | 16/16 | 1.98e-05 | 58.0 | 0.5 | 58.7 |
| HippocampusMR_0010 | 25.09 / 27.73 | 0.00310 / 0.00168 | 0.5723 → 0.8106 | 3.70 → 3.56 | 2.29e-05 | 16/16 | 1.65e-05 | 57.8 | 0.5 | 58.5 |
| HippocampusMR_0011 | 25.20 / 28.09 | 0.00302 / 0.00155 | 0.5961 → 0.7824 | 3.37 → 2.63 | 2.29e-05 | 16/16 | 1.93e-05 | 57.3 | 0.5 | 58.0 |
| HippocampusMR_0012 | 25.29 / 28.18 | 0.00296 / 0.00152 | 0.4509 → 0.7513 | 4.55 → 3.62 | 1.72e-04 | 16/16 | 1.96e-05 | 57.6 | 0.5 | 58.3 |
| HippocampusMR_0013 | 25.63 / 28.09 | 0.00274 / 0.00155 | 0.6678 → 0.7858 | 2.70 → 2.58 | 0 | 16/16 | 1.99e-05 | 58.0 | 0.5 | 58.7 |
| HippocampusMR_0014 | 24.85 / 27.33 | 0.00327 / 0.00185 | 0.6554 → 0.7887 | 3.39 → 2.82 | 0 | 15/16 | 2.00e-05 | 58.2 | 0.5 | 58.9 |
| HippocampusMR_0015 | 26.33 / 28.64 | 0.00233 / 0.00137 | 0.6633 → 0.7447 | 4.18 → 3.82 | 0 | 16/16 | 1.90e-05 | 57.0 | 0.5 | 57.6 |
| HippocampusMR_0016 | 25.36 / 28.17 | 0.00291 / 0.00152 | 0.6923 → 0.8111 | 3.08 → 2.62 | 3.43e-05 | 16/16 | 1.99e-05 | 57.8 | 0.5 | 58.5 |
| HippocampusMR_0017 | 28.08 / 26.89 | 0.00156 / 0.00205 | 0.5921 → 0.6309 | 3.30 → 4.92 | 3.81e-06 | 16/16 | 1.99e-05 | 57.7 | 0.5 | 58.3 |
| HippocampusMR_0018 | 27.08 / 27.72 | 0.00196 / 0.00169 | 0.5801 → 0.6800 | 4.00 → 5.15 | 0 | 16/16 | 1.98e-05 | 58.1 | 0.5 | 58.7 |
| HippocampusMR_0019 | 25.84 / 28.15 | 0.00261 / 0.00153 | 0.6498 → 0.8150 | 2.72 → 2.62 | 0 | 16/16 | 1.79e-05 | 56.5 | 0.5 | 57.2 |
| HippocampusMR_0020 | 24.79 / 27.87 | 0.00332 / 0.00163 | 0.6127 → 0.7854 | 3.56 → 2.87 | 1.53e-05 | 16/16 | 2.00e-05 | 57.5 | 0.5 | 58.1 |
| HippocampusMR_0021 | 25.24 / 28.09 | 0.00299 / 0.00155 | 0.5577 → 0.7380 | 5.00 → 3.66 | 0 | 16/16 | 1.94e-05 | 57.5 | 0.5 | 58.1 |

### Abdomen CT

| Pair | PSNR src/tgt (dB) | Endpoint MSE src/tgt | Dice (pre → post) | HD95 mm (pre → post) | Folding | PCG conv. | Max resid. | Train s | Recover s | Total s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AbdomenCTCT_0002 | 21.90 / 22.11 | 0.00646 / 0.00615 | 0.2920 → 0.5930 | 26.35 → 16.66 | 1.02e-04 | 16/16 | 1.99e-05 | 64.0 | 24.8 | 92.6 |
| AbdomenCTCT_0003 | 20.68 / 21.49 | 0.00855 / 0.00710 | 0.2134 → 0.3875 | 54.68 → 38.78 | 5.59e-05 | 10/16 | 2.14e-05 | 57.4 | 26.3 | 87.3 |
| AbdomenCTCT_0004 | 21.98 / 21.83 | 0.00634 / 0.00656 | 0.2047 → 0.4674 | 36.07 → 30.72 | 8.65e-06 | 11/16 | 2.06e-05 | 57.9 | 25.0 | 86.6 |
| AbdomenCTCT_0005 | 21.79 / 21.31 | 0.00663 / 0.00740 | 0.0572 → 0.1956 | 86.20 → 78.46 | 8.41e-05 | 6/16 | 2.42e-05 | 58.3 | 26.3 | 88.3 |
| AbdomenCTCT_0006 | 22.54 / 22.06 | 0.00558 / 0.00623 | 0.3307 → 0.4700 | 27.99 → 23.83 | 1.06e-05 | 15/16 | 2.06e-05 | 57.4 | 24.2 | 85.1 |
| AbdomenCTCT_0007 | 21.01 / 21.61 | 0.00792 / 0.00690 | 0.1887 → 0.2910 | 45.47 → 54.54 | 2.76e-05 | 14/16 | 2.09e-05 | 58.3 | 26.5 | 88.6 |
| AbdomenCTCT_0008 | 22.76 / 22.13 | 0.00530 / 0.00612 | 0.2834 → 0.4328 | 31.58 → 30.87 | 6.61e-06 | 16/16 | 2.00e-05 | 57.9 | 22.6 | 84.2 |
| AbdomenCTCT_0009 | 20.87 / 21.31 | 0.00818 / 0.00739 | 0.1399 → 0.2472 | 89.09 → 73.84 | 1.78e-05 | 3/16 | 2.64e-05 | 58.3 | 26.7 | 88.6 |
| AbdomenCTCT_0010 | 21.90 / 21.78 | 0.00645 / 0.00663 | 0.3386 → 0.4904 | 35.28 → 28.94 | 2.67e-06 | 13/16 | 2.07e-05 | 57.8 | 24.1 | 85.4 |
| AbdomenCTCT_0011 | 21.16 / 22.00 | 0.00766 / 0.00631 | 0.2669 → 0.5289 | 34.28 → 25.91 | 1.87e-05 | 14/16 | 2.03e-05 | 58.2 | 24.1 | 85.9 |

### Lung CT

| Pair | PSNR src/tgt (dB) | Endpoint MSE src/tgt | Dice (pre → post) | HD95 mm (pre → post) | Folding | PCG conv. | Max resid. | Train s | Recover s | Total s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LungCT_0001 | 21.86 / 25.16 | 0.00651 / 0.00305 | 0.7086 → 0.4699 | 49.00 → 109.70 | 4.09e-03 | 16/16 | 1.99e-05 | 63.3 | 25.9 | 92.5 |
| LungCT_0002 | 21.36 / 22.21 | 0.00731 / 0.00602 | 0.8378 → 0.8343 | 19.75 → 66.50 | 3.43e-03 | 16/16 | 1.98e-05 | 57.9 | 23.7 | 84.6 |
| LungCT_0003 | 20.81 / 23.26 | 0.00830 / 0.00472 | 0.8394 → 0.5913 | 56.74 → 110.84 | 4.19e-03 | 14/16 | 2.05e-05 | 57.1 | 22.6 | 83.0 |
| LungCT_0004 | 22.98 / 24.98 | 0.00504 / 0.00317 | 0.9043 → 0.9663 | 10.89 → 3.50 | 2.29e-04 | 16/16 | 2.00e-05 | 58.0 | 21.2 | 82.0 |
| LungCT_0005 | 21.48 / 23.70 | 0.00711 / 0.00427 | 0.8727 → 0.9328 | 17.90 → 19.37 | 1.11e-03 | 15/16 | 2.01e-05 | 57.6 | 21.3 | 81.8 |
| LungCT_0006 | 21.14 / 23.72 | 0.00769 / 0.00425 | 0.8448 → 0.9167 | 22.75 → 26.25 | 8.31e-04 | 16/16 | 1.98e-05 | 56.7 | 22.4 | 82.1 |
| LungCT_0007 | 22.30 / 24.84 | 0.00589 / 0.00328 | 0.9176 → 0.9732 | 13.29 → 3.72 | 6.25e-04 | 16/16 | 2.00e-05 | 57.9 | 21.1 | 81.7 |
| LungCT_0008 | 22.49 / 25.12 | 0.00564 / 0.00308 | 0.8462 → 0.6866 | 38.68 → 102.44 | 3.83e-03 | 15/16 | 2.02e-05 | 57.8 | 24.8 | 85.6 |
| LungCT_0009 | 21.84 / 23.45 | 0.00655 / 0.00451 | 0.8649 → 0.9543 | 16.20 → 5.25 | 3.80e-04 | 16/16 | 1.96e-05 | 58.3 | 20.5 | 81.7 |
| LungCT_0010 | 24.02 / 25.91 | 0.00396 / 0.00256 | 0.8814 → 0.9590 | 14.60 → 3.50 | 4.96e-05 | 16/16 | 1.98e-05 | 58.1 | 20.2 | 80.9 |
| LungCT_0011 | 22.73 / 24.35 | 0.00534 / 0.00367 | 0.8867 → 0.9660 | 17.29 → 3.50 | 7.72e-04 | 16/16 | 1.96e-05 | 57.6 | 20.9 | 81.3 |
| LungCT_0012 | 21.97 / 23.74 | 0.00635 / 0.00422 | 0.8637 → 0.8893 | 17.50 → 45.03 | 2.39e-03 | 16/16 | 1.96e-05 | 57.7 | 22.5 | 83.1 |
| LungCT_0013 | 21.84 / 24.88 | 0.00654 / 0.00325 | 0.8830 → 0.9568 | 21.00 → 7.00 | 8.52e-04 | 16/16 | 2.00e-05 | 57.5 | 22.0 | 82.2 |
| LungCT_0014 | 22.72 / 24.41 | 0.00535 / 0.00362 | 0.8760 → 0.9559 | 12.21 → 3.50 | 1.64e-04 | 16/16 | 1.98e-05 | 57.7 | 20.4 | 80.8 |
| LungCT_0015 | 21.72 / 24.78 | 0.00672 / 0.00332 | 0.8310 → 0.5962 | 56.46 → 120.39 | 4.68e-03 | 16/16 | 1.98e-05 | 57.3 | 23.1 | 83.6 |
| LungCT_0016 | 23.82 / 26.09 | 0.00415 / 0.00246 | 0.9349 → 0.9601 | 6.07 → 3.91 | 5.61e-06 | 16/16 | 1.99e-05 | 56.9 | 20.0 | 79.5 |
| LungCT_0017 | 21.42 / 24.07 | 0.00721 / 0.00392 | 0.7262 → 0.4609 | 44.50 → 131.14 | 4.75e-03 | 9/16 | 2.16e-05 | 57.6 | 25.4 | 86.6 |
| LungCT_0018 | 23.94 / 25.53 | 0.00404 / 0.00280 | 0.9293 → 0.9742 | 7.94 → 2.15 | 2.62e-04 | 16/16 | 1.99e-05 | 57.3 | 19.8 | 79.7 |
| LungCT_0019 | 22.08 / 23.98 | 0.00620 / 0.00400 | 0.7349 → 0.6389 | 47.69 → 93.16 | 3.16e-03 | 13/16 | 2.05e-05 | 58.1 | 24.8 | 85.9 |
| LungCT_0020 | 21.89 / 24.99 | 0.00647 / 0.00317 | 0.8599 → 0.5634 | 40.48 → 109.51 | 4.00e-03 | 15/16 | 2.02e-05 | 57.8 | 23.9 | 84.7 |

## Failure, convergence and denominator reporting

* **Failures:** none. Every one of the 69 attempted pairs wrote a complete `status.json` and a
  scored `metrics.json`; no pair was skipped for a gate failure because the packaged presets
  explicitly skip the endpoint gate. Per-cohort attempted/completed/scored denominators are
  9/9/9 (LPBA40), 10/10/10 (OASIS), 20/20/20 (Hippocampus), 10/10/10 (Abdomen CT) and
  20/20/20 (Lung CT).
* **Endpoint PSNR:** all values are diagnostic. Hippocampus, Abdomen CT and Lung CT contain
  endpoint reconstructions below the nominal 25 dB threshold (minimum 24.10 dB, 20.68 dB and
  20.81 dB respectively). This is visible, not hidden; the gate status is `null` throughout.
* **PCG convergence:** 1,027 of 1,104 interval solves met the `2e-5` relative tolerance
  (139/144 LPBA40, 146/160 OASIS, 319/320 Hippocampus, 118/160 Abdomen CT, 305/320 Lung CT).
  Fully converged pairs: 6/9, 3/10, 19/20, 2/10 and 14/20. Most misses are marginal
  (≈2.00–2.16e-5), but Abdomen CT pair 0005 (6/16) and Lung CT pair 0017 (9/16) have
  materially incomplete solves; they are reported with their Dice/HD95 rather than discarded.
* **Folding:** LPBA40 has exactly zero folded voxels. OASIS and Hippocampus have small
  nonzero fractions (max 8.3e-6 and 1.7e-4). Abdomen CT max is 1.0e-4. Lung CT has the largest
  folding (max 4.75e-3, i.e. 0.48%); those pairs are exactly the low-Dice/high-HD95 cases, so
  the Lung CT aggregate is not a clean success.

## Bounded tuning study

The tuning study was declared before execution in
`results/default_method_tuning_20261112/declaration.json`. It was run after the full benchmark
finished, sequentially on the same GPU, so timing is not contaminated by concurrent benchmark
work.

### Tuning subset

For each cohort, pairs were sorted by identifier and the three central indices
`n//2 − 1, n//2, n//2 + 1` were selected. The rule is deterministic and was fixed before any
tuning result was inspected:

| Cohort | Tuning pairs | Held-out pairs |
|---|---|---|
| LPBA40 | S05, S06, S07 | S02, S03, S04, S08, S09, S10 |
| OASIS | 0006, 0007, 0008 | 0002, 0003, 0004, 0005, 0009, 0010, 0011 |
| Hippocampus MR | 0011, 0012, 0013 | the other 17 |
| Abdomen CT | 0006, 0007, 0008 | the other 7 |
| Lung CT | 0010, 0011, 0012 | the other 17 |

### Candidate grid

| Dimension | Candidates | Current |
|---|---|---|
| PD-LK coefficient \(c\) | 2, 4, 6 | 4 |
| Local box fraction | 0.4, 0.5, 0.6 | 0.5 |
| Recovery smoothness \(s\) | 0.005, 0.01, 0.02 | 0.01 |

This is 27 candidates per cohort and 135 candidate-cohort cells. Everything else is fixed:
PD-LK weight 0.1, `rho=0`, `eta=1e-6`, 8³ cells, two pairs per cell, pair offset 0.2,
`sigma_cells=0.75`, registration weight 0.05, 4,096 particles, two Euler substeps, 5,000 Adam
steps, seed 42, 16 recovery intervals, ridge 1e-6, 500 PCG iterations, tolerance 2e-5 and four
RK2 substeps. The packaged PSNR-gate policy (explicit skip, pass status null) is preserved for
every candidate.

### Execution design

Training depends on \((c,\text{box})\) only. Each of the nine \((c,\text{box})\) training runs per
cohort was performed once and retained; the three smoothness candidates were then recovered
from the **same checkpoints** with `medical recover --run ... --recovery-config ...`. This makes
the smoothness comparison paired at the checkpoint level and avoids retraining.

### Criterion (declared before execution)

Cohort-balanced, gate-constrained Dice-first score:

1. Average Dice, HD95, folding, PCG convergence and pair wall-clock time within each cohort over
   its three tuning pairs.
2. Average those five cohort means with equal cohort weight (so LPBA40 and Hippocampus cannot
   dominate by pair count).
3. Eligibility: all 15 pairs completed with finite endpoint metrics; no pair above 1% folded
   voxels; cohort-balanced mean folding no worse than the current configuration by more than
   1e-4 absolute; cohort-balanced all-interval-converged fraction not lower than current; gate
   policy unchanged.
4. Among eligible candidates choose the highest cohort-balanced Dice. **Freeze and rerun** only
   if it exceeds the current configuration by at least 0.002 Dice, its cohort-balanced HD95 is no
   worse by more than 0.05 mm, and its cost increase is at most 25%. Otherwise retain the current
   configuration. Cost is reported and inspected but cannot promote a candidate that fails the
   accuracy/folding/convergence conditions.

### Tuning execution and completeness

The sweep ran after the full benchmark, sequentially on the same GPU. It completed **45 training
runs** (nine \((c,\text{box})\) combinations per cohort, three tuning pairs each) and **405
recovery evaluations** (27 candidates × 15 tuning pairs) with **zero failed runs**. Wall clock was
4 h 39 min (2026-11-11 19:21:35 to 2026-11-12 00:02:14 UTC), inside the 6-hour tuning budget.
Each \((c,\text{box})\) checkpoint set was reused by all three smoothness candidates, as declared.
All 135 training pair fits and 405 recovery pairs wrote complete status and finite metrics.
Per-pair tuning metrics are retained at
`results/default_method_tuning_20261112/recovery/<candidate>/<method>/<pair>/metrics.json`;
compact extracts are `results/default_method_benchmark_20261112/tuning_balanced.csv` and
`tuning_per_cohort.csv`, and the declared grid is
`results/default_method_tuning_20261112/declaration.json`.

### Cohort-balanced tuning results

Each row averages within a cohort over its three tuning pairs, then weights the five cohort means equally. Folding is the mean fraction; the max is the worst single pair. Convergence counts pairs whose 16 PCG intervals all met tolerance. `s/pair` is training + recovery + loading/evaluation wall clock for that candidate.

| c | box | s | Dice | HD95 (mm) | Folding mean | Folding max | Conv. pairs | s/pair |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.4 | 0.005 | 0.7103 | 12.226 | 3.61e-04 | 1.85e-03 | 13/15 | 17.8 |
| 2 | 0.4 | 0.01 | 0.7139 | 11.873 | 2.00e-04 | 1.59e-03 | 6/15 | 19.5 |
| 2 | 0.4 | 0.02 | 0.7141 | 11.399 | 1.07e-04 | 7.74e-04 | 2/15 | 21.7 |
| 2 | 0.5 | 0.005 | 0.7105 | 11.631 | 3.18e-04 | 1.70e-03 | 12/15 | 17.7 |
| 2 | 0.5 | 0.01 | 0.7137 | 11.084 | 1.87e-04 | 1.51e-03 | 8/15 | 19.5 |
| 2 | 0.5 | 0.02 | 0.7147 | 10.539 | 1.20e-04 | 1.11e-03 | 4/15 | 21.7 |
| 2 | 0.6 | 0.005 | 0.7057 | 14.169 | 3.64e-04 | 2.96e-03 | 14/15 | 17.6 |
| 2 | 0.6 | 0.01 | 0.7092 | 13.701 | 2.47e-04 | 2.58e-03 | 8/15 | 19.5 |
| 2 | 0.6 | 0.02 | 0.7110 | 12.981 | 1.14e-04 | 1.07e-03 | 4/15 | 21.6 |
| 4 | 0.4 | 0.005 | 0.7150 | 11.036 | 3.25e-04 | 1.60e-03 | 14/15 | 17.6 |
| 4 | 0.4 | 0.01 | 0.7175 | 10.583 | 1.76e-04 | 1.26e-03 | 10/15 | 19.4 |
| 4 | 0.4 | 0.02 | 0.7168 | 10.189 | 8.74e-05 | 6.13e-04 | 4/15 | 21.6 |
| 4 | 0.5 | 0.005 | 0.7157 | 12.243 | 3.10e-04 | 2.04e-03 | 12/15 | 17.6 |
| 4 **current** | 0.5 | 0.01 | 0.7187 | 11.379 | 1.68e-04 | 1.37e-03 | 7/15 | 19.3 |
| 4 | 0.5 | 0.02 | 0.7188 | 10.668 | 4.44e-05 | 4.13e-04 | 4/15 | 21.4 |
| 4 | 0.6 | 0.005 | 0.7113 | 13.425 | 3.15e-04 | 2.84e-03 | 13/15 | 17.6 |
| 4 | 0.6 | 0.01 | 0.7131 | 13.171 | 2.12e-04 | 2.42e-03 | 10/15 | 19.5 |
| 4 | 0.6 | 0.02 | 0.7136 | 12.608 | 1.15e-04 | 1.27e-03 | 4/15 | 21.4 |
| 6 | 0.4 | 0.005 | 0.7191 | 11.048 | 2.60e-04 | 1.44e-03 | 13/15 | 17.6 |
| 6 | 0.4 | 0.01 | 0.7205 | 10.792 | 1.71e-04 | 1.36e-03 | 6/15 | 19.4 |
| 6 | 0.4 | 0.02 | 0.7188 | 10.411 | 8.95e-05 | 8.10e-04 | 5/15 | 21.6 |
| 6 | 0.5 | 0.005 | 0.7189 | 12.129 | 2.48e-04 | 1.75e-03 | 13/15 | 17.4 |
| 6 | 0.5 | 0.01 | 0.7196 | 11.473 | 1.51e-04 | 1.42e-03 | 8/15 | 19.2 |
| 6 | 0.5 | 0.02 | 0.7180 | 10.825 | 6.64e-05 | 5.73e-04 | 4/15 | 21.2 |
| 6 | 0.6 | 0.005 | 0.7198 | 12.722 | 2.87e-04 | 2.25e-03 | 12/15 | 17.4 |
| 6 | 0.6 | 0.01 | 0.7224 | 11.988 | 1.82e-04 | 1.70e-03 | 9/15 | 19.3 |
| 6 | 0.6 | 0.02 | 0.7229 | 11.033 | 6.64e-05 | 5.21e-04 | 4/15 | 21.1 |

### Per-cohort tuning Dice by candidate

| c | box | s | LPBA40 | OASIS | Hippocampus | Abdomen CT | Lung CT |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.4 | 0.005 | 0.7039 | 0.7413 | 0.7787 | 0.3955 | 0.9320 |
| 2 | 0.4 | 0.01 | 0.7050 | 0.7485 | 0.7738 | 0.4044 | 0.9378 |
| 2 | 0.4 | 0.02 | 0.7046 | 0.7506 | 0.7653 | 0.4070 | 0.9432 |
| 2 | 0.5 | 0.005 | 0.7078 | 0.7289 | 0.7840 | 0.3888 | 0.9432 |
| 2 | 0.5 | 0.01 | 0.7088 | 0.7363 | 0.7770 | 0.3981 | 0.9482 |
| 2 | 0.5 | 0.02 | 0.7083 | 0.7434 | 0.7674 | 0.4018 | 0.9525 |
| 2 | 0.6 | 0.005 | 0.7067 | 0.7171 | 0.7814 | 0.3986 | 0.9246 |
| 2 | 0.6 | 0.01 | 0.7083 | 0.7246 | 0.7750 | 0.4064 | 0.9315 |
| 2 | 0.6 | 0.02 | 0.7084 | 0.7316 | 0.7676 | 0.4092 | 0.9382 |
| 4 | 0.4 | 0.005 | 0.7086 | 0.7500 | 0.7765 | 0.3960 | 0.9436 |
| 4 | 0.4 | 0.01 | 0.7087 | 0.7538 | 0.7724 | 0.4040 | 0.9489 |
| 4 | 0.4 | 0.02 | 0.7076 | 0.7537 | 0.7613 | 0.4086 | 0.9527 |
| 4 | 0.5 | 0.005 | 0.7075 | 0.7466 | 0.7770 | 0.4066 | 0.9409 |
| 4 | 0.5 | 0.01 | 0.7085 | 0.7518 | 0.7717 | 0.4124 | 0.9490 |
| 4 | 0.5 | 0.02 | 0.7076 | 0.7535 | 0.7637 | 0.4141 | 0.9548 |
| 4 | 0.6 | 0.005 | 0.7108 | 0.7246 | 0.7790 | 0.4094 | 0.9328 |
| 4 | 0.6 | 0.01 | 0.7120 | 0.7300 | 0.7753 | 0.4098 | 0.9383 |
| 4 | 0.6 | 0.02 | 0.7116 | 0.7374 | 0.7662 | 0.4090 | 0.9437 |
| 6 | 0.4 | 0.005 | 0.7074 | 0.7569 | 0.7770 | 0.4093 | 0.9447 |
| 6 | 0.4 | 0.01 | 0.7076 | 0.7606 | 0.7718 | 0.4125 | 0.9498 |
| 6 | 0.4 | 0.02 | 0.7063 | 0.7604 | 0.7616 | 0.4129 | 0.9531 |
| 6 | 0.5 | 0.005 | 0.7088 | 0.7535 | 0.7752 | 0.4163 | 0.9407 |
| 6 | 0.5 | 0.01 | 0.7094 | 0.7567 | 0.7712 | 0.4139 | 0.9470 |
| 6 | 0.5 | 0.02 | 0.7083 | 0.7565 | 0.7618 | 0.4110 | 0.9523 |
| 6 | 0.6 | 0.005 | 0.7108 | 0.7362 | 0.7760 | 0.4369 | 0.9391 |
| 6 | 0.6 | 0.01 | 0.7117 | 0.7418 | 0.7724 | 0.4408 | 0.9455 |
| 6 | 0.6 | 0.02 | 0.7111 | 0.7465 | 0.7640 | 0.4421 | 0.9508 |

### Per-cohort tuning HD95 (mm)

| c | box | s | LPBA40 | OASIS | Hippocampus | Abdomen CT | Lung CT |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.4 | 0.005 | 6.04 | 3.13 | 2.90 | 35.59 | 13.47 |
| 2 | 0.4 | 0.01 | 6.02 | 3.07 | 2.95 | 35.00 | 12.33 |
| 2 | 0.4 | 0.02 | 5.98 | 3.00 | 2.97 | 34.49 | 10.54 |
| 2 | 0.5 | 0.005 | 6.14 | 3.26 | 2.93 | 36.59 | 9.24 |
| 2 | 0.5 | 0.01 | 6.06 | 3.17 | 2.90 | 35.50 | 7.79 |
| 2 | 0.5 | 0.02 | 6.01 | 3.09 | 2.95 | 34.33 | 6.32 |
| 2 | 0.6 | 0.005 | 6.14 | 3.56 | 2.95 | 37.26 | 20.94 |
| 2 | 0.6 | 0.01 | 6.04 | 3.43 | 2.93 | 36.44 | 19.66 |
| 2 | 0.6 | 0.02 | 5.98 | 3.26 | 2.92 | 35.27 | 17.48 |
| 4 | 0.4 | 0.005 | 6.07 | 3.06 | 2.77 | 34.85 | 8.44 |
| 4 | 0.4 | 0.01 | 6.01 | 2.97 | 2.80 | 34.14 | 6.98 |
| 4 | 0.4 | 0.02 | 5.98 | 3.00 | 2.86 | 33.11 | 5.99 |
| 4 | 0.5 | 0.005 | 6.13 | 3.21 | 3.12 | 36.24 | 12.52 |
| 4 | 0.5 | 0.01 | 6.06 | 3.07 | 3.06 | 35.38 | 9.32 |
| 4 | 0.5 | 0.02 | 6.02 | 3.03 | 3.11 | 34.39 | 6.78 |
| 4 | 0.6 | 0.005 | 6.06 | 3.48 | 2.90 | 36.07 | 18.61 |
| 4 | 0.6 | 0.01 | 5.99 | 3.31 | 2.89 | 35.56 | 18.10 |
| 4 | 0.6 | 0.02 | 5.92 | 3.22 | 2.86 | 34.77 | 16.26 |
| 6 | 0.4 | 0.005 | 6.07 | 2.97 | 2.96 | 34.66 | 8.59 |
| 6 | 0.4 | 0.01 | 6.03 | 2.93 | 3.04 | 34.32 | 7.64 |
| 6 | 0.4 | 0.02 | 6.00 | 2.92 | 2.96 | 33.69 | 6.48 |
| 6 | 0.5 | 0.005 | 6.08 | 3.09 | 3.11 | 35.48 | 12.88 |
| 6 | 0.5 | 0.01 | 6.03 | 3.02 | 3.05 | 34.99 | 10.28 |
| 6 | 0.5 | 0.02 | 5.98 | 3.01 | 3.00 | 34.06 | 8.08 |
| 6 | 0.6 | 0.005 | 6.08 | 3.33 | 2.89 | 33.52 | 17.79 |
| 6 | 0.6 | 0.01 | 6.00 | 3.19 | 2.96 | 33.03 | 14.77 |
| 6 | 0.6 | 0.02 | 5.95 | 3.09 | 2.92 | 32.06 | 11.15 |

### Per-cohort tuning convergence (fully converged pairs / 3)

| c | box | s | LPBA40 | OASIS | Hippocampus | Abdomen CT | Lung CT |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.4 | 0.005 | 3/3 | 3/3 | 3/3 | 2/3 | 2/3 |
| 2 | 0.4 | 0.01 | 1/3 | 0/3 | 3/3 | 1/3 | 1/3 |
| 2 | 0.4 | 0.02 | 0/3 | 0/3 | 2/3 | 0/3 | 0/3 |
| 2 | 0.5 | 0.005 | 2/3 | 1/3 | 3/3 | 3/3 | 3/3 |
| 2 | 0.5 | 0.01 | 1/3 | 0/3 | 3/3 | 1/3 | 3/3 |
| 2 | 0.5 | 0.02 | 0/3 | 0/3 | 3/3 | 0/3 | 1/3 |
| 2 | 0.6 | 0.005 | 3/3 | 2/3 | 3/3 | 3/3 | 3/3 |
| 2 | 0.6 | 0.01 | 2/3 | 0/3 | 3/3 | 0/3 | 3/3 |
| 2 | 0.6 | 0.02 | 0/3 | 0/3 | 3/3 | 0/3 | 1/3 |
| 4 | 0.4 | 0.005 | 3/3 | 3/3 | 3/3 | 2/3 | 3/3 |
| 4 | 0.4 | 0.01 | 2/3 | 2/3 | 3/3 | 0/3 | 3/3 |
| 4 | 0.4 | 0.02 | 0/3 | 0/3 | 3/3 | 0/3 | 1/3 |
| 4 | 0.5 | 0.005 | 3/3 | 2/3 | 3/3 | 1/3 | 3/3 |
| 4 | 0.5 | 0.01 | 1/3 | 0/3 | 2/3 | 1/3 | 3/3 |
| 4 | 0.5 | 0.02 | 0/3 | 0/3 | 3/3 | 0/3 | 1/3 |
| 4 | 0.6 | 0.005 | 3/3 | 2/3 | 3/3 | 2/3 | 3/3 |
| 4 | 0.6 | 0.01 | 2/3 | 1/3 | 3/3 | 1/3 | 3/3 |
| 4 | 0.6 | 0.02 | 0/3 | 0/3 | 3/3 | 0/3 | 1/3 |
| 6 | 0.4 | 0.005 | 3/3 | 3/3 | 2/3 | 3/3 | 2/3 |
| 6 | 0.4 | 0.01 | 0/3 | 2/3 | 3/3 | 0/3 | 1/3 |
| 6 | 0.4 | 0.02 | 0/3 | 0/3 | 3/3 | 0/3 | 2/3 |
| 6 | 0.5 | 0.005 | 3/3 | 3/3 | 3/3 | 1/3 | 3/3 |
| 6 | 0.5 | 0.01 | 0/3 | 1/3 | 3/3 | 1/3 | 3/3 |
| 6 | 0.5 | 0.02 | 0/3 | 0/3 | 3/3 | 0/3 | 1/3 |
| 6 | 0.6 | 0.005 | 2/3 | 3/3 | 3/3 | 1/3 | 3/3 |
| 6 | 0.6 | 0.01 | 3/3 | 1/3 | 3/3 | 0/3 | 2/3 |
| 6 | 0.6 | 0.02 | 0/3 | 0/3 | 3/3 | 0/3 | 1/3 |

### Declared selection outcome

| Quantity | Current (4, 0.5, 0.01) | Best eligible candidate |
|---|---:|---:|
| Candidate | c=4, box=0.5, s=0.01 | c=6, box=0.6, s=0.01 |
| Cohort-balanced Dice | 0.7187 | 0.7224 |
| Cohort-balanced HD95 (mm) | 11.379 | 11.988 |
| Cohort-balanced folding mean | 1.68e-04 | 1.82e-04 |
| Worst single-pair folding | 1.37e-03 | 1.70e-03 |
| Fully converged tuning pairs | 7/15 | 9/15 |
| Mean s/pair | 19.3 | 19.3 |

* Dice gain: **+0.0038** (promotion threshold +0.0020).
* HD95 change: **+0.608 mm** (allowed up to +0.05 mm).
* Cost change: **-0.2%** (allowed up to +25%).
* Decision: **retain current: HD95 worse by > 0.05 mm**. `promote=False`.

Eligible candidates ranked by Dice:

| Rank | Candidate | Dice | HD95 (mm) | Conv. pairs | s/pair |
|---:|---|---:|---:|---:|---:|
| 1 | c=6, box=0.6, s=0.01 | 0.7224 | 11.988 | 9/15 | 19.3 |
| 2 | c=6, box=0.5, s=0.01 | 0.7196 | 11.473 | 8/15 | 19.2 |
| 3 | c=6, box=0.4, s=0.005 | 0.7191 | 11.048 | 13/15 | 17.6 |
| 4 | c=6, box=0.5, s=0.005 | 0.7189 | 12.129 | 13/15 | 17.4 |
| 5 | c=4, box=0.5, s=0.01 | 0.7187 | 11.379 | 7/15 | 19.3 |
| 6 | c=4, box=0.4, s=0.01 | 0.7175 | 10.583 | 10/15 | 19.4 |
| 7 | c=2, box=0.5, s=0.01 | 0.7137 | 11.084 | 8/15 | 19.5 |
| 8 | c=4, box=0.6, s=0.01 | 0.7131 | 13.171 | 10/15 | 19.5 |
| 9 | c=2, box=0.6, s=0.01 | 0.7092 | 13.701 | 8/15 | 19.5 |

Ineligible candidates and reasons:

* cohort-balanced folding worse by >1e-4: 7 candidate(s) — 2.0:0.4:0.005, 2.0:0.5:0.005, 2.0:0.6:0.005, 4.0:0.4:0.005, 4.0:0.5:0.005, 4.0:0.6:0.005, 6.0:0.6:0.005
* fewer fully converged pairs: 11 candidate(s) — 2.0:0.4:0.01, 2.0:0.4:0.02, 2.0:0.5:0.02, 2.0:0.6:0.02, 4.0:0.4:0.02, 4.0:0.5:0.02, 4.0:0.6:0.02, 6.0:0.4:0.01, 6.0:0.4:0.02, 6.0:0.5:0.02, 6.0:0.6:0.02


### Selection decision

The pre-declared rule retains the current configuration. No candidate satisfied every promotion
condition:

* **Highest eligible Dice: c=6, box=0.6, s=0.01.** It gained **+0.0038** cohort-balanced Dice
  (0.7224 versus 0.7187) and 9/15 fully converged pairs (versus 7/15), at essentially unchanged
  cost (−0.2%). However, its cohort-balanced HD95 was **+0.608 mm worse** (11.988 versus
  11.379 mm), far outside the +0.05 mm guard. The degradation is concentrated in Lung CT
  (9.32 → 14.77 mm) and OASIS (3.07 → 3.19 mm), partly offset by Abdomen CT (35.38 → 33.03 mm).
  Because the guard is part of the declared criterion, this candidate was not frozen.
* **c=6, box=0.6, s=0.02** reached +0.0042 Dice and −0.346 mm HD95, but only **4/15** tuning
  pairs fully converged (current 7/15), so it failed the convergence eligibility condition.
* **c=6, box=0.4, s=0.01** was the most HD95-favourable near-miss (−0.587 mm, Dice +0.0018), but
  its convergence was 6/15, below the current 7/15, so it was ineligible; it was also below the
  +0.002 Dice threshold.
* **c=6, box=0.4, s=0.005** was the best-accuracy member of the 13/15-converged group (HD95
  −0.331 mm) but gained only **+0.0004** Dice, below the +0.002 freeze threshold. Its folding
  mean was also higher (2.60e-4 versus 1.68e-4), though still inside the 1% per-pair guard. The
  single highest-convergence cell, c=4/box 0.4/s=0.005 (14/15), was rejected by the folding guard
  (mean 3.25e-4), so it was never a promotion candidate.

The smoothness dimension shows a consistent trade-off rather than a free improvement: the
`s=0.005` candidates generally converge more often but fold more (seven of them were rejected by
the folding guard), while every `s=0.02` candidate was rejected for losing convergence. Seven of
the nine eligible candidates use `s=0.01`; only two `s=0.005` cells remain eligible, and no
`s=0.02` cell does. The cost dimension never bound: eligible candidates span 17.4–19.5 s per
pair against 19.3 s for the current configuration, and the best-Dice candidate was not more
expensive.

**No candidate met all promotion conditions, so no configuration was frozen and no additional
full-cohort rerun was performed.** The packaged default `(c=4, box=0.5, s=0.01)` and the
five-cohort benchmark above stand as the final evidence. The packaged JSON files were not edited.

### Held-out-only view of the current default

The full-cohort means above include the 15 tuning pairs. For transparency, the same current
configuration restricted to the remaining held-out pairs gives:

| Cohort | Held-out pairs | Dice | HD95 (mm) | Folding mean | Fully converged |
|---|---:|---:|---:|---:|---:|
| LPBA40 | 6 | 0.7127 | 6.19 | 0 | 5 / 6 |
| OASIS | 7 | 0.7743 | 2.78 | 1.29e-06 | 2 / 7 |
| Hippocampus MR | 17 | 0.7708 | 3.14 | 1.12e-05 | 16 / 17 |
| Abdomen CT | 7 | 0.4157 | 41.90 | 4.14e-05 | 1 / 7 |
| Lung CT | 17 | 0.7901 | 54.02 | 2.15e-03 | 11 / 17 |

These are descriptive subsets of the same one-seed benchmark, not a validation of a promoted
configuration. No candidate was run on held-out pairs, because none met the promotion rule, so
no tuning-then-held-out comparison exists for a new setting.

## Objective-weight screen (one case per cohort, 2026-11-12)

The earlier sweep fixed the objective weights at the packaged PD-LK 0.1 and box-NCC 0.05. To test
whether those weights are over-regularized, a one-hour screen used the **same central case** of the
earlier tuning subset for each cohort (LPBA40 S06, OASIS 0007, Hippocampus 0012, Abdomen CT 0007,
Lung CT 0011) and changed **one weight at a time**:

* PD-LK \(w_{\mathrm{PDLK}} \in \{0.1, 0.05, 0.02, 0.01\}\) with box NCC fixed at 0.05;
* box NCC \(w_{\mathrm{NCC}} \in \{0.05, 0.02, 0.01\}\) with PD-LK fixed at 0.1.

All other settings stayed at the packaged values: \(c=4\), box fraction 0.5, `sigma_cells=0.75`,
5,000 steps, seed 42, recovery \(s=0.01\), gate skipped. The six distinct candidates were trained
from scratch (30 `register` runs, no checkpoint reuse because the weights change). The grid and
case rule were declared in `results/default_method_weight_sweep_1d_20261112/declaration.json`
before execution. All 30 runs completed with no failure in 43 minutes (03:40:35–04:23:37 UTC).
No full-cohort benchmark was run.

### PD-LK sweep (box NCC fixed at 0.05)

| PD-LK w | NCC w | Dice | Δ Dice | HD95 (mm) | Δ HD95 | PCG pairs | Endpoint PSNR (dB) | Endpoint MSE | s/pair |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 **current** | 0.05 | 0.7041 | — | 13.22 | — | 3/5 | 25.98 | 3.35e-03 | 82.3 |
| 0.05 | 0.05 | 0.6992 | −0.0049 | 13.84 | +0.62 | 3/5 | 26.08 | 3.30e-03 | 82.4 |
| 0.02 | 0.05 | 0.6865 | −0.0176 | 14.47 | +1.25 | 1/5 | 26.06 | 3.32e-03 | 82.0 |
| 0.01 | 0.05 | 0.6774 | −0.0267 | 15.26 | +2.04 | 2/5 | 25.96 | 3.35e-03 | 81.3 |

Lowering PD-LK **hurts registration** and does not improve reconstruction. OASIS falls
0.7618 → 0.7577 → 0.7281 → 0.6966 and Abdomen CT 0.3436 → 0.3201 → 0.2828 → 0.2805;
Hippocampus improves slightly (peak 0.7583 at 0.02) and LPBA40 is flat. Endpoint PSNR moves by at
most +0.1 dB. The packaged PD-LK weight 0.1 is therefore **not** the over-regularizer.

### Box-NCC sweep (PD-LK fixed at 0.1)

| PD-LK w | NCC w | Dice | Δ Dice | HD95 (mm) | Δ HD95 | PCG pairs | Endpoint PSNR (dB) | Endpoint MSE | s/pair |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 **current** | 0.05 | 0.7041 | — | 13.22 | — | 3/5 | 25.98 | 3.35e-03 | 82.3 |
| 0.1 | 0.02 | 0.7056 | +0.0015 | 13.76 | +0.54 | 3/5 | 26.74 | 2.71e-03 | 81.1 |
| 0.1 | 0.01 | 0.7101 | **+0.0060** | 13.00 | **−0.22** | **5/5** | 26.95 | 2.56e-03 | 79.6 |

NCC 0.02 is intermediate and is **dominated by 0.01** on the balanced metrics: Dice 0.7056 versus
0.7101, HD95 13.76 versus 13.00 mm, convergence 3/5 versus 5/5, endpoint PSNR 26.74 versus
26.95 dB. Per cohort, 0.02 gains OASIS +0.024, Hippocampus +0.009 and Lung +0.003, leaves LPBA40
almost unchanged (−0.001), and loses Abdomen CT (−0.026 Dice, +3.2 mm HD95); its single OASIS
convergence miss is one interval at 2.0014e-5, just above the 2e-5 tolerance. Its only advantage
over 0.01 is the smaller LPBA40 Dice loss.

Lowering box NCC to 0.01 improves the cohort-balanced Dice, HD95, PCG convergence, endpoint
reconstruction and runtime at the same time. Endpoint PSNR rises 0.97 dB and endpoint MSE falls
24%; the improvement is present in every cohort (PSNR +0.44 to +1.44 dB). Per cohort:

| Cohort (case) | Dice .05 → .01 | HD95 .05 → .01 | PSNR .05 → .01 | PCG .05 → .01 |
|---|---:|---:|---:|---:|
| LPBA40 (S06) | 0.7044 → 0.6954 | 6.19 → 6.26 | 30.47 → 30.91 | 0/1 → 1/1 |
| OASIS (0007) | 0.7618 → **0.7948** | 3.29 → **3.06** | 27.84 → 28.83 | 1/1 → 1/1 |
| Hippocampus (0012) | 0.7486 → **0.7614** | 3.56 → 3.56 | 26.81 → 27.64 | 1/1 → 1/1 |
| Abdomen CT (0007) | 0.3436 → 0.3301 | 48.96 → 49.08 | 21.28 → 22.71 | 0/1 → 1/1 |
| Lung CT (0011) | 0.9619 → **0.9688** | 4.11 → **3.05** | 23.50 → 24.65 | 1/1 → 1/1 |

The gain is concentrated in OASIS (+0.033 Dice, −0.23 mm), Hippocampus (+0.013) and Lung CT
(+0.007, −1.06 mm); LPBA40 (−0.009) and Abdomen CT (−0.014) lose some Dice. The per-cohort signs
therefore disagree, and the balanced +0.006 is not a uniform improvement.

### What this screen supports

* The user's hypothesis is **half right**: PD-LK 0.1 is not too high, but the **box-NCC weight
  0.05 does look too high**. The reconstruction cost of the objective is carried mainly by the
  NCC term (PSNR flat across the PD-LK sweep, +1 dB across the NCC sweep).
* **Recommendation:** keep PD-LK at 0.1; consider box NCC 0.01 as the one candidate worth a full
  five-cohort benchmark later. The defaults are not changed, and no full benchmark was run.
* **This is a screen, not validation.** There is one central case per cohort and one seed. The
  same current configuration retrained on these cases differs from its earlier runs by up to
  ~0.05 Dice on the hard Abdomen 0007 case (0.291–0.344 across earlier runs), so the LPBA40 and
  Abdomen deltas of ~0.01 are not resolved. The OASIS (+0.033) and Hippocampus (+0.013) gains are
  larger than the observed run-to-run spread on those cases, but a full-cohort rerun is required
  before any promotion.

## Limitations and what is not claimed

* **One seed per pair.** Every SIREN result is a single seed-42 fit. No repeated-seed variance is
  estimated for the full benchmark. The tuning rerun of the *same* current configuration on the
  tuning pairs already differs from the full benchmark by up to +0.036 Dice (Abdomen 0007 and
  Lung 0012) and by −24.3 mm HD95 (Lung 0012); cohort-mean differences are smaller but not zero.
  Candidate differences below roughly 0.005 Dice are therefore not resolved by this design.
* **Skipped gate, diagnostic PSNR.** The packaged presets explicitly skip the endpoint PSNR gate;
  `endpoint_quality_pass` is `null` everywhere. Hippocampus, Abdomen CT and Lung CT contain
  reconstructions below the nominal 25 dB threshold (minimum 24.10, 20.68 and 20.81 dB). Those
  outcomes are accepted by the packaged policy, not certified by it.
* **Classical rows are reused, not rerun.** They come from revision `44f1ca2` with the same data
  and metric code, but classical toolkits are not deterministic and their forward map is an
  approximate inversion that failed to converge on every pair. No classical Lung CT row exists.
* **Lung CT remains a failed cohort.** Its aggregate Dice is below identity and its HD95 is
  roughly doubled; the per-pair table shows the collapse. It should not be pooled into a
  headline efficacy average.
* **Tuning is low-power.** Three deterministic pairs per cohort, one seed, 27 bounded candidates.
  The grid is not exhaustive and the tuning images are not independent validation. The
  cohort-balanced criterion deliberately weights cohorts equally, which lets the small
  Hippocampus cohort count as much as the 20-pair cohorts.
* **No superiority claim.** The comparisons are descriptive. On LPBA40 the reused ANTs row has
  higher Dice; on OASIS ANTs and Demons are higher; on Hippocampus and Abdomen CT the SIREN
  default leads the reused rows. One seed and reused classical evidence do not establish a
  general ranking.

## Artifact preservation and verification

* **Dataset bytes are unchanged.** `Data/` was hashed before the study (2,343 files, about
  3.2 GiB, `/tmp/data_manifest_before_20261112.sha256`) and after all runs
  (`/tmp/data_manifest_after_20261112.sha256`). The two sorted SHA-256 manifests are identical;
  no dataset file was added, removed, moved or rewritten.
* **Inputs and existing artifacts preserved.** All benchmark, tuning and recovery outputs are new
  directories under ignored `results/`; no prior result directory was overwritten. Recovery read
  immutable endpoint checkpoints. No packaged default JSON was edited.
* **Tests.** The medical synthetic suite (`python -m unittest discover -s medical/tests -t .`)
  ran 123 tests successfully before the benchmark; it was rerun after the study.
* **No dense outputs.** No deformation maps, warped volumes, dense trajectories or persistent
  velocity arrays were written. Recovery velocity scratch under `results/.scratch/` was removed
  after each pair; only final endpoint weights and compact scalar evidence remain.

## Appendix: reused classical per-pair evidence

These rows are copied from the 2026-11-10 `classical` runs at revision `44f1ca2` and are not new
measurements. They use the same pair membership, preprocessing and evaluation code as the SIREN
tables above. Lung CT has no classical row. HD95 is in millimetres.

| Cohort | Pair | ANTs Dice | ANTs HD95 | elastix Dice | elastix HD95 | Demons Dice | Demons HD95 |
|---|---|---:|---:|---:|---:|---:|---:|
| LPBA40 | S02 | 0.7202 | 6.38 | 0.6869 | 6.53 | 0.6650 | 7.01 |
| LPBA40 | S03 | 0.7181 | 6.63 | 0.7091 | 6.31 | 0.6618 | 7.24 |
| LPBA40 | S04 | 0.7430 | 5.22 | 0.6879 | 6.07 | 0.7139 | 5.45 |
| LPBA40 | S05 | 0.7213 | 5.86 | 0.6729 | 6.44 | 0.6606 | 6.58 |
| LPBA40 | S06 | 0.7051 | 6.43 | 0.6909 | 6.40 | 0.6012 | 7.95 |
| LPBA40 | S07 | 0.7172 | 6.46 | 0.7033 | 5.99 | 0.6847 | 6.96 |
| LPBA40 | S08 | 0.7088 | 6.45 | 0.6805 | 6.58 | 0.6495 | 7.07 |
| LPBA40 | S09 | 0.7130 | 6.63 | 0.6817 | 6.95 | 0.6759 | 6.97 |
| LPBA40 | S10 | 0.7363 | 5.79 | 0.6966 | 5.97 | 0.6704 | 6.59 |
| OASIS | OASIS_0002_0000 | 0.7872 | 3.22 | 0.7023 | 3.53 | 0.7802 | 3.58 |
| OASIS | OASIS_0003_0000 | 0.8343 | 1.77 | 0.7586 | 2.42 | 0.8321 | 2.08 |
| OASIS | OASIS_0004_0000 | 0.7949 | 2.59 | 0.7340 | 3.14 | 0.8062 | 2.60 |
| OASIS | OASIS_0005_0000 | 0.7742 | 3.52 | 0.6710 | 3.79 | 0.7705 | 3.55 |
| OASIS | OASIS_0006_0000 | 0.7953 | 2.72 | 0.6848 | 4.06 | 0.7862 | 2.97 |
| OASIS | OASIS_0007_0000 | 0.7935 | 3.06 | 0.6785 | 6.27 | 0.7892 | 3.15 |
| OASIS | OASIS_0008_0000 | 0.8006 | 2.23 | 0.7059 | 3.10 | 0.7455 | 2.79 |
| OASIS | OASIS_0009_0000 | 0.8102 | 2.04 | 0.7388 | 2.50 | 0.8124 | 2.35 |
| OASIS | OASIS_0010_0000 | 0.7726 | 2.53 | 0.6947 | 3.21 | 0.7688 | 2.59 |
| OASIS | OASIS_0011_0000 | 0.8051 | 2.77 | 0.7283 | 3.32 | 0.8047 | 2.96 |
| Hippocampus | HippocampusMR_0002_0000 | 0.7649 | 2.62 | 0.7070 | 2.70 | 0.7953 | 2.50 |
| Hippocampus | HippocampusMR_0003_0000 | 0.7964 | 2.50 | 0.7563 | 2.62 | 0.7969 | 2.41 |
| Hippocampus | HippocampusMR_0004_0000 | 0.8032 | 2.62 | 0.7466 | 3.50 | 0.7954 | 3.00 |
| Hippocampus | HippocampusMR_0005_0000 | 0.7232 | 3.80 | 0.6540 | 3.56 | 0.7270 | 3.66 |
| Hippocampus | HippocampusMR_0006_0000 | 0.7505 | 3.00 | 0.6939 | 3.16 | 0.7500 | 3.00 |
| Hippocampus | HippocampusMR_0007_0000 | 0.7130 | 3.98 | 0.6533 | 4.50 | 0.7487 | 3.82 |
| Hippocampus | HippocampusMR_0008_0000 | 0.7958 | 2.00 | 0.7086 | 4.06 | 0.7872 | 2.18 |
| Hippocampus | HippocampusMR_0009_0000 | 0.6573 | 4.54 | 0.6395 | 7.76 | 0.6329 | 4.90 |
| Hippocampus | HippocampusMR_0010_0000 | 0.7861 | 3.16 | 0.7337 | 3.80 | 0.8073 | 2.91 |
| Hippocampus | HippocampusMR_0011_0000 | 0.7468 | 2.43 | 0.6637 | 3.80 | 0.7444 | 3.50 |
| Hippocampus | HippocampusMR_0012_0000 | 0.7293 | 3.56 | 0.6162 | 4.55 | 0.6859 | 3.82 |
| Hippocampus | HippocampusMR_0013_0000 | 0.7686 | 2.87 | 0.6901 | 3.62 | 0.7653 | 3.06 |
| Hippocampus | HippocampusMR_0014_0000 | 0.7889 | 2.80 | 0.7143 | 3.50 | 0.7698 | 3.18 |
| Hippocampus | HippocampusMR_0015_0000 | 0.7271 | 3.87 | 0.7063 | 3.68 | 0.7445 | 3.87 |
| Hippocampus | HippocampusMR_0016_0000 | 0.7588 | 3.08 | 0.6740 | 4.56 | 0.7561 | 3.00 |
| Hippocampus | HippocampusMR_0017_0000 | 0.6478 | 4.50 | 0.5266 | 12.02 | 0.6010 | 3.64 |
| Hippocampus | HippocampusMR_0018_0000 | 0.6545 | 4.77 | 0.5129 | 8.51 | 0.7011 | 3.66 |
| Hippocampus | HippocampusMR_0019_0000 | 0.8046 | 2.58 | 0.7667 | 3.62 | 0.8128 | 2.50 |
| Hippocampus | HippocampusMR_0020_0000 | 0.7542 | 3.07 | 0.7400 | 3.00 | 0.7591 | 3.41 |
| Hippocampus | HippocampusMR_0021_0000 | 0.7223 | 4.00 | 0.6368 | 5.34 | 0.7196 | 4.00 |
| Abdomen CT | AbdomenCTCT_0002_0000 | 0.4704 | 22.72 | 0.5193 | 34.16 | 0.4448 | 22.76 |
| Abdomen CT | AbdomenCTCT_0003_0000 | 0.3276 | 51.24 | 0.2746 | 60.46 | 0.3006 | 50.72 |
| Abdomen CT | AbdomenCTCT_0004_0000 | 0.2764 | 35.57 | 0.3894 | 36.86 | 0.2681 | 34.93 |
| Abdomen CT | AbdomenCTCT_0005_0000 | 0.0855 | 86.26 | 0.1177 | 103.92 | 0.0729 | 86.20 |
| Abdomen CT | AbdomenCTCT_0006_0000 | 0.5666 | 23.24 | 0.4732 | 27.06 | 0.4352 | 26.18 |
| Abdomen CT | AbdomenCTCT_0007_0000 | 0.2470 | 44.00 | 0.2404 | 56.22 | 0.2345 | 45.79 |
| Abdomen CT | AbdomenCTCT_0008_0000 | 0.3975 | 28.49 | 0.3415 | 38.08 | 0.3601 | 29.71 |
| Abdomen CT | AbdomenCTCT_0009_0000 | 0.2489 | 85.87 | 0.1801 | 109.35 | 0.1687 | 88.43 |
| Abdomen CT | AbdomenCTCT_0010_0000 | 0.4518 | 32.18 | 0.3877 | 40.48 | 0.3969 | 33.99 |
| Abdomen CT | AbdomenCTCT_0011_0000 | 0.4655 | 32.68 | 0.4400 | 33.52 | 0.3669 | 31.87 |

The classical forward inversion was approximate and did not converge on any pair; no SIREN
inversion diagnostic is claimed as exact either. Deterministic classical reproduction was never
claimed.

## Appendix: exact packaged objective and recovery block

The five packaged configs have byte-identical `model`, `runtime`, `training`, `objective`,
`recovery`, `evaluation` and `baselines` sections; they differ only in `dataset` and
`preprocessing`. The shared objective and recovery blocks are reproduced verbatim below from
`medical/configs/lpba.json`. The per-cohort `dataset`/`preprocessing` values are summarized in
the cohort table above and are stored in each run's `config.json` and `manifest.json`.

```json
{
  "objective": {
    "weight": 0.1,
    "c": 4.0,
    "eta": 1e-06,
    "rho": 0.0,
    "points": 4096,
    "differentiate_solve": false,
    "weight_schedule": "constant",
    "gaussian_registration": {
      "sampling": "symmetric_pair",
      "bidirectional": true,
      "pair_offset_fraction": 0.2,
      "grid_size": 8,
      "particles": 4096,
      "substeps": 2,
      "integrator": "time_midpoint_euler",
      "sigma_cells": 0.75,
      "chunk": 4096,
      "recompute": false,
      "compile_velocity": true,
      "weight": 0.05,
      "gradient_budget": null,
      "activation_step": 0,
      "max_seconds": 0.25,
      "startup_seconds": 60.0,
      "runtime_patience": 3,
      "hard_seconds": 2.0,
      "crop_fraction": 0.5,
      "crop_min_valid": 32,
      "crop_max_skipped": 20,
      "pairs_per_cell": 2,
      "gaussian_kernel": "three_tap",
      "particle_sampling_fraction": 0.8,
      "crop_fraction_range": null,
      "random_timesteps": true,
      "allow_pdlk": true,
      "single_backward": true,
      "pdlk_sampling": "paired_grid",
      "registration_similarity": "ncc",
      "reconstruction_at_particles": false
    }
  },
  "recovery": {
    "intervals": 16,
    "smoothness": 0.01,
    "ridge": 1e-06,
    "iterations": 500,
    "tolerance": 2e-05,
    "substeps": 4
  }
}
```
