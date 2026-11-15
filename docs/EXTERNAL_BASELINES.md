# External baseline results

Collected 2026-11-12. Central reference for the completed three-method run; ccIDIR was not run.

## Provenance and scope

- Source run: `baselines/results/three_baselines/`; aggregate `summary.json`, commands/seeds/pins in `plan.json`, frozen presets in `configs/`, per-pair metrics and logs in dataset/method directories. These generated artifacts remain ignored.
- Summary SHA256: `e301e0d9db9fed5170c44c386f77268d933d640264d39ccecf55aa0f1e39629c`.
- 222 intended jobs; all 222 now returned successfully after three NIR retries. Single configured seed per pair. Cohorts: 10 DIR-Lab, 45 AbdomenCTCT and 19 OASIS directed pairs.
- These are the current audited presets, not claims of reproducing published headline results. See [implementation and recipe audit](../baselines/AUDIT.md) and [dataset protocols](DATASETS.md). FireANTs uses its native unfused fallback and declared generic schedule; ConvexAdam uses native MIND defaults without task self-configuration; NIR is the 3000-step single-field released-code adaptation, not the paper hybrid recipe.

## DIR-Lab

TRE is the mean of the ten case means (300 landmarks each), rounded landmark convention, mm. Standard deviation below is across case means, not pooled landmark spread. Primary direction only; no bidirectional average.

| Method | TRE mm, mean ± case SD ↓ | Scored | Cases >2 mm |
| --- | ---: | ---: | ---: |
| fireants | 4.040 ± 2.612 | 10/10 | 7/10 |
| convexadam | 1.402 ± 0.545 | 10/10 | 1/10 |
| nir | 1.561 ± 0.310 | 10/10 | 1/10 |

Per-case mean rounded TRE (mm):

| Case | FireANTs | ConvexAdam | NIR |
| --- | ---: | ---: | ---: |
| 1 | 1.254 | 0.982 | 1.455 |
| 2 | 1.150 | 0.766 | 0.943 |
| 3 | 1.996 | 1.024 | 1.276 |
| 4 | 3.343 | 1.365 | 1.388 |
| 5 | 2.991 | 1.435 | 1.612 |
| 6 | 5.351 | 1.333 | 1.572 |
| 7 | 6.068 | 1.485 | 1.794 |
| 8 | 10.400 | 2.895 | 2.154 |
| 9 | 4.096 | 1.327 | 1.671 |
| 10 | 3.755 | 1.406 | 1.743 |

## AbdomenCTCT

Official public-validation metrics: DSC ↑, HD95 in voxel units ↓, and shifted SDlogJ (`LogJacDetStd`) ↓. Values are mean ± population SD over scored pair-level means; label averaging follows the official evaluator.

| Method | DSC ↑ | HD95 voxels ↓ | SDlogJ ↓ | Scored/intended |
| --- | ---: | ---: | ---: | ---: |
| fireants | 0.3543 ± 0.0867 | 18.1551 ± 5.5840 | 0.0516 ± 0.0034 | 45/45 |
| convexadam | 0.4128 ± 0.0971 | 16.4884 ± 4.2799 | 0.1151 ± 0.0118 | 45/45 |
| nir | 0.1033 ± 0.1196 | ∞ (42 finite pairs: 44.7550 ± 15.6253) | 0.5140 ± 0.2095 | 45/45; HD95 finite in 42/45 |

## OASIS

Official public-validation metrics: DSC ↑, HD95 in voxel units ↓, and shifted SDlogJ (`LogJacDetStd`) ↓. Values are mean ± population SD over scored pair-level means; label averaging follows the official evaluator.

| Method | DSC ↑ | HD95 voxels ↓ | SDlogJ ↓ | Scored/intended |
| --- | ---: | ---: | ---: | ---: |
| fireants | 0.7437 ± 0.0416 | 2.4643 ± 0.6291 | 0.0294 ± 0.0015 | 19/19 |
| convexadam | 0.7633 ± 0.0304 | 2.3260 ± 0.5419 | 0.0596 ± 0.0028 | 19/19 |
| nir | 0.7089 ± 0.0615 | 3.1032 ± 0.9100 | 0.1864 ± 0.0150 | 19/19 |

## Completed evaluation retries

All three original NIR AbdomenCTCT evaluation errors were resolved by installing
Google DeepMind's official surface-distance revision
`1f805cea446805e559badc80dd27f34175e4173d`. Only obsolete NumPy alias spellings
change in the metric code; no method, loss, metric formula or local monkey patch
was introduced. Dependency manifests pin this revision.

Three fits were repeated with byte-identical frozen configs and the same seed 42,
on GPUs 0–2. All succeeded. Original artifacts remain intact; final cohort scores
come from `baselines/results/nir_abdomen_eval_retry/combined_summary.json`.
That file merges successful retries with the original 219 successful jobs.
Exact package and config provenance is in `dependency_provenance.json` there.

The three retried pairs have nonfinite official HD95 because predicted label
surfaces are empty. The existing JSON policy stores infinity as null. All 45
pairs contribute to DSC and SDlogJ; the summary's finite-only HD95 mean remains
44.7550 over 42 pairs. The full 45-pair HD95 mean is infinite, as shown above;
these cases must not be silently excluded when comparing complete-cohort HD95.

Retried pairs:

- `abdomenctct/nir/0007_to_0010`
- `abdomenctct/nir/0013_to_0022`
- `abdomenctct/nir/0019_to_0022`

## Measured compute cost

Sum of child-process elapsed times, including startup, fitting, inference and evaluation. Original failed attempts are included; the three retry attempts additionally took 4.29 serial-equivalent minutes. This is serial-equivalent time, not measured launcher wall time.

| Method | Jobs attempted | Serial-equivalent minutes |
| --- | ---: | ---: |
| fireants | 74 | 15.79 |
| convexadam | 74 | 10.41 |
| nir | 74 | 123.55 |
