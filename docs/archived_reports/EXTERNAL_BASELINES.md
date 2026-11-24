# External baseline results

Collected 2026-11-12. Central reference for the completed three-method run; ccIDIR was not run.

## Provenance and scope

- Source run: `baselines/results/three_baselines/`; aggregate `summary.json`, commands/seeds/pins in `plan.json`, frozen presets in `configs/`, per-pair metrics and logs in dataset/method directories. These generated artifacts remain ignored.
- Summary SHA256: `e301e0d9db9fed5170c44c386f77268d933d640264d39ccecf55aa0f1e39629c`.
- 222 intended jobs; all 222 now returned successfully after three NIR retries. Single configured seed per pair.
- These are the current audited presets, not claims of reproducing published headline results. See [implementation and recipe audit](../baselines/AUDIT.md) and [dataset protocols](DATASETS.md). FireANTs uses its native unfused fallback and declared generic schedule; ConvexAdam uses native MIND defaults without task self-configuration; NIR is the 3000-step single-field released-code adaptation, not the paper hybrid recipe.

## OASIS

Official public-validation metrics: DSC ↑, HD95 in voxel units ↓, and shifted SDlogJ (`LogJacDetStd`) ↓. Values are mean ± population SD over scored pair-level means; label averaging follows the official evaluator.

| Method | DSC ↑ | HD95 voxels ↓ | SDlogJ ↓ | Scored/intended |
| --- | ---: | ---: | ---: | ---: |
| fireants | 0.7437 ± 0.0416 | 2.4643 ± 0.6291 | 0.0294 ± 0.0015 | 19/19 |
| convexadam | 0.7633 ± 0.0304 | 2.3260 ± 0.5419 | 0.0596 ± 0.0028 | 19/19 |
| nir | 0.7089 ± 0.0615 | 3.1032 ± 0.9100 | 0.1864 ± 0.0150 | 19/19 |

## Completed evaluation retries

Only obsolete NumPy alias spellings
change in the metric code; no method, loss, metric formula or local monkey patch
was introduced. Dependency manifests pin this revision.

Three fits were repeated with byte-identical frozen configs and the same seed 42,
on GPUs 0–2. All succeeded. That file merges successful retries with the original 219 successful jobs. Exact package and config provenance is in `dependency_provenance.json` there.

The three retried pairs have nonfinite official HD95 because predicted label
surfaces are empty. The existing JSON policy stores infinity as null. All 45
pairs contribute to DSC and SDlogJ; the summary's finite-only HD95 mean remains
44.7550 over 42 pairs. The full 45-pair HD95 mean is infinite, as shown above;
these cases must not be silently excluded when comparing complete-cohort HD95.

Retried pairs:

## Measured compute cost

Sum of child-process elapsed times, including startup, fitting, inference and evaluation. Original failed attempts are included; the three retry attempts additionally took 4.29 serial-equivalent minutes. This is serial-equivalent time, not measured launcher wall time.

| Method | Jobs attempted | Serial-equivalent minutes |
| --- | ---: | ---: |
| fireants | 74 | 15.79 |
| convexadam | 74 | 10.41 |
| nir | 74 | 123.55 |
