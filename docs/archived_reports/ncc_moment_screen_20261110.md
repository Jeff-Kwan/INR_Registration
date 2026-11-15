# NCC moment-weight screen: LPBA S02

On this single S02→S01 pair, **0.01 has the highest Dice and lowest HD95** in the screen. This is a one-pair, one-seed result, not evidence of cohort-level superiority or a validated weight for other subjects.

| Moment weight | Dice ↑ | HD95 mm ↓ | Min–max PSNR moving / fixed, dB ↑ | Folding % | PCG converged intervals |
|---:|---:|---:|---:|---:|---:|
| 0.0001 | .662309 | 6.79251 | 21.51 / 21.71 | 0 | 0/16 |
| 0.001 | .689775 | 6.67717 | 22.14 / 25.19 | .000020 | 16/16 |
| 0.005 | .690486 | 6.63146 | 27.75 / 28.34 | .003560 | 16/16 |
| **0.01** | **.694369** | **6.44561** | **27.93 / 28.68** | .002645 | 16/16 |
| 0.05 | .675544 | 6.57483 | 26.46 / 27.29 | .002563 | 16/16 |
| 0.1, previous run | .667706 | 6.79266 | 26.60 / 26.71 | .001790 | 16/16 |
| 0.2 | .643075 | 6.87264 | 24.24 / 23.10 | .002665 | 16/16 |
| 0.5 | .652100 | 7.02232 | 23.72 / 27.29 | .003621 | 16/16 |
| 1.0 | .654593 | 6.88323 | 25.47 / 24.95 | .004944 | 16/16 |

The existing **MSE+PD-LK** reference scored Dice **.698229**, HD95 **6.56515 mm**, with **zero folding** and **15/16** flow intervals meeting tolerance; it was not retrained. The earlier NCC+PD-LK control with moment weight zero scored Dice .644834 / HD95 6.88864 mm, zero folding and 0/16 converged intervals. The 0.01 screen result is .003860 Dice below and .11954 mm better HD95 than MSE, with 130 folded voxels. Small single-case differences cannot establish general superiority.

## Protocol

The original screen tested **0.01, 0.2, 0.5, 1.0**; the quick follow-up added **0.0001, 0.001, 0.005, 0.05**. Each used the same LPBA S02 images, seed42, 5,000-step shared-initialization paired SIREN training, global minibatch NCC on [0,1] targets, strict PD-LK, original Adam schedule and unchanged trajectory PCG+RK2 recovery from the [previous pilot](ncc_moments_20261110.md). No weight is frozen as a general setting from this case. The 0.0001 run had no flow intervals meet the convergence tolerance (maximum relative residual 1.38034e-4); the other three follow-up runs converged in all 16 intervals.

The user-authorized NCC PSNR gate was **skipped** for every condition: `endpoint_quality_pass=null`, `psnr_gate_skipped=true`. The table's PSNR is calculated afterward by independently scaling each entire reconstructed endpoint volume with `(reconstruction-min)/(max-min)` and comparing it to its original normalized target. This is the same diagnostic as the earlier comparison; it is **not a reconstruction gate** or an intensity transformation applied to training, PCG, RK2 or image/label warping. Because min–max scaling uses extrema, its PSNR can be sensitive to outliers.

All 56 union labels are included in MONAI1.6.1 macro Dice and maximum directed HD95, with distances in millimetres; no label, pair or failed run was omitted. All four endpoint losses and decoded reconstructions remained finite. Recovery uses 16 global flow systems, true residual tolerance2e-5, at most500 PCG iterations and midpoint RK2 with four substeps. Maximum true flow residuals were 1.95057e-5, 1.98195e-5, 1.97758e-5 and 1.96596e-5 in table order for the four new weights. The 0.1 row is a previously completed reference; its min–max PSNR was recomputed read-only from the same immutable checkpoint.

The folding rates correspond to **130, 131, 178 and 243 out of 4,915,200 voxels** for the newly tested weights .01/.2/.5/1.0, respectively. Mean inverse-composition errors are .14621/.15384/.15686/.16573 mm, and inverse-map outside fractions are 7.908%/7.736%/7.995%/8.158%. Folding is not zero despite all flow systems converging.

## Endpoint and trajectory diagnostics

| Moment weight | Full endpoint means, moving / fixed | Full endpoint variances, moving / fixed | Full endpoint NCC, moving / fixed | Sampled path variance range |
|---:|---:|---:|---:|---:|
| .01 | .03225 / -.03856 | .84720 / .76521 | .99613 / .99629 | .77240–.85161 |
| .2 | .01462 / -.02708 | .98287 / .98298 | .99404 / .99367 | .97598–.99561 |
| .5 | .00297 / -.01157 | 1.00580 / .99977 | .99608 / .99578 | 1.00107–1.01163 |
| 1.0 | .00457 / -.00149 | 1.00249 / 1.00430 | .99593 / .99583 | 1.00847–1.01704 |

Path ranges are evaluated on the same fixed 65,536 independent continuous-coordinate sites at nine times. They are sparse diagnostics, not dense maps or proof of anatomical transport. Larger weights constrain moments closer to their target, but on this pair that did not give better registration. Endpoint NCC remains high in all four runs; neither high endpoint NCC nor near-unit endpoint variance alone establishes trajectory recoverability. Because scaling changes the effective relative strength of the fixed flow smoothness/ridge and the strict PD-LK term, the screen does not isolate a unique causal mechanism.

Training took **37.86–39.76 seconds** per fit; recoveries took **13.34–15.77 seconds**. These are observations from sequential single runs without repeated-run uncertainty. The four fits plus recoveries and compact diagnostics took **239.46 seconds** in total. The architecture, initialization, inputs, recovery settings and data normalization were identical; the independently trained endpoint weights differ.

## Provenance and preservation

`results/ncc_moment_screen/20261110T132700_429803Z` retains the explicit four-weight protocol, inputs, exact executed runner, source hashes/snapshot, final endpoint weights, scalar training/path/PSNR diagnostics and fresh recovery evidence. The runner inherited a two-condition docstring from the prior pilot; its executable condition list and the separately recorded protocol both specify the four conditions actually run. The [reviewed scalar record](ncc_moment_screen_20261110.json) contains all settings, metrics and previous references. All **1,620 earlier result files** were verified unchanged after the experiment. No training outside S02, new code change, cohort expansion, commit or push occurred.

All 250 logged random-time, learning-rate and PD-LK-weight rows match across the four conditions. Source hashes match the frozen snapshot, and only four final endpoint checkpoints were created; no dense image, map, trajectory or velocity file was retained. All **2,343 Data files / 3,405,406,449 bytes** and **830 earlier benchmark files / 83,266,043 bytes** match their earlier hashes, and recovery scratch is empty. Timing context: NVIDIA GeForce RTX 5090, PyTorch 2.14.1+cu132, CUDA 13.2, four CPU threads, compiled analytical PD-LK/recovery derivatives. Dataset presets and source code were unchanged.

## Anchored mutual-moment pilot

One additional S02/seed42 fit used weight .01 with `moment_mode="mutual_anchored"`: squared endpoint mean difference, squared log-variance difference, and squared deviation of average variance from one (anchor coefficient 1). All three terms share weight .01. The 5,000-step schedule, minibatch sampling, strict PD-LK and final PCG+RK2 settings match the fixed-moment screen. No coefficient search was performed; the fixed Gaussian-moment mode remains the default.

| Moment mode, weight .01 | Dice ↑ | HD95 mm ↓ | Min–max PSNR moving / fixed, dB ↑ | Folding % | PCG converged |
|---|---:|---:|---:|---:|---:|
| Fixed N(0,1), previous run | .694369 | 6.44561 | 27.93 / 28.68 | .002645 | 16/16 |
| Mutual + average variance anchor | .670297 | 6.74528 | 29.13 / 29.16 | .001811 | 16/16 |

The mutual mode improved min–max-scaled endpoint PSNR but worsened Dice and HD95 on this pair. Full endpoint means were -.49427/-.55040 and variances .89074/.85940; the shared mean is intentionally unanchored. Folding affected 89/4,915,200 voxels. PSNR remains a reporting diagnostic; the gate was explicitly skipped and no intensity recalibration was applied during recovery. Equal outer weights do not imply equal penalty strength between the two formulations.

Training plus sparse/endpoint diagnostics took 41.73 s, recovery itself 15.48 s, and the complete pilot 61.95 s. Evidence is in `results/ncc_mutual_moments/20261110T141958_218373Z` and the [compact scalar record](ncc_mutual_moments_20261110.json). All 1,835 earlier result files were verified unchanged. Only the requested single-pair pilot ran; datasets and previous checkpoints were preserved. Both synthetic suites passed (51 medical + 1 Cameraman); wheel contents and whitespace checks passed. No commit or push occurred.
