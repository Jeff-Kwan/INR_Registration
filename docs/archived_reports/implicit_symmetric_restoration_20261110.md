# Restored symmetric implicit-image recovery

**Known-case outcome:** both restored configurations completed on an immutable S02 MSE+PD-LK checkpoint. The pyramid reduced recovery from **154.10 s to 82.84 s (1.86×)**, with lower Dice and higher HD95. Both have small nonzero folding and midpoint-inversion tolerance misses. Keep this backend experimental and the pyramid opt-in; default PCG+RK2 remains unchanged.

**Retired after review:** implementation, four dedicated tests and active hooks are archived locally. This endpoint-only solver ignores the SIREN interpolation trajectory and does not assess the central trajectory-based registration contribution. Historical NCC recovery success is not evidence for that contribution. `implicit_symmetric` is no longer supported; the experimental record below is retained. See the [retirement/design record](../archive/methods/IMPLICIT_SYMMETRIC_RETIREMENT.md).

## Scope and method review

`medical/registration/implicit_gradient.py` restores the archived symmetric endpoint-image solver as an optional recovery backend. It adapts to current `PairedSiren` models and version-two checkpoints. The old endpoint trainer, benchmark runner, historical checkpoint formats and NCC gate bypass are not restored. Default datasets, strict PD-LK, endpoint fitting and default PCG+RK2 remain unchanged.

This is **SyN-inspired symmetric Demons-style registration**, not ANTs SyN. Two fields map midpoint coordinates into the fixed and moving domains. Each iteration uses the endpoint intensity difference and native-coordinate SIREN gradients, applies opposite signed stabilized Demons forces, then Gaussian-smooths both accumulated fields. The update is additive, rather than exponentiating/composing a stationary velocity. It has no guaranteed diffeomorphism or inverse convergence. Original-image warping, folding, inverse composition and label evaluation remain essential.

Despite its historical use with NCC-trained representations, its recovery force minimizes endpoint intensity discrepancy, **not NCC**. It ignores the intermediate weight-linear image trajectory. That difference can explain why its behavior differs from brightness-flow recovery, but does not establish NCC invariance. Current reconstruction gates remain enforced for every recovery method. The [archived NCC comparison](../archive/methods/ARCHITECTURES.md#ncc-endpoint-training-2026-11-09) used separately refitted models and bypassed those gates; those scores are not current implementation validation.

## Changes addressing cost

| Archived implementation | Restored implementation |
|---|---|
| Autograd input derivative, followed by another image forward, for each endpoint | Shared analytical image-value/input-gradient pass; explicit sine reverse chain rule, no graph |
| Recreate Gaussian kernels for both fields every iteration | Cache per-level kernels and smooth both fields in one six-channel batch |
| Every iteration on the full native lattice | Single-level compatibility default; optional explicit coarse-to-fine schedule |
| Full-volume chunks up to 262,144 sites | Separate bounded solver chunks, default 16,384, independent of decoding chunk |
| Archived runner numerically inverts the composite inverse to obtain forward flow | Invert each midpoint map and construct both directions directly; report both fixed-point defects |
| No maintained recovery/diagnostic interface | Shared `RecoveryResult`, distinct method identifiers and backend-specific scalar diagnostics |

The analytical query removes one redundant image forward **per endpoint per query**; it still performs forward and reverse matrix products. Actual speed depends on kernels, chunking and hardware; it is not assumed to be twice as fast. Compilation of the pure query is optional and off by default. Startup costs must be reported separately when enabled.

The optional pyramid `[4,2,1]` with `[100,100,200]` iterations retains 400 total updates, but allocates only 200 to full resolution. On large 3D grids its scheduled endpoint query count is approximately `(100/64+100/8+200)/400 = .535` of 400 full-resolution updates, before lattice rounding. This changes the optimization sequence and is not an exact acceleration of the single-level trajectory. Sparse stochastic scatter updates were not introduced; their smoothing/normalization would require separate validation.

Pyramid grids span the original native extent exactly. Continuous SIRENs are queried directly; no medical input is resampled or modified. Fields retain native displacement units on every lattice and interpolate with align_corners=True. Sigma is in native voxels, divided by each level's actual axis spacing. The last level must be full resolution. The native axes, gradient scale and border-gradient convention match the archive. Importantly, the old solver's `source` argument meant fixed, whereas current `pair.source` is moving; the restored backend explicitly queries fixed=`pair.target`, moving=`pair.source`.

## Interface and safeguards

Select a recovery-only JSON object `{"method":"implicit_symmetric"}` through the existing `recover --recovery-config` CLI, using an immutable current endpoint checkpoint and fresh output directory. [Full settings](../archive/methods/IMPLICIT_SYMMETRIC_RETIREMENT.md#optional-symmetric-implicit-recovery) and [equations](../archive/methods/IMPLICIT_SYMMETRIC_RETIREMENT.md#optional-symmetric-implicit-image-recovery) record the historical interface. The legacy runner remains archived.

Defaults: one level, 400 iterations, step=1, sigma=1 native voxel, alpha=1, chunk=16,384, no compilation, up to 30 midpoint inversion iterations and max-defect tolerance 1e-4 native voxel. Options validate finite/positive values, matching iteration counts and strictly decreasing factors ending at one. The brightness-flow stationarity training extension cannot consume this backend's settings.

No optimizer, neural network, endpoint training, input-volume write, map exporter or recovery scratch writer is introduced. Non-finite fields/residuals abort. Stage diagnostics record actual lattice sizes, endpoint query counts, synchronized stage time and last pre-update endpoint MSE. Midpoint inversion defects and convergence are recorded separately; they are not PCG convergence. Flow convergence fields are null for this backend. Inversion misses do not silently drop pairs.

## Synthetic validation

Both README suites pass: **51 medical tests + one Cameraman test**. Four restored-solver tests check analytical values/native derivatives against autograd in 2D/3D float64, chunk equivalence, archived force signs, Gaussian smoothing against SciPy, combined/separate smoothing, native-unit field resizing, midpoint inversion/nonconvergence, schema validation, pair direction and dispatch with deformation updates mocked. Existing pipeline checks also cover distinct method names, non-PCG diagnostics and `recover --run` selecting checkpoint directories from source manifests. No endpoint fitting or image-deformation optimization runs in these tests.

A separate tiny CUDA derivative check compares eager and compiled analytical queries with autograd, passing at atol=1e-6 / rtol=1e-5 without optimizer steps or saved-model recovery. These checks establish algebra and interfaces, not medical registration efficacy or an old/new performance ratio.

## Known-case inference

The user subsequently authorized one paired single-resolution/pyramid inference check on LPBA S02→S01 with the **same existing MSE+PD-LK checkpoint**, no training. The fixed comparison uses 400 single-level iterations versus 100/100/200 at factors 4/2/1, with identical force parameters, eager queries, solver chunks and inversion controls. Both conditions completed; this does not reproduce the historical NCC-trained experiment.

The first instrumented attempt failed at about 106 seconds because the progress wrapper used `unittest.mock` with `side_effect`, retaining all call arguments. At 153,472 logged endpoint chunks, the coordinate tensors alone account for about 28.1 GiB, consistent with the observed allocation. The subsequent pyramid attempt in that process also failed allocation. No quality scores were obtained; neither failure is scored as a solver result. The failed run and diagnosis remain under `results/implicit_symmetric/20261110T123515_451854Z`. The fixed rerun uses a plain wrapper via `new=instrument`, without Mock call history, in a fresh process. Solver code, endpoint weights and both configurations are unchanged. This instrumentation error is recorded separately from method limitations.


### Same-checkpoint results

| Measure | Single level [1], 400 updates | Pyramid [4,2,1], 100/100/200 updates |
|---|---:|---:|
| Recovery seconds, including both inversions | 154.096 | **82.836** |
| Iterative stage seconds | 153.927 | **82.805** |
| Whole pair seconds, including gates/evaluation | 159.577 | **87.153** |
| Endpoint queries | 3,932,160,000 | **2,109,837,200** |
| Dice | **.689953** | .686068 |
| HD95 mm | **6.70714** | 6.74747 |
| Recovered folding fraction | .000301921 (0.03019%) | .000294393 (0.02944%) |
| Mean inverse-composition error mm | .131677 | .123042 |
| Fixed/moving midpoint max defect, native voxels | 6.87644 / 3.91317 | 7.23828 / 3.93660 |
| Midpoint inversions meeting 1e-4 tolerance | **0/2** | **0/2** |
| Final endpoint PSNR moving/fixed dB | 32.2247 / 32.5119 | 32.2247 / 32.5119 |

All 56 union labels are scored by MONAI 1.6.1; none are missing and neither condition is omitted. Endpoint MSE is .000599145 / .000560801 for both. No non-finite safeguard fired. Solver settings were identical except the explicit level/iteration allocation: chunk 16,384, step=1, sigma=1, alpha=1, no compilation, 30 inverse iterations. No coefficients were selected using labels and no further settings were tested.

The pyramid uses **53.656%** of single-level endpoint queries and **53.756%** of recovery time, a **46.24% reduction / 1.860× speedup**. Its Dice change is **-.003885** and HD95 change **+.04033 mm**. The stage times are .723 s at factor4, 5.031 s at factor2 and 77.052 s at full resolution. Thus the saving largely comes from halving full-resolution updates; coarse levels add about 5.75 seconds. This is a speed–quality tradeoff, not identical convergence achieved faster.

For context, the **existing** PCG+RK2 evaluation of these same weights has Dice **.698229**, HD95 **6.56515 mm**, zero folding, and recorded recovery time **19.58 s**. That timing was not freshly rerun under this comparison's conditions. Both symmetric configurations score below that flow reference on this MSE-trained case; the experiment does not reproduce the historical NCC-trained benefit. The restored implementation is not promoted as a replacement for default flow.

The inversion misses are substantial maximum local defects despite small folding fractions and modest mean composition error. Additive midpoint updates do not enforce invertibility; using more inverse iterations is not assumed to correct folding or a noncontractive fixed-point map. Establishing a compositional/diffeomorphic update or robust inverse would require a separately controlled method change, not hidden tuning during this timing comparison.

Local provenance: `results/implicit_symmetric/20261110T123800_326081Z` retains fixed settings, audited runner, scalar metrics/diagnostics and source/checkpoint hashes. The [reviewed scalar record](implicit_symmetric_inference_20261110.json) contains both conditions, the existing flow reference and metric definitions. Source and checkpoint hashes match before/after; no new endpoint checkpoint, image, displacement or velocity file was written. GPU process memory was observed stable at approximately 1.35–1.41 GiB after fixing the progress wrapper; these observations are not peak-memory benchmarks. Datasets and historical results remain unchanged. No training or cohort run occurred.

Timing context: NVIDIA GeForce RTX 5090, PyTorch 2.14.1+cu132 / CUDA 13.2, four CPU threads, eager queries and solver chunk 16,384. Conditions ran sequentially, single then pyramid, once each. There is no repeat/order uncertainty estimate or matched timing of the archived implementation. The clean wheel has 33 Python modules, six JSON configs, one JPEG and four metadata files, excluding tests/data/results/archives; both isolated CLI help checks pass. Final checks confirm all **2,343 Data files / 3,405,406,449 bytes** and **830 earlier benchmark files / 83,266,043 bytes** unchanged, no persistent maps or new checkpoints, and empty recovery scratch. No commit or push was made.
