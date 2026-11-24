> Current classified cross-method findings: [baseline protocol review](../docs/BASELINE_PROTOCOL_REVIEW.md). This review corrects earlier claims of best-map selection for NODEO and separates ambiguous NIR localization, deliberate ccIDIR spacing choices, and confirmed numerical/source/reporting errors.

# Comparison implementation audit

The adapters invoke pinned native implementations. That establishes source
alignment of their algorithmic paths; it does not establish reproduction of every
published experiment or its runtime. Each result now records the recipe and
limitations from `audit.json`, plus its actual seed and pinned source revision.
No training, optimizer step, benchmark or saved-model recovery was run for this
audit. Synthetic source-math checks and mocked adapter checks were used.

| Comparison | Source alignment | Experiment alignment |
| --- | --- | --- |
| FireANTs-SyN | Native `SyNRegistration.optimize` and composed normalized coordinates | Constructor defaults with an explicit generic multiresolution schedule; not a dataset-tuned paper recipe or the quickstart's affine + Greedy variant |
| ConvexAdam–MIND | Native `convex_adam_pt`; MIND-SSC, coupled convex, inverse consistency, Adam; native XYZ voxel displacement retained | API defaults, without the paper's self-configuration search; not a leaderboard-reproduction claim |
| ccIDIR | Native paired-INR fit and native **second-order** cycle consensus | Whole-image fitting and dense cycle-consensus inference with 2500 updates and static cycle weight |
| NIR single field | Native Gaussian/SIREN field, spatial sampler, IEEE FP32 training loop, objective and released output conversion | Explicit released-code adaptation, not the paper's hybrid recipe or reported brain-MRI scores |

## Corrections made

**ccIDIR inference:** the general source CLI and paper use second-order Taylor
consensus. Our presets select `inference_order=2`. Dense voxel coordinates use
the native slice inference scale of (shape−1)/2, with arrays passed to the native
networks in ZYX order. Training uses 2500 updates, static cycle weight, native
balanced Jacobian regularization and `voxel_size=[1,1,1]` for cycle errors.

**NIR output coordinates:** the released test script uses
`util.warp_3d_volume`, which converts normalized output with
`normalized/2 * maxlen + (shape-1)/2`. Our old adapter instead used
`dataio.unnormalize_coords`, which scales by `maxlen-1`. The adapter now follows
the actual test script. This retains an inconsistency within the upstream code;
it does not silently substitute a more convenient identity-preserving conversion.
Synthetic samples match the actual native warp helper. Native input coordinate
normalization, interpolation's align-corners behavior and the model are unchanged.
The logging interval now also applies the CLI's batch-size divisor, and dataset
maxlen metadata is set as in the CLI.

**Seed propagation:** baseline subprocesses now use `baselines.case_runner`,
which applies an explicit seed override to the resolved config before invoking
the shared runner. Saved config, metrics and each planned job record that seed.
Python, NumPy and PyTorch RNGs are seeded. SIRENPath has no new seed override
and remains single-seed.

## Remaining recipe differences

**FireANTs:** its [official SyN documentation](https://github.com/rohitrango/FireANTs/blob/f8a9cc259ab9113fc43bd99797000fa8fc34eb59/docs/docs/reference/syn.md)
supports the selected symmetric compositive optimizer. Native composition and
numerical inversion are used. The selected [constructor](https://github.com/rohitrango/FireANTs/blob/f8a9cc259ab9113fc43bd99797000fa8fc34eb59/fireants/registration/syn.py)
defaults define the API; our values now follow its author OASIS SyN test
branch: scales [4,2,1], iterations [200,100,50], Adam LR 0.1, CC width 5,
gradient sigma 0.5 and warp sigma 0.25. Its OASIS cohort differs from official
Learn2Reg validation. The same preset is used on the comparison cohorts without an extra affine stage. The official PyTorch
fallback is used because fused operations are unavailable; published speed claims
are not applicable to an unmeasured fallback run.

**ConvexAdam:** our [native API](https://github.com/multimodallearning/convexAdam/blob/b229e52e44b114e2040a503334c92269750c16b2/src/convexAdam/convex_adam_MIND.py)
options match its defaults apart from logging/device selection. The returned
components already follow the input XYZ array axes and are in voxels; no extra
axis flip, spacing factor or normalized-grid scale is applied. Its source offers
separate self-configuration workflows; we do not claim to have run them or to
have obtained task-selected hyperparameters. The selected path uses raw images,
not nnUNet semantic features.

**NIR.** The corrected single-field MRI recipe uses native local NCC, Gaussian
scale 3 with 2π encoding, sine frequency 30 and IEEE FP32. The native learning-rate
schedule and whole-loader passes are retained, giving 904 updates for a request
of 900 at batch size 1. The original AMP/MSE run is described in the
[diagnostic audit](../docs/NEURAL_BASELINE_DIAGNOSTICS.md).
Requested and native-loop counts are stored in metrics. The downsize sampler's
three-voxel stride sets sample count; sample_exp=18 is unused by that branch.
Its OASIS cohort, crop and evaluation labels differ from Learn2Reg validation.
The connected variant is single-field NIR-D. The paper's hybrid model requires
its separate two-phase, two-field protocol.

## Verification scope

- Exact upstream revisions and recorded compatibility diffs are checked on import.
- Config keys and native API signatures/defaults are inspected without fitting.
- Native ccIDIR nonlinear first-/second-order inference is checked on synthetic
  CPU fields, including a nonzero higher-order correction.
- NIR output conversion is compared with the actual native inference warp helper.
- Mocked runner checks cover seed overrides, output paths and missing-fit denominators.

These checks establish integration behavior and source-math parity on synthetic
inputs. They do not establish numerical convergence, accuracy, reproducibility
across CUDA kernels, or agreement with published scores on real datasets.

## Evaluation dependency compatibility (2026-11-12)

The PyPI surface-distance 0.1 package uses NumPy aliases removed in NumPy 2.
Pin the official google-deepmind/surface-distance revision
`1f805cea446805e559badc80dd27f34175e4173d` in /venv/main and both dependency manifests.
Its metrics.py changes from the installed PyPI source replace only np.Inf/np.NaN
with np.inf/np.nan; the HD95 algorithm is unchanged. No monkey patch, replacement
metric or NIR method modification is applied. Empty predicted surfaces retain
infinite HD95; existing JSON serialization represents nonfinite metrics as null.
These cases must remain visible in reporting rather than being described as
successful finite HD95 scores.

## Classical baseline source recipes

`demons` uses the official SimpleITK DemonsRegistration1 example with 50 iterations,
sigma 1.0 and MRI histogram matching.

`niftyreg` uses the pinned v2.1.1 reg_f3d and reg_transform tools with their public
optimization defaults. The build uses Release CPU/OpenMP. A synthetic native
reg_transform check verified the physical coordinate convention on an oblique,
anisotropic image. Mocked tests cover optimization entry points and scratch cleanup.

An authorized OASIS pair-1 profile used four CPU threads per process. Demons
registration took 11.50 seconds, with 18.09 seconds total process time. NiftyReg
registration took 26.99 seconds, with 36.90 seconds total process time. The two
jobs ran concurrently with a third profiling process. These are rough single-case
measurements. Detailed settings and metrics are in the ignored profile results.


## Finalized neural-field implementations

The selected neural-field methods are ccIDIR, NIR, NODEO and DNVF. The source revisions and native
entry points are pinned in sources.json. The dedicated neural launcher selects
these four methods on all four official cohorts with standardized_v1 scoring.

NODEO uses Registration.registration without rewriting its fitting loop or ODE
solver. The two changes recorded in nodeo-shapes.patch derive the encoder linear
input dimension from the image shape and fit the final decoder output to that
shape. An exact comparison with unmodified source on its original supported
160×192×144 grid verified both parameter initialization and forward output.
Meta-device checks cover every prepared ACDC depth and both brain shapes. Native
CUDA forward/adjoint gradients passed on a small synthetic odd-depth grid.
Its native final-50-iteration unsupervised map-selection criterion is preserved.

DNVF uses opt_nf_diff.main, including native Adam AMSGrad, learning-rate decay,
NCC/gradient/Jacobian losses and seven-step scaling and squaring. Its source
run_diff.sh single-scale recipe defines all fitting hyperparameters. The loader
is replaced with a single in-memory pair on the prepared grid. Native validation
segmentation logging receives placeholder labels; real labels enter only the
shared post-fit evaluator. Captured weights match the final pre-update field
used by native validation. A small synthetic CUDA backward pass through the
source SIREN and integrator passed. The corrected DNVF runs completed all 309 cases.

ccIDIR uses the existing source adapter with a whole-image ROI for the four
selected cohorts and native second-order cycle consensus. NIR retains its
recorded single-field source/paper adaptation and pass-rounded step count.
Both retain their existing optimizer implementations and gradient paths.
