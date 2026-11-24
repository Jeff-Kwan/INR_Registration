# External registration baselines

All comparison code lives here, separate from `medical/` and SIRENPath's
development. `upstream/` contains code-only sparse checkouts of the authors'
repositories. These are ignored by our Git repository; tracked `sources.json`
records the exact revisions and source entry points. The adapters call those
implementations rather than translating their algorithms into SIRENPath.

Use `/venv/main/bin/python`. On a Vast instance, first read the complete
`/workspace/AGENTS.md` and the repository's `AGENTS.md`.

| Adapter | Author source | Pinned revision | Connected implementation |
| --- | --- | --- | --- |
| `baselines.fireants` | [FireANTs](https://github.com/rohitrango/FireANTs) | `f8a9cc259ab9113fc43bd99797000fa8fc34eb59` | `SyNRegistration.optimize` and native composed coordinates |
| `baselines.convexadam` | [ConvexAdam](https://github.com/multimodallearning/convexAdam) | `b229e52e44b114e2040a503334c92269750c16b2` | `convex_adam_pt`, MIND-SSC features, coupled convex + inverse consistency + Adam |
| `baselines.ccidir` | [cycle_consistent_INR](https://github.com/Louisvh/cycle_consistent_INR) | `f6a624c1f2347fd44eb7d212e950ec12041baa61` | `ImplicitRegistrator.fit` and second-order Taylor cycle-consensus inference |
| `baselines.nir` | [NIR: Medical Image Registration via Neural Fields](https://github.com/uci-cbcl/NIR) | `45c0d5ccbe22c72603e4782560f487ff1e20bcfd` | Released Fourier/SIREN field, spatial sampler and `train_efficient_pairwise` |

The checkouts retain upstream notices and licenses where supplied. Bundled example
data/results are excluded. Do not edit a checkout silently: verification rejects
revision changes or tracked changes beyond the recorded compatibility patches
and line omissions in `sources.json`.

## Setup and checks

```bash
uv pip install --python /venv/main/bin/python -r baselines/requirements.txt
/venv/main/bin/python -m baselines.setup
/venv/main/bin/python -m baselines.setup --check
/venv/main/bin/python -m unittest baselines.tests.test_adapters baselines.tests.test_planning -v
```

Setup fetches and imports source; it never constructs a model or runs registration.
It uses the available PyTorch installation. Tests use temporary synthetic inputs
and mock every native fit/optimize call. They check axis order, coordinate scale,
transformation direction, cycle consensus, strict configuration options, plan-only
execution and scratch cleanup on failures. Numerical registration accuracy,
convergence and GPU memory have not been tested for these adapters.

FireANTs currently uses its official PyTorch fallback because its optional fused
CUDA extension is not built. Every result records whether fused operations were
used. Runtime comparisons must account for the backend; this setup makes no speed
claim. No CUDA extension build is triggered by an import or setup check.

## Presets, planning and outputs

The [baseline protocols](../docs/BASELINE_PROTOCOLS.md) describe the recipes and
local adaptations. ## Classical CPU baselines

| Adapter | Public source recipe | Backend |
| --- | --- | --- |
| `demons` | SimpleITK 2.5.7 DemonsRegistration1, 50 iterations, displacement sigma 1.0 | CPU |
| `niftyreg` | NiftyReg v2.1.1 reg_f3d defaults | CPU/OpenMP |

Demons enables 1024-level, seven-point histogram matching. NiftyReg uses native
NMI, conjugate gradients, three pyramid levels, 150 final-level iterations,
five-voxel control spacing, bending weight 0.001 and linear-energy weight 0.01.
Both adapters convert native physical transformations to moving voxel
coordinates on the original fixed evaluation grid. Metrics record native
registration time separately from adapter and evaluation time.

Install the requirements above, then use CMake and a C++17 compiler for NiftyReg.

```bash
/venv/main/bin/python -m baselines.setup --methods demons niftyreg
/venv/main/bin/python -m baselines.setup --check --methods demons niftyreg
```

## Finalized neural-field comparison set

The selected methods are ccIDIR, NIR, NODEO and DNVF. Additional literature
methods are deferred until the first paper draft establishes available space.
Original IDIR and SINR are outside this launch set.

```bash
bash scripts/launch_neural_comparisons.sh --output baselines/results/official_neural --plan-only
```

The launcher plans 309 fits per method across Neurite-OASIS, ACDC, LPBA40 test
and IXI test, for 1,236 fits. It uses the common standardized_v1 evaluator and
seed 42 per pair. An explicit experiment request can replace `--plan-only` with
`--run`. Existing output directories are refused. Shared GPU workers and
independent CPU queues retain the common dispatch behavior.

ccIDIR uses whole-image fitting on these cohorts, preserving native SIRENs,
cycle training, symmetric Jacobian regularization and second-order consensus.
NIR retains the documented single-field recipe and native 904-step pass rounding.

NODEO calls the native registration function directly. Its custom adjoint,
Euler/RK implementations, Adam AMSGrad, loss coefficients, smoothing and
final-50-iteration map selection are preserved. The recorded
`patches/nodeo-shapes.patch` derives the encoder flatten size from the five
stride-2 convolutions and makes the final decoder interpolation match the input
shape. The original supported grid produces exactly matching model parameters
and forward values. Odd ACDC depths are retained on their original evaluation
grids. Its native function returns selected dense outputs, so checkpoints retain
provenance only. The native Euler step-size interpretation remains source-defined.

DNVF calls the released single-scale `opt_nf_diff.main` optimization loop. The
`run_diff.sh` recipe uses 200 epochs, five hidden layers of width 512, Adam
AMSGrad at 1e-4, Jacobian weight 50 and downsample factor 2. Native losses,
learning-rate decay, initialization and seven-step scaling and squaring are
preserved. Only data loading and segmentation diagnostic logging are localized.
The actual evaluation labels are read by the standardized evaluator after the
fit. The final pre-update map and matching network state follow native validation
semantics. Native CSV scratch is cleaned on success and exceptions.

Compatibility and mocked-entry tests validate source pins, map axes and scratch
cleanup. Full optimization performance and memory usage require an authorized
run. The four neural adapters preserve prepared image values.

## Corrected MRI replacements

The invalid official NIR/DNVF results have been removed for in-place replacement.
The corrected NIR MRI recipe uses native LNCC, Gaussian sigma3 with2π features,
sine30 and IEEE FP32; prepared intensities are preserved. DNVF keeps the native
recipe with TF32 disabled. Both full reruns are prepared, not launched.
[Replacement commands and ETA](../docs/NEURAL_RERUNS.md).
