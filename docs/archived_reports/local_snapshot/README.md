# External registration baselines

All comparison code lives here, separate from `medical/` and our method's
development. `upstream/` contains code-only sparse checkouts of the authors'
repositories. These are ignored by our Git repository; tracked `sources.json`
records the exact revisions and source entry points. The adapters call those
implementations rather than translating their algorithms into ours.

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
revision changes or tracked changes beyond the recorded NIR import patch.

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

## Connected presets

`configs/` contains explicit native arguments, not our SIREN/PD-LK settings. These
are starting configurations; no DIR-Lab parameter search or comparison has run.

**FireANTs:** native compositive SyN with Adam and rectangular local CC; the
constructor defaults are spelled out, with a generic `[4,2,1]` scale schedule and
`[200,100,50]` iterations. No affine stage, reorientation or extra normalization
is added. This is an explicit SyN baseline, not a claim of reproducing a
dataset-specific FireANTs experiment. Native image metadata handling and native
forward/reverse warp composition are retained, including numerical inversion.

**ConvexAdam–MIND:** native public API defaults: MIND radius 1/dilation 2, convex
grid spacing 6/search half-width 4, inverse consistency enabled, Adam grid spacing
2, 80 iterations and regularization 1.25. No semantic features, extra intensity
normalization, mask replication or self-configuration sweep. Input axes and
returned displacement components follow nibabel XYZ order, in voxel units.

**ccIDIR:** the official DIR-Lab driver explicitly says its 1250-epoch example
uses a shortened demo schedule. Our preset follows its reproduction instruction:
2500 epochs, no cycle-weight schedule, Adam at `1e-4`, cycle weight `1e-3`, native
balanced Jacobian regularizer and paired `[3,256,256,256,3]` SIRENs with omega 32.
Second-order consensus is explicitly selected, following the paper and general source CLI. The DIR-Lab helper defaults to first order; this discrepancy is recorded in [AUDIT.md](AUDIT.md). Intensities remain raw.
The native driver's default cycle scaling `voxel_size=[1,1,1]` is preserved;
physical TRE uses the actual prepared affine independently. Original one-based
DIR-Lab landmarks are restored before the native ZYX normalization and converted
back to zero-based XYZ for our shared evaluator.

ccIDIR samples fixed and moving ROIs separately. It requires both adjacent
`T00_lung_mask.nii.gz` and `T50_lung_mask.nii.gz`, each binary and on its image's
grid. Prepare the separate T50 R231 masks with `scripts/prepare_dirlab_ccidir_masks.py`. Both masks are verified before ccIDIR runs; T00 masks are never copied into the moving domain. Using R231 is our preparation choice; exact identity
with the authors' original mask assets is not claimed.

**NIR:** this is Sun et al.'s MedIA 2024 neural-field registration, not NODEO.
The released CLI's default `model_type=sine` has no matching `EfficientMLP` input
encoding branch, and its default random-point tensors do not match the spatial
tensor layout expected by the released forward method. Our preset explicitly
selects the implemented `gaussian_sine` network and `downsize_unified_random`
sampler. It uses the released single-field path, MSE plus discrete-Jacobian
penalty 10, 3000 optimizer steps, LR `2e-4`, Gaussian scale 1 and 3 hidden layers.
These are declared adaptation settings, not a reproduction claim for the paper's
hybrid/two-field experiments or its brain-MRI scores. The alternative
`ode_gaussian_sine` uses the released ODE construction if selected in config.

NIR's `Single_CT_Brain` loader assumes prepared intensities and applies Identity.
Our preset min/max scales each DIR-Lab volume to `[0,1]` before passing an
equivalent in-memory dataset into the native wrapper. Set `preprocessing` to
`prepared_0_1` for already normalized data. Native sampling, coordinate
normalization, interpolation, loss, AMP, Adam and learning-rate scheduling remain
unchanged. The released loop completes whole DataLoader passes; the adapter
rejects step counts it would overshoot.

`patches/nir-imports.patch` only replaces a removed private NumPy import with the
public equivalent and makes the unused hash-encoding dependency `tinycudann`
optional. Selecting a native hash encoder still raises an explicit dependency
error. No network, loss, sampler or solver equations are patched. Hash variants
are not exposed by this adapter.

## Runner and outputs

The existing single-case runner accepts each adapter through `--method`, for
example `--method baselines.convexadam` with `--config
baselines/configs/convexadam.json`. Inputs remain the prepared NIfTI pair. Current
evaluation is connected to DIR-Lab's 300 adjacent landmark correspondences; the
adapters reject missing landmark inputs before optimization. AbdomenCTCT label evaluation is also connected through the corresponding `NAME_abdomenctct.json` configs; see [the abdomen protocol](../docs/ABDOMENCTCT.md). OASIS public-validation evaluation is also connected; see [OASIS](../docs/OASIS.md).

To inspect all planned jobs without launching or creating result directories:

```bash
/venv/main/bin/python -m baselines.run_dirlab --gpus 0 1 2 3
```

The plan includes commands and missing inputs. A later explicitly authorized
experiment can add `--run --output results/baselines_new_run`; `--methods` and
`--cases` select subsets. Each worker handles one case at a time and has one
physical GPU exposed as logical `cuda:0`, preserving the authors' bare `.cuda()`
calls. Each method/case runs in its own process, with independent logs and a
return-code summary. Existing output directories are refused.

Scores use the same IDIR-style rounded and continuous physical TRE as our method.
Dense folding metrics are explicitly marked `not_evaluated`, not reported as zero.
`elapsed_seconds` includes native registration and landmark inference, excluding
initial input/config validation. No deformation maps or warped images are saved.

ccIDIR and NIR return native network weights in `checkpoint.pt`. FireANTs and
ConvexAdam's optimized state consists of dense fields, so their `checkpoint.pt`
contains provenance/configuration only (`parameter_checkpoint=false`) and cannot
reconstruct their transformation. Native temporary logs and NIR's intermediate
checkpoint stay in a temporary directory cleaned on success and exceptions;
NIR's SummaryWriter is closed before cleanup. Only compact metrics, provenance,
configs and permitted network checkpoints persist through the case runner.

## Publications

- FireANTs: Jena et al., *Adaptive Riemannian optimization for multi-scale
  diffeomorphic matching*, Nature Communications (2026).
  [Paper](https://doi.org/10.1038/s41467-026-72508-3).
- ConvexAdam: Siebert et al., *ConvexAdam: Self-Configuring Dual-Optimisation-Based
  3D Multitask Medical Image Registration*, IEEE TMI (2024).
  [Paper](https://ieeexplore.ieee.org/document/10681158).
- ccIDIR: van Harten et al., *Robust Deformable Image Registration Using
  Cycle-Consistent Implicit Representations*, IEEE TMI 43(2), 784–793 (2024).
  [Paper](https://doi.org/10.1109/TMI.2023.3321425).
- NIR: Sun et al., *Medical image registration via neural fields*, Medical Image
  Analysis 97, 103249 (2024). [Paper](https://doi.org/10.1016/j.media.2024.103249).

The [implementation audit](AUDIT.md) distinguishes native algorithm paths from paper recipes, records the corrected ccIDIR/NIR inference conventions, and specifies external repetition policies. `python -m baselines.run_dirlab` now plans 50 fits per case for ccIDIR and scores both directions; all jobs remain unlaunched by default. Our method remains single-seed.

For the prepared all-dataset FireANTs/ConvexAdam/NIR runs, use `python -m baselines.run_all`. It defaults to planning, excludes ccIDIR, and uses all four GPUs when executed with `--run`. Results default to `baselines/results/`. Exact combined and per-method commands are in [BASELINE_RUNS.md](../docs/BASELINE_RUNS.md).
