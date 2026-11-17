# IXI and LPBA40 training setup

Canonical cohort, preparation and metric choices: [Dataset protocols](DATASETS.md).

Both datasets are prepared and connected to our per-pair registration method.
No training, optimizer steps or saved-model recovery were executed during setup.
The launcher prints a plan by default; fitting requires an explicit `--run`.
The NIR/Mindboggle reproduction is no longer pursued.

## Prepared inputs and frozen pairs

| Dataset | Prepared root | Array shape | Test fits | Validation fits |
| --- | --- | --- | ---: | ---: |
| IXI/TransMorph | `Data/IXI_TransMorph/prepared` | 160×192×224 | 115 | 58 |
| LPBA40/SITReg | `Data/LPBA40/prepared` | 160×192×160 | 45 | 10 |

Each root contains `images/*.nii`, `labels/*.nii`, `protocol.json`,
`preparation.json` and `validation.json`. Original archives and extracted inputs
remain separate and unchanged. These ignored data files must not be staged.

IXI preserves the author's preprocessed arrays exactly. Its atlas is the moving
image and each held-out subject is fixed. The frozen 403/58/115 subject split
is retained. The supplied pickles have no original world affine, so their array
grid is recorded explicitly in a unit-voxel benchmark frame.

LPBA40 uses the original SITReg inverse AIR interpolation, white-matter mean
normalization, tissue masking of intensities, and exact crop. Manual anatomical
labels are unmasked. The prepared affine includes the crop-origin shift and
negative first two axes. Its frozen 25/5/10 subject split defines 45 test and
10 validation combinations in author order.

Our method optimizes a fresh paired model for each registration pair. The
author training cohorts are preserved for provenance; this launcher does not
fit a shared network on those subjects. Validation pairs remain separate from
test pairs for any later recipe selection.

## Default method

`medical.brain_method` uses `medical/configs/default.json`:

- Paired SIRENs, width 512, four layers, omega 30 and output initialization ×30.
- 5,000 steps, seed 42, OneCycle peak LR 1e-4 and AdamW decay 0.001.
- Endpoint NCC with moment matching 0.01, shared-grid PD-LK weight 0.1,
  registration NCC weight 0.01, 32,768 separate particles per direction.
- Native voxel units, four Euler training quarter-steps with one stochastic
  field time per quarter, an 8³ velocity grid
  and three-tap smoothing with sigma 0.75 grid cells.
- Full native-grid PCG recovery and midpoint RK2, using the existing frozen
  checkpoint inference implementation.

Protocol-prepared image values enter fitting directly. Evaluation labels never
enter fitting, crop selection or particle sampling. All training derivatives,
solves, smoothing and trajectories retain the existing full-gradient behavior.
Recovery scratch is removed on success and exceptions. Only checkpoints,
metrics and logs persist; maps and warped registration images are not written.

The default [four-quarter Euler training recipe](TRAINING_INTEGRATORS.md) is
also named by `medical/configs/ablations/native_siren_euler4_5k.json`. Set
`training.integrator: "rk2"` explicitly to use the earlier two-half-step recipe.

## Plan commands

Run from the repository root with `/venv/main/bin/python`:

```bash
/venv/main/bin/python -m medical.brain_experiments --plan-only

/venv/main/bin/python -m medical.brain_experiments \
  --datasets ixi --split validation --plan-only

/venv/main/bin/python -m medical.brain_experiments \
  --datasets lpba40 --pairs 1 2 --gpus 0 1 2 3 --plan-only
```

The full test plan contains 160 jobs and currently reports no missing inputs.
Planning creates no output directory, GPU workers or CUDA work. `--pairs` uses
one-based indices in the frozen pair list and requires one dataset. No quick4
selection has been invented for these cohorts.

For a later authorized launch, add `--run` to the chosen command and omit
`--plan-only`. Use a fresh `--output` directory outside `Data`; the default is
`results/brain_benchmarks`. Execution uses one persistent worker per selected
GPU, a shared queue, and one seed per pair. It snapshots the resolved config,
protocol hash and exact pair plan before fitting. Existing output directories
are rejected.

`progress.json` and `summary.json` retain success denominators, pair runtimes,
PCG convergence, primary scores and reverse scores. IXI's primary mean is the
author's equally weighted 17-group DSC, with both group/case SD and per-case SD.
LPBA40 retains all 56 labels, voxel HD95, the seeded million-point Jacobian
metric and inverse consistency. Its forward and inverse maps come from one
paired fit; the author SITReg experiment also performs a separate swapped-input
model call, which is not represented as another fit here.

## Dependencies, preparation and checks

The optional `brain-protocols` extra pins `composable-mapping==0.1.2` and
`deformation-inversion-layer==1.1.2`; both are installed in `/venv/main`.
LPBA40 preparation and fitting reject a different mapping implementation
version. The vendored AIR and metric helpers are hash-checked against the
[protocol manifest](../medical/protocols/brain_benchmarks.json).

For a fresh acquisition only, preparation is:

```bash
/venv/main/bin/python scripts/prepare_brain_benchmarks.py \
  --datasets ixi lpba40 --workers 2
```

It refuses existing `prepared/` or staging directories. Do not rerun it over
the completed local inputs. The current preparation was checked by exact
NIfTI readback, full source/prepared SHA-256 and geometry validation. Original
author preprocessing methods on temporary copies of LPBA40 S01 and S11 produced
exactly the same image and label values. No learned registration was involved.

Ten small tests check frozen lists, read-only planning, streamed IXI label
warping, group weighting, SITReg boundary/label/Jacobian conventions, adapter
directions, intensity preservation and scratch cleanup. Fitting and recovery
are mocked in adapter tests:

```bash
/venv/main/bin/python -m unittest medical.tests.test_brain_benchmarks
```

The detailed [protocol audit](BRAIN_BENCHMARK_PROTOCOLS.md) links the pinned
author sources and distinguishes benchmark scoring from reproducing either
author's registration algorithm.

During the authorized generalist sweep, a shared `runtime.recovery_max_axis` is
accepted only when it is at least the brain volume's native axis lengths
(IXI maximum 224; LPBA40 maximum 192). Thus cap256 is allowed and has no effect
on brain recovery sampling. Smaller caps remain rejected. This permits one
shared speed configuration across all five medical cohorts while retaining
full native brain recovery and the same evaluation protocol.
