# Reproducing SIRENPath

Use [the canonical method](docs/METHOD.md), [default configuration](medical/configs/default.json), [dataset protocols](docs/DATASETS.md) and [case runner](docs/CASE_RUNNER.md). The official comparisons comprise 49 Neurite-OASIS, 100 ACDC, 45 LPBA40 and 115 IXI directed cases. The canonical recipe uses width 512, 5,000 updates, NCC endpoints and cosine learning-rate decay.

## Environment

The SIRENPath inventory below was measured on 22 November 2026 (HKT) from `/venv/main/bin/python`. It covers training, frozen recovery, scoring, dataset preparation and figure generation for the four official comparison cohorts and the Learn2Reg OASIS ablations. The current project name and version are `sirenpath==0.2.0.dev0`. Dependency declarations live in [pyproject.toml](pyproject.toml), and the [reproducibility archive](docs/reproducibility/README.md) retains the earlier full research environment. The package versions below describe the live installation. A fresh install has yet to be verified against this inventory.

### Host and build tools

| Component | Observed version |
| --- | --- |
| Python | 3.12.15 |
| OS and architecture | Ubuntu 24.04.5 LTS, x86-64, glibc 2.39 |
| GPU and driver | NVIDIA GeForce RTX 5090 (compute capability 12.0), driver 595.99.02 |
| PyTorch CUDA backend | 13.2 |
| System CUDA compiler | nvcc 13.2.86 |
| Host compilers and build runner | GCC/G++ 13.3.0, Ninja 1.11.1 |
| Shell and source control | Bash 5.2.21, Git 2.43.0 |
| Python packaging tools | pip 26.2.2, uv 0.13.4, wheel 0.48.0 |

### SIRENPath Python dependencies

| Role | Installed packages |
| --- | --- |
| Training and recovery | `torch==2.14.1+cu132`, `triton==3.8.0`, `numpy==2.5.4` |
| Medical I/O and scoring | `nibabel==5.5.0`, `scipy==1.18.1`, `surface-distance==0.1` |
| Declared by the base package | `pillow==13.0.0`, `tqdm==4.70.1` |
| ACDC preparation and protocol scoring | `scikit-image==0.26.0`, `medpy==0.5.2`, `simpleitk==2.5.7` |
| LPBA40 preparation and protocol scoring | `composable-mapping==0.1.2`, `deformation-inversion-layer==1.1.2` |
| Figures and trajectory analysis | `matplotlib==3.11.2` |
| Build backend | `setuptools==84.2.1` |

- `surface-distance` reports version `0.1`. Use the upstream Git commit `1f805cea446805e559badc80dd27f34175e4173d` to identify the exact source.
- The maintained Learn2Reg scorer uses the checked-in metric functions with the pinned `surface-distance` package.

### Resolved transitive Python dependencies

These versions come from installed package metadata on this Linux/Python environment, including PyTorch's CUDA extras and the dataset and plotting packages listed above.

**CUDA packages**

```text
cuda-bindings==13.4.4
cuda-pathfinder==1.8.3
cuda-toolkit==13.2.2
nvidia-cublas==13.4.1.3
nvidia-cuda-cupti==13.2.86
nvidia-cuda-nvrtc==13.2.86
nvidia-cuda-runtime==13.2.86
nvidia-cudnn-cu13==9.24.0.43
nvidia-cufft==12.2.0.57
nvidia-cufile==1.17.1.22
nvidia-curand==10.4.2.66
nvidia-cusolver==12.2.0.11
nvidia-cusparse==12.7.10.12
nvidia-cusparselt-cu13==0.8.1
nvidia-nccl-cu13==2.30.7
nvidia-nvjitlink==13.4.92
nvidia-nvshmem-cu13==3.4.5
nvidia-nvtx==13.2.86
```

**Other packages**

```text
absl-py==2.5.1
contourpy==1.4.0
cycler==0.12.1
filelock==4.2.4
fonttools==4.68.0
fsspec==2026.9.0
imageio==2.38.1
jinja2==3.1.6
kiwisolver==1.5.1
lazy-loader==0.6
markupsafe==3.0.4
mpmath==1.3.0
networkx==3.7
packaging==26.3
pyparsing==3.3.3
python-dateutil==2.9.0.post0
six==1.17.0
sympy==1.14.0
tifffile==2026.11.12
typing-extensions==4.16.0
```

External baseline setup is documented in the [baseline guide](baselines/README.md).

## Inputs and execution

Obtain datasets from their documented original releases and follow each dataset's preparation instructions. Compare prepared manifests, ordered pairs, geometry and recorded checksums against the archived metadata. Prepared inputs, labels and checkpoints are intentionally excluded from Git. Labels remain evaluation-only.

The SIRENPath entry point for the four official comparison datasets is scripts/launch_sirenpath_comparisons.sh, which delegates to medical.comparison_experiments. Inspect its command-line help and use its default plan-only mode before explicitly authorizing execution with --run. Use medical/configs/default.json and fresh output directories. Dataset adapters and shared scoring are documented in docs/CASE_RUNNER.md. Baseline launchers, pinned revisions, patches and localization choices are documented in baselines/README.md and docs/BASELINE_PROTOCOL_REVIEW.md.

## Recorded results

The default official run is results/official_ours_cosine_20261116 (309 cases), including per-case checkpoint.pt, run.json and metrics.json plus root-level plans/configurations/summaries. Preserve that whole folder separately. Its file sizes and SHA-256 checksums are recorded in docs/reproducibility/default_run_inventory.jsonl. Checksum verification confirms backup integrity; reproduction also requires the matching prepared inputs.

The reproducibility archive retains compact metadata for the corrected external-baseline results, tuning and ablations. It also records dataset selection/preparation and analysis selections. It contains no checkpoint weights, image arrays, deformation fields, credentials or raw process logs. Current aggregate tables remain in docs/BASELINES_REPORT.md and docs/ABLATIONS_REPORT.md.

See the [supplementary material index](docs/SUPPLEMENT_INDEX.md) for existing tables/figures, evidence sources, generation tools and remaining artifact dependencies.
