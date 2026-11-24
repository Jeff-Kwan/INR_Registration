# SIRENPath

Fit two coordinate SIRENs independently for each image pair. Linear interpolation
of their parameters supplies a continuous intensity path. Local velocity solves on a shared grid
estimate training velocities; frozen global brightness-flow PCG and midpoint RK2
recover the maps used for evaluation.

**[Current method and canonical recipe](docs/METHOD.md)** ·
**[Dataset protocols](docs/DATASETS.md)** ·
**[Data preparation](docs/DATA_PREPARATION.md)** ·
**[Case runner](docs/CASE_RUNNER.md)** ·
**[Development](docs/DEVELOPMENT.md)**

There is one maintained recipe: [medical/configs/default.json](medical/configs/default.json).
It uses width 512, 5,000 updates, cosine LR 1e-4→2e-5 without warmup,
endpoint NCC with moment weight 0.005,
local transport loss (LT) weight 0.1 with c=4, endpoint parameter penalty weight
0.01, and trajectory-based registration weight cosine-decaying from0.03 to0.01. Training uses two spatial midpoint RK2 half-steps,
Sobol temporal draws, two orthogonal query pairs per cell, sigma 0.5-cell velocity
smoothing, 32,768 reconstruction samples and 65,536 independent particles per
direction. Frozen recovery uses 8 intervals ×8 RK2 substeps with an axis cap 256.
Each loss component can be disabled by setting its documented weight to zero.
Reproduced results may vary even with the same seed because of nondeterministic CUDA operations and differences in how PyTorch compiles GPU kernels.

The dataset reference covers the official Learn2Reg OASIS ablation cohort and
the four comparison cohorts, Neurite-OASIS, IXI, LPBA40 and ACDC. It records
preparation, runner readiness and the frozen Learn2Reg OASIS quick4 subset.

On this Vast instance, read `/workspace/AGENTS.md` and use `/venv/main/bin/python`.
Training, optimizer steps, benchmarks and checkpoint recovery require explicit
user authorization. Synthetic checks must be audited before execution. Outputs
must stay outside `Data/`; never stage datasets, checkpoints or results. Do not
persist deformation maps or warped images; clean recovery scratch on exceptions.

[External baselines](baselines/README.md) retain their separate pinned protocols.
[Research reports](docs/archived_reports/) document earlier experiments.
Do not commit or push without authorization.

The seven-run [SIRENPath L2R-OASIS ablation study](docs/SIRENPATH_ABLATIONS.md)
is complete: [aggregate results](docs/ABLATIONS_REPORT.md). Its plan-only-by-default
launcher includes the Raw crossfade and Fitted crossfade controls.
