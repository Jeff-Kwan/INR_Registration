# SIRENPath — paired-SIREN registration

Fit two coordinate SIRENs independently for each image pair. Linear interpolation
of their parameters supplies a continuous intensity path. Shared-grid PD-LK
estimates training velocities; frozen global brightness-flow PCG and midpoint RK2
recover the maps used for evaluation.

**[Current method and canonical recipe](docs/METHOD.md)** ·
**[Dataset protocols](docs/DATASETS.md)** ·
**[Case runner](docs/CASE_RUNNER.md)** ·
**[Development](docs/DEVELOPMENT.md)**

There is one maintained recipe: [medical/configs/default.json](medical/configs/default.json).
It uses width 512, 5,000 updates, cosine LR 1e-4→2e-5 without warmup,
endpoint NCC with moment weight 0.005,
PD-LK weight 0.1 with c=4, inter-SIREN L2 weight 0.01, and GW-NCC registration
cosine-decaying from0.03 to0.01. Training uses two spatial midpoint RK2 half-steps,
Sobol temporal draws, two orthogonal query pairs per cell, sigma 0.5-cell velocity
smoothing, 32,768 reconstruction samples and 65,536 independent particles per
direction. Frozen recovery uses 8 intervals ×8 RK2 substeps with an axis cap 256.
Each loss component can be disabled by setting its documented weight to zero.

The dataset reference distinguishes DIR-Lab, official Learn2Reg OASIS,
AbdomenCTCT, IXI/TransMorph, LPBA40/SITReg, ACDC/KAN-IDIR and the49-pair
Neurite-OASIS cohort following Jena et al. (*Magic or Mirage?*). It records preparation and runner readiness;
data preparation does not authorize an experiment. Frozen quick4 subsets are
unchanged.

On this Vast instance, read `/workspace/AGENTS.md` and use `/venv/main/bin/python`.
Training, optimizer steps, benchmarks and checkpoint recovery require explicit
user authorization. Synthetic checks must be audited before execution. Outputs
must stay outside `Data/`; never stage datasets, checkpoints or results. Do not
persist deformation maps or warped images; clean recovery scratch on exceptions.

[External baselines](baselines/README.md) retain their separate pinned protocols.
Superseded method descriptions and research reports are in `docs/archived_reports/`;
their commands and configurations are unsupported. Experimental source branches
and presets are not maintained. Do not commit or push without authorization.

The seven-run [SIRENPath L2R-OASIS ablation study](docs/SIRENPATH_ABLATIONS.md)
is complete: [aggregate results](docs/ABLATIONS_REPORT.md). Its plan-only-by-default
launcher includes both raw-image and canonical-reconstruction crossfade controls.
