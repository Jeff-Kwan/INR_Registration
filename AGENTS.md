# Agent guide

Read README.md, docs/METHOD.md and relevant current documentation before editing.
On this Vast instance read the complete /workspace/AGENTS.md and use
/venv/main/bin/python. Inspect Git status, jobs and symlink boundaries first.

## Canonical method

The method is called **SIRENPath**. The user's finalized recipe is medical/configs/default.json, specified in
**docs/METHOD.md**. It supersedes earlier defaults and all research presets:
paired SIRENs, width 512/four sine layers/omega 30/output initialization ×30;
5,000 updates; AdamW decay 0.001; cosine LR decay 1e-4→2e-5 without warmup;
endpoint NCC and moment weight 0.005; PD-LK weight 0.1, c=4, ridge1e-6;
GW-NCC registration cosine0.03→0.01; inter-SIREN parameter L2 weight 0.01;
32,768 reconstruction samples and 65,536 separate particles per direction;
shared half-size crop; two orthogonal query pairs per cell;8³ shared velocity
grid; sigma 0.5-cell three-tap smoothing; two spatial midpoint RK2 half-steps
with scrambled 2D Sobol temporal draws. Native voxel units and captured CUDA
FP32 training are the sole maintained training path. Frozen recovery uses
brightness-flow PCG and midpoint RK2,8 intervals × 8 substeps, axis cap 256.

Keep all gradients through derivatives, pooled solves, smoothing and trajectories
live. Only logging values may detach. Inference freezes checkpoints. Endpoint,
moment, PD-LK, brightness, differential, registration, parameter-L2, AdamW decay and PCG
smoothness coefficients may be zero for ablations; see the exact key table in
docs/METHOD.md. Zero registration weight must override the entire schedule.
Zero PD-LK loss weight still allows velocity estimation for registration.

Do not retain experimental method implementations or presets as compatibility
branches. No particle/cycle objective, Euler training, inverse-metric objective,
physical working grids, physical-unit training, structure samplers, alternative
endpoint objectives or experimental optimizers remain outside the explicitly
requested [L2R-OASIS ablation study](docs/SIRENPATH_ABLATIONS.md). That study alone
permits MSE endpoint fitting in runs 1–4 and a no-training raw crossfade control;
the canonical recipe remains NCC. Scalar ablations use
copies of the one canonical configuration. Numerical eager helpers remain for
validation of the same equations. Superseded documentation belongs in
**docs/archived_reports/** and is not current instruction or runnable guidance.

## Execution and artifacts

Do not run training, optimizer steps, benchmarks or saved-model recovery without
an explicit user request. Synthetic validation is permitted after auditing side
effects. Remain single-seed; no multiseed validation without explicit request.
Preparation and plan inspection do not authorize registration experiments.
Do not commit, push or stage artifacts without explicit authorization.

Preserve Data inputs and prepared data, checkpoints and results. Never stage
data, archives, masks, generated volumes, checkpoints, results, credentials or
machine-local configuration. Do not persist deformation maps or warped images.
Clean recovery scratch on success and exceptions. Reject unsupported physical
checkpoint geometry rather than recovering it silently in voxel units.
No Cameraman development. Keep external baselines isolated and follow their
own pinned protocols; their author seed policies do not change our method.

## Dataset protocols

**docs/DATASETS.md** is the sole authority for cohorts, preparation, directions,
label handling and evaluation. **docs/CASE_RUNNER.md** specifies execution.
Labels/landmarks remain evaluation-only. Preserve original evaluation grids.

DIR-Lab: ccIDIR/IDIR native preparation,10 cases,300 landmarks each; report
T00→T50 and reverse TRE separately. Choose T00/T50 masks equally to place the
shared crop by sampling a mask voxel and shifting the centered box to fit.
No occupancy rejection, loss gating or mask filtering. Every PD-LK cell and NCC
particle contributes, including crop exits. Border extension and a live full-image
boundary penalty apply. Masks may define evaluation-only lung metrics.

Learn2Reg OASIS:19 official directed public-validation pairs, labels1–35.
AbdomenCTCT:45 official directed public-validation pairs,13 labels. Both retain
official voxel HD95 and shifted SDlogJ conventions; labels are never crop masks.
Their launchers are plan-only by default. DIR-Lab executes by default: use
--plan-only for inspection. Frozen quick4 selections are DIR-Lab01,03,04,05;
OASIS1,4,8,9; abdomen2,8,18,41. Never redraw them or choose using scores.

IXI/TransMorph:115 atlas-to-subject test pairs. LPBA40/SITReg:45 frozen test
combinations; one fit returns both maps, qualified against the author swapped
model-call convention. Preserve their prepared intensities and dataset-specific
label warps/Jacobians. Their launcher is plan-only by default.

ACDC uses KAN-IDIR only: official testing patients101–150,100 directed ED/ES
fits, frame IDs from Info.cfg; ES header spacing for both phases; array rescale
to1.5×1.5×3.15mm; cubic/edge MRI, nearest/edge labels; independent minmax before
center crop128×128×min(Z,32); no explicit padding. Preparation/runner readiness
and scoring caveats are in docs/ACDC.md and docs/ACDC_KAN_IDIR_PROTOCOL.md.

Neurite-OASIS/GaussianDIR is separate from Learn2Reg OASIS: final50 of414
lexicographically sorted subjects,49 consecutive earlier-fixed/next-moving pairs.
Preserve released aligned_norm/aligned_seg35 bytes. Labels1–35 are evaluation-only;
cohort agreement does not establish author metric parity.
