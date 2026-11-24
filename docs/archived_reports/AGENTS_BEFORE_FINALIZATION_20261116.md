# Agent guide

Read README.md and relevant current documents before editing. On this Vast instance read the complete `/workspace/AGENTS.md` and use `/venv/main/bin/python`.

The retained method consists of configurable paired SIRENs; separate MSE or global endpoint NCC with moment matching; full-gradient shared-grid PD-LK; box or GW-NCC registration; and global brightness-flow PCG with midpoint RK2 recovery. The default is native voxel units, endpoint NCC with moment weight 0.01 + PD-LK 0.1 + box registration NCC 0.01, with 32768 separate registration particles, four stochastic-quarter Euler training steps and sigma 0.75-cell velocity smoothing. Zero loss weights remain available for ablations. All training gradients through derivatives, the solve, smoothing and trajectories stay live. Inference uses frozen checkpoints. Detaching logging values is permitted.

Outside the opt-in particle objective documented in docs/PARTICLE_REGISTRATION.md, pointwise PD-LK remains unsupported. No rho relaxation, covariant alternatives, brightness-patch/differential-patch objectives, or enhanced/adaptive recovery remain supported. Remove obsolete implementations and interfaces rather than preserving compatibility. Historical documentation stays archived; historical source/configurations are not maintained.

Preserve current Data inputs and prepared data. Never stage data, archives, masks, generated volumes, checkpoints, results, credentials or machine-local configuration. No persistent deformation-map or warped-image writers. Recovery scratch must be cleaned, including exceptions.

Do not run training, optimizer steps, benchmarks or saved-model recovery without an explicit user request. Synthetic validation is permitted after auditing its side effects. Use `/venv/main`, inspect Git status/jobs/symlink boundaries before edits, and audit checks before running them. Do not commit or push without explicit authorization. No Cameraman development.

Current dataset protocols are in docs/DATASETS.md; the new runner is documented in docs/CASE_RUNNER.md. Historical documents are not current instructions.

Endpoint reconstruction and registration share a crop and draw samples independently for each direction. PD-LK uses all grid cells. NCC uses every particle, including trajectories that leave the crop. The full-image boundary penalty remains live.

Evaluation labels are never training masks.

OASIS uses the unchanged official Learn2Reg release, 19 directed public-validation pairs and labels 1–35. Use medical.oasis_method or baseline *_oasis.json configs; the shared evaluator retains official conventions. See docs/OASIS.md. Labels remain evaluation-only; launcher is plan-only and single-seed.

This exception does not change our method’s single-seed restriction or authorize launching experiments. Baseline implementation/recipe audit: baselines/AUDIT.md.

The optional inverse-metric velocity-agreement training alternative is documented
in docs/INVERSE_METRIC.md. It retains the same shared-grid velocity estimator and
PCG+RK2 inference. Its initial preset uses fixed metric weight 1 and box NCC 0.05;
step-500 calibration is deferred. The default PD-LK preset remains unchanged.

For quick OASIS prototyping, reuse the frozen --subset quick4 selection (official
pairs 1,4,8,9; selection seed 42), documented in docs/OASIS_QUICK4.md. Do not redraw
it or select cases based on scores. This does not authorize launching experiments.

Do not redraw it. No experiment is authorized by this selection.

The experimental physical-working-grid branch adds optional 1 mm preprocessing;
see docs/PHYSICAL_WORKING_GRID.md. Preserve original evaluation grids and reuse
frozen reference geometry across resampling comparisons. Experiments remain opt-in.

Physical-grid training remains at 1 mm; `working_grid.recovery_spacing_mm`
independently selects recovery sampling (default 1 mm). Recovery uses physical
units and spacing-weighted PCG. Returned maps use the recorded recovery geometry;
convert through that geometry for original-grid evaluation. See the physical-grid doc.

Subset preparation does not authorize experiments.

Masks only select crop centers. Training trajectories
use four Euler quarter-steps, each with a field sampled uniformly within its quarter. Inference remains
PCG with midpoint RK2.

Optional structure-weighted endpoint MSE sampling is documented in
`docs/STRUCTURE_SAMPLING.md`. It remains global and independent of registration
crops. If retained in the published method, cite INSPIRE and the documented
sampling precedents; the 2 mm endpoint adaptation is our experimental choice.

Deferred idea: evaluate whole-volume endpoint reconstruction ignoring endpoint
masks; not enabled now. See docs/STRUCTURE_SAMPLING.md.

Optional `training.tie_registration_points` reuses one shared continuous set for
both endpoint reconstructions and both registration directions. It requires equal
reconstruction/registration counts and no structure sampler; crop-center masks are supported.
Default remains separate samples. See docs/PHYSICAL_WORKING_GRID.md for the
32k tied-points experiment and its central-region sampling convention.

Optional endpoint `gw_ncc` reuses our global/window 50/50 GW-NCC with unchanged
Gaussian moments (default 0.01). Reconstruction bins cover the full crop (8³,
sigma 0.75 cells, three-tap kernel); registration NCC remains separately selected.
Use physical_working_grid_endpoint_gwncc_32k.json for the separate-32k experiment.

`objective.endpoint_gw_grid_size` independently sets endpoint GW-NCC moment bins
(default 8); it does not change the velocity-estimation grid. Sigma stays 0.75
cells, so changing this grid also changes the physical window scale.

Physical-grid recovery defaults to the frozen original reference spacing (`recovery_spacing_mm: "original"`), including anisotropic axes. Training remains at 1 mm. Explicit numeric recovery overrides remain supported; preserve recorded checkpoint geometry when restoring.

Optional native-grid `spatial_units: "mm"` standardizes training derivatives and
velocities without image resampling; recovery uses native spacing-weighted PCG.
See docs/NATIVE_PHYSICAL_UNITS.md. Preserve physical_units checkpoint geometry and
restore with medical.physical_units.restore_checkpoint; never silently recover
such checkpoints in voxel units. Cannot combine with working_grid. Experiments
remain opt-in.

Optional objective.registration.sigma_mm sets local velocity smoothing bandwidth
in mm (overriding sigma_cells), converted per crop using ZYX cell-center spacing.
See docs/NATIVE_PHYSICAL_UNITS.md; the sigma7p5mm preset is an opt-in ablation,
not authorization to launch. PCG regularization is separate.

Physical-unit and metric-smoothing presets remain opt-in ablations; do not select
them for subsequent runs unless requested. This selection does not authorize runs.

`medical/configs/default.json` uses the width512 OneCycle recipe: output initialization ×30, peak1e-4, AdamW decay0.001. By explicit user selection (2026-11-16), training now defaults to four-quarter Euler. The original `native_siren_onecycle_5k.json` explicitly selects RK2. The historical `full_suite/06_native_grid.json` stays unchanged. Both losses remain active.

The original shared 5k quick4 reference preset is
`medical/configs/native_siren_onecycle_5k.json` (width 512, uniform omega 30,
output initialization ×30, OneCycle peak 1e-4, AdamW decay 0.001). Experimental ablation presets are grouped in
`medical/configs/ablations/`.

When proposing research ideas in discussion or documentation, start from a
conceptual explanation of the observed behavior. Explain how the proposed
change acts on that mechanism and what observation would distinguish it from
a competing explanation. Motivate a combination by how its components interact.

The ten-hour research comparison is recorded in `docs/LONG_RESEARCH_REPORT.md`.
Its explicit shared accuracy candidate is
`medical/configs/native_siren_gw_orthogonal_sobol_5k.json`: captured full-gradient
native training, GW-NCC, two orthogonal query pairs, scrambled Sobol time sampling,
and fixed cap-256/eight-time PCG+RK2 recovery. This accuracy preset explicitly
retains RK2 training and its recorded execution/sampling recipe. Per-case folding regressions remain a qualification; see the report.
The execution interface and guards are in `docs/CAPTURED_TRAINING.md`.
These presets do not authorize new runs.

The user-requested experimental particle objective is documented in
`docs/PARTICLE_REGISTRATION.md`; its preset is
`medical/configs/ablations/particle_velocity.json`. It solves pointwise weighted
brightness/differential LK, continuously Gaussian-averages velocities by mm
distance, and transports through a stochastic split with two Euler steps.
Brightness and differential weights can each be zero. The selected particle
preset uses independent bidirectional starts in one shared crop, averaged LK/NCC
on both directions, and crop-relative sigma linearly decreasing from 0.4 to 0.1
(`sigma_box_fraction`, `final_sigma_box_fraction`). Omit cycle options entirely;
zero cycle weight still selects the optional loop protocol and is not this preset.
Absolute `sigma_mm` and optional `final_sigma_mm` remain available for ablations.
Training must use at least 4096 particles per direction; smaller arrays are only
for synthetic validation. The default shared-grid objective and frozen PCG/RK2
inference remain unchanged. This addition does not authorize experiments.

The user explicitly authorized `particle_velocity.detach_return_starts` for the
experimental transferred-start bidirectional variant. Detach only the outward
arrival coordinates when starting the return trip; outward LK/NCC keep their
graphs, and return LK/NCC/cycle/boundary gradients remain live within that trip.
Both legs contribute averaged LK and NCC. See docs/PARTICLE_REGISTRATION.md.

The selected particle preset additionally enables `capture_objective: true` and
`training.optimizer: {"name": "adam", "fused": true}`. Its dedicated CUDA graph
replays loss/backward/clipping with fresh sample/time/sigma buffers; sampling and
the finite-gradient guard before the ordinary optimizer remain eager. Zero
effective outer weights use the ordinary path to preserve inactive-loss skips.
Unsupported particle ablations must disable capture_objective. The complete
shared-grid captured trainer still rejects particles. See the particle docs.

Optional shared-grid crop-relative velocity smoothing is documented in
`docs/GRID_BOX_SIGMA.md`, with `objective.registration.sigma_box_fraction` and
`final_sigma_box_fraction`, `gaussian_kernel: "full"`. It uses the original
pooled grid solve and selected training integrator, precomputed reflected convolution matrices
and all live field gradients. The default remains unchanged.

Current default selection (2026-11-16, explicit user request): use the width512
OneCycle recipe with four stochastic-quarter Euler training steps in default.json.
This supersedes the earlier width256 default selection. Keep its box NCC,
32768 registration particles, sigma0.75-cell smoothing and native16-interval
PCG/RK2 recovery. Crop-relative sigma remains an opt-in ablation. Recent
width256 synthetic/medical results must not be presented as width512 results.
This default change does not authorize additional experiments.

Default shared-grid `training.integrator: "euler4"` samples one time uniformly
within each quarter, treats each field as representing that quarter and takes
four Euler steps of size 1/4. Reverse transport reuses the same fields in reverse
order with negative steps. LK is averaged over all four times; NCC, boundary
terms and all training gradients remain live. `random_timesteps: false` uses
quarter midpoints. Ordinary and captured training support it; captured Sobol
uses four dimensions. Explicit `"rk2"` selects two half-steps. Omitted shared-grid
integrators also use Euler. Frozen PCG/RK2 recovery is unchanged. The separate pointwise particle objective rejects
this option. See docs/TRAINING_INTEGRATORS.md;
medical/configs/ablations/native_siren_euler4_5k.json matches default.json.
Changing this default does not authorize training.

The eight-hour five-cohort generalist study selected the opt-in
`medical/configs/native_generalist_euler4_4k.json`: width512, Euler4/4k,
c8, endpoint moment.003, LK.1, GW-NCC.03, velocity sigma.4cells and parameter
L2.01, with unchanged AdamW decay.001. All234 final pairs succeeded at89.39s
mean. This does not change default.json or authorize new experiments. The small
positive-L2 advantage is uncertain; folding, cold-start runtime and single-seed
qualifications are in docs/GENERALIST_SWEEP_REPORT_20261116.md.

Dataset protocol authority: docs/DATASETS.md centralizes all seven benchmarks,
including distinct Learn2Reg OASIS (19 pairs) and Neurite-OASIS/GaussianDIR
(last50 of414 lexicographically, 49 consecutive earlier-fixed/next-moving pairs).
Neurite preparation preserves released aligned_norm/aligned_seg35 bytes; labels
1–35 are evaluation-only. Do not claim author metric parity from cohort agreement.
ACDC follows only KAN-IDIR revision 3ac12e200159811eafb6192f3f8db3afef2a9d1a:
patients101–150, 100 directed fits, ES header spacing, array rescale to
1.5×1.5×3.15mm (cubic/edge MRI, nearest/edge GT), minmax before centered
128×128×min(Z,32) crop, no explicit padding. Preserve existing Data inputs and
prepared data. See docs/ACDC.md
and docs/ACDC_KAN_IDIR_PROTOCOL.md for readiness and scoring caveats. Preparation
alone does not authorize training or recovery; our method remains single-seed.
