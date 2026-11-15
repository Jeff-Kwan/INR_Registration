# Agent guide

Read README.md and relevant current documents before editing. On this Vast instance read the complete `/workspace/AGENTS.md` and use `/venv/main/bin/python`.

The retained method consists of configurable paired SIRENs; separate MSE or global endpoint NCC with moment matching; full-gradient shared-grid PD-LK; box or GW-NCC registration; and global brightness-flow PCG with midpoint RK2 recovery. The default is native voxel units, endpoint NCC with moment weight 0.01 + PD-LK 0.1 + box registration NCC 0.01, with 32768 separate registration particles, RK2 training trajectories and sigma 0.75-cell velocity smoothing. Zero loss weights remain available for ablations. All training gradients through derivatives, the solve, smoothing and trajectories stay live. Inference uses frozen checkpoints. Detaching logging values is permitted.

No pointwise PD-LK, rho relaxation, covariant alternatives, brightness-patch/differential-patch objectives, or enhanced/adaptive recovery remain supported. Remove obsolete implementations and interfaces rather than preserving compatibility. Historical documentation stays archived; historical source/configurations are not maintained.

Preserve current Data inputs and prepared data. Never stage data, archives, masks, generated volumes, checkpoints, results, credentials or machine-local configuration. No persistent deformation-map or warped-image writers. Recovery scratch must be cleaned, including exceptions.

Do not run training, optimizer steps, benchmarks or saved-model recovery without an explicit user request. Synthetic validation is permitted after auditing its side effects. Use `/venv/main`, inspect Git status/jobs/symlink boundaries before edits, and audit checks before running them. Do not commit or push without explicit authorization. No Cameraman development.

Current dataset protocols are in docs/DATASETS.md; the new runner is documented in docs/CASE_RUNNER.md. Historical documents are not current instructions.

DIR-Lab retains ccIDIR data preparation and separately reported forward/reverse TRE. Training uses T00/T50 masks only to sample crop centers: choose an endpoint equally, draw a mask voxel, center the box and shift it to fit the image. No mask occupancy rejection or mask-based loss gating. Endpoint reconstruction and registration share that crop; direction samples remain independent. PD-LK uses all grid cells. Every particle contributes to NCC, including trajectories leaving the crop; velocity/image interpolation uses border extension and a live full-image boundary penalty. Masks may still define evaluation-only lung metrics. Training remains single-seed; do not add or run multiseed validation unless explicitly requested.

AbdomenCTCT uses the unchanged official release and exactly 45 public-validation directed pairs. Use medical.abdomen_method or the baseline *_abdomenctct.json configs. Evaluation labels are never training masks. Official metric details (including voxel HD95 and shifted SDlogJ) are in docs/ABDOMENCTCT.md. The abdomen launcher is plan-only by default; single-seed only.

OASIS uses the unchanged official Learn2Reg release, 19 directed public-validation pairs and labels 1–35. Use medical.oasis_method or baseline *_oasis.json configs; the shared evaluator retains official conventions. See docs/OASIS.md. Labels remain evaluation-only; launcher is plan-only and single-seed.

External baseline seed policies may follow their published protocols: ccIDIR DIR-Lab plans 50 seeds and evaluates both directions. This exception does not change our method’s single-seed restriction or authorize launching experiments. Baseline implementation/recipe audit: baselines/AUDIT.md.

The optional inverse-metric velocity-agreement training alternative is documented
in docs/INVERSE_METRIC.md. It retains the same shared-grid velocity estimator and
PCG+RK2 inference. Its initial preset uses fixed metric weight 1 and box NCC 0.05;
step-500 calibration is deferred. The default PD-LK preset remains unchanged.

For quick OASIS prototyping, reuse the frozen --subset quick4 selection (official
pairs 1,4,8,9; selection seed 42), documented in docs/OASIS_QUICK4.md. Do not redraw
it or select cases based on scores. This does not authorize launching experiments.

AbdomenCTCT quick prototyping likewise uses its frozen `--subset quick4` selection
(official pairs 2,8,18,41; selection seed 42), documented in
docs/ABDOMENCTCT_QUICK4.md. Do not redraw it. No experiment is authorized by this selection.

The experimental physical-working-grid branch adds optional 1 mm preprocessing;
see docs/PHYSICAL_WORKING_GRID.md. Preserve original evaluation grids and reuse
frozen reference geometry across resampling comparisons. Experiments remain opt-in.

Physical-grid training remains at 1 mm; `working_grid.recovery_spacing_mm`
independently selects recovery sampling (default 1 mm). Recovery uses physical
units and spacing-weighted PCG. Returned maps use the recorded recovery geometry;
convert through that geometry for original-grid evaluation. See the physical-grid doc.

DIR-Lab quick prototyping uses frozen `--subset quick4`: cases 01,03,04,05,
the four smallest prepared native pairs by voxel count (no random selection).
See docs/DIRLAB_QUICK4.md. Preview the DIR-Lab launcher with `--plan-only`;
it executes by default. Subset preparation does not authorize experiments.

Default endpoint reconstruction again samples within the current registration
crop, including DIR-Lab. Masks only select crop centers. Training trajectories
use two spatial midpoint RK2 half-steps, reusing the two existing fields. Inference remains
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

Current DIR-Lab development selection (2026-11-13): return to native voxel units
with `medical/configs/full_suite/06_native_grid.json`, NCC endpoint fitting,
32k separate registration particles, RK2 training trajectories, and velocity
smoothing sigma 0.75 grid cells. Keep the lung-mask crop-center-only policy.
Physical-unit and metric-smoothing presets remain opt-in ablations; do not select
them for subsequent runs unless requested. This selection does not authorize runs.

`medical/configs/default.json` now matches `full_suite/06_native_grid.json`. Both losses are restored; temporary single-loss presets were removed (resolved configs remain in ignored experiment outputs).

The original shared 5k quick4 reference preset is
`medical/configs/native_siren_onecycle_5k.json` (width 512, uniform omega 30,
output initialization ×30, OneCycle peak 1e-4, AdamW decay 0.001).
See docs/SIREN_CONVERGENCE.md for the comparison with the native reference and
the DIR-Lab-specific recipe. Experimental ablation presets are grouped in
`medical/configs/ablations/`.

When proposing research ideas in discussion or documentation, start from a
conceptual explanation of the observed behavior. Explain how the proposed
change acts on that mechanism and what observation would distinguish it from
a competing explanation. Motivate a combination by how its components interact.

The ten-hour research comparison is recorded in `docs/LONG_RESEARCH_REPORT.md`.
Its explicit shared accuracy candidate is
`medical/configs/native_siren_gw_orthogonal_sobol_5k.json`: captured full-gradient
native training, GW-NCC, two orthogonal query pairs, scrambled Sobol time sampling,
and fixed cap-256/eight-time PCG+RK2 recovery. Default execution and sampling remain
unchanged. Per-case folding regressions remain a qualification; see the report.
The execution interface and guards are in `docs/CAPTURED_TRAINING.md`.
These presets do not authorize new runs.
