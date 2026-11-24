# Native grid with physical derivative units

Opt in with top-level `"spatial_units": "mm"`. Omission retains native voxel
units. This option is separate from `working_grid`: using both is rejected to
prevent ambiguous geometry. The existing working-grid implementation is unchanged.

Preset: `medical/configs/native_physical_units_endpoint_ncc_32k.json`, copied from
the native-grid full-suite NCC configuration. It retains NCC reconstruction and
its moment prior, PD-LK 0.1, registration NCC 0.01, 32k separate registration
particles, two spatial RK2 half-steps and standard 16-interval PCG+RK2 recovery.
No coefficients have been tuned for this mode.

The experiment retains native image and evaluation grids, reconstruction targets, SIRENs and sampling budgets.

For native spacing h in sampler-component order XYZ, the SIREN derivative scale
is `2 / ((N_xyz - 1) * h)`. Consequently g is per mm, H per mm squared and m
per mm per unit time; q is unchanged. The shared-grid solve and PD-LK energy
use these physical derivatives and mm/time velocities, including the existing
c and eta terms. Inverse-metric regularization uses the same unit path.
Before field smoothing/interpolation, divide velocity components by h to express
them in native voxels/time for particle coordinates. This fixed linear conversion
commutes with Gaussian smoothing and preserves gradients through the solves,
smoothing and trajectories. Endpoint NCC/MSE intensities are unchanged.

Recovery evaluates derivatives at the original native grid sites, converts g to
per mm, and supplies native ZYX spacing to the existing spacing-weighted PCG
Laplacian. Warm starts are velocities in mm/time. Divide recovered velocities by
XYZ spacing before RK2. The optional gradient-aware recovery path also uses physical H/m and spacing.

Axes may be rotated or reflected, but must be orthogonal and nondegenerate. Vector
components are distances along the native oriented image axes, not world RAS
components. Adapters supply affines with columns in tensor ZYX order. Do not pass
an ordinary NIfTI XYZ-column affine directly to `fit_pair`.

## Geometry and checkpoints

`fit_pair` requires an affine for this mode and records `physical_units_metadata`
on the model. Checkpoints and result metrics from all three dataset adapters
contain a `physical_units` section recording the version, mm units, native ZYX
shape, ZYX-column affine and XYZ spacing. The ordinary model state and resolved
configuration are retained as before. Generic `medical.artifacts` save/load also
preserve this section.

Restore dataset-runner checkpoints using
`medical.physical_units.restore_checkpoint(payload, device)`. It requires and
validates geometry; do not restore only the state dict and silently assume voxel
units. `recover` rejects missing required metadata or a shape mismatch; an explicit
`physical_units_metadata` keyword is available for synthetic/custom callers.
Working-grid and native physical-unit metadata cannot be combined. Historical
checkpoints are not relabeled as physical-units fits.

## Limits and validation

This standardizes derivative and velocity units. It does not recover missing
through-plane image information or make all hyperparameters physically identical:
fractional crop extents, cell widths and Gaussian smoothing bandwidths still
follow the native image geometry. The sampling probe radius already uses mm.
Therefore this mode does not guarantee resampling invariance. Both training
units and recovery regularization units change together in the provided preset;
it is not a training-only ablation.

Synthetic checks: `medical.tests.test_physical_units`, plus existing recovery-grid,
working-grid and derivative regressions. They cover g/q/H/m scaling, a nontrivial
local solve and energy, parameter gradients, unit-spacing equivalence, rotated
and reflected anisotropic axes, native recovery velocity conversion, spacing
routing, checkpoint round trips, invalid/missing geometry and scratch cleanup
on success and exceptions. Optional CUDA coverage compares eager and compiled
physical fields and parameter gradients. No optimizer updates or real-case
recovery are needed for these checks.

Completed quick4 experiments are summarized below. No deformation or warped-image writer is added.

## Optional metric velocity smoothing

`objective.registration.sigma_mm` selects an isotropic physical Gaussian sigma
for the local training velocity field. When present it overrides `sigma_cells`.
Each crop recomputes sigma in lattice samples using its cell-center spacing:
`cell_width_xyz_mm = (crop_shape_xyz - 1) * spacing_xyz_mm / grid_size`.
The smoothing array receives `sigma_mm / cell_width_zyx_mm`, accounting for
the reversed tensor-axis order. Crop origin changes do not change the bandwidth;
crop size changes do. This works with native voxel units, native physical units
and the working-grid path. It does not change derivative probes, GW-NCC windows
or global PCG regularization.

Preset `native_physical_units_sigma7p5mm_endpoint_ncc_32k.json` differs from
`native_physical_units_endpoint_ncc_32k.json` only by `sigma_mm: 7.5`. This value
matches the nominal 0.75-cell sigma of the former 80 mm / 8-cell physical crop.
It is an explicit starting ablation value, not a tuned optimum. All kernel
weights and field operations retain full gradients.

The preset retains the existing three-tap Gaussian and reflection boundary
handling. Sigma defines Gaussian weights in mm; finite kernel support and
lattice discretization mean the achieved kernel variance need not equal
sigma squared exactly. This does not guarantee equivalent smoothing at arbitrary
grid resolutions. No new experiment was launched when adding this option.

## Quick4 outcome and current selection (2026-11-13)

It retains the corrected mask-centered shared crops and full-gradient losses. The physical-unit options remain available for explicit ablations. Their fixed
coefficients were not retuned; these results do not establish that physical
units are intrinsically worse. Metric sigma also changed smoothing strength.
