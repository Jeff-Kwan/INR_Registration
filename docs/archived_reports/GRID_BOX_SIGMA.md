# Crop-relative shared-grid velocity smoothing

The original SIREN-Reg estimator pools local derivative observations before its
PD-LK solve. A broad blur can suppress disagreement between neighboring solved
cells early in fitting; narrowing the blur later restores spatial variation.
This opt-in experiment changes only that velocity-grid smoothing. It retains
the shared-grid solve and raw minimized PD-LK residual, the selected training integrator,
the endpoint/registration objectives, masks/crops, and frozen PCG/RK2 recovery.

`objective.registration.sigma_box_fraction` and optional
`final_sigma_box_fraction` specify Gaussian sigma as a fraction of each crop's
voxel-center span, not FWHM. For a G³ cell-centered velocity grid, spacing per
axis is `crop_extent / G`, so the cell sigma is simply `fraction * G`. This
conversion holds independently for crop width, height and length, including
anisotropic physical spacing and rectangular crops. At G=8, 0.4→0.1 gives
3.2→0.8 cells. Initial and final values are reached at steps 1 and N; N=1 uses
the initial value. Omitting the final fraction gives a constant bandwidth.

The original fixed sigma0.75 cells is already 0.09375 of a box side at G=8.
The main experimental changes are the broader early smoothing, its schedule,
and full Gaussian support; the final fraction0.1 is close to the old scale.

Use `gaussian_kernel: "full"` with these options. Three-tap weights cannot
represent a 3.2-cell Gaussian. The new path folds the wider one-dimensional
Gaussian taps onto a tiny G×G reflected-convolution matrix, then applies it
separably along all three axes. Reflection matches the existing smoothing
boundary convention and repeats correctly even when support exceeds the grid.
The fixed radius is ceil(4 * largest scheduled cell sigma); weights are normalized
at each sigma. Taps beyond four current sigmas may remain when sigma shrinks,
so the schedule avoids changing support shape. Constant fields remain constant.

All schedule matrices are precomputed once, in FP64 on CPU, then converted to
the field's device/dtype. Training selects a tensor view for the current step,
so sigma changes do not specialize/recompile the smoothing graph. Gradients
through the field remain live. The native decoder CUDA graph remains available.
Gaussian-window NCC moments use their original independent settings.

Preset: `medical/configs/ablations/grid_box_sigma_0p4_0p1.json`. This stays opt-in;
the repository default retains sigma0.75 cells/three taps. Relative grid sigma
cannot combine with registration.sigma_mm or the particle objective.
Ordinary and complete captured training both support it. Zero objective weights remain supported.

Synthetic tests compare explicit reflected convolution and all field VJPs,
including support wider than G, constant fields, linear schedules, crop resizing,
anisotropic physical conversion and model gradients. The CUDA test compares
compiled values/gradients with changing matrix views and verifies 100 different
step weights create no extra compiled graphs. Synthetic fitting, quick4 tuning
and full public-validation evidence are recorded in the experiment outputs.

## Completed staged comparison, 2026-11-15

The synthetic far128 comparison used the native shared-grid default: width256,
5000steps, seed42, 32768 registration particles per direction, pooled PD-LK,
RK2 training and unchanged full-grid sixteen-interval PCG/RK2 recovery. Only
velocity smoothing differed.

| Smoothing | Forward/reverse EPE vox | Dice | Training seconds |
|---|---:|---:|---:|
| Original sigma0.75-cell three-tap | 4.380/3.915 | 0.97800 | 64.40 |
| Box sigma0.2→0.1 | 3.373/2.990 | 0.97885 | 64.10 |
| Box sigma0.4→0.1 | 3.300/2.920 | 0.97897 | 64.65 |
| Box sigma0.6→0.1 | 3.486/3.081 | 0.97839 | 65.07 |

CPU/CUDA convolution and gradient checks passed, including wide reflected
support and unchanged compilation count over changing schedule weights. The
small-field synchronized forward/backward probe measured0.82ms versus0.72ms
for the original three-tap implementation; complete training time was similar.
All synthetic PCG solves converged. Every fit, including the reference, had
1–5 whole-grid folded voxels out of2097152. Consequently the initial exact-zero
synthetic gate admitted none. Before quick4 results, it was changed to a
1e-5 whole-grid fraction bound; the two lowest-error sweeps advanced. This
is adaptive development, not a preregistered confirmatory trial.

Quick4 used the frozen subsets of all three medical datasets. All576 PCG intervals
converged.

| Smoothing | OASIS Dice |
|---|---:|
| Original | 0.81881 |
| Box sigma0.4→0.1 | 0.82091 |
| Box sigma0.2→0.1 | 0.82211 |

The selected0.2→0.1 sweep had mean case-rank1.583 versus2.167 for0.4→0.1 and
2.250 for the original. Its grid sigma is1.6→0.8 cells. The published inference and all other method settings stayed
fixed. Preset: `medical/configs/ablations/grid_box_sigma_0p2_0p1.json`.

The selected12 quick4 fits were reused; the remaining62 ran in1166seconds
on four RTX5090 GPUs. All74 succeeded and all1184 PCG intervals converged.

| Dataset | Full-cohort score | Training/case | Recovery/case | Total/case |
|---|---:|---:|---:|---:|
| OASIS19 | Dice0.80691; HD95 1.8426vox | 62.09s | 6.13s | 69.30s |

No hidden-test performance is claimed.

They are a different case mix, not a
matched baseline comparison. There is no newly run original-method full-cohort
control in this experiment; only quick4 provides a direct before/after test. The repository default remains unchanged. Input/source checks passed and all
recovery scratch was cleaned. Only compact checkpoints, configs and metrics
were retained.

[Experiment manifest](../results/grid_box_sigma_20261115T110551Z/experiment.json)
· [Synthetic scores](../results/grid_box_sigma_20261115T110551Z/synthetic_analysis.json)
· [Quick4 selection](../results/grid_box_sigma_quick4_20261115T111037Z/selection.json)
· [All74 scores and runtimes](../results/grid_box_sigma_full_20261115T111226Z/analysis.json).

Captured training now also supports this full Gaussian schedule. It precomputes
the same reflected matrices once and copies the current 8×8 matrix into a fixed
GPU buffer before each update. The captured objective smooths all four Euler
fields through that buffer with live field gradients. Sigma changes reuse the
compiled graph. The original three-tap path is unchanged when crop-relative
sigma is omitted. Pure CPU value/all-parameter VJP checks match ordinary training
at the beginning, middle and end of the schedule. The complete six-update CUDA
graph schedule/L2 synthetic optimizer check passed in 30.292 seconds before
the authorized generalist campaign continued to its refinement batches.
