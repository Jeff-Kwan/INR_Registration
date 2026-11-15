# Paired-SIREN registration

Each image pair gets its own fit. Two coordinate networks represent the moving and fixed intensities. Linear interpolation of their **parameters** defines a continuous intensity function in space and time. Its analytical derivatives supply the motion estimates used in training and recovery.

## From images to maps

1. **Prepare the endpoints.** Keep both images on their prepared native voxel grid and independently min–max scale each to [0, 1]. Initialize two identical SIRENs, then optimize their parameters separately in one joint objective. Moving is time 0 and fixed is time 1. At time `t` the network uses weights and biases `(1 − t) θ_moving + t θ_fixed`.
2. **Sample one crop per update.** Both recipes below use half the image length along each axis. OASIS and AbdomenCTCT sample an origin uniformly among fitting boxes. DIR-Lab chooses the T00 or T50 mask with equal probability, samples one foreground voxel, centers the box there and shifts it to fit. Both endpoint losses use the same 32,768 sampled voxel indices. Independently draw 32,768 continuous particle starts **per direction** from the central 80% of the crop along each axis. Evaluation labels and landmarks stay out of fitting.
3. **Fit endpoint intensities.** Sum the two endpoint Pearson NCC losses against their respective observed images. NCC leaves output offset and scale underdetermined. A Gaussian moment penalty with weight 0.01 anchors each network output toward mean zero and variance one on the minibatch.
4. **Estimate local motion.** Evaluate the interpolated SIREN at one early and one late time, one from each half of the interval. At each time, every cell of an 8³ crop grid uses four sites arranged as two symmetric pairs. Write `g = ∇I`, `q = ∂I/∂t`, `H = ∇²I` and `m = ∇q`, with spatial derivatives in voxel units. Solve one three-component velocity per cell by minimizing the site average of `(q + g·v)² + 4 ‖m + Hv‖²` plus `10⁻⁶ ‖v‖²`. The minimized energy, averaged over cells and times, gives the PD-LK loss. The brightness term asks which motion explains the intensity change. The differential term also asks whether that motion explains how the local gradient changes.
5. **Train against alignment.** Smooth each velocity grid with a separable three-tap Gaussian (σ = 0.75 cells). Use the two fields in order for moving→fixed and in reverse order with negative time steps for fixed→moving. Each half-step uses spatial midpoint RK2. Compare the actual source intensities at the starts with target intensities at the transported positions. Average the two directional similarity losses and full-image boundary penalties. Every trajectory counts, including crop exits, with border extension for both velocity and image interpolation. The total is **endpoint loss + 0.1 × PD-LK + 0.01 × registration**. Gradients flow back through the analytical derivatives, cell solves, smoothing and trajectories to both networks.
6. **Recover dense maps from the frozen pair.** At evenly spaced interval midpoints, evaluate the spatial gradient and time derivative across the volume. Solve global brightness flow with spatial smoothness 0.01 and ridge 10⁻⁶ by preconditioned conjugate gradients (PCG), warm-starting each interval from the previous velocity. Midpoint RK2 integrates the fields in both directions. Recovery scratch is temporary. The evaluator consumes both map coordinates and spacing in the original image geometry.
7. **Score each pair.** DIR-Lab reports T00→T50 and T50→T00 landmark TRE separately in mm, including continuous and IDIR-rounded values. The primary T00→T50 point map is the inverse of the model's moving→fixed trajectory. OASIS and AbdomenCTCT use the fixed→moving pullback for official Dice, voxel HD95 and shifted SDlogJ. Report the unshifted nonpositive-Jacobian fraction separately to expose folding. The full single-seed cohorts contain 10, 19 and 45 pairs respectively. [Dataset protocols](DATASETS.md)

## Which configuration reproduces which pipeline?

| Setting | Repository default | Leading shared accuracy candidate |
| --- | --- | --- |
| SIREN per endpoint | Four sine layers, width 256, ω = 30 | Four sine layers, width 512, ω = 30, output layer initialization ×30 |
| Per-pair fit | 5,000 steps, Adam, exponential LR decay from 2 × 10⁻⁴ toward 5 × 10⁻⁵ | 5,000 steps, AdamW, decay 0.001, OneCycle peak 10⁻⁴ after 10% warmup |
| Cell query directions | Two independently drawn directions | Two orthogonal directions |
| Early/late time draws | Independent uniform draws | Scrambled 2D Sobol draws mapped to the same half-intervals |
| Registration similarity | One Pearson NCC over the box | Equal mix of global and Gaussian-window NCC |
| CUDA execution | Ordinary training loop with compiled kernels | Captured full-gradient training step |
| Frozen recovery | Full native grid, 16 time evaluations × 4 RK2 substeps | At most 256 samples per axis, 8 evaluations × 8 substeps, returned maps on the original grid |

Omitting the config selects the repository default. To reproduce the latest full-cohort accuracy run, select the candidate explicitly. The earlier width-512 OneCycle preset is the comparison reference for that report. The 4k speed run is a separate research operating point. [Default](../medical/configs/default.json) · [Accuracy candidate](../medical/configs/native_siren_gw_orthogonal_sobol_5k.json) · [Original shared reference](../medical/configs/native_siren_onecycle_5k.json) · [Full results and folding audit](LONG_RESEARCH_REPORT.md)

## Supported experiments

Endpoint MSE, endpoint GW-NCC and explicit window LNCC are independent of the choice of registration similarity. PD-LK and registration weights can independently be zero. Other documented experiments include inverse-metric velocity agreement, structure sampling and physical units. The physical-grid protocol specifies the checkpoint geometry required for recovery. [Inverse metric](INVERSE_METRIC.md) · [Structure sampling](STRUCTURE_SAMPLING.md) · [Working grid and window objectives](PHYSICAL_WORKING_GRID.md) · [Native physical units](NATIVE_PHYSICAL_UNITS.md)

Sampler locations are XYZ, while PyTorch volume axes are ZYX. Normalized-coordinate derivatives use the scale `2/(axis_size−1)` per spatial axis. Dataset adapters convert maps to landmark or label geometry for scoring. [Case runner](CASE_RUNNER.md) · [Captured training and capped recovery](CAPTURED_TRAINING.md)
