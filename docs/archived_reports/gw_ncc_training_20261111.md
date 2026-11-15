# Gaussian-weighted window NCC training pilot

The original pilot used a fixed local-box grid. Subsequent sections identify
landed-point or source-coordinate binning for each follow-up.

## Setup

Implemented `registration_similarity="gw_ncc"` as an opt-in replacement for the
local registration NCC in the symmetric-pair branch. LPBA S02 was trained once
with seed 42 and the packaged 5,000-step schedule. The existing MSE endpoint
objective, shared-grid strict PD-LK (`weight=.1`), local registration weight
`.05`, bidirectional 4,096 particles, 8³ velocity grid, two midpoint-Euler
fields, crop and final recovery configuration were left as configured. The
window filter used `gw_ncc_sigma_cells=.75`; velocity smoothing separately
used `sigma_cells=.75`. These equal values are independent parameters.

Each valid directional particle contributes six raw moments to its nearest
cell on a fixed 8³ local-box grid. Six-channel scatter-add and separable
three-tap filtering yield local population moments. Smoothed particle count
weights local NCC; empty windows are masked and variance epsilon is 1e-8. The
registration scalar is half global NCC loss plus half weighted local NCC loss,
averaged over directions. No new derivative, local solve, particle or
integration evaluation was added.

## Result

Training completed all 5,000 steps. Total pair time was 76.27 s, including
startup/compilation; the final logged step block averaged 14.09 ms per update,
with the registration branch taking 12.47 ms. Final endpoint metrics were:

| Endpoint | MSE | PSNR |
| --- | ---: | ---: |
| Moving | 0.001383 | 28.59 dB |
| Fixed | 0.001109 | 29.55 dB |

The PSNR gate was explicitly skipped, so endpoint pass status is null. At step
5,000 the unweighted GW-NCC registration loss was 0.20059: mean global loss
0.02756 and mean local loss 0.37362. Its `.05` weighted contribution was
0.01003. Shared-grid PD-LK was 0.01032 before its `.1` weight (0.00103 after
weighting). These are training diagnostics, not registration-quality metrics.

The subsequent unchanged PCG+RK2 recovery took 13.11 s (18.09 s total
evaluation time). It scored Dice **0.70642**, HD95 **6.2439 mm**, and zero
folding. All 16 PCG intervals converged; the maximum relative residual was
1.98e-5. Warped-image MSE changed from .02090 to .002312 and global NCC from
.8890 to .9862. Compared with the existing S02 MSE + shared-grid PD-LK + crop-wide
NCC run (.71441 Dice / 6.251 mm HD95), GW-NCC was .00799 lower in Dice
and .007 mm lower in HD95. This single-pair comparison does not establish a
general improvement; the small HD95 difference is effectively unchanged.

The PSNR training gate was explicitly skipped, so endpoint pass status remains
null despite measured endpoint PSNR above 25 dB. The run artifacts are kept in
the ignored `results/gw_ncc_s02_20261111/` and
`results/gw_ncc_s02_recovery_20261112/` directories; no data or checkpoints
were staged.

The full medical synthetic suite passed (119 tests). The focused tests cover
six-channel grouped smoothing, statistic accumulation, local/global loss
composition, live trajectory gradients, and finite gradients reaching both
endpoint SIRENs. See [method](../METHOD.md#bidirectional-symmetric-pair-training)
and [configuration](../DEVELOPMENT.md#bidirectional-symmetric-pair-training)
for the selectable option.

## S02: landed-point grid follow-up (2026-11-12)

The GW-NCC grid was changed to span the valid endpoint coordinates' per-axis
minimum and maximum, independently for each direction and minibatch. The grid
assignment and fitted bounds are detached; gradients continue through the
warped intensities and trajectories. The run used seed 42, 5,000 steps, MSE
endpoints, shared-grid PD-LK weight .1, GW-NCC weight .01, three-tap filtering
at .75 cells, and the same PCG+RK2 recovery settings. This is the first training
run with the landed-point grid.

Training took 74.05 s. Endpoint PSNR was 30.61/31.19 dB; the configured PSNR
gate was skipped. At step 5,000 the unweighted GW-NCC loss was .21962 and its
weighted contribution was .002196. Recovery took 12.16 s and scored Dice
.71310 / HD95 6.3041 mm with zero folding. Fifteen of 16 PCG intervals
converged; maximum relative residual was 2.0071e-5. The preceding fixed-box
.01 GW-NCC S02 run scored .71448 / 6.2237 mm, with zero folding and 15/16
intervals converged. Those runs differ in grid assignment only and use the
same seed and loss settings. Results are single-pair evidence.

Training and recovery artifacts are in the ignored `results/gw_ncc_s02_landed_grid_20261112/`
and `results/gw_ncc_s02_landed_grid_recovery_20261112/` directories.

## S02: landed-point grid, weight .05 (2026-11-12)

With the landed-point grid, seed 42, and all other settings matched to the .01
run above, GW-NCC weight .05 completed 5,000 steps in 74.55 s. Endpoint PSNR
was 28.44/29.30 dB; the configured gate was skipped. The final unweighted
GW-NCC loss was .21239 and its weighted contribution was .01062. Unchanged
PCG+RK2 recovery took 13.11 s and scored Dice .70341 / HD95 6.3514 mm, zero
folding, with 16/16 intervals converged (maximum relative residual 1.9939e-5).
The matched landed-grid .01 run scored .71310 / 6.3041 mm, zero folding and
15/16 intervals converged. This is one seed and pair.

Artifacts are in the ignored `results/gw_ncc_s02_landed_grid_weight_005_20261112/`
and `results/gw_ncc_s02_landed_grid_weight_005_recovery_20261112/` directories.

## S02: source-coordinate grid, weight .05 (2026-11-12)

Valid correspondences are binned by their starting coordinates in a fixed 8³
grid spanning the central 80% of the local box. The seed, MSE endpoints,
shared-grid PD-LK (.1), GW-NCC (.05), three-tap window sigma (.75 cells) and
recovery settings match the landed-point .05 run above. Training took 74.85 s;
endpoint PSNR was 28.92/29.49 dB and its gate was skipped. Recovery took
13.47 s and scored Dice .70736 / HD95 6.3313 mm, zero folding, and 14/16 PCG
intervals converged (maximum relative residual 2.0173e-5). The final
unweighted/weighted GW-NCC losses were .22246/.01112. This is one pair and seed.

Artifacts are in the ignored `results/gw_ncc_s02_source_grid_weight_005_20261112/`
and `results/gw_ncc_s02_source_grid_weight_005_recovery_20261112/` directories.

## S02: source-coordinate grid, weight .01 (2026-11-12)

Using the same source-coordinate grid, seed 42 and other settings as the .05
run, weight .01 completed training in 73.86 s. Endpoint PSNR was 30.54/31.28
dB; the gate was skipped. The final unweighted and weighted GW-NCC losses were
.22553 and .002255. PCG+RK2 recovery took 12.11 s and scored Dice .71154 /
HD95 6.3166 mm with zero folding. Fourteen of 16 PCG intervals converged; the
maximum relative residual was 2.0128e-5. This is one pair and seed.

Artifacts are in the ignored `results/gw_ncc_s02_source_grid_weight_001_20261112/`
and `results/gw_ncc_s02_source_grid_weight_001_recovery_20261112/` directories.

## S02: shared reconstruction and registration particles (2026-11-12)

With source-coordinate GW-NCC (.01), shared-grid PD-LK (.1), seed 42 and
unchanged recovery, each direction sampled 16,384 continuous starting points.
Moving and fixed endpoint MSE used the respective forward and reverse starts,
with trilinear original-image targets. All starts contributed to reconstruction;
registration excluded trajectories outside the local box. This changes particle
count, reconstruction sampling support and target interpolation together compared
with the earlier 4,096-particle and 32,768-voxel-per-endpoint run.

Training took 66.72 s. Endpoint PSNR was 23.07/21.89 dB, below 25 dB; the
gate was explicitly skipped and its pass field is null. Recovery took 11.76 s
and scored Dice .71340 / HD95 6.2295 mm, zero folding, and 15/16 PCG intervals
converged (maximum relative residual 2.0116e-5). The prior source-grid .01 run
scored .71154 / 6.3166 mm, with 30.54/31.28 dB endpoint PSNR and 14/16 PCG
intervals converged. Results are one seed and pair.

Artifacts are in the ignored `results/gw_ncc_s02_shared_recon_16k_20261112/`
and `results/gw_ncc_s02_shared_recon_16k_recovery_20261112/` directories.
