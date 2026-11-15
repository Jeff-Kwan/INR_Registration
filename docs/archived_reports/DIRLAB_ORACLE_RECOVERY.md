# ConvexAdam oracle-trajectory recovery ablation

Frozen DIR-Lab quick4: cases 01,03,04,05, seed 42. This is a recovery diagnostic, not a new method fit. ConvexAdam is the best of our completed ten-case baseline runs (1.402 mm rounded TRE). Dense maps were not retained, so the pinned native ConvexAdam API was rerun with the unchanged baseline config. Small rerun differences occur; every paired comparison below uses the exact same regenerated map within each case.

## Trajectory and controls

Let d(x) be ConvexAdam displacement, mapping fixed T00 coordinates into moving T50 coordinates. Set I_t(x)=M(x+t d(x)), where M is the min/max-normalized T50 image. I_0=M and I_1=M composed with the ConvexAdam map. I_1 approximates T00 anatomically but does not insert its different intensities. This appearance-preserving construction makes the desired inverse time-flow map exactly x+d(x). It is an oracle-informed image trajectory, not landmark-supervised training or a guarantee that the baseline deformation is ground truth.

Use the existing 16 intervals, interval-midpoint fields, four midpoint RK2 substeps per interval, warm-started brightness PCG, smoothness 0.01, ridge 1e-6, tolerance 2e-5, cap 500, native voxel units and FP32 fields. No solver or integration production code changed.

- **Known velocity:** J_t=Id+t Dd; v_t= -solve(J_t,d). This checks map direction, trajectory and RK2 without estimating velocity from image derivatives. Dd uses central finite differences (one-sided edges). All sampled path Jacobians are positive in these cases.
- **Rendered-frame PCG:** render 16 interval-midpoint images; take their finite-difference spatial gradients. Temporal derivatives use differences of the adjacent interval-boundary renderings divided by 1/16 (17 boundary frames evaluated transiently). The primary paired comparison scales all derivative evidence by 1/std(M), giving unit-variance image intensities.
- **Transport-consistent PCG:** a=sample(gradient(M), x+t d); g=J_t^T a and q=a dot d. The source gradient uses finite differences, but the chain rule is applied consistently: g dot v_t+q=0 by construction. This isolates the recovery equations from inconsistency introduced by separately differencing rendered frames. These are not claimed to be exact derivatives of the piecewise-trilinear rendered image at every point.

The chain-rule control uses oracle deformation derivatives; the rendered-frame test uses only the rendered intensities for g and q. Neither provides the known velocity directly to PCG.

## Paired results

| Case | Direct ConvexAdam | Known velocity + RK2 | Rendered-frame PCG + RK2 | Consistent-derivative PCG + RK2 |
| --- | ---: | ---: | ---: | ---: |
| 01 | 0.983 | 0.984 | 1.468 | 0.881 |
| 03 | 1.023 | 1.029 | 2.109 | 1.051 |
| 04 | 1.356 | 1.368 | 2.781 | 1.369 |
| 05 | 1.435 | 1.455 | 2.410 | 1.559 |
| Mean | 1.199 | 1.209 | 2.192 | 1.215 |

Values are forward IDIR-rounded TRE in mm. Continuous TRE and all interval diagnostics are in the raw summaries. Both paired PCG variants converge in all 64 intervals. Retained improvement means (initial TRE minus recovered TRE)/(initial TRE minus direct ConvexAdam TRE), using cohort means. It does not mean a fraction of perfectly recovered landmarks.

- known_velocity: 99.83% of baseline TRE reduction retained; mean map-point error relative to direct ConvexAdam 0.0184 mm; mean diagnostic runtime 1.96 s/case.
- unit_variance: 82.99% of baseline TRE reduction retained; mean map-point error relative to direct ConvexAdam 1.9626 mm; mean diagnostic runtime 4.97 s/case.
- unit_variance_chain: 99.73% of baseline TRE reduction retained; mean map-point error relative to direct ConvexAdam 0.2187 mm; mean diagnostic runtime 6.36 s/case.

An initial additional [0,1]-intensity rendered-frame run gave 2.337 mm versus 1.202 mm direct, with only 43/64 intervals converged; it cannot cleanly isolate the converged recovery objective. A separately rerun unit-variance chain-rule control agreed with the paired conclusion. Initial FP64-to-FP32 conversion failures were corrected in the disposable driver before successful runs; failure logs are preserved.

## Interpretation

RK2 retains the oracle map closely. With transport-consistent derivative evidence, the current brightness PCG objective also retains nearly all ConvexAdam landmark accuracy. This argues against a large unavoidable loss from the PCG+RK2 formulation on these four cases. Separately finite-differenced image frames lose substantial accuracy even when PCG converges. The g/q construction and its transport consistency matter. This does not establish that derivatives of our trained SIRENs provide equivalent transport evidence, or that all ten cases behave similarly. It does not isolate temporal discretization from spatial derivative/resampling errors. No new regularization weight was tuned.

Raw results: `results/dirlab_quick4_convexadam_oracle_20261113/paired/summary.json`; initial intensity controls in the parent directory. Each output includes the disposable driver, source revision, config, timings, TRE and interval diagnostics. Intermediate images, displacements and velocities were kept in memory or exception-safe temporary scratch; all successful runs verified scratch cleanup. No dense maps, warped images or interpolation images were retained. No SIREN training, abdomen experiment or commit was performed.
