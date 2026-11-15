# Finite-operator consistency: formulation and one S02 pilot

**Outcome:** feasible in this bounded test. Both endpoint reconstructions passed the existing gate; held-out operator consistency improved; gradients stayed finite and bounded; unchanged PCG+RK2 recovery improved both Dice and HD95 slightly. This is one pair, one seed and one fixed coefficient setting, not evidence of cohort-level superiority. No coefficient tuning or additional cohorts were run.

## Formulation and mathematical limits

For the existing linear weight path, use four auxiliary forward maps `psi_k: t_k -> t_{k+1}` and minimize the **sum** of spatial mean squared differences `I(psi_k(x),t_{k+1})-I(x,t_k)`. Forward composition is `Phi=psi_3 o psi_2 o psi_1 o psi_0`. Original-image anchoring compares `M_data(a)` to `F_data(Phi(a))`; it does not use SIREN outputs as endpoint data. This direction is distinct from the inverse map used for the final moving-to-fixed image warp.

Each `psi=id+u` is a compact, zero-extended tensor-product cubic B-spline field, with six controls per axis and three displacement components: **2,592 coefficients** across four maps. Displacements use native voxel units and (W,H,D) component order. The operator branch needs no neural deformation network, dense training map, image reconstruction grid, derivative-derived velocity, global solve or ODE integration; the original local PD-LK calculation remains in the baseline objective.

The prototype uses a conservative sufficient diffeomorphism bound, derived directly here. If every component coefficient has magnitude at most `a`, quadratic derivative-basis partition of unity gives `|partial_j u_i| <= 2a/h_j`. Setting `a=.25*min(h)/(2d)` bounds `||Du||_2 <= ||Du||_F <= .25` globally. Therefore `x=y-u(x)` is a contraction, `id+u` is globally invertible on R^d with positive Jacobian determinant, and the four-map composition has derivative norm at most `1.25^4=2.4414`. This is not a guarantee that the image box maps into itself: border sampling and a squared normalized boundary-overshoot penalty handle escaping samples explicitly. Auxiliary-map guarantees do not imply any guarantee for the separately recovered PCG map.

For small increments `u=dt*v`, the operator residual linearizes to brightness transport. It is **not identical to strict PD-LK**: spatial differentiation of exact brightness transport includes `(Dv)^T grad(I)`, while the maintained pointwise frozen-velocity gradient residual omits this term. Both remain soft objectives; a common zero is not assumed. The fixed number of intervals matters for scaling: a sum of squared unscaled increments changes with time resolution.

Exact brightness-preserving operators can also conflict with contrast changes or topology changes in the parameter-interpolated image path. Geometry is nonunique along intensity level sets; low operator residual alone can reflect smoothing, background agreement or incorrect anatomical alignment. Endpoint reconstruction constrains endpoint coadaptation, coarse smooth fields limit capacity, and original-image similarity anchors geometry, but anatomical correctness still requires evaluation. Auxiliary geometry need not match the global flow recovered from the SIRENs.

Cubic B-spline FFDs are established non-neural registration models; coefficient constraints are an established route to invertibility. See [Rueckert et al., 1999](https://webdocs.cs.ualberta.ca/~vis/readingMedIm/papers/RueckertFreeForm.pdf) and [Sdika, 2019](https://doi.org/10.1090/mcom/3419). The bound above is this prototype's deliberately conservative derivation, not an implementation of the latter paper's sharper constraints. Smooth kernel coefficients or a spline stationary-velocity flow are alternatives, but would require separate bounds or integration/convergence checks; they were not tested here.

## Alternating updates and safeguards

One geometry Adam update holds both SIRENs fixed and minimizes operator consistency + original-image MSE + .01 times coefficient membrane regularization + boundary penalty. One endpoint update holds all maps fixed and preserves endpoint reconstruction, strict PD-LK (.1), architecture, initialization, interpolation and the original Adam schedule. It adds operator weight **.1** after step 2,000. There is no differentiation through geometry optimization or composed geometry into the endpoints. The original-image term is constant in the endpoint block; it influences endpoints indirectly through the geometry fitted in the other block.

The weighted operator gradient is separately capped at **.1 times the baseline gradient norm before mixing**, followed by the original global clipping and Adam step. This safeguard makes the update different from unconstrained exact joint minimization. It prevents the particle failure's clipping starvation without changing the maintained PD-LK-only path. Geometry coefficients are projected after each update; their learning rate is .05 native voxels.

Each block samples 1,024 continuous positions using independent RNG. Every 20 steps, fixed probes check 65,536 rounded continuous locations against 25 dB and evaluate operator/data losses at 2,048 independent continuous locations. Training aborts on a failing probe, non-finite gradients or its 180-second budget. The pair runner has a separate 450-second limit. Sparse training probes do not certify the full-volume gate: the original full endpoint gate runs after training. The prototype exposes no persistent map or auxiliary-coefficient writer; only final SIREN weights and compact evidence are saved.

## Fixed test and results

Prepared LPBA S02 moving to S01 fixed, seed 42, 5,000 steps, original packaged recovery settings. The PD-LK control is the valid earlier S02 run. Logged warm-up losses through step 2,000 match that control **exactly**, so the independent sampling/hooks did not alter the pre-activation trajectory.

| Measure | PD-LK control | Operator prototype |
|---|---:|---:|
| Moving endpoint PSNR dB | 32.2247 | **32.1715** |
| Fixed endpoint PSNR dB | 32.5119 | **32.4493** |
| Moving / fixed endpoint MSE | .0005991 / .0005608 | .0006065 / .0005689 |
| Final full-volume gate | pass | **pass** |
| Dice | .698229 | **.700765** |
| HD95 mm | 6.56515 | **6.52270** |
| Measured recovered folding fraction | 0 | **0** |
| Training seconds | 43.78 | 94.47 |
| Recovery seconds | 19.58 | 17.99 |
| Flow intervals meeting tolerance | 15/16 | **15/16** |

Dice increases **.002537** and HD95 decreases **.04245 mm**. Both endpoint PSNRs are slightly lower than the control but comfortably above 25 dB. MONAI 1.6.1 evaluates all 56 union labels, with macro Dice and maximum directed HD95 in physical millimetres. No label was missing and no case was omitted.

Both conditions have one narrowly non-converged recovery interval: maximum true relative residual is **2.0195e-5 control / 2.0081e-5 prototype**, against tolerance **2e-5**. Settings and recovery code were unchanged; this qualification remains even though Dice and HD95 improved. No inference tuning or surrogate-map substitution occurred.

Held-out operator loss decreases from **.008335 at activation to .002472 at termination**. On the final prototype SIRENs, identity operators give .008341: the learned smooth operators reduce it by about **70.4%**. The original-image auxiliary endpoint MSE decreases from **.019216 to .007740**. These are auxiliary losses, not final recovered-image metrics. For context, applying the prototype's learned maps to the control SIRENs gives operator loss .003274 versus .009013 with identity; those maps were not separately optimized for the control, so this is not a matched geometry-fitting comparison.

All active gradients passed finite checks. Across the 150 logged active samples, nominal weighted operator gradient norms are **.01430–.02391**, baseline norms **.04116–.07638**, and applied operator norms **.00412–.00764**. Budget multipliers are .1898–.4497. These ranges are sampled logs, not claimed maxima over every update; the bound is enforced on every active update. Lowest sampled endpoint PSNR stays above 30 dB. No safeguard fired. Observed settled active-step time is about **26 ms**, versus about 7 ms for PD-LK-only; these are run observations, not an independent speed benchmark.

## Implementation, validation and interpretation

`medical/registration/operators.py` owns sparse operators, loss algebra and alternating-block safeguards. The shared trainer calls its hooks only for `objective.operator_consistency`; dataset presets remain PD-LK-only, weight zero skips the operator branch, and combining it with particle loss is rejected. Final PCG/RK2 inference and evaluation remain separate. The [method](../archive/methods/METHOD_BEFORE_DEFAULT_CONSOLIDATION_20261112.md#experimental-finite-operator-consistency) and [development guide](../DEVELOPMENT.md#experimental-finite-operator-consistency) define the interfaces and settings.

Synthetic checks cover native coordinates, direction/noncommuting composition, projected spatial Jacobians and positive determinants, finite differences, endpoint/geometry gradient separation, gradient budget, reconstruction/runtime guards and schema rejection. Both README suites pass (42 medical tests plus one Cameraman test). A full-width CUDA block-gradient check reaches both SIRENs without optimizer steps. Packaging includes only maintained packages/configs/JPEG; no tests, data or result artifacts.

The single initial test meets the requested numerical success criteria, with the flow-tolerance qualification above. It establishes feasibility and a small S02 improvement, not robustness across pairs/seeds or the necessity of every component. No coefficient selection used Dice/HD95, no additional coefficient was tested, and no cohort was launched. The next scientific question is whether this fixed protocol reproduces on another authorized pair; the current evidence does not authorize that expansion automatically.

Local provenance: `results/operator_consistency/20261110T095724_102075Z` retains fixed config/protocol, source snapshot/hashes, audited runner, final endpoints and scalar evidence. [Reviewed scalar summary](operator_consistency_20261110.json) records comparisons. Prior experiments were hashed before/after and remained unchanged. Medical data and the earlier benchmark remain byte-for-byte unchanged; recovery scratch is empty. No commit or push was made.
