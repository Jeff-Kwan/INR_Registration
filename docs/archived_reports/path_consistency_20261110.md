# Nonlocal consistency and solver recoverability: independent S02 tests

**Decision:** neither tested extension should replace the adjacent-only operator control. A gives mixed anatomical metrics and does not improve held-out nonlocal consistency over that control. B slightly improves native-grid motion alignment but worsens Dice and HD95; its continuous held-out residual also worsens. These are bounded negative/mixed results for the fixed implementations, not general refutations of either research direction.

## Source facts and fixed protocol

The source uses four auxiliary maps and **five knots**, at 0, .25, .5, .75 and 1. Final recovery uses 16 midpoint global brightness-flow solves, a positive graph Laplacian with missing boundary edges omitted, scalar Jacobi preconditioning, and RK2. Architecture, initialization, normalization, native-voxel derivatives, strict PD-LK, recovery constants and metric definitions were unchanged. No labels entered training or coefficient selection.

We ran one fresh adjacent-only control, one A fit and one B fit on prepared LPBA S02 moving to S01, seed 42, 5,000 steps, activation after step 2,000. The immutable PD-LK checkpoint was evaluated without retraining. The fresh adjacent control reproduced **all logged losses and final scores exactly** from its previous run. All three fits matched PD-LK warm-up losses exactly through step 2,000. No combined A+B, coefficient search or cohort run occurred.

Settings were frozen before scoring: existing operators (six controls/axis, Lipschitz .25, 1,024 samples/block, operator weight .1, original-data weight 1, membrane .01, boundary 1), original optimizer schedules, original .1 operator-gradient budget, 25 dB full endpoint gate, sampled checks every 20 steps, 180-second training and 450-second pair limits. All conditions finished without a triggered safeguard.

## A: mathematical deduction and implementation choice

For ordered existing knots i<j, compose the existing maps forward: `Psi_{i,j}=psi_{j-1} o ... o psi_i`. Along `x_{k+1}=psi_k(x_k)`, let `r_k=I(x_{k+1},t_{k+1})-I(x_k,t_k)`. Then

`I(Psi_{i,j}(x),t_j)-I(x,t_i) = sum_{k=i}^{j-1} r_k(x_k)`.

This telescoping identity means nonlocal residuals add **weighting and coupling**, not independent constraints when adjacent consistency already holds everywhere. Uniform start-space samples transport to different intermediate densities, so practical finite-sample objectives also reweight space. The cocycle property is already built into composition; no cycle loss or independent time-pair transform was added.

Long residuals can cancel opposite adjacent errors. We therefore used, in both alternating blocks,

`L_A = .5 L_adj + .5 K E_balanced[(r_{i,j}/(j-i))^2]`.

Separation d is uniform over 2,...,K, then start is uniform over valid starts. This avoids pair-count bias toward d=2 and preserves the old sum scale for coherent equal per-interval residuals. It measures squared error per knot increment, not Brownian error growth. Keeping adjacent loss prevents total blindness to cancellation, but halves its weight and does not guarantee it cannot worsen. Evaluation enumerates all six nonadjacent pairs with the same separation-balanced weights. Endpoint updates detach the entire composed geometry; geometry updates retain original-image anchoring and projection.

This is one justified mixture, not a claim that all temporal normalizations are equivalent.

## B: velocity deduction and selected residual

For interval k of width 1/K, interpolate `F_tau(x)=x+tau*u_k(x)`. Its Eulerian velocity is

`v(y,t)=K*u_k(F_tau^-1(y))`.

This is consistent with the forward composition: previous interval maps cancel in the Eulerian inverse expression. Since `||Du|| <= .25`, inversion is a contraction with factor tau*.25, and

`Dv = K Du (Id+tau Du)^-1`.

At the 16 recovery midpoints, tau is .125, .375, .625 or .875, and the spatial velocity-Jacobian bound is at most **1.28**, unlike the large derivative-derived local velocities in the failed particle experiment. Twelve detached contractions and a native-voxel inverse-defect check were used. Measured maximum defects are below **1.72e-5 voxel**. No gradient passes through inversion, geometry optimization or PCG.

Naively using `K*u_k(y)` has a pointwise relative-error bound tau*.25 in exact arithmetic; it is not the same velocity. Measured mean relative L2 discrepancy is about **1.30%**, averaged across the 16 times, so the shortcut was not used. Auxiliary velocity can jump at operator knots; the composed map remains continuous in time. Queries avoid knots, and no temporal-derivative penalty or interpolation of a velocity across the jumps was introduced.

**Selected strong formulation.** Define L as the repository's positive native graph stencil, with only valid one-voxel neighbours:

`R = g(q+g dot v) + s L v + eta v`,

`D_c = g_c^2 + s*degree + eta`,

`L_B = E_{x,t} sum_c R_c^2 / D_c`.

Geometry updates stay exactly adjacent/control-only and do not optimize stationarity; only the SIREN block receives the added condition. Weight is **1**, fixed before labels. R and D both differentiate through the current SIREN g/q. Velocities and their stencil values are detached. D matches the actual scalar PCG preconditioner; no arbitrary relative-RHS denominator or fitted surrogate is introduced. The retained operator gradient keeps its .1 baseline-norm budget, and B has its own .1 budget against the **original** reconstruction+PD-LK gradient. Thus total auxiliary norm can reach .2 baseline norm; the original control budget is not silently reduced.

At native voxel centres, this residual reproduces the actual graph operator, including faces, edges and corners. **Training samples continuous positions and is a shifted-lattice surrogate**, not exact optimization of the one fixed native grid. No continuous B-spline Laplacian is claimed to equal the discrete stencil. Training uniformly samples one of the 16 recovery times per step. Final held-out evaluation averages all 16.

**Why not the weak form first?** Integration by parts needs consistent boundary terms; an unconstrained spline velocity does not automatically satisfy continuous natural Neumann conditions. The exact discrete weak form uses unordered graph-edge differences. A finite test basis only constrains a projection, with blind residual directions; its Gram/energy normalization must also control basis support and scale. Squaring a Monte Carlo estimate adds variance bias unless separately controlled. The pointwise graph residual avoids those additional choices and gives a nonnegative sample estimator. No weak-loss search was run.

**Conditioning qualification.** `R^T D^-1 R` is an inexpensive energy-error surrogate, not `R^T A^-1 R`. Even the unweighted norm only bounds error using the small ridge lower bound on A. A block Jacobi metric would capture within-voxel cross-component coupling but still miss spatial modes. We selected the existing scalar diagonal and directly measured the relationship to recovered motion instead of assuming it.

## Empirical endpoint and unchanged-recovery results

| Condition | Endpoint PSNR moving / fixed, dB | Dice | HD95 mm | Folding | Converged intervals |
|---|---|---:|---:|---:|---:|
| PD-LK checkpoint | 32.2247 / 32.5119 | .698229 | 6.56515 | 0 | 15/16 |
| Adjacent control | 32.1715 / 32.4493 | **.700765** | 6.52270 | 0 | 15/16 |
| A: nonlocal | 32.1668 / 32.4593 | .700642 | **6.51420** | 0 | 16/16 |
| B: recoverability | 32.0722 / 32.4056 | .699961 | 6.56553 | 0 | 16/16 |

Full-volume endpoint MSE (moving/fixed): PD-LK .00059915/.00056080; adjacent .00060653/.00056895; A .00060719/.00056763; B .00062056/.00057470. All passed the unchanged 25 dB gate. Lowest sampled active PSNR remained above 30 dB in both extensions. All 56 union labels were evaluated using MONAI 1.6.1 macro Dice and maximum directed HD95 in physical millimetres; no missing label or omitted case.

PD-LK and adjacent control each narrowly miss tolerance on one interval (maximum residual **2.01950e-5 / 2.00814e-5**, tolerance 2e-5). A/B maxima are **1.99080e-5 / 1.95556e-5**. Improved numerical convergence does not establish improved anatomy.

## Held-out losses and cost

Temporal losses use the same 2,048 continuous sites. Continuous recoverability uses 256 shared sites and all 16 times; the native diagnostic uses 512 shared uniformly sampled voxel centres and all times. These are distinct sampled probes, not full-grid certificates.

| Condition | Adjacent | Nonlocal | Continuous recoverability | Native recoverability | Auxiliary original-data MSE | Training seconds | Active ms/step |
|---|---:|---:|---:|---:|---:|---:|---:|
| PD-LK† | .0032744 | .0022552 | .0044231 | .0029628 | .0077401 | 43.78‡ | 6.96‡ |
| Adjacent | .0024719 | .0020131 | **.0028391** | .0018925 | .0077401 | 93.29 | 25.62 |
| A | .0025929 | .0020217 | .0033455 | .0020434 | .0077643 | 110.63 | 32.11 |
| B | **.0023535** | **.0019265** | .0030833 | **.0017176** | **.0077098** | 116.02 | 33.09 |

† PD-LK has no auxiliary geometry; these diagnostics use the common adjacent-control geometry, not separately optimized PD-LK maps. They are contextual probes. ‡ Retained original fit timing; PD-LK was not retrained. Active timing excludes the first activation block and includes logging/guards; observed costs are not a separate speed benchmark. The smallest nonlocal value in this table belongs to B; A did not improve its own held-out nonlocal target over the adjacent control.

## Measured auxiliary versus PCG motion

During each unmodified recovery solve, velocity was sampled before disposal. No auxiliary dense field or velocity was saved. Alignment is averaged over the 16 times at the same 512 integer sites. Energy error is the Monte Carlo per-voxel `e^T A e`, using brightness and ridge node terms plus half-weighted directed edge differences; it requires no extra solve.

| Condition | Velocity RMSE, native voxels/time | Relative L2 | Cosine | Energy error/voxel |
|---|---:|---:|---:|---:|
| PD-LK† | 3.55958 | .97996 | .72490 | .0120866 |
| Adjacent | 3.41400 | .90770 | .75080 | .0081413 |
| A | 3.48698 | .93933 | .74250 | .0086445 |
| B | **3.39262** | **.90036** | **.75506** | **.0074228** |

B improves sampled native energy error by about **8.8%** and native preconditioned residual by **9.2%**, but continuous held-out residual worsens **8.6%**. Different point sets and the shifted-grid/native-grid distinction limit interpretation; neither probe alone establishes full-volume stationarity. More importantly, better sampled native alignment does **not** translate to better Dice/HD95 here. Uniform global motion energy is not an anatomical correspondence criterion.

## Gradients, validation and decisions

All active updates passed finite-gradient checks. Across 150 logged active samples, nominal weighted operator gradients are .0143–.0239 (control), .0111–.0215 (A), and .0138–.0227 (B). The added B gradient is much larger and variable: **.1556–6.2844**, versus baseline .0420–.0987. Budget multipliers are **.000708–.03431**, with applied B norm **.00420–.00987**. Thus B's stability depends on the separate budget; finite raw gradients alone would not establish safety. These ranges are logged samples, not claimed per-update maxima. Every active update enforces the bound. Increasing a nominal coefficient is not justified by this heavily capped result.

Synthetic validation covers transported telescoping/cancellation and duration scaling; exact inverse-based affine velocity; graph stencil equality at all synthetic faces/edges/corners; finite-difference parameter gradients into both endpoints with geometry frozen; independent RNG and mutually exclusive config; and separate gradient budgets. Both README suites pass: **47 medical + 1 Cameraman tests**. Full-width CUDA recoverability gradients agree with eager execution (relative difference 1.78e-7, cosine 1). Wheel/import/CLI checks exclude tests, archives, data and results, and launch no optimization.

**A decision:** do not promote this mixture. Relative to adjacent control, Dice is down .000124, HD95 improves only .00851 mm, and both adjacent and nonlocal losses are slightly worse. One pair does not establish any anatomical advantage or rule out other justified nonlocal formulations.

**B decision:** reject this fixed loss as an anatomical improvement over the adjacent control. Native motion alignment improves slightly, while Dice drops .000805 and HD95 worsens .04283 mm. The continuous/native residual disagreement also prevents claiming general solver-stationarity improvement. Possible roles of sampling, background dominance, auxiliary-geometry ambiguity and conditioning remain hypotheses, not diagnosed causes. No further coefficient or cohort run follows from these results.

The opt-in branches remain available as reproducible research implementations; neither changes production defaults or becomes a recommended protocol. The adjacent-only prototype and PD-LK controls remain intact.

## Reproducibility and preservation

[Reviewed scalar evidence](path_consistency_20261110.json) accompanies the report. Ignored local root `results/path_consistency/20261110T105356_113906Z` contains the frozen inputs, task, source snapshot/hashes, audited runner, three final endpoint checkpoints, scalar logs and per-time alignment diagnostics. Source stayed frozen during the study. No geometry coefficients, dense maps, warped volumes or persistent velocities were written; temporary recovery scratch was cleaned. Earlier experiments, medical data and the earlier benchmark were hash-checked unchanged. No commit or push was made.

Final preservation audit: 2,343 dataset files (3,405,406,449 bytes) and 1,312 earlier result files remain unchanged, including the 830-file prior benchmark. A post-study test assertion also explicitly checks that both new losses leave auxiliary coefficients unused during endpoint differentiation; the maintained research code still matches the frozen study snapshot. Current documentation links and `git diff --check` pass.
