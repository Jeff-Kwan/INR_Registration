# Research evolution

The project moved from endpoint image fitting toward a continuous trajectory
whose intermediate states can support registration. The practical loop is quick
2D experimentation on synthetic Cameraman deformations, followed by controlled
3D medical-image evaluation.

## Endpoint models need a trajectory principle

A pair of endpoint INRs can reconstruct the moving and fixed images while their
linear parameter interpolation produces poor intermediate images. This made
trajectory regularization the central question. Shared initialization became
the default convention because independent initialization performed worse in a
limited matched Cameraman comparison; that evidence is one seed and does not
establish a general result.

## From patch transport to differential transport

Early patch objectives compared image neighborhoods at two times while allowing
local motion. This exposed a tradeoff: patch matching imposes neighborhood
agreement but introduces sampling, boundary and scale choices. Differential
Lucas–Kanade instead differentiates brightness transport in space and uses the
image Hessian and mixed space-time derivative. For a local constant velocity,
this yields a small least-squares problem at each sampled point. The local
velocity is eliminated from the training penalty; the final spatially coupled
map is still recovered by a global optical-flow solver.

The patch formulation remains a historical reference, not a maintained LPBA
pipeline. The concise record and sweep evidence are in
[`historical_methods/LPBA_H1_SWEEP.md`](historical_methods/LPBA_H1_SWEEP.md).

## Relaxation as a local affine model

Standard differential LK assumes velocity is locally constant. Allowing an
affine velocity and penalizing its Jacobian leads, after elimination, to an
adaptive coefficient on the gradient-transport residual. The ratio
coordinate `rho=c/beta` makes the standard objective explicit at `rho=0` and
controls attenuation for positive values. With `c=4`, the earlier `beta=10`
setting corresponds to `rho=0.4`.

The current implementation applies this independently at each training query;
it does not couple neighboring local velocities. This is a methodological
extension of PD-LK, not a spatial regularizer. The current priority is to derive
its scaling and conditioning properties and establish when this local model
improves a trajectory.

## Keep learning and recovery distinct

The synthetic Cameraman setup has a known deformation and oracle frames. It can
separate endpoint reconstruction, intermediate trajectory quality, recovered
flow and final-map error. An oracle-frame flow remains subject to texture
ambiguity; the true deformation does not. In LPBA, labels support Dice and HD95
evaluation but no true deformation field is available. Endpoint quality, folding
and flow-solver convergence must accompany those label scores.

## Current evidence is exploratory

The completed four-hour LPBA sweep found near ties between the matched
seed-42 `rho=0` and `rho=0.4` candidates, and the seed-44 comparison slightly
favored `rho=0`. Its highest individual score came from a seed-43 `rho=0` run.
Several full-cohort subjects had unconverged flow intervals. Those observations
do not show that relaxation improves registration, and they do not identify a
stable best value. The detailed values and gates are recorded in
[`BEST_RESULTS.md`](BEST_RESULTS.md); full artifacts remain ignored local files.

The direct time-coordinate SIREN, H1 patch transport, metamorphosis and alternate
architectures remain historical ideas. Small objective references and research
notes retain their formulation. The maintained method path is paired endpoint
SIRENs, standard/relaxed pointwise PD-LK, analytical image derivatives and
separate global flow recovery.

## Retired spatial-coupling experiments

We tested whether nearby locations should admit a coherent explanation of
the SIREN trajectory, using shared velocities for jittered query pairs and
Gaussian-process priors. Exact GP variants compared random queries,
sectioned sampling and translated grids. RFF and fixed inducing-point FITC
approximations allowed denser queries with smaller global systems.

The single-seed results were mixed and showed no significant, consistent
improvement over standard/relaxed pointwise PD-LK. Some variants improved
Dice while worsening deformation EPE. Practical exact-GP query sets were
too sparse for the intended 3D setting and the dense solve was expensive;
low-rank approximations traded covariance fidelity for speed, especially at
short correlation lengths. A full-image exact GP run was not completed,
and these methods were never validated in 3D.

We discontinued this direction and removed its code, configs and tests.
The maintained method remains pointwise standard/relaxed PD-LK. Retain the
[comparison tables](CAMERAMAN_COMPARISON.md) and
[historical equations](historical_methods/SPATIAL_COUPLING.md) as research
evidence, without interpreting the decision as a general negative result
for spatial priors.

## Contour motion without a fitted field

The next 2D pilot treats intensity levels as moving contours and penalizes
normal speed through the coarea-motivated energy mean(I_t²/h). An optional
term smooths normal speed tangentially to contours, with small-gradient
downweighting. This removes auxiliary velocity elimination from training.
It does not remove brightness ambiguity, constrain tangential motion, or
guarantee a diffeomorphic recovered map. Downstream optical-flow recovery
remains a common evaluator. Epsilon, tangential beta and outer weight need
controlled selection before any 3D transfer or method claim.

## Coherent material trajectories

The material-deformation experiment introduces one shared space-time map
Phi(X,t)=X+t*d(X,t), trained jointly with the image endpoints. Its transport
residual follows moving material locations; kinetic and small strain penalties
balance motion and distortion. This is an explicit coherent trajectory model,
with no independently eliminated local velocities. It adds model capacity and
optimization coupling, so comparison must retain the same image architecture,
reconstruction and downstream recovery. Strain does not guarantee a valid
invertible map; outside-domain and Jacobian diagnostics accompany the pilot.

The material-map pilot was subsequently discontinued and its objective code,
network, configs and tests removed. [Results](MATERIAL_PILOT.md) remain as
historical evidence; the maintained fitter supports PD-LK and level-set energy.

## Supervising the endpoint derivatives

The next experiment changes the endpoint reconstruction objective rather
than the trajectory prior. Intensity MSE anchors values and additive bias;
first-order supervision aligns analytical SIREN gradients with signed Sobel
estimates from the input images. The hypothesis is that more faithful
endpoint derivatives may improve the interpolated trajectory and downstream
registration. This is tested with the existing relaxed PD-LK prior fixed.
Improved gradient reconstruction is a supervised result; it must be assessed
separately from true-map EPE, Dice and intermediate-oracle diagnostics.

The [first pilot](SOBEL_RECONSTRUCTION.md) reduced endpoint Sobel-target
gradient RMSE substantially, but not deformation EPE. This separates improved
endpoint derivative reconstruction from trajectory/recovery quality; no
combined-loss setting is promoted on the single-seed evidence.

## Local velocity is not the learned deformation

[Frozen-model diagnostics](VELOCITY_STRUCTURE.md) find similar rough but
nonwhite local PD-LK velocities in one 2D and one 3D pair. Spatial recovery
provides much greater coherence through its prior. Gradient-supervised camera
endpoints yield smoother intermediate g/q but no better implied velocities.
The distinction between a useful local temporal penalty and a faithful
material explanation remains important. Close-frame oracle comparisons must
respect the generating geometry: smoothing an interpolated map can create
folds even when its sampled mesh passed discrete checks.

### Analytical derivative of the local optimum

Following the frozen-field diagnostic, a standard rho=0 Cameraman pilot trained
on the complete brightness-residual gradient, differentiating through the old
pointwise optimum and its spatial Jacobian. At the existing lambda=.1,c=4,
eta=1e-6, it lost endpoint reconstruction (PSNR7.62/7.74dB versus28.23/30.58
for a fresh matched standard control), with large repeatedly clipped gradients.
This rejects that unchanged-strength surrogate pilot, not the corrected
field-coupled variational formulation. See
[the analysis and training evidence](ANALYTICAL_VELOCITY_JACOBIAN.md).

The full256×256 grid follow-up replaces the analytical velocity Jacobian with
unit-pixel finite differences. It avoids catastrophic training collapse and
passes reconstruction (25.29/26.74dB), but gives Dice.9321/EPE4.530px versus
.9450/4.351 for standard PD-LK, at10.2× training cost. Dense sampling and the
coarser derivative scale both change, so the stability difference is not isolated.
This remains a surrogate at independent local optima; no default is promoted.
