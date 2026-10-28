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
