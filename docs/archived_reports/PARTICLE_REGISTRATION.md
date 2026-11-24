# Experimental particle registration objective

The grid estimator pools derivative observations before solving for motion, then
smooths and interpolates a velocity lattice. This alternative estimates motion
at the particles themselves and couples those estimates with a continuous
Gaussian. Its second derivative evaluation follows the particles to a randomly
chosen intermediate time. This tests whether a velocity estimate that follows
the moving samples can preserve local motion that a fixed lattice averages away.

The hypothesis has a qualification: solving each point before averaging can
amplify poorly constrained estimates. Post-solve blurring cannot supply missing
constraints to an individual solve. The ridge remains active. Compare local
velocity disagreement and known synthetic transport errors as well as endpoint
NCC; lower NCC alone would not show that the motion estimate improved.

## Selected particle setup

Use [particle_velocity.json](../medical/configs/ablations/particle_velocity.json):
independent moving-source and fixed-source starts in one shared crop, 4096
particles per direction, both brightness/differential LK terms, and averaged
LK/NCC over both directions. Gaussian sigma decreases linearly from **0.4 to 0.1
of each crop side**. The selected execution uses fused Adam and
`particle_velocity.capture_objective: true` for CUDA objective/backward replay.
Sampling and the optimizer stay outside the graph, with a finite-gradient check
before every update. Cycle options are omitted entirely, so no round-trip
trajectory or transferred return starts are constructed. Independent source
crops remain an opt-in ablation. This selection applies to the experimental
particle objective; the repository default remains the shared-grid method.

The single-seed far-phantom sampling comparison gave forward/reverse EPE
2.905/2.625 vox and Dice 0.97291 for this setup. Keeping transferred starts with
zero cycle weight gave 2.938/2.661 vox; independently placed source crops gave
3.008/2.716 vox. These results motivate the simpler shared-crop sampling choice,
without establishing a general accuracy advantage.
[Comparison](../results/particle_sampling_ablations_20261115T084959Z/analysis.json).

## Point velocity and LK residual

At each current particle position and time, evaluate the interpolated SIREN's
`g = ∇I`, `q = ∂t I`, `H = ∇²I` and `m = ∇q`. With brightness weight `b`,
differential weight `c` and positive ridge `η`, solve independently at every point:

```text
A_i = b g_i g_iᵀ + c H_iᵀ H_i + η I
u_i = −A_i⁻¹ (b g_i q_i + c H_iᵀ m_i)
```

Both terms are on initially: `b = 1`, `c = 4`. Set
`objective.registration.particle_velocity.brightness_weight` to zero for the
differential-only ablation. Set `objective.c` to zero for brightness-only LK;
this also skips Hessian/mixed-derivative evaluation and uses the exact rank-one
ridge solution without a 3×3 LU solve. Either outer loss weight
can still be zero. Both inner weights can be zero, yielding zero velocity and
zero LK residual; this intentionally removes learned transport from that branch.

Blur the solved velocities first. The LK penalty assesses the blurred velocity
used to move the particles:

```text
E_i = b (q_i + g_i · v_i)² + c ‖m_i + H_i v_i‖² + η ‖v_i‖²
```

This makes the residual sensitive to the coherence imposed by the blur. It is
not the minimized energy of the independent raw point solves. Neither term adds
spatial derivatives of the velocity.

## Continuous Gaussian in mm

For XYZ voxel positions `x_i` and XYZ voxel spacing `s`, use

```text
w_ij = exp(−‖s ⊙ (x_i − x_j)‖² / (2 σ_mm²))
v_i  = Σ_j w_ij u_j / Σ_j w_ij
```

Every sampled particle in the same direction contributes, including the particle
itself. The kernel has no binning, nearest-neighbour assignment or hard support
cutoff. A constant velocity remains constant, and anisotropic voxel spacing is
respected. Kernel positions are the current particle positions, so weights are
recomputed after the first Euler step. Directions use independently sampled
point sets and are blurred separately.

Sigma describes **physical distance**, independently of whether the LK solve
uses the default voxel derivatives or optional native-mm derivatives. In mm
mode, blur velocities in mm/time and convert them to voxels/time for transport.
The existing physical geometry and frozen-recovery rules still apply.

CUDA FP32 uses an exact tiled Gaussian with one forward and one backward kernel.
It retains `O(N)` state and recomputes pair weights inside the backward kernel,
including both query and source coordinate contributions. No `N × N` matrix is
saved. CPU/FP64 use the chunked differentiable reference; higher derivative and
learnable-geometry requests use a live torch fallback. Arithmetic remains `O(N²)`;
this is not a sparse approximation. [Timing and attention equivalence](PARTICLE_PERFORMANCE.md)

## Two Euler steps

Draw one shared split `τ ~ Uniform[0,1)` per update. For each direction, starting
at endpoint time `t_a` and ending at `t_b`:

1. Evaluate derivatives and solve point velocities at `(x_a, t_a)`, then blur.
2. Move to `x_τ = x_a + (τ − t_a) v_a`.
3. Reevaluate derivatives at `(x_τ, τ)`, solve again, and blur at the new positions.
4. Move to `x_b = x_τ + (t_b − τ) v_τ`.
5. Compare the observed image intensities at `x_a` and `x_b` using the separately
   selected registration NCC. Both box NCC and the existing GW-NCC are available.

The independent shared-crop implementation batches both middle-time derivative
evaluations, then splits the results before separate directional Gaussian
blurring. This reduces dispatches while preserving both coordinate gradients
and independent smoothing neighborhoods. See [performance evidence](PARTICLE_PERFORMANCE.md).

The reverse direction uses `t_a = 1`, `t_b = 0` with its own points. The directional
LK penalty is `|τ − t_a| mean(E_a) + |t_b − τ| mean(E_τ)`, and directions are averaged.
This is a time-length-weighted two-stage penalty, not unbiased temporal quadrature.
`random_timesteps: false` fixes the split at 0.5. These are Euler steps; no RK2
accuracy claim applies to this training alternative.

The crop and mask policies remain shared with endpoint reconstruction. Every particle is scored, including crop
exits. Image interpolation and derivative queries extend at the full-image
border, without clipping the transported coordinates. The final full-image
boundary penalty remains live. Gradients flow through the point solves, both
uses of kernel positions, the first arrival, the second derivative evaluation
and the endpoint NCC. No velocity lattice is evaluated along this path.

## Bandwidth configuration

The [selected preset](../medical/configs/ablations/particle_velocity.json) uses
4096 particles per direction, crop-relative sigma 0.4→0.1, reference blur chunks
of 256 and no derivative recomputation. `recompute: true` remains available for
memory-constrained execution. The validator rejects fewer than 4096 training
particles; smaller arrays are reserved for tiny synthetic checks.

```json
"particle_velocity": {
  "brightness_weight": 1.0,
  "sigma_box_fraction": 0.4,
  "final_sigma_box_fraction": 0.1,
  "blur_chunk": 256,
  "capture_objective": true
}
```

Sigma interpolates linearly from the initial value at step 1 to the final value
at the last step; a one-step run uses the initial value. Omitting the final value
keeps sigma constant. Absolute bandwidth remains available by replacing the two
fraction keys with `sigma_mm` and optional `final_sigma_mm`. The two unit systems
cannot be combined. Logs and checkpoint metadata record the chosen bandwidth,
blur backend and training integrator.

Broad early averaging may reduce disagreement between noisy point estimates;
reducing sigma later may retain more spatial variation. If narrowing only
increases disagreement or folding without better transport, that would favour
insufficient local constraints over excessive smoothing as the explanation.

This branch uses standard training execution. The separate complete captured
shared-grid trainer and its compiled decoder are rejected. The particle-specific
`capture_objective` option replays endpoint/registration loss, backward and
gradient clipping with fresh input buffers, while preserving ordinary crop/point/
time sampling and the guarded optimizer. It requires CUDA FP32, compiled NCC
endpoints/registration, shared fixed-size crops, ordinary parameters and no
recomputation. Disable it for cycle, independent-source-crop, tied-point or
structure/window ablations. When either effective outer loss weight is zero,
training falls back to the ordinary path and skips inactive losses; subsequent
active steps safely reuse the graph. Fused Adam is selected with
`training.optimizer: {"name": "adam", "fused": true}`. Omitting `fused` preserves
the existing optimizer behavior. [Execution evidence](PARTICLE_PERFORMANCE.md). The shared schema's grid size, probe-pair
geometry, cell sigma and grid kernel do not set particle velocities or their
blur; the existing GW-NCC settings still control the similarity windows when
selected. Frozen inference remains the recorded PCG plus midpoint RK2 pipeline.
The default preset is unchanged. Implementing this alternative does not launch
training, benchmarks or saved-model recovery.

## Synthetic validation

`medical.tests.test_particle_registration` checks the point solve against
augmented least squares, independent term ablations, physical Gaussian values,
constant fields, permutations, first/second gradients of the blur, arrived-point
reevaluation, both time directions, schedule endpoints, full loss gradients,
recomputation, boundary penalties and configuration/dispatch guards. Its dispatch
check stops before the first endpoint loss. An opt-in CUDA test checks eager
versus compiled values and parameter gradients for all three term combinations.
All checks use synthetic tensors and perform no optimizer steps.

## Optional same-particle cycle

`particle_velocity.cycle_weight` selects a different sampling scheme: a fair
Bernoulli draw chooses fixed or moving as the starting endpoint on each update.
One set of at least 4096 particles travels to the other endpoint, then returns
from those exact arrived positions, without resampling or detaching. Both legs
reevaluate the velocity estimator and its Gaussian at their current positions.
They share the stochastic split time. This costs four velocity evaluations,
the same count as the ordinary independent bidirectional branch.

Registration NCC and the LK regularization loss use the outward leg only.
The return velocities still use the same brightness/differential LK solve, with
live gradients, but no additional return-leg LK residual is added. Boundary
loss averages the two final endpoint violations. Add
`cycle_weight * mean(||spacing * (returned - starts)||²)` to the registration
objective. Distance is Euclidean in mm, squared, and the existing outer
registration weight multiplies this term too. The initial cycle preset uses
`cycle_weight: 0.01`, so its effective coefficient is 0.0001 with outer weight
0.01. Setting cycle_weight to zero keeps the same loop sampling as a control;
omitting it selects ordinary independent bidirectional sampling.

Cycle coupling asks matching particles to recover their identity after the
round trip. Image NCC remains necessary because identity motion can have perfect
cycle consistency. With two Euler steps per leg, the penalty also measures
numerical integration error; a lower cycle loss alone does not establish better
anatomical alignment or invertibility of the frozen PCG maps. The synthetic
comparison reports recovered errors and Dice separately. Endpoint reconstruction
and crop-center mask policies are unchanged. Logs include raw `cycle_mm2` and
`cycle_starts_fixed`; checkpoints record the sampling convention. See
`medical/configs/ablations/particle_cycle.json`.

The first requested far-phantom cycle trial completed at width256, seed42,
5000steps,4096 loop particles and sigma7.5mm. The matched loop control
(cycle_weight0) reached forward/reverse EPE16.711/15.718vox and Dice0.59119;
weight0.01 reached18.392/17.077vox and Dice0.57692. Whole-grid folding was
approximately0.85–1.04%, and all recovery intervals converged. On the same eight
frozen crop/RNG draws, RMS training-loop return distance was7.93mm for control
and8.39mm with penalty. Both loop fits had extremely large pre-clipping gradient
norms, including the zero-penalty control. This indicates that the changed loop
training scheme itself needs an instability diagnosis before interpreting or
tuning the cycle penalty; it is not evidence against cycle consistency in
general. The earlier independent bidirectional fixed-sigma result was
4.033/3.644vox and Dice0.96893. The cycle preset remains an opt-in negative
prototype. [Trial evidence](../results/particle_cycle_20261115T074947Z/analysis.json).

### Huber option and return-LK diagnosis

The non-detached cycle control disables the return-leg LK loss. The explicitly
requested detached transferred-start variant below restores it. Its
original zero-cycle-weight control was already unstable because return LK
backpropagated through the outward arrival positions. At step300, an independent
live-graph diagnostic found weighted return-LK parameter gradient norm36688,
with almost all of it from this coordinate chain; the direct parameter part was
about47. Outward LK remained about167 and endpoint reconstruction about12.5.
The outward-only zero-penalty ablation recovered stability and reached
5.200/4.735vox EPE and Dice0.96033 on the same far phantom. This does not remove
return-motion derivatives from the cycle or boundary losses.

Optional `cycle_loss: "huber"` with `cycle_delta_mm: 1.0` applies radial Huber
to the Euclidean physical return error r. It is r² for r≤delta and
2*delta*r−delta² otherwise, matching the previous L2 coefficient near zero.
It is rotationally invariant and caps the residual-space gradient at2*delta;
it does not cap parameter gradients through the trajectory Jacobian. Raw
`cycle_mm2` stays available for comparison, and `cycle_penalty` records the
selected robust/L2 objective. Omitting cycle_loss preserves L2. The Huber preset
is `medical/configs/ablations/particle_cycle_huber.json`.

Two-Euler transport also biases loop closure: exact spatially constant
v(t)=a+bt gives returned−start=−b*(tau²+(1−tau)²), despite an invertible
continuous flow. A unit check verifies this. Robustifying the cycle distance
therefore cannot by itself distinguish integration error from inconsistent
motion. [Gradient diagnosis](../results/cycle_gradient_probe_20261115T075713Z/summary.json).

The focused Huber diagnostic fits retained both brightness and differential LK
on the outward leg and outward NCC. With return LK removed, the zero-cycle
control reached EPE5.200/4.735vox and Dice0.96033. At cycle_weight0.01, L2 reached
16.785/15.748vox and Dice0.59403; radial Huber delta1mm reached17.499/16.409vox
and Dice0.57510. Huber with the original two-leg LK failed the finite-gradient
check at4524. Thus Huber reduced the residual gradient but did not fix the
remaining long-chain cycle-gradient instability at this coefficient. The user
selected the simple outward-only LK patch; return motion and all cycle/boundary
gradients remain live, and the default cycle distance remains L2 unless Huber
is explicitly selected. [Controlled fit evidence](../results/cycle_huber_diagnosis_20261115T075911Z/analysis.json).

## Detached transferred starts: explicit user-requested variant

The optional `particle_velocity.detach_return_starts: true` variant requires
an explicit `cycle_weight` and restores both LK terms and registration NCC on
both trips while using outward landing positions to sample the return trip:

```text
arrived = outward(starts)          # outward LK and NCC retain gradients
return_starts = arrived.detach()   # the only new gradient boundary
returned = reverse(return_starts)  # fresh live velocity solves and smoothing
LK = mean(outward_LK, return_LK)
NCC = mean(outward_NCC, return_NCC)
cycle = mean(||spacing * (returned - starts)||²)
```

Both directions share the sampled intermediate time, and a fair50/50 endpoint
draw chooses which is outward each update. The two directional image losses
use observed image values at their own starts and arrivals. Return boundary and
cycle losses remain differentiated through the return velocity solves, internal
Euler stages and smoothing, but do not flow through the detached arrival into
the outward trajectory. Shared network parameters are still trained by both
legs. Endpoint reconstruction is unchanged. The former zero-cycle transferred
preset was removed when independent bidirectional shared-crop sampling was
selected. This variant remains available only through explicit cycle options.

This detach is an explicit exception requested by the user; it is not applied
to standard independent bidirectional training or the historical live cycle
control. Gradient-boundary tests verify unchanged point values, live outward
NCC, live return LK/NCC/cycle parameter gradients, and the missing cross-leg
coordinate gradient. Frozen PCG/RK2 recovery is unchanged.

The detached-return sweep completed all three5000-step far-phantom fits with
seed42,4096particles,sigma7.5mm, both-leg LK and NCC. Weights0/0.001/0.01 gave
forward/reverse EPE3.895/3.579,3.761/3.448,4.757/4.371vox and Dice
0.97004,0.97035,0.96379. All recovered maps had zero sampled-map folding and
all recovery intervals converged. Maximum logged pre-clipping gradient norms
were166,37.5,259, respectively. Matched eight-crop frozen training-loop closure
RMS was3.767,3.614,3.468mm. Increasing weight improved closure here, but0.01
worsened recovered alignment. These are single-case single-seed development
results, not evidence of a universal best weight.

GPU3 reproduced the previous outward-LK-only live-cycle failure atweight0.001.
Atstep1100, weighted cycle parameter-gradient norm was about34624; the outward
arrival-coordinate chain contributed34624, versus7.93 from direct return-model
dependence. Earlier step700 showed the same separation33.15versus0.80, before
large eager/compiled differences emerged. No particles left the full image.
This explains why the detach addresses the observed amplification without
removing coordinate gradients inside either trip. Later exact norm values are
approximate because ill-conditioning increases eager/compiled discrepancies.
[Detached trial evidence](../results/detached_return_sweep_20261115T081538Z/analysis.json)
· [Previous-variant gradient attribution](../results/previous_cycle_gradient_20261115T081236Z/summary.json).

The requested detached-return Huber sweep used delta1mm and weights0.001,
0.01,0.1, with the same seed42/5000steps/4096particles/sigma7.5mm settings.
Forward/reverse EPE was3.869/3.515,3.844/3.527,12.060/10.954vox; Dice was
0.97045,0.96941,0.74496. All fits stayed finite and sampled-map folding was
zero. Low-weight recoveries fully converged; weight0.1 had nonconverged PCG
intervals, so its map scores carry that qualification. Paired frozen-loop RMS
was3.900,3.446,5.096mm. The small Huber weights were similar, with no EPE gain
over the earlier L2 weight0.001 result3.761/3.448vox. These remain single-case,
single-seed development results. [Huber trial evidence](../results/detached_huber_sweep_20261115T082235Z/analysis.json).

Crop-relative particle bandwidth uses `sigma_box_fraction` instead of `sigma_mm`,
with optional `final_sigma_box_fraction` for a linear schedule. These specify
Gaussian sigma, not FWHM. Per XYZ axis, sigma is the fraction times the crop's
voxel-center span `(crop_shape_xyz - 1) * spacing_xyz`; rectangular crops therefore
have an ellipsoidal physical Gaussian. The kernel evaluates distances normalized
by each crop axis and reuses the fused CUDA implementation. Crop translations
do not change the bandwidth; crop resizing updates it. Both units cannot be
specified together, and initial/final units must match. Logs include
`velocity_sigma_box_fraction` and `velocity_sigma_mm` (the geometric mean of
the physical per-axis sigmas; equal to each axis on a physical cube).
For the 64³, 1 mm synthetic crop, fractions 0.1, 0.2, 0.4 and 0.6 correspond
to sigmas 6.3, 12.6, 25.2 and 37.8 mm.

For sampling ablations, retaining `cycle_weight: 0` and
`detach_return_starts: true` preserves the transferred-start protocol (a fair
50/50 starting endpoint, then return from detached outward arrivals), with LK
and NCC averaged over both trips. Removing all cycle options instead selects
ordinary bidirectional sampling: independent moving-source and fixed-source
particle sets, with LK and NCC averaged over both directions. The two directions
share the stochastic split time and crop by default. Both keep all within-trip
gradients live.

In ordinary bidirectional mode, `particle_velocity.independent_source_crops: true`
selects independent, equal-size moving and fixed source boxes. Each endpoint's
reconstruction samples its own source box, and the corresponding registration
particles sample the central fraction of that same box. Each direction's crop
origin is also used for any crop-relative registration statistics/diagnostics.
Arrivals use the full opposite image with border extension; they are not gated
to either crop. Relative sigma uses the common crop size, so changing crop
location does not change bandwidth. A dedicated fixed-source crop RNG preserves
the moving-source crop stream. This option requires cropped MSE/NCC endpoint
fitting, separate particles and no structure samplers, and cannot be combined
with any cycle options. Defaults and the retained shared-grid method are unchanged.
