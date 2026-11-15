# Single Euler step with a 32³ Gaussian local-velocity lattice

**Decision: keep on hold.** Single-step Euler achieves the intended cost reduction,
but gives substantially worse endpoint maps and folding on the existing trained
LPBA S02 checkpoints. No training or optimizer updates were performed.

## Cost

Same RTX 5090, MSE+PD-LK and NCC+fixed moments (.01)+PD-LK checkpoints as the
previous profiles. 32³ lattice, 4 mm physical Gaussian, 256 training-loss particles,
4096-point derivative chunks, compiled local solve, activation recomputation and
separate endpoint gradient budgets. Euler uses one t=0 velocity lattice, versus
eight time queries for four-step RK2. Both endpoints remain differentiable through
q and m, although the integration samples only the initial time.

Median of five warmed passes, with the existing endpoint/PD-LK backward and clipping:

| Model | Euler full forward/backward | Registration portion | Earlier 32³ RK2 full pass |
|---|---:|---:|---:|
| MSE | 48.5 ms | 41.3 ms | 321.5 ms |
| NCC | 49.1 ms | 41.3 ms | 322.3 ms |

No Adam update or periodic reconstruction monitor is included. Initial compiled
Euler registration pass took 4.95 s; warmed no-grad evaluation of 65,536 particles
took about 9.8–9.9 ms versus 78.0 ms for four-step Gaussian-RK2. The first dense
Euler map took 3.70 s including no-grad compilation; NCC reused that graph and took
10.8 ms. Dense Gaussian-RK2 maps took about 80 ms. These are distinct forward-only
and forward/backward measurements. Standard PCG recovery (forward and inverse)
took 19.45/15.17 s and is not timed as a like-for-like training step.

## Direct deformation evaluation

All maps were estimated directly from the immutable trained endpoints; no fitting
of an independent deformation and no reconstructed-intensity registration.
Euler is phi(a)=a+G_sigma[v_local(a,0)] sampled from the 32³ lattice. Four-step
Gaussian-RK2 uses the same lattice, filter and local-velocity formulation. PCG+RK2
uses the unchanged maintained recovery with 16 midpoint intervals, four RK2
substeps per interval and saved checkpoint settings.

**Evaluation direction:** compare original moving M(a) with original fixed
F(phi(a)); sample fixed labels onto the moving grid with each forward map. This
avoids assuming that negating an Euler displacement gives its inverse. These HD95
and Dice values are not the standard earlier fixed-grid inverse-map evaluation.
All three methods here use the same direction and existing metric implementation.
Identity Dice=.563048, HD95=7.837058 mm. Macro labels=56, no missing labels.

| Model | Map | Dice | HD95 mm | Folded voxels / 4,915,200 | Folding % |
|---|---|---:|---:|---:|---:|
| MSE | PCG+RK2 | .698464 | 7.1934 | 0 | 0 |
| MSE | Gaussian four-step RK2, 32³ | .680858 | 7.0372 | 6 | .000122 |
| MSE | Gaussian one-step Euler, 32³ | .632834 | 7.5044 | 41,249 | .839213 |
| NCC | PCG+RK2 | .696908 | 7.0489 | 95 | .001933 |
| NCC | Gaussian four-step RK2, 32³ | .676926 | 7.1475 | 1,388 | .028239 |
| NCC | Gaussian one-step Euler, 32³ | .617781 | 8.1273 | 394,597 | 8.028097 |

Folding uses det(Dphi)<=0 from native-grid finite differences, not an analytic
Jacobian proof. Euler sampling outside the image was 2.58%/4.06%, versus
.75%/1.07% for Gaussian-RK2 and .20%/.34% for PCG. Original images use linear border
sampling; labels use nearest sampling and zero outside. NCC Euler HD95 is worse
than identity despite higher Dice. MSE Gaussian-RK2 has a lower HD95 than PCG in
this direction, but lower Dice; do not infer superiority from that isolated metric.

## Endpoint differences

On the original foreground union, displacement errors against PCG's forward map:

| Model | Euler RMS error | Gaussian-RK2 RMS error | Euler displacement cosine vs PCG | RK2 cosine |
|---|---:|---:|---:|---:|
| MSE | 3.65 mm | 1.79 mm | .693 | .926 |
| NCC | 4.94 mm | 2.49 mm | .640 | .894 |

Euler versus Gaussian-RK2 endpoint RMS difference is 3.36/4.72 mm. Thus equal
spatial resolution with a single initial-time field loses substantial endpoint
agreement. These comparisons do not independently separate temporal quadrature
error, large-step spatial integration error and properties of the t=0 local field.
The previously identified coarse-grid approximation error remains present in both
Gaussian variants. This does not prove that an Euler-based regularizer cannot
help training, but it is not an accurate endpoint-registration surrogate here.

An independent 65,536 continuous-particle check versus interpolation of the dense
forward map had RMS differences .052/.085 mm for Euler and .021/.034 mm for RK2.
Composing/interpolating discrete maps need not equal direct continuous trajectory
evaluation; those discrepancies are much smaller than the measured map differences.
No inverse Euler map was assumed or used for the registration scores.

## Preservation and validation

Both trained parameter states were identical before/after profiling and recovery.
Checkpoint, input-image and annotation file hashes were unchanged. Only scalar
metrics and per-label tables were written. No deformation, velocity or warped
volume was persisted. Maintained PCG temporary velocity scratch was removed;
`results/.scratch` was empty afterward. MSE PCG converged at 15/16 intervals (the
previous slight tolerance miss); NCC converged at 16/16. All convergence diagnostics
are retained in the compact JSON. The complete map comparison took 61.1 s.

Opt-in configuration adds `integrator="euler",substeps=1` within the Gaussian block;
default Gaussian RK2, all maintained presets and final PCG+RK2 remain unchanged.
66 synthetic checks passed (65 medical, one Cameraman), including a test that Euler
queries t=0 exactly once and has the correct displacement direction. Frozen real
checkpoint passes yielded finite nonzero registration gradients at both endpoints;
weighted raw norms ranged .0077–.6847, then were capped at 10% of each base norm.
That cap does not guarantee bounded Adam updates or stable long-term training.
Wheel audit passed (47 entries; no datasets/results/tests/archives).
No training, checkpoint writes, commit or push occurred.

Local evidence: `results/gaussian_euler_probe/20261111/cost_runner.py`,
`map_runner.py`, `cost.json`, `maps.json` and per-label scalar JSON tables.
