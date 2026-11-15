# Two-step Euler and RK2 on frozen LPBA S02 endpoints

Both two-step variants improve over single Euler, but remain below standard PCG
recovery and have appreciable NCC folding. No training or optimizer updates were
performed. Existing Gaussian training remains on hold; no preset was changed.

All four surrogate integrators were rerun using the retained packed analytic
derivative chain, same 32³ Cartesian grid, 4 mm Gaussian and frozen MSE/NCC+moments
(.01) PD-LK checkpoints. Standard PCG+RK2 uses its unchanged checkpoint settings.
Euler2 queries times 0,.5; RK2 with two substeps queries 0,.25,.5,.75. Derivative
queries per pass: 32768/65536/131072/262144 for Euler1/Euler2/RK2_2/RK2_4.

## Warmed full forward/backward cost

RTX5090, 256 particles, 32768 endpoint samples, 4096 PD-LK samples, compiled
analytic local solve and activation recomputation. Medians of five warmed passes;
includes base objective, per-endpoint registration gradient budgets and global
clipping. Excludes Adam, periodic reconstruction checks, loading and recovery.

| Integrator | MSE | NCC |
|---|---:|---:|
| Baseline endpoint+PD-LK | 6.9 ms | 7.2 ms |
| Euler, one step | 42.0 ms | 42.0 ms |
| Euler, two steps | 74.3 ms | 74.8 ms |
| RK2, two steps | 139.3 ms | 139.5 ms |
| RK2, four steps | 270.6 ms | 271.2 ms |

Five passes are preliminary timing, not sustained training throughput. One MSE
four-step RK2 registration pass had a .503 s outlier, and one NCC pass .482 s;
normal registration-only passes were about .263–.265 s. Compiler startup is
excluded. These post-packing timings replace earlier unoptimized costs for this
comparison; do not attribute all differences from older reports to integrators.
All active registration gradient norms were finite and nonzero at both endpoints.

## Direct forward-map evaluation

Use moving-domain comparisons M(a) versus F(phi(a)), warping fixed labels onto
the moving grid. This evaluates the estimated forward map directly without
constructing or assuming an Euler inverse. Values differ from historical fixed-grid
inverse-map benchmark scores. Macro MONAI v1.6.1 Dice/HD95, 56 labels, no missing
labels; HD95 in mm. Identity Dice=.563048, HD95=7.837058 mm. Folding is finite-
difference det(Dphi)<=0 over 4,915,200 native voxels. No maps/volumes were saved.

| Integrator | Dice MSE/NCC | HD95 mm MSE/NCC | Folding % MSE/NCC |
|---|---:|---:|---:|
| Euler, one step | .632836 / .617784 | 7.504 / 8.124 | .83927 / 8.02816 |
| Euler, two steps | .668999 / .662752 | 7.176 / 7.503 | .01213 / .32833 |
| RK2, two steps | .675175 / .671747 | 7.250 / 7.252 | .02350 / .21273 |
| RK2, four steps | .680853 / .676920 | 7.037 / 7.148 | .000122 / .02828 |
| Standard PCG+RK2 | .698464 / .696908 | 7.193 / 7.049 | 0 / .001933 |

Two-step Euler is the cheaper substantial improvement. Two-step RK2 gives higher
Dice and better NCC HD95/folding, but costs about twice as much. MSE folding/HD95
are not monotonic in integrator order, so do not claim that higher order guarantees
a better registration or invertibility. Four-step Gaussian-RK2 remains closer to
PCG overall; all Gaussian variants retain spatial approximation error. These are
frozen-map diagnostics, not evidence of training convergence or improved models.

## Validation and preservation

Synthetic v(x,t)=t integration checks passed: Euler1 gives zero displacement,
Euler2 gives .25, and RK2_2/RK2_4 give .5, matching their intended time queries.
No maintained code changed in this experiment; existing options already support
these substep counts. Maintained suites were not repeated (the prior packed-chain
change passed 66 medical + one Cameraman checks). The map comparison took 74.2 s.

Loaded weights and checkpoint/image/annotation hashes were unchanged. PCG converged
15/16 MSE intervals (known near-tolerance miss) and 16/16 NCC intervals. Temporary
PCG velocity scratch was removed; scratch root was empty afterward. Scalar and
per-label JSON only, no persistent deformation/velocity/warped-volume writers.
No dataset modifications, training, checkpoint writes, commit or push occurred.

Exact local runners, timings and scalar evidence:
`results/gaussian_integrator_probe/20261111/`; source hashes record the exact packed
derivative, Gaussian sampler and maintained recovery code. The accompanying JSON
contains per-time PCG diagnostics and endpoint displacement comparisons.
