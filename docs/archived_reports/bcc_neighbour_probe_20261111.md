# BCC sampling and immediate-neighbour filtering: quick frozen probes

Neither change warrants promotion from this probe. No training, optimizer steps,
RK2 integration, deformation-map recovery or registration scoring was performed.
Existing LPBA S02 MSE+PD-LK and NCC+moments (.01)+PD-LK endpoints were immutable.
No maintained implementation, configuration or test code changed; prototypes remain
in ignored run folders. This is one lattice phase and two checkpoints, not a
universal claim about BCC sampling.

## BCC versus Cartesian

Same local analytic strict PD-LK c=4, eta=1e-6, physical 4 mm Gaussian, three times
(.21875,.46875,.71875), and the earlier 65,536 native-centre probes. Foreground
union=17,322; strong edges=4,331; foreground at least 16 mm inside boundaries=17,020.
The primary interior mask avoids comparing truncated-domain normalization against
the dense control's reflected boundaries. LPBA's existing grid is orthogonal 1 mm.

Both lattices use the same normalized Gaussian-weighted quadrature at query sites.
BCC numerator/denominator combine BOTH shifted sublattices; there is no independent
filtering of sublattices. Cartesian spacing is isotropic; the old 32³ separable
filter/trilinear sampler is retained as a separate reconstruction control. Gaussian
support is ±4 sigma per coordinate. Constant-field reconstruction passed synthetic
checks for every lattice. This is a Gaussian quadrature prototype, not an optimal
BCC box-spline reconstruction or a verification of band-limited sampling theory.

| Pattern | Unique sites | Actual padded derivative queries | PCG cosine MSE/NCC, interior foreground | Dense-Gaussian cosine MSE/NCC, interior |
|---|---:|---:|---:|---:|
| Cartesian | 32,400 | 32,768 | .823 / .775 | .871 / .829 |
| BCC equal compute budget | 31,516 | 32,768 | .813 / .758 | .860 / .813 |
| BCC reduced budget | 22,491 | 24,576 | .775 / .723 | .818 / .772 |
| Existing Cartesian 32³ filter/interpolation | 32,768 | 32,768 | .838 / .793 | .883 / .852 |

Integer lattice boundaries prevent identical unique counts; equal-compute BCC has
2.7% fewer unique sites than the Cartesian control. Last chunks are padded with
duplicate coordinates to 4096, then sliced, so actual derivative evaluations are
explicit. Reduced BCC saves 30.6% unique sites / 25% evaluated sites. Equal-budget
BCC is slightly worse in this phase; the smaller version degrades further.

Warmed forward/backward medians for a shared 256-site velocity probe, excluding the
base training loss, budgets and optimizer: Cartesian 40.0/40.1 ms; equal BCC
40.5/40.3 ms; reduced BCC 30.7/30.8 ms (MSE/NCC). Three warmed repetitions, one
cold pass excluded. A denser 259,200-site Cartesian reference (262,144 evaluated)
took 308 ms. An NCC reference pass had a 575 ms outlier.

For parameter-gradient alignment, use the SAME velocity MSE against a detached PCG
field at 256 fixed interior-foreground sites, t=.46875. Compare each endpoint's
parameter gradients against the denser Cartesian quadrature, NOT gradients through
PCG and NOT the original endpoint registration objective. Reference quadrature is
itself approximate; no native-resolution gradient reference was constructed.

| Pattern | Source/target cosine, MSE checkpoint | Source/target cosine, NCC checkpoint |
|---|---:|---:|
| Cartesian | .174 / .205 | -.173 / -.159 |
| BCC equal | .042 / .048 | .007 / -.002 |
| BCC reduced | .023 / .030 | .005 / .016 |

All gradients were finite/nonzero, but gradient direction was poorly reproduced by
ALL coarse patterns. Relative gradient errors were about 7.7–16.1, so vector-field
agreement must not be read as faithful training gradients. BCC does not establish a
useful gradient-alignment improvement here; the reduced version's speed comes with
lower field fidelity. The complete BCC probe took 34.3 s. PCG reference converged
at 2/3 MSE times and 3/3 NCC times, retaining the known near-tolerance MSE miss.

## Immediate-neighbour convolution

On the existing Cartesian 32³ Euler surrogate, replace full Gaussian taps with
radius-one Gaussian taps on each axis: three separable 3-tap convolutions,
equivalent to a 3×3×3 neighbourhood (centre plus 26 axis/edge/corner neighbours).
Keep physical sigma=4 mm and normalize the shortened kernels. This tests the
Cartesian proposal; it is not a BCC nearest-neighbour graph implementation.

Five warmed registration-only forward/backward passes, same 256 continuous points,
compiled derivatives, single Euler step, original-image endpoint MSE and unchanged
boundary penalty. No base objective, gradient budgets or optimizer are timed.

| Checkpoint | Full filter | Immediate neighbours | Velocity cosine vs full | Parameter-gradient cosine source/target |
|---|---:|---:|---:|---:|
| MSE | 40.70 ms | 39.83 ms | .99905 | .909 / .678 |
| NCC | 40.06 ms | 39.81 ms | .99898 | .454 / .667 |

Velocity alignment is at t=.46875 over 65,536 foreground sites; gradient alignment
is for the t=0 Euler endpoint loss. They are different diagnostics. Relative
velocity error is about 5.7%; parameter-gradient error is 44–89%. Filtering alone
cost about .136 ms with either kernel (20 warmed repetitions). There is negligible
practical speed benefit: derivative construction/backward remains dominant. Small
forward changes can alter gradients in this nonlinear loss; this probe does not
isolate whether image-sampling gradients or other paths dominate that difference.
Do not interpret close velocity cosine as a gradient-equivalence guarantee.

## Preservation

Both probes verified unchanged loaded parameters and checkpoint/input image hashes.
No annotations were loaded or metrics selected by labels. No maps, velocities,
reconstructions or warped volumes were written; only compact scalar JSON evidence.
No pipeline changes, optimizer steps, training, commit or push occurred. Synthetic
constant-field checks ran within the BCC script; maintained suites were not rerun
because maintained code was unchanged. Existing Gaussian training stays on hold.
Exact local runners: `results/bcc_probe/20261111/{runner,lattice}.py` and
`results/gaussian_neighbour_probe/20261111/runner.py`.
