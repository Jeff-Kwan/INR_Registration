# Shared analytical SIREN derivative audit

Retained one optimization: pack value, spatial tangents and temporal tangent into
contiguous matrix rows, then jointly propagate reverse adjoint tangents for H/m.
No architecture, objective, sample count, derivative units, relaxation, precision
policy or interpolation change. Basic g/q-only recovery retains its previous chain.
No training, optimizer construction/steps, real images, labels or saved checkpoints
were used in this round. Synthetic tensors only; pipeline training remains on hold.

## Why this helps

The existing implementation already shared sine activations, interpolated weights
and a forward pass among g/q/H/m. It was not four independent network evaluations.
However, spatial tangent/adjoint matrices repeatedly transposed around linear layers,
and mixed derivatives used separate temporal contractions at each layer.

The retained differential chain keeps a leading lane axis [value, dx0, dx1, dx2, dt].
A single linear operation handles the lanes together. The time lane includes the
parameter-direction term; its reverse adjoint includes adjoint times DeltaW. The
readout yields q, while the reverse chain yields g/H/m. This shares matrix operations
and avoids separate per-layer mixed-derivative contractions. It remains the exact
analytic chain rule with ordinary differentiable parameter operations, not a custom
approximate gradient. Live coordinate/time gradients are retained when requested.

An eager forward/backward operator trace at 4096 sites saw separate mm calls fall
65→47, multiplies 157→124 and sums 42→27. Matrix multiplication and elementwise work
were prominent. A layout-only alternative changed little, so copying/transposes
alone were not the principal optimization. These eager counts explain implementation
structure; they are not percentages of compiled execution time.

## Warmed synthetic timing, RTX 5090

Four sine layers, width256, omega30; two endpoint networks with fixed synthetic
parameter perturbations, 4096 differential queries, medical-shaped scale factors.
Full PD-LK includes 32768 endpoint samples and the default envelope solve, but no
Adam update, global clipping, logs or image loading. Medians exclude compiler startup.

| Workload | Original | Packed | Reduction |
|---|---:|---:|---:|
| g/q/H/m plus parameter backward | 2.98 ms | 2.47 ms | 17% |
| Endpoint MSE + strict PD-LK forward/backward | 6.36 ms | 5.86 ms | 8% |
| Local velocity objective forward/backward | 3.73 ms | 3.37 ms | 10% |
| Gaussian 32³ single-Euler registration branch | 40.24 ms | 33.88 ms | 16% |
| Gaussian 16³ four-RK2 registration branch | 43.65 ms | 37.14 ms | 15% |

First three rows compare original, layout-only and packed candidates in one process
with eight post-startup passes (summary above excludes one additional settling pass).
Local-velocity timings showed more variation than PD-LK. Last two rows are separate
production verification with five warmed fixed-input passes, including activation
checkpoint recomputation but excluding the base loss and gradient budgets. Both
Gaussian cases use 32768 unique derivative queries over their time/chunk layouts.
Synthetic random images provide interpolation gradients; these are not measured
clinical training or registration outcomes. No claim of a multi-fold speedup.

Memory increased: derivative-only peak allocation ~425→473 MiB; full PD-LK
~948→1025 MiB. The separate Gaussian verification increased by ~49 MiB. These
figures exclude Adam state and are process/workload dependent. Packing uses wider
intermediate tangents; it trades modest additional memory for fewer operations.

## Agreement and validation

Both candidates passed float64 agreement with the original for g/q/H/m and all
endpoint, coordinate and time gradients in 2D/3D. The retained implementation then
passed the independent nested-autograd/JVP oracle in the maintained tests, including
rho=0 and positive rho, one/three-layer models, and added live coordinate/time VJP
checks at t=0,.41,1. Existing particle, normals and other synthetic checks pass.

Compiled float32 random-VJP comparison: packed output relative errors below 1e-6,
parameter-gradient relative error 8.4e-7, cosine >.9999998. Production Gaussian
verification showed gradient relative errors 4.7e-6 (Euler) and 1.4e-5 (RK2), cosine
>.9999999. Basic g/q and spatial-only outputs were bitwise identical to the old
implementation in the separate verification. Floating-point operation ordering is
changed in the differential path; mathematical equivalence is not bitwise identity.

66 medical + one Cameraman synthetic tests passed. Wheel audit passed (47 entries,
no datasets/results/tests/archives); CLI help and git diff --check passed. No datasets
or checkpoints were opened or modified. No commit/push or training occurred.

## Remaining opportunities and limits

Within one derivative call, weights/activations are already shared. Gaussian fields
are already shared across particles at each time. Interpolated weights and parameter
directions are still rebuilt for each spatial chunk; caching them within a single
time evaluation could reduce overhead, but was not implemented or measured here.
Do not cache activations across different positions/times or parameter updates.

Further large gains would likely need fewer required derivative queries or a more
specialized fused kernel, with corresponding approximation/maintenance costs. We did
not change precision, approximate Hessians, remove q/m, or alter strict PD-LK.
The retained optimization does not address the Gaussian branch's field-fidelity or
integration failures, and does not make its training validated.

Exact local candidates, original source snapshot, checks, timings and traces are
under `results/derivative_audit/20261111/`. Only the winning analytic implementation
and one independent regression test were added to maintained code.
