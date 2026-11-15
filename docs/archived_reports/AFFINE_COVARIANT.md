# Opt-in affine-covariant shared-grid PD-LK

One LPBA S02 fit and recovery is recorded in the [pilot report](reports/affine_covariant_s02_20261112.md). The maintained default remains strict PD-LK.

In a copy of any packaged default config, add
`"affine_covariant": true` to `objective.gaussian_registration`. Keep
`objective.rho=0`, positive PD-LK and registration weights, `allow_pdlk=true`,
`pdlk_sampling="paired_grid"`, `derivative_mode="analytic"`,
`velocity_model="differential"`, and `single_backward=true`. The normal two-pair
layout, endpoint loss, local NCC and recovery settings remain selectable as before.
The method identifier adds `_affine_covariant` so its evidence is distinguishable.
No packaged config enables the option.

## Conditional affine model

Let g, q, H, m denote the existing native-voxel analytical SIREN derivatives.
Each sampled field time uses the following single correction pass:

1. Query each paired site once. Build the ordinary shared-grid matrix
   \(A_c=\langle gg^T+cH^TH\rangle+\eta I\), factor it once and solve for the
   strict centre velocities. Angle brackets average sites within a cell.
2. Smooth that preliminary lattice with the configured Gaussian filter, then
   derive \(B_c=\nabla v\) at each lattice centre.
3. Use the actual native-voxel pair offsets \(\delta_j=x_j-x_c\) and let
   \(v_j=u_c+B_c\delta_j\). Set
   \[
   a_j=q_j+g_j^TB_c\delta_j,\qquad
   d_j=m_j+H_jB_c\delta_j+B_c^Tg_j.
   \]
   Reuse the same matrix factorization to solve
   \[
   u_c^*=-A_c^{-1}\langle g_ja_j+cH_j^Td_j\rangle.
   \]
4. Evaluate
   \[
   E_c=\langle(q_j+g_j^Tv_j)^2+
   c\|m_j+H_jv_j+B_c^Tg_j\|^2\rangle+\eta\|u_c^*\|^2.
   \]
   Average over cells and the two field times for the PD-LK contribution.
5. Smooth the corrected centre velocities to supply the existing bidirectional
   particle integration and original-image NCC loss.

Use the configured c and eta (defaults 4 and 1e-6). The ridge penalizes the
centre velocity, rather than all four site velocities; changing that would
change the requested normal equations. B is held fixed only with respect to u
inside this conditional solve. It remains a live function of the SIREN weights.

## Jacobian convention and units

Field arrays use [D,H,W,3], vector channels use (W,H,D), and
\(B_{ij}=\partial v_i/\partial x_j\). Native cell widths are
`domain_extent_xyz / grid_size`. Pair offsets already include physical-spacing
conversion; apply no further spacing factor to B, H or the residuals.

A trilinear field has different one-sided derivatives at its lattice knots.
At a centre we explicitly average those slopes: central differences internally
and half the interior slope at outer centres, because the registration decoder
extends its border values into the outer half-cell. This is the symmetric
derivative convention for that same border-extended interpolant. It is exact for
affine fields at interior centres; no unique classical derivative exists at a
general knot. The operator is linear in the field values and differentiable in
the SIREN parameters, without differentiating a grid-sampler backward operation.

## Gradients, cost and limits

For the PD-LK energy, detach only the optimum u. All residuals, B, the preliminary
strict solve, smoothing and the effective dependence on the SIREN derivatives
remain live. This gives the full conditional value-function derivative, including
the chain through B. For NCC, the corrected solve remains live as well.

`registration/affine_covariant.py` owns the pure Jacobian, factorization, correction
and energy operations. `SymmetricPairRegistration` constructs both passes while
retaining derivative-site chunk limits. Analytical derivative evaluations are
unchanged in count; each cell adds a second RHS solve using the existing LU
factorization, a small-lattice Jacobian, and a second Gaussian filter. With the
default `recompute=false`, no derivative sites are evaluated again. Compiled seed
queries and optional explicit checkpoint recomputation are supported. The single-case pilot measured training cost and endpoint quality.

B comes from the preliminary field, not from a converged coupled affine solve.
The final smoothed corrected field can have a different Jacobian. The centre
linearization also approximates a trilinear field across the paired sites.
The model therefore has a precise conditional objective but does not ensure
self-consistency, diffeomorphism or better registration. Existing numerical,
gradient, reconstruction-policy and runtime safeguards still apply. Rho relaxation,
independent pointwise PD-LK, brightness-only and finite-difference branches cannot
be combined with this option.

Synthetic checks cover known affine transport with nonsymmetric B, native-axis
and border conventions against sampled trilinear differences, the B=0 strict
limit, finite differences of the full value-function gradient, nonzero gradients
into both endpoints through each loss, derivative-query reuse, invalid configs,
and compiled/eager values and parameter adjoints. These checks use no medical data
or optimizer steps and do not establish medical performance.
