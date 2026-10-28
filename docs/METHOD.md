# Method and conventions

## Paired endpoint trajectory

For a moving image and a fixed image, the model fits two spatial SIRENs with
parameters `theta_M` and `theta_F`. The continuous image path is obtained by
linear interpolation in parameter space:

\[
\theta(t)=(1-t)\theta_M+t\theta_F,\qquad I(x,t)=f_{\theta(t)}(x),\quad t\in[0,1].
\]

Endpoint reconstruction losses fit the two input images. They do not by
 themselves guarantee a useful path between them. The PD-LK regularizer acts on
that path. Labels and ground-truth deformation are evaluation signals only;
LPBA provides no ground-truth deformation field.

## Standard pointwise differential LK

At a sampled space-time point define

\[
g=\nabla_x I,\quad q=\partial_t I,\quad
H=\nabla_x^2 I,\quad m=\partial_t\nabla_x I.
\]

Assuming locally constant velocity, first-order brightness and gradient
transport give the minimized energy

\[
E_0=\min_v\;(q+g^\top v)^2+c\lVert m+Hv\rVert_2^2+\eta\lVert v\rVert_2^2.
\]

The ridge `eta` makes the small 2D or 3D system well posed in flat or
rank-deficient regions. The velocity is an auxiliary local solution. By default,
the solve is held fixed in reverse mode and gradients flow through the residual
and analytical SIREN derivatives into both endpoint networks (the
envelope-theorem path). An optional through-solve gradient is retained as an
ablation setting.

Each training step samples one shared uniform time and independent continuous
spatial points. Each point solves its own velocity system. Pointwise PD-LK does
not impose spatial agreement between neighboring auxiliary velocities; `H`
encodes image curvature, not a velocity-smoothness penalty.

## Relaxed, locally affine PD-LK

The relaxed model permits a local velocity Jacobian and penalizes its magnitude.
Eliminating that Jacobian gives the pointwise coefficient

\[
 c_{\mathrm{eff}}=\frac{c}{1+\rho\lVert g\rVert_2^2},\qquad
 E_\rho=\min_v\;(q+g^\top v)^2+c_{\mathrm{eff}}\lVert m+Hv\rVert_2^2
 +\eta\lVert v\rVert_2^2.
\]

Here `rho = c / beta`, where `beta` is the local Jacobian penalty. `rho=0`
recovers standard PD-LK exactly. A positive `rho` attenuates gradient transport in
proportion to local image-gradient energy. With `c=4`, `rho=0.4` corresponds to
`beta=10`. The implementation also accepts legacy `beta` in old saved configs,
but new configs use `rho`.

The product `rho*|g|^2` is dimensionless; `rho` itself depends on the spatial and
intensity units used for `g`. A numerical rho value is therefore not invariant
under image-intensity or coordinate rescaling.

The parameter `c` is the differential-gradient weight, `eta` is the velocity
ridge, and `lambda` multiplies the minimized energy in the training objective.
Do not merge these roles. The 3D config uses `c=4`, `lambda=0.1`, `eta=1e-6`,
`rho=0.4`; the matched standard control sets `rho=0`.

## Global deformation recovery is a separate stage

After endpoint fitting, recovery estimates a spatial velocity field using the
trajectory's midpoint image derivatives and a global regularized optical-flow
solve. Piecewise velocity fields are integrated with midpoint RK2 to obtain
forward and inverse maps. The inverse map samples the original moving image onto
the fixed grid. Thus the local velocities used to define the training loss are
not directly assembled into the final deformation.

The synthetic Cameraman case provides a known deformation, oracle frames, and
an oracle trajectory. Comparing learned trajectory, oracle-frame flow, and
true-map error helps separate representation error from flow ambiguity. LPBA
uses image reconstruction gates, label Dice/HD95, folding and flow diagnostics;
it has no true deformation for endpoint error measurement.

## Spatial units and array axes

- Cameraman coordinates enter the SIREN in normalized `[-1,1]` coordinates;
  analytical derivatives are rescaled to native pixels.
- NIfTI arrays use `[D,H,W]`. Sampler/vector channels are `(x=W,y=H,z=D)`.
  Derivatives and recovered displacement are in native voxels, per unit
  normalized time. Exported world displacement vectors are RAS millimetres
  obtained through the NIfTI affine.
- LPBA inputs must share shape and affine. Runtime MRI normalization maps the
  nonzero foreground 1st/99th percentiles to `[0,1]` and leaves background zero.
  Stored data is not modified by the loader.
- LPBA labels are evaluation-only. Dice is MONAI `compute_dice(ignore_empty=False)`
  averaged equally over nonzero labels. HD95 is symmetric Euclidean surface
  distance in millimetres using NIfTI array-axis spacing. A missing label scores
  Dice zero and makes aggregate HD95 unavailable rather than silently dropping
  the label.

The implementation is in `cameraman/core/` and `medical/registration/`. See
[the development notes](DEVELOPMENT.md) for supported entry points and checks.
