# Full-gradient covariant residual

The experimental full-gradient strict baseline is packaged for each dataset as
`medical/configs/<dataset>_fullgrad_pdlk_ncc001.json`. It sets alpha to zero,
PD-LK weight to 0.1 and registration NCC weight to 0.01.

Set `objective.gaussian_registration.covariant_residual_alpha` to a number in
`[0,1]` in a copy of a shared-grid joint training config. An absent or null
setting retains the strict default. Require analytical derivatives, `rho=0`,
`pdlk_sampling="paired_grid"`, a differential velocity model, and the uncapped
single-backward path. This option cannot be combined with `affine_covariant`.
The existing `objective.differentiate_solve=false` option remains required by
the shared-grid configuration validator; the covariant-residual implementation
explicitly differentiates its own velocity solve regardless of that setting.

At each sampled time, the unchanged strict pooled solve gives cell velocities
`v`. The existing Gaussian filter produces the registration decoder field. Its
trilinear spatial Jacobian `B` is evaluated at each paired site in native voxel
coordinates. The energy averages `(q + g·v)^2 + c |m + H v + alpha B^T g|^2`
across sites and adds `eta |v|^2` per cell. The velocity, filter, Jacobian and
all SIREN derivatives remain live for backpropagation. At alpha zero this has
the strict shared-grid value and its exact-solve gradient, up to numerical error.
The registration NCC and final PCG+RK2 recovery are unchanged.

The trilinear derivative uses the right-hand slope at interior lattice knots and
zero beyond the outer centres, matching the decoder's border extension away from
knots. The brightness residual uses the raw cell velocity while the correction
uses the interpolated field gradient, so this is a partial covariant correction,
not a fully affine-consistent flow equation.
