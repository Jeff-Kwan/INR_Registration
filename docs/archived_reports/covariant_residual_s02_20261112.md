# Full-gradient covariant residual: LPBA S02

Three seed-42, 5,000-step fits used the maintained MSE + shared-grid PD-LK +
bidirectional box NCC configuration, changing only `covariant_residual_alpha`.
The strict velocity solve and smoothed registration decoder remained unchanged;
the PD-LK residual used full solve and field-Jacobian gradients. Recovery used
the unchanged PCG+RK2 pipeline. See [method and limitations](../COVARIANT_RESIDUAL.md).

| Alpha | Dice | HD95 mm | Folding | Endpoint PSNR source / target dB | Training s | Recovery s | PCG converged |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | .710539 | 6.2394 | 0% | 30.37 / 30.83 | 96.19 | 11.89 | 13/16 |
| .5 | .712256 | 6.2509 | 0% | 30.31 / 30.94 | 95.05 | 11.50 | 16/16 |
| 1 | .713283 | 6.2059 | 0% | 30.24 / 31.02 | 95.38 | 11.63 | 15/16 |

The maximum relative PCG residuals were `2.01185e-5`, `1.99876e-5` and
`2.01102e-5`, respectively, against tolerance `2e-5`. The endpoint PSNR gate
was explicitly skipped in every run; PSNR is diagnostic. All 250 recorded
training rows per run had finite losses and gradient norms.

The existing independently run strict baseline scored .709180 / 6.2863 mm in
61.93 seconds of training, with no folding and 16/16 converged intervals. The
alpha-zero value and derivative match strict PD-LK in synthetic checks, but its
S02 fit differs numerically from that earlier reference; the three new fits are
the closer comparison. Alpha 1 improved Dice by .002745 and HD95 by .0335 mm
relative to alpha 0. This is one case and seed, with a skipped gate and near
PCG tolerance misses; it does not establish a cohort-level benefit. The full
gradient path cost roughly 95–96 seconds of training, versus 62 seconds for the
existing strict reference. All new artifacts are in ignored fresh directories
`results/covariant_residual_s02_a0_20261112/`, `..._a05_20261112/` and
`..._a1_20261112/`.
