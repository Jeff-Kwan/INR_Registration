# Affine-covariant shared-grid PD-LK: LPBA S02

One 5,000-step seed-42 fit enabled `objective.gaussian_registration.affine_covariant`
in the maintained MSE + shared-grid PD-LK + bidirectional local NCC configuration.
All other settings matched the existing LPBA S02 reference, including `rho=0`,
the endpoint gate skip, and final PCG+RK2 recovery. The run used a fresh ignored
directory at `results/affine_covariant_s02_20261112/`; no existing checkpoint or
result was overwritten.

| Measure | Strict PD-LK | Affine-covariant | Change |
|---|---:|---:|---:|
| Dice | .709180 | .709909 | +.000729 |
| HD95 (mm) | 6.2863 | 6.2788 | −.0074 |
| Folded voxels | 0 | 0 | 0 |
| Minimum Jacobian | .09592 | .10405 | +.00813 |
| Endpoint PSNR source / target (dB) | 30.42 / 30.96 | 30.24 / 30.74 | −.18 / −.22 |
| Training (s) | 61.93 | 89.59 | +27.66 (+44.7%) |
| Recovery (s) | 11.62 | 11.91 | +.28 |
| Total (s) | 78.18 | 106.11 | +27.92 |
| PCG intervals converged | 16/16 | 15/16 | −1 |

The affine run's maximum relative PCG residual was `2.02399e-5`; interval 16
missed the `2e-5` tolerance narrowly. Its mean inverse-composition error was
0.1557 mm versus 0.1601 mm for strict PD-LK. The 25 dB endpoint gate was
explicitly skipped in both runs; reported endpoint PSNR is diagnostic.

On this single subject and seed, the anatomical score change is small while
training cost increased by about 45%. This is preliminary evidence only: it
does not establish a reliable gain or justify changing the strict default.
