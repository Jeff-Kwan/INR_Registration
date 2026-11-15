# Full-gradient covariant residual: LPBA S02 and S03

Matched seed-42 fits used 5,000 steps of MSE endpoint reconstruction, shared-grid
PD-LK plus bidirectional box NCC. Only `covariant_residual_alpha` changed within
each subject. All used the strict pooled velocity solve, full gradients through
the corrected PD-LK residual, and unchanged PCG+RK2 recovery. The endpoint PSNR
gate was explicitly skipped for all six runs.

| Subject | Alpha | Dice | HD95 (mm) | Folding | Endpoint PSNR (moving/fixed dB) | Train (s) | PCG converged |
|---|---:|---:|---:|---:|---:|---:|---:|
| S02 | 0 | .710539 | 6.2394 | 0% | 30.37 / 30.83 | 96.19 | 13/16 |
| S02 | .5 | .712256 | 6.2509 | 0% | 30.31 / 30.94 | 95.05 | 16/16 |
| S02 | 1 | .713283 | 6.2059 | 0% | 30.24 / 31.02 | 95.38 | 15/16 |
| S03 | 0 | .716784 | 6.3716 | 0% | 31.28 / 31.01 | 95.40 | 16/16 |
| S03 | .5 | .716333 | 6.4114 | 0% | 31.32 / 31.09 | 94.90 | 16/16 |
| S03 | 1 | .717358 | 6.3528 | 0% | 31.12 / 31.14 | 94.06 | 15/16 |

Alpha 1 had the highest Dice and lowest HD95 within each subject, while alpha .5
was mixed against alpha 0. Across these two subjects, the mean Dice was .71366,
.71429 and .71532 for alpha 0, .5 and 1; mean HD95 was 6.3055, 6.3312 and
6.2793 mm. All runs had zero measured folding. Alpha 1 missed the PCG tolerance
on the final interval of both cases (maximum relative residuals 2.01102e-5 for
S02 and 2.00864e-5 for S03; tolerance 2e-5). Alpha 0 missed one S02 interval;
alpha .5 converged on all intervals.

Training took 94–96 seconds per fit, compared with about 62 seconds for the
existing strict reference. This two-subject screen favors alpha 1 descriptively,
but the gains are small and do not establish a cohort-level benefit. The PSNR
gate skip and PCG near-tolerance misses remain part of the result. Per-run
checkpoints and logs are under fresh ignored `results/covariant_residual_s02_*`
and `results/covariant_residual_s03_*` directories.
