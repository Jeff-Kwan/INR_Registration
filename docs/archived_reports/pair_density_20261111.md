# Symmetric-pair density and gradient-budget pilot

LPBA S02, seed 42; same 5,000-step paired SIREN settings, 0.01 bidirectional
registration weight after step 2,000, 4,096 trajectory particles per direction,
2 mm symmetric offsets, and unchanged PCG+RK2 recovery. All runs used MSE endpoint
reconstruction, PD-LK disabled throughout, and global gradient norm clipping at
1.0. The 8³ uncapped run stopped at the reconstruction guard, so it has no
recovery score. A capped 8³ rerun checked the stated fallback. These are one-seed
exploratory results, not evidence for a general setting.

These runs used the historical Gaussian-supported interior for particle sampling
and validity. The later central-half sampling and full-box validity policy has
not been trained, so its stability and rejection rates are unknown.

| Setting | Registration gradient budget | Endpoint PSNR (moving/fixed, dB) | Dice | HD95 (mm) | Folded voxels | Train / recovery (s) |
|---|---:|---:|---:|---:|---:|---:|
| Maintained MSE+PD-LK reference | — | 32.20 / 32.48 | 0.70245 | 6.605 | 0 | 37.03 / 18.35 |
| 16³ whole image, 1 pair | 50% | 33.44 / 33.87 | 0.64978 | 6.991 | 4 | 90.75 / 11.08 |
| 16³ local box, 1 pair | 50% | 31.39 / 31.71 | 0.66473 | 6.823 | 9 | 87.87 / 9.71 |
| 16³ local box, 1 pair | uncapped | 31.25 / 31.81 | 0.66510 | 6.838 | 3 | 87.74 / 9.97 |
| 12³ local box, 2 pairs pooled per cell | uncapped | 31.26 / 31.79 | 0.66301 | 6.804 | 9 | 94.36 / 9.54 |
| 8³ local box, 2 pairs pooled per cell | uncapped | 24.09 at abort | — | — | — | aborted / — |
| 8³ local box, 2 pairs pooled per cell | 50% | 31.33 / 31.75 | 0.65241 | 6.873 | 3 | 56.84 / 9.65 |

The uncapped 16³ test passed the reconstruction gate, had no skipped updates, and
recovered Dice/HD95 within 0.00038 / 0.015 mm of the 50%-capped local-box run.
It was safe to carry the uncapped budget into 12³: that run also passed the gate,
had no skipped updates, and all PCG intervals converged. The 8³ uncapped run
collapsed shortly after activation (source PSNR 24.09 dB at step 2,119) and was
stopped by the reconstruction safeguard. Restoring the 50% budget allowed the
8³ run to complete, but its registration score remained below the 12³ results.

For the successful uncapped 16³ and 12³ runs, all 150 active training log checks
had zero skipped updates; average registration-branch time was about 21 ms per
logged active step. The capped 8³ run averaged about 10 ms per active branch
check. Recovery converged for every completed run. Folding remained very sparse,
though the minimum sampled Jacobian was negative in each completed surrogate run.

The 8³ result suggests uncapped gradients are not stable across grid densities.
The 16³ evidence supports uncapped use at that setting, while 12³ was also stable
in this case. Keep 50% for 8³ unless a separate controlled test establishes a
stable alternative. Every surrogate configuration remained below the maintained
PD-LK reference in Dice and HD95; no promotion is warranted.

Compact ignored run artifacts are under `results/pair_bidirectional_20261111/`:
`whole/`, `box/`, `box16_uncapped/`, `box12_2pairs_uncapped/`,
`box8_2pairs_uncapped/`, and `box8_2pairs_cap50/`. The two-pair implementation
uses four derivative sites per cell in one pooled solve; `chunk` bounds derivative
sites, not cell count.
