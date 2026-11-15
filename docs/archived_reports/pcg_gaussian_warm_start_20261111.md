# Gaussian velocity warm start for original PCG (2026-11-11)

On the existing LPBA S02 factor-8 pyramid checkpoint, original brightness-flow
PCG was recovered with three initializations. The reference starts each interval
at zero. The other variants start the first interval at zero and subsequently
reuse the preceding solved velocity, either unchanged or after a separable
Gaussian with sigma 1 native voxel, radius 4 and replicated boundaries. Each
variant solves the same PCG system to the existing tolerance and uses the same
16 intervals and RK2 integration. This is an initialization experiment: the final
velocity is not blurred after the solve.

| Initialization | PCG iterations, total | PCG solve s | Recovery s | Dice | HD95 mm | Folding fraction | Max relative residual |
|---|---:|---:|---:|---:|---:|---:|---:|
| Zero each interval | 6260 | 12.51 | 16.69 | 0.706715 | 6.3953 | 0 | 2.0065e-5 |
| Previous velocity | 3700 | 7.45 | 11.58 | 0.706719 | 6.3960 | 0 | 2.0064e-5 |
| Gaussian previous velocity | 3730 | 7.52 | 11.30 | 0.706718 | 6.3960 | 0 | 1.9898e-5 |

The Gaussian warm start reduced measured PCG time by 40% and recovery time by
32% against zero initialization, with essentially unchanged metrics. The
unfiltered warm start achieved nearly the same result and slightly fewer
iterations: the speedup is attributable to reusing the preceding velocity, with
no demonstrated extra gain from Gaussian smoothing in this case. The zero and
unfiltered runs narrowly missed the 2e-5 true residual tolerance; the Gaussian
run met it for all intervals. All passed the endpoint PSNR gate.

These are single-run timing observations on an RTX 5090. Derivative and solver
paths were warmed before timing; the unfiltered control ran in a separate
process, so small timing differences are not conclusive. PCG timings include
initialization and synchronization; recovery timings include derivatives,
integration and scratch I/O, but exclude scoring. The factor-8 checkpoint and
medical data remained unchanged; hashes were checked before and after. No
training ran. Compact results and the experiment runners are in the ignored
`results/pcg_gaussian_warm_start/20261111/` directory. This does not establish
the effect on other cases or a general improvement in registration accuracy.
