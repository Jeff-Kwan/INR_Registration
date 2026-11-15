# Central-half symmetric-pair retraining

LPBA S02, seed 42. Both 5,000-step runs used MSE endpoint reconstruction,
bidirectional symmetric-pair registration loss (weight .01 after step 2,000),
PD-LK disabled, uncapped registration gradients, and the unchanged PCG+RK2
recovery. Sampling starts uniformly in the central 50% of each local-box
dimension; a particle remains valid through the full velocity box. Both runs
used the full Gaussian smoothing kernel to isolate the sampling-policy change.
The endpoint PSNR floor remained 25 dB. This is single-case exploratory evidence.

| Setting | Endpoint PSNR (moving/fixed, dB) | Dice | HD95 (mm) | Folded voxels | Train / recovery (s) |
|---|---:|---:|---:|---:|---:|
| 16³ local box, 1 pair per cell | 31.38 / 31.78 | .66819 | 6.79850 | 5 | 87.40 / 9.88 |
| 8³ local box, 2 pairs per cell | 31.28 / 31.69 | .66938 | 6.77514 | 1 | 56.78 / 9.66 |
| Maintained MSE+PD-LK reference | 32.20 / 32.48 | .70245 | 6.605 | 0 | 37.03 / 18.35 |

Both runs passed the reconstruction gate and completed recovery; every PCG
interval converged. At every logged active step, all 4,096 particles in each
direction remained within the velocity box, and no registration batches were
skipped. The 16³ and 8³ registration branches averaged about 20.8 ms and 10.5 ms
per active logged step, respectively (20.6 ms and 10.4 ms averaged over 150
active checkpoints).

This supports the suspicion that the former 8³ failure was related to the old
narrow supported interior: before its step-2,119 reconstruction abort, the old
uncapped run rejected 32–73% of forward and 41–61% of reverse trajectories at
logged points. The new rule retained all particles. Since the start distribution
and validity rule changed together, this does not isolate their effects. The
new runs also remain below the maintained PD-LK reference on this case; they do
not establish a cohort-level benefit.

Ignored run artifacts are under
`results/pair_bidirectional_central50_20261111/box16_uncapped/` and
`results/pair_bidirectional_central50_20261111/box8_twopair_uncapped/`.
