# Uncapped crop registration pilot: LPBA S02

One seed-42 LPBA S02 run repeated the PD-LK-free half-volume crop pilot using
24³ local velocity grids, Gaussian smoothing, Euler ×2, 256 particles and the
same .01 registration weight, activated after step 2,000. Endpoint MSE used
32,768 samples per crop. The local velocity retains the existing PD-LK-form
solve; the PD-LK training loss stayed zero. The previous per-endpoint gradient
budget of .1 was removed (`gradient_budget: null`), and combined global norm
clipping was also disabled (`training.grad_clip=0`). All other settings and the
existing finite, particle-support, runtime and reconstruction guards were kept.

The run **failed at step 2,020**: the moving endpoint's independent sampled
reconstruction monitor was **21.98 dB**, below the 25 dB threshold. This check
occurs before the step-2,020 optimizer update, after 19 registration-active
updates. The last logged warm-up minibatch at step 2,000 had moving/fixed
reconstruction MSE .001325/.001195; these crop minibatch values are not the
whole-volume reconstruction monitor and cannot be directly compared to 21.98 dB.
The run terminated before a final endpoint checkpoint or PCG+RK2 recovery, so
Dice, HD95 and folding were not measured. The first active gradients were not
persisted by the existing 20-step logger; their magnitude and the precise cause
of the drop are unknown. Removing both caps also changes the first 2,000 fitting
updates, so this is not an isolated test of the separate registration budget.

The prior capped pilot passed its endpoint gate and obtained Dice .634556 and
HD95 7.05269 mm, close to the existing global MSE-only result. The failed uncapped
run is evidence of instability for this configuration, not a registration-score
comparison or proof of which cap caused it. No further coefficients or runs were
tested. Input data and previous artifacts were preserved; the fresh ignored
`results/cropped_gaussian_registration/uncapped_20261111/` contains the config,
training log and failure status.

## Restored clipping with a 100% registration budget

A subsequent S02/seed42 pilot restored global norm-1 clipping and capped each
endpoint's registration gradient at 100% of its reconstruction-gradient norm.
All other settings matched the prior 10%-budget crop pilot. The independent
moving-endpoint PSNR monitor fell to **24.81 dB at step 2,320**, before that
step's optimizer update; the 25 dB guard stopped training after 319 active
registration updates. At the last logged check, step 2,300, moving/fixed PSNR
was 25.015/27.464 dB and registration-gradient scale factors were .393/.806.

This ran longer than the fully uncapped attempt but did not preserve the gate.
The 10% pilot completed and scored Dice .634556 / HD95 7.05269 mm. The 100%
pilot produced no final checkpoint or recovery, so no registration-performance
comparison is available. Its config and scalar logs are in ignored
`results/cropped_gaussian_registration/budget100_20261111/`.

## 50% registration budget

One S02/seed42 run retained global norm-1 clipping and set the per-endpoint
registration budget to 50%; all other settings matched the 10% pilot. It completed
5,000 steps with PD-LK zero in all logged steps. Full-volume endpoint PSNR was
31.410/31.800 dB, passing the 25 dB gate. The unchanged PCG+RK2 recovery yielded
Dice **.644948**, HD95 **6.92715 mm** and 14 folded voxels out of 4,915,200
(0.000285%). All 16 PCG intervals converged; maximum relative residual was
1.9827e-5 against tolerance 2e-5. Training took 129.49 s and recovery 9.76 s.

The earlier 10% pilot scored Dice .634556, HD95 7.05269 mm and 15 folded voxels;
the 50% result improves Dice by .01039 and HD95 by .12554 mm for this pair. The
50% recovery used the current previous-velocity PCG warm start, whereas the
historical 10% result predates that default. This single-seed comparison also
lacks a matched crop-only control and does not establish a cohort-wide benefit.
The existing MSE+PD-LK reference remains higher at Dice .698229 / HD95 6.56515 mm.

Across 150 logged active batches, the moving/fixed registration-gradient scale
factors had medians .0609/.2219, so the 50% budget remained influential despite
being less restrictive than 10%. Median retained particles were 89.1%; no logged
batch was skipped. Monitored moving/fixed PSNR minima were 27.95/28.92 dB. The
run completed, so the conditional 20% run was not started. Compact scalar evidence
and the final endpoint weights are in ignored
`results/cropped_gaussian_registration/budget50_20261111/`.

## 4,096-particle follow-up

The next S02/seed42 run changed only particles=256 to 4,096, retaining the 24³
half-box lattice, Euler2, 50% gradient budget, global clipping and all guards.
It completed 5,000 steps, with PD-LK zero in all logs. Both runs used the current
PCG previous-velocity warm start. The tetrahedral whole-image proposal was not
implemented or trained.

| Particles | Dice | HD95 mm | Folded voxels | Training s | Recovery s |
|---|---:|---:|---:|---:|---:|
| 256 | .644948 | 6.92715 | 14 | 129.49 | 9.76 |
| 4,096 | .648504 | 7.01892 | 4 | 130.14 | 9.82 |

Full endpoint PSNR31.470/31.763 dB passed the gate. All16 PCG intervals converged,
maximum residual1.77225e-5. Median active registration pass33.29 ms; median whole
active training step37.72 ms. Median surviving particles3,638.5/4,096 (88.8%),
logged range3,282–3,848. Median moving/fixed registration-gradient scale factors
.1324/.5222. Increasing samples added little observed runtime and slightly
improved Dice/folding, but HD95 was worse; one seed does not establish a useful
registration gain. Compact logs and final weights remain in ignored
`results/cropped_gaussian_registration/particles4096_20261111/`.
