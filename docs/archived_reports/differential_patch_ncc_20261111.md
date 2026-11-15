# Differential patch-NCC: LPBA S02 pilots (2026-11-11)

Three opt-in 5,000-step MSE-endpoint pilots used seed 42, the width-256,
four-layer paired SIRENs, shared initialization, linear weight interpolation,
Adam/LR schedule and norm-1 clipping. PD-LK was disabled for every step. The loss
uses 64 continuous-origin 4³ patches at one shared random time, with one-native-
voxel spacing, normalized projected constant-velocity normal equations, detached
velocities and live projection/variance gradients. No geometry, Hessians, PCG or
integration entered training. Below-floor patches contribute zero to the mean
over all 64. Defaults are eta=1e-6 and variance floor=1e-6.

| Quantity | Weight .1 | Weight .01 | .01 ramp after step 2,000 |
|---|---:|---:|---:|
| Moving endpoint MSE / PSNR | .0058023 / 22.364 dB | .0044379 / 23.528 dB | .0095765 / 20.188 dB |
| Fixed endpoint MSE / PSNR | .0059015 / 22.290 dB | .0050227 / 22.991 dB | .0110154 / 19.580 dB |
| Endpoint 25 dB gate | Failed | Failed | Failed |
| Training time | 38.92 s | 37.04 s | 30.67 s |
| Registration Dice / HD95 / folding | Not evaluated | Not evaluated | Not evaluated |

Reducing the constant weight improved both endpoint fits, but both still failed
the gate. Delaying and ramping the .01 coefficient made the final endpoint fits
worse than either constant-weight run. At step 2,000, logged endpoint minibatch
MSEs were .000606/.000575 and patch loss/weight were zero. At step 2,020, raw
patch loss was 14.87, its weighted contribution .000992, and endpoint minibatch
MSEs had risen to .00153/.00321. At step 5,000, raw/weighted patch loss was
1.596/.01596 and endpoint minibatch MSEs were .00963/.01106. This is evidence of
an activation-associated failure for this seed, not a diagnosis of its cause.
The original training runs skipped recovery at the endpoint gate. A subsequent
explicit diagnostic bypass evaluated all three checkpoints with unchanged PCG+RK2;
see the [separate recovery report](differential_patch_ncc_recovery_20261111.md).
No component-gradient decomposition was performed.

For the constant .1 run, median logged block time after startup was 7.00 ms/step;
final sampled raw/weighted patch loss was .029158/.0029158, and total gradient
norm median/maximum before clipping was 48.64/7174.09. Its raw patch loss was
5.22 at step 500 and 6.24 at step 1,000, with weighted contributions .522/.624
against summed endpoint MSE .201/.212. Losses and logged gradients stayed finite,
and PD-LK remained zero. These scales and timing are observations; they do not
isolate objective conflict, normalization sensitivity or optimizer effects.

Dataset bytes, baseline checkpoints and existing results were preserved. Final
weights and scalar logs remain in ignored `results/differential_patch_ncc/`
subdirectories. Synthetic validation checks explicit projected systems,
minimized-objective finite differences, photometric nuisance removal,
constant-patch stability, native coordinates, and finite nonzero gradients into
both endpoint SIRENs. The .01-after-2,000 ramp schedule is also covered by a
synthetic unit test.
