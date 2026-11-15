# Compiling the Gaussian local-velocity surrogate

**On hold:** the [coarse-grid probe](gaussian_coarse_probe_20261111.md) finds
substantial loss of dense-field agreement despite faster execution.

Frozen LPBA S02 MSE/NCC checkpoints; no optimizer steps, training or inference
recovery. The reported passes include backward and gradient budgeting, not merely
batched inference. Same RTX 5090, 256 particles, four RK2 steps, 4 mm Gaussian,
4096-point derivative chunks and checkpoint recomputation as the eager profile.

| Grid | Earlier eager MSE/NCC | Compiled MSE/NCC |
|---|---:|---:|
| 16³ | 70.3 / 67.2 ms | 52.0 / 51.8 ms |
| 24³ | 233.0 / 222.5 ms | 155.6 / 154.5 ms |
| 32³ | 435.6 / 440.3 ms | 321.5 / 322.3 ms |

Compiled values are medians of five warmed full forward/backward passes, versus
three in the earlier eager profile; stochastic sites differ between these timing
runs. No Adam update or periodic reconstruction monitor is included. Compilation
improves observed full-pass throughput by about 1.3–1.5×, without eliminating the
eight derivative-field evaluations per active step. An NCC 24³ registration pass
had a .426 s outlier; its other warmed passes were .146–.147 s. Peak allocated
memory was about 975–979 MiB, excluding Adam state.

The first 16³ chunk forward/backward took 7.30 s; the 24³ shape introduced a
1536-point remainder chunk and took 9.34 s. These include compiler startup with
an existing local cache, not a clean-cache compilation benchmark. 32³ reused the
4096-point compiled shape. Time remains a tensor, avoiding separate compilation
for each RK2 time. Endpoint parameters remain live inputs, not frozen constants.

A separate identical-particle comparison checked all three grids on both models.
Registration loss relative disagreement was at most 0.0065%; endpoint gradient
relative L2 disagreement ranged 0.015–0.319%, with cosine at least .999995.
Both endpoint gradients were finite and nonzero. The agreement is numerical,
not bitwise, and does not establish stability through training. Compilation used
Inductor fullgraph on only analytical derivatives and the local solve; Gaussian
filtering and integration remain eager. No silent fallback is configured.

Enable explicitly with `objective.gaussian_registration.compile_velocity=true`.
The default stays false pending a training decision. When compilation is enabled,
the default first-use allowance becomes 60 s (an explicit startup_seconds overrides
it); the .25 s steady guard remains unchanged. No warm-up optimizer steps are used.
Final PCG+RK2 inference and maintained training presets are unchanged.

Validation: 64 medical and one Cameraman synthetic tests passed, including mocked
lazy compiler dispatch. Actual GPU compiled backward was exercised by the explicit
frozen-checkpoint experiment, not routine tests. Wheel audit passed (47 entries;
no datasets/results/tests/archives). Loaded parameter states and checkpoint/input
file hashes remained unchanged. No commit or push was made. Training is on hold.

Local runners: `results/gaussian_compile_profile/20261111/runner.py` and `parity.py`.
Timings and agreement evidence are in the accompanying compact JSON. Spatial
accuracy of the coarse lattice is a separate question, tested in the coarse probe.
