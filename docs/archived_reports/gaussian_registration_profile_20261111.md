# Gaussian local-velocity registration loss: profiling only

Follow-up: compilation improves speed, but the [coarse-grid probe](gaussian_coarse_probe_20261111.md)
invalidates treating 16³ as a faithful dense-smoothing approximation.

Status: opt-in pipeline prepared; **training on hold**. No optimizer steps,
checkpoint writes, recovered maps or registration evaluation were performed.

LPBA S02→S01, existing MSE+PD-LK and NCC+moments (.01)+PD-LK checkpoints,
RTX 5090. Full-size endpoint minibatch 32,768; PD-LK samples 4,096.
The extra branch uses 256 particles, four midpoint RK2 steps (eight velocity
lattices), local strict PD-LK c=4/eta=1e-6, a 4 mm Gaussian, analytic derivative
chunks of 4,096 and nonreentrant activation recomputation. No derivative compilation
in the extra branch; existing compiled PD-LK remains unchanged.

| Lattice | MSE forward/backward | NCC forward/backward |
|---|---:|---:|
| Baseline | 7.0 ms | 7.7 ms |
| 16³ | 70.3 ms | 67.2 ms |
| 24³ | 233.0 ms | 222.5 ms |
| 32³ | 435.6 ms | 440.3 ms |

Medians of three warmed passes, including base and registration gradients,
separate per-endpoint gradient budgets and global clipping. No Adam update,
periodic reconstruction monitor, image loading or recovery is included. This is
an approximate active-step cost, not measured end-to-end training throughput.
Registration-only warmed ranges were 59–230 ms (16³), 215–233 ms (24³),
and 427–585 ms (32³); occasional latency outliers make three repeats preliminary.
First compiled baseline pass took 2.32 s; first Gaussian MSE 16³ pass took .129 s.
Peak allocated CUDA memory was approximately 975–976 MiB including endpoint data,
models and base graphs; optimizer state was absent. These runs do not establish
training memory consumption.

All registration gradients were finite and nonzero at both endpoints. Weighted
raw endpoint gradient norms across warmed passes ranged .0196–.339; the separate
budget capped each applied contribution at 10% of that endpoint's base gradient
norm. This bounds a gradient norm, **not an Adam parameter update**. Both loaded
parameter-state hashes were identical before/after profiling. Inputs and saved
checkpoints were only read. MSE loss and normalized out-of-bounds penalty are
reported separately in the scalar JSON; border sampling is explicit.

The branch is much faster than the historical roughly 62 s/step experiment,
but still about 9–10× baseline at 16³ and 29–63× at larger lattices. A 16³
lattice is spaced roughly 11–13 mm on this grid: it can undersample local
velocities and does not inherit the earlier dense Gaussian-versus-PCG agreement.
No claim of PCG equivalence, stability over training, or registration improvement
is supported. Keep training on hold pending a deliberate choice to test this
tradeoff. If resumed, 16³ is the initial runtime-feasible candidate; 32³ exceeds
the default .25 s active-branch guard. No coefficient was selected by labels.

Implementation: `medical/registration/gaussian_registration.py`; opt-in
`objective.gaussian_registration`. Original training presets, endpoint objective,
strict PD-LK and final PCG+RK2 recovery remain unchanged. Activation is after step
2000; defaults weight .01 and per-endpoint gradient budget .1 are unvalidated.
The implementation aborts on nonfinite values, runtime excess, and sampled MSE
reconstruction below 25 dB (NCC respects the explicit skipped PSNR gate).
The sampled gate checks before an update every 20 steps, so it cannot guarantee
that a subsequent update is safe. Warm-up and weight zero skip registration.

Synthetic tests cover constant-field direction, Gaussian preservation, checkpoint
recomputation agreement, finite differences through RK2/solves to both endpoints,
shear rejection, configuration validation, and inactive gradient equivalence.
This is not a long-trajectory test or a repeat of the earlier collapse diagnosis.

Exact no-update runner and scalar evidence:
`results/gaussian_registration_profile/20261111/runner.py` and `summary.json`
(ignored local evidence). The first launch used the filename `profile.py`, which
shadowed Python's standard module and failed before any measured pass; renaming
the runner resolved it. No optimizer existed in either launch.

Validation after integration: 63 medical synthetic tests and one Cameraman
synthetic test passed. CLI help and `git diff --check` passed. The wheel contains
47 entries, including the new module, with datasets/results/tests/archives excluded.
No commit or push was made.
