# Complete OASIS training-step profile

Measured 2026-11-12 on one idle RTX 5090, PyTorch 2.14.1+cu132, FP32/highest matmul
precision, four CPU threads. Input is official OASIS pair 0395→0396, normalized
exactly as the current adapter. No dataset run or recovery was launched.

## Methodology

The current fit_pair function was copied via AST into a temporary profiler.
Only its loop length and profiling hooks were changed: 500 disposable-model
optimizer updates per timing run, with the first 100 excluded from steady-state
statistics. The configured 5000-step learning-rate schedule and every loss,
sampling, clipping and model setting remained unchanged. Seed 42 was reset by
fit_pair for each run. Report callbacks serialized the ordinary diagnostic rows.
No checkpoints, reconstructed images or deformation maps were saved.

Timing runs used one GPU sequentially: baseline, baseline repeat, fused-Adam
candidate and stage-instrumented baseline. Each starts a fresh model. An additional
120-update run traced five active steady-state steps; profiler overhead makes
that trace unsuitable for absolute wall-time claims. These short runs sample
early training rather than the complete 5000-update trajectory.

## Full-step timing

| Variant | Steady-state ms/update | Peak allocated MiB |
| --- | ---: | ---: |
| baseline | 17.013 | 1011.5 |
| baseline_repeat | 17.044 | 1011.5 |
| fused_adam | 16.599 | 1011.5 |
| instrumented | 17.284 | 1011.5 |

The first 500-update baseline run took 15.42 s versus 8.47 s on repetition,
consistent with roughly 7 s of first-use compilation/library setup. This uses
existing compiler disk caches, not a deliberately empty cache. The 17.0 ms warm
step predicts about 85 s for 5000 updates; adding first-use overhead is consistent
with the previously observed OASIS mean training time of 93.66 s.

## Stage breakdown

CUDA-event spans include kernel execution and gaps waiting for host dispatch;
CPU wall spans can include waiting for earlier GPU work. They must not be added
together. Event instrumentation raised the full step only from ~17.04 to 17.28 ms.

| Stage | Mean CUDA-event span, ms | Mean CPU wall span, ms |
| --- | ---: | ---: |
| sampling | 0.409 | 0.390 |
| endpoint_forward | 1.021 | 0.512 |
| registration_forward | 5.592 | 6.018 |
| finite_check | 0.066 | 0.052 |
| zero_grad | 0.054 | 0.044 |
| backward | 9.387 | 8.774 |
| gradient_clip | 0.152 | 0.608 |
| adam_and_lr | 0.360 | 0.343 |
| logging | 0.013 | 0.005 |

Registration forward and combined backward account for approximately 87% of the
instrumented elapsed step. Backward covers all losses together; this measurement
does not pretend to separate per-loss gradients by timing extra backward passes.

## Operator evidence

The five-step trace contains approximately 1069 GPU kernels per update and 5.34 ms
of summed kernel execution per update. Correlation IDs attribute approximately:

| Phase | Kernels/update | Summed kernel execution/update |
| --- | ---: | ---: |
| Sampling | 37 | 0.058 ms |
| Endpoint forward | 19 | 0.737 ms |
| Registration forward | 455 | 1.191 ms |
| Combined backward | 533 | 3.239 ms |
| Clipping | 14 | 0.026 ms |
| Adam and LR | 7 | 0.080 ms |

The trace itself is much slower (~34 ms/update) than the untraced baseline, so
its idle gaps cannot quantitatively establish untraced GPU utilization. However,
the kernel counts and untraced event spans identify substantial framework/kernel
dispatch overhead around small registration operations in addition to real GEMM
work. GEMMs dominate device execution among individual kernels. GPU sampling and
Gaussian/trajectory/NCC operations are distributed across many launches.

Explicit CUDA launch API calls took ~3.59 ms/update in the instrumented trace;
CPU stream synchronization ~0.10 ms/update. These are trace-overhead-affected
numbers and overlap other timings. Removing safety checks alone is not supported
as the principal optimization. The math kernels already have substantial fusion.

## Value-for-effort conclusions

1. Highest potential: investigate a compiled tensor-only boundary around shared
   field smoothing, bidirectional particle transport and box NCC, plus their
   backward. Dynamic valid-particle indexing/count decisions currently interrupt
   such fusion. Fixed-size masked reductions could help, but require exact
   validity, skip, epsilon and gradient parity checks. Preserve all safeguards.
2. Fused Adam is easy but small: 17.04→16.60 ms/update (~2.6%), about 2.2 s over
   5000 steps. It was tested only as a disposable profiling candidate, not enabled
   in production and not validated as a numerical reproduction.
3. Sampling, clipping and logging are minor elapsed contributors. They are not
   the best first targets. Retain gradient clipping and finite-value checks.
4. Custom SIREN kernels or reduced precision are not justified by this profile
   before reducing fragmentation around the existing compiled kernels.

No production training implementation or configuration was changed by this
profile. Results and trace: results/oasis_training_profile/ (ignored). Temporary
drivers: /tmp/profile_training_full.py and /tmp/profile_training_trace.py.

## Shared reconstruction/registration particles, 2026-11-13

Disposable real-input profile on OASIS 0395→0396, one idle RTX 5090, four
CPU threads, current FP32 compiled kernels. Endpoint objective: NCC plus
Gaussian moments 0.01, PD-LK 0.1 and registration NCC 0.01; two Euler
training half-steps, 80 mm crop, 8³ lattice and two antipodal probe pairs/cell.
No production sampler or preset was changed by this profile.

Three fresh-model runs per variant used 50 warmup updates then 100 measured
updates. Confirmation used 50 warmups and 500 measured updates in reverse
variant order. The 5000-step learning-rate schedule, finite checks, gradient
clipping and Adam were retained. Full wall timings include sampling, forward,
backward, clipping, optimizer and CUDA-event recording; compilation is excluded
from measured windows. Events introduce some overhead. Early disposable fitting
does not establish full-run convergence or registration accuracy.

| Variant | Median of three short runs (ms/step) | 500-step confirmation | Peak allocated MiB |
| --- | ---: | ---: | ---: |
| 32k reconstruction, separate 4k particles/direction | 17.92 | 18.75 | 1011.49 |
| 32k reconstruction, separate 32k particles/direction | 16.49 | 16.87 | 1011.71 |
| Shared 32k points for reconstruction and both directions | 17.00 | 18.38 | 1011.83 |

Shared means **32768 unique continuous start positions**, reused for both endpoint
reconstruction and both directions (65536 directed registration trajectories).
Targets are trilinearly sampled from the observed endpoint volumes. Reconstruction
therefore samples the central 64 mm particle-start region, instead of discrete
voxels throughout the 80 mm reconstruction crop. Separate-32k retains independent
directional starts. The branch's configured particle counts are asserted in the
confirmation. Its prototype draws both ordinary directional arrays then reuses
the first in shared mode; no cached cross-step tensors or graphs are retained.

The longer shared measurement's CUDA stage means: sampling 0.45 ms, endpoint
forward 1.04 ms, registration forward 5.94 ms, finite-check/zero-grad/backward
10.38 ms, clipping/LR/Adam 0.55 ms. Wall and event spans need not sum exactly.

Practical estimate: **17–19 ms/update**, roughly 85–95 s for 5000 warm updates.
Observed run-to-run spread prevents claiming a speedup from tying samples.
Memory changes by less than 1 MiB: derivative probes remain the same 2048
points per time, and particle interpolation/NCC adds relatively little work.
Tying reconstruction and registration points does not merge the SIREN endpoint
forwards with the analytical derivative computation at separate probe locations.

Timing-only results and exact disposable scripts:
`results/shared_particles_profile_20261113/` and
`results/shared_particles_profile_20261113_confirmation/`. No checkpoints,
deformation maps, warped images or recovery results were written.

## Registration forward/backward investigation, 2026-11-13

Current RK2, separate 32768 particles per direction, global endpoint NCC preset.
One OASIS 0395→0396 trained checkpoint, current physical 80 mm crop, FP32/highest,
one RTX 5090, four CPU threads. Registration-only loss .1 PD-LK + .01 box NCC;
no endpoint loss, optimizer steps, recovery, checkpoints or images written.
Fresh forward/backward each iteration; random probes/times/particles retained.
20 warmups + 100 measured iterations per block, interleaved/reversed mode order.
Event timings include CPU dispatch gaps. Early blocks fluctuated substantially
(roughly 12–20 ms), so raw logs are retained and no whole-step speed claim made.
Later interleaved baseline/both/baseline/both blocks were stable:

| Registration path | Forward ms | Backward ms | Total wall ms |
| --- | ---: | ---: | ---: |
| Current baseline | 6.61 | 4.95 | 11.58 |
| Compiled smoothing + masked decoder prototype | 3.68 | 3.61 | 7.30 |

Potential saving: 4.28 ms (~37%) for registration plus backward, approximately
21 seconds per 5000 updates if retained in whole-step execution. This remains
an unaccepted prototype, not a production speedup. Allocated peak was essentially
unchanged (~500 MiB). Single later smoothing-only block: 11.24 ms; decoder-only:
7.87 ms. The decoder accounts for most observed benefit.

Smoothing prototype replaces tiny grouped convolutions with reflect-indexed
three-tap stencils compiled together. Value max difference 1.19e-7; VJP relative
L2 ~8e-8. Decoder prototype compiles fixed-size masked centered moments with
bidirectional RK2 per direction, avoiding dynamic particle indexing. Finite
trajectory checks remain outside compilation. Only unmasked cropped box NCC is
covered; DIR-Lab masks, skip cases, uncropped boundaries and full config support
require dedicated acceptance before implementation.

Complete parameter VJP on identical trained-checkpoint probes/times/particles
matched at relative L2 4.95e-7; loss difference 4.66e-10. Nevertheless, large random
field stress tests FAILED the 1e-3 VJP gate. A fixed-seed diagnostic found eager
reformulated decoder vs reference ~5e-7, compiled vs reference up to 0.0266.
Loss differences were below 5e-7 and valid counts agreed. Disabling compiler
floating-point contraction through emulate_precision_casts did not fix this.
Interpolation coordinate-gradient discontinuities are a hypothesis, not an
established root cause; ordinary small rounding noise does not justify accepting
the observed VJP discrepancy.

Recommended next work: isolate the compiled interpolation/trajectory mismatch
with intermediate coordinate and input-gradient comparisons; retain native
sampling kernels at a compiler boundary if needed. Then validate fixed-size
masked reductions, min-count skip semantics, masks and nonfinite safeguards;
reprofile the complete training step. Compiled smoothing alone offers a smaller,
cleaner independent candidate. No production implementation/configuration changed.
Exact scripts, parity outputs and timing blocks:
`results/registration_speed_round_20261113/`.

## Accepted compilation boundary (2026-11-13)

Production now compiles three-tap Gaussian smoothing and fixed-size masked box
NCC statistics when CUDA and compile_velocity are enabled. RK2 trajectory arithmetic,
normalization and native image/field sampling remain eager. GW-NCC retains its
previous path. Existing finite checks, direction masks, min-valid count, skip and
uncropped boundary behavior remain outside compiled reductions. No gradients detached.

Keeping sampling alone native did not resolve the 2.7% stress VJP discrepancy,
even with emulate_precision_casts. Keeping trajectory arithmetic native as well
reduces stress VJP errors below 6e-7. This isolates a stable boundary but does not
establish the exact compiler-level cause of the fully compiled trajectory failure.
Do not re-enable the full RK2 compiled prototype without resolving that issue.
Complete endpoint-parameter VJP relative error: 2.421e-7. Compiled/native smoothing
and masked statistics passed CPU/CUDA value/VJP checks including 0,1,31,32,257
valid samples; existing RK2, masked crop and derivative checks passed (20 tests total).

Registration-only timing blocks (200 repetitions after 30 warmups, no optimizer):
reference/compiled = 14.32/11.85, 13.58/13.46, 11.51/10.46 ms. Significant
run-to-run host noise remains; final pair suggests about 9% gain, roughly 5 s
per 5000 updates if sustained. This is smaller than the unaccepted fully compiled
prototype's 37% improvement and is not a measured whole-training speedup.
Scripts and outputs remain in results/registration_speed_round_20261113/.

## Full decoder default and quick4 rollout (2026-11-13)

At the user's explicit request, CUDA compile_velocity now enables the full box-NCC
RK2 decoder, superseding the narrower accepted boundary. Three-tap smoothing is
also compiled. The eager reference remains available by disabling compile_velocity.
CPU and GW-NCC retain eager decoding. Fixed-size masked centered statistics preserve
independent direction masks, minimum valid count, skipped-direction policy and
uncropped boundary term; nonfinite trajectory detection remains a host decision.
Disposable synthetic RK2 tests showed ordinary-field VJP parity ~1e-6, but extreme
random fields amplify rounding differences across stages and can fail 1e-3 VJP
parity. This limitation remains; the full decoder is not bitwise interchangeable.
No gradient detaches or precision reduction were introduced.

Frozen OASIS quick4, seed 42, same best global NCC separate-32k config, 5000 steps,
1 mm training/recovery, standard brightness-only PCG+RK2:

| Measure | Previous native decoder | Full compiled decoder |
| --- | ---: | ---: |
| DSC | 0.818590 | 0.816500 |
| HD95 | 1.915278 | 1.942831 |
| SDlogJ | 0.079323 | 0.079428 |
| Folding (%) | 0.003014 | 0.002945 |
| Training seconds/case | 101.81 | 80.84 |
| End-to-end seconds/case | 127.82 | 107.11 |
| PCG convergence | 64/64 | 64/64 |

| Pair | Previous DSC | Compiled DSC |
| --- | ---: | ---: |
| 0395→0396 | 0.820054 | 0.818804 |
| 0398→0399 | 0.793508 | 0.790872 |
| 0402→0403 | 0.826576 | 0.822144 |
| 0403→0404 | 0.834221 | 0.834179 |

Training is 20.6% faster in these separate runs; this is not an interleaved speed
benchmark. Mean DSC decreases 0.00209 (0.21 percentage points), with all four
slightly lower; one seed cannot establish unchanged accuracy or statistical
significance. No crashes, nonfinite failures or PCG convergence regression.
Synthetic decoder direction/mask/all-skipped checks and CPU/CUDA reduction and
smoothing checks passed. Source: results/oasis_quick4_fullcompiled_decoder_ncc_20261113/.

Full RK2 decoder compilation was subsequently disabled by default at user request. Optional `objective.registration.compile_decoder: true` explicitly enables it on CUDA; absence means false. Existing compile_velocity continues to control compiled derivative/solve, smoothing and masked box-NCC statistics, with native RK2 and sampling. No new training run was launched.
