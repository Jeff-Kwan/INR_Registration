# Particle objective performance diagnosis

The initial implementation spent much of its time dispatching and recomputing
small operations around a dense Gaussian. The velocity model also has an
intrinsic cost: 4096 particles in each direction at two stages require 16,384
SIREN derivative sites, versus 4096 derivative sites for the default shared grid.
Optimizing the Gaussian can remove avoidable overhead but cannot eliminate that
model-evaluation difference.

## Controlled frozen-model measurements

These measurements use one RTX 5090, FP32, PyTorch 2.14.1+cu132, a frozen
width-256/four-layer checkpoint from the far phantom, 4096 particles per
direction and 32768 fixed endpoint samples. They measure objective plus backward
without optimizer steps, new fitting or recovery. Each stage is warmed up, then
reported as the median of ten synchronized repetitions. Compilation is excluded
from the warm numbers and retained separately in the JSON evidence.

| Implementation | Objective + backward | Derivative recomputation |
|---|---:|---|
| Original chunked Gaussian | 66.27 ms | On |
| Original with PyTorch efficient attention only | 30.01 ms | On |
| Exact fused Gaussian and deferred solve checks | 24.42 ms | On |
| Exact fused Gaussian, deferred checks, no repeated derivatives | **17.60 ms** | Off |

The final frozen objective is **3.76× faster**. True CUDA kernel launches in a
profiled objective/backward fall from 3547 to 1365. Tensor scalar reads fall from
16 to zero. The profiler also records annotations and memory transfers; those
are excluded from these kernel-launch counts. Peak allocated memory in the
combined timing rises from 1030 MiB to 2086 MiB when recomputation is disabled.
This trades modest additional memory for less repeated derivative work.

These are frozen-objective timings, not a measured new 5000-step training time.
The earlier registration scores do not establish the accuracy of a fresh fit
with the optimized execution.

## Why Gaussian averaging is attention

With centered physical coordinates `r_i = (x_i − mean(x)) ⊙ spacing / sigma`, set

```text
Q_i = [r_i, 1]
K_j = [r_j, −½ ||r_j||²]
V_j = [u_j, 0]
```

Use scaled-dot-product attention with `scale=1`, no dropout, and keep the first
three output components. Its logits equal the Gaussian distance logits plus a
query-only constant, which cancels in softmax. Coordinate centering reduces FP32
cancellation; both coordinate roles and the centering operation retain gradients.

PyTorch's efficient-attention CUDA backend supports FP32 on this installation.
The independent 4096-point probe measured about 1.53 ms forward/backward with a
four-component head, versus about 12.99 ms for the original Gaussian. Flash and
cuDNN attention reject FP32 here; no half-precision conversion was introduced.
Output and coordinate/velocity VJP relative errors against FP64 were roughly
1–2 × 10⁻⁶ at sigma 7.5 mm, increasing toward 2 × 10⁻⁵ at sigma 2.5 mm.

The production particle path specializes this same normalized Gaussian to three
coordinates and three velocity components. It uses one tiled forward and one
tiled backward launch, sums query and source position contributions without
atomics, and saves only positions, velocities, outputs, row normalization and
geometric scale. The integrated public blur plus its synthetic reduction/backward
measures 0.57 ms in the main harness; the standalone fused blur/VJP probe measures
0.28 ms. They are different timing scopes. No spatial cutoff, adjacency cap,
smoothing approximation or lowered precision is used.

## Other removed overhead

- Replace checked local LU reads with the existing shared-grid pattern:
  `solve_ex(check_errors=False)`, mask failed statuses to NaN, and let the
  trainer's pre-update finite checks reject the update. A fault-injection check
  covers failed solves that return finite values.
- Use the exact rank-one ridge formula for brightness-only velocity.
- Validate GPU geometric scales asynchronously instead of reading scalars back
  four times per objective. Shape validation remains immediate.
- Avoid five concatenations when derivatives already fit in a single chunk.
- Disable derivative recomputation in the particle preset, retaining the option.
  Existing compiled SIREN derivatives and NCC statistics remain reused. Frozen
  inference-only PCG/RK2 kernels are not substituted into live training graphs.

## Gradient checks and evidence

At the same trained checkpoint and identical 4096-particle positions, the
combined local objective differs by 5.03 × 10⁻⁷ relative. The concatenated VJP
for every model parameter and both particle arrays differs by 2.79 × 10⁻⁵
relative L2. Small floating-point changes can still alter a new training path.
CUDA checks include anisotropic spacing, odd/noncontiguous inputs, constant
fields, query/source coordinate gradients, higher derivatives, learnable geometry
and both LK ablations. The reference path remains live for higher-order requests.

- [Full before/after timings and parity](../results/particle_optimization_20261115T072646Z/comparison.json)
- [Reproducible frozen profiler](../results/particle_optimization_20261115T072646Z/measure_particle.py)
- [PyTorch attention backend audit](../results/particle_speed_audit_sdpa/report.json)
- [Head size and bandwidth audit](../results/particle_speed_audit_sdpa/heads.json)
- [Standalone fused blur/VJP evidence](../results/particle_blur_kernel_20261115/benchmark.json)

Input data and frozen reference geometry were preserved. No new training,
optimizer steps or recovery were run during this diagnosis. Sparse adjacency or
approximate smoothing remains a possible later scaling experiment; it was not
needed for the measured reduction at 4096 particles.

## Requested optimized training reruns

A subsequent user-authorized run completed three parallel fits of the far phantom
with both LK terms, width 256, 5000 updates and seed 42. The third arm inherits
the second arm's sigma schedule. All 48 recovery intervals converged, scratch
was cleaned, and sampled-map folding was zero in both directions for all arms.

| Particles per direction | Sigma mm | Forward/reverse foreground EPE, vox | Dice | Training including startup |
|---|---|---:|---:|---:|
| 4096 | Fixed 7.5 | 4.033 / 3.644 | 0.96893 | 112.66 s |
| 4096 | Linear 8→2 | 10.239 / 9.238 | 0.81337 | 113.12 s |
| 8192 | Linear 8→2 | 5.359 / 4.838 | 0.95164 | 160.79 s |

The fixed-sigma optimized fit is 3.29× faster than the earlier 370.57-second
combined-LK fit. Slight score changes remain consistent with the previously
measured floating-point differences; this is not a bitwise-reproduction claim.
Narrowing sigma worsens this particular case. Doubling particles partly repairs
that loss but does not beat fixed sigma. The scheduled 4096 run also loses about
10 dB reconstruction PSNR and reaches a logged pre-clipping gradient norm around
1.4 million, versus about 162 for fixed sigma. These are single-seed development
observations, not a general conclusion about sigma schedules.

GPU 3 separately profiled 160 real optimizer updates, retaining the 5000-step
schedule, with a warmed 20-update trace. Its unprofiled steady loop median is
17.91 ms/update, matching 17.94 ms/update from the full fixed-sigma run's logs.
Peak allocated memory is 2089 MiB. Of summed CUDA kernel time, matrix operations
use 60.2%, sine/cosine derivative kernels 16.5%, the exact Gaussian 9.3%
(1.11 ms/update), other kernels/reductions 13.7%, interpolation 0.18% and local
solves 0.075%. There are 1438 true CUDA launches/update. SIREN/derivative matrix
work is the dominant remaining compute cost.

The trace's 837 tensor scalar reads are mostly CPU Adam counters: 800 events.
Actual GPU reads comprise the 20 required pre-update finite guards and 17 scalar
logging values. Five synchronous host-to-device scalar copies remain per update:
four Euler endpoint-time tensors and a tensor created to log the known sigma.
Reusing device time constants and recording sigma directly on the host are
specific next dispatch improvements. The finite guard must remain. Profiler
instrumentation inflates trace wall time; its apparent busy fraction must not
be interpreted as normal training GPU utilization.

[Fit comparison](../results/particle_optimized_phantom_20261115T073952Z/analysis.json)
· [Training curves](../results/particle_optimized_phantom_20261115T073952Z/loss_curves.png)
· [Actual-loop profile](../results/profile_particle_training_20261115T074002Z/summary.json)
· [Dispatch attribution](../results/profile_particle_training_20261115T074002Z/dispatch_audit.json)

## Shared-middle batching for the selected particle preset

The selected independent bidirectional/shared-crop protocol evaluates both
middle clouds at the same stochastic time and with the same interpolated SIREN.
The implementation now concatenates these coordinates for one derivative/point
solve evaluation (up to twice the ordinary derivative chunk), then splits all
results before applying the two directional Gaussian kernels and LK means.
The number of evaluated sites and the mathematical objective are unchanged;
three derivative passes replace four for 4096 particles per direction. Gradients
to both middle coordinates and both endpoint networks stay live. Optional cycle
and independent-source-crop paths keep their separate transport evaluation.

Endpoint time tensors are allocated once per branch. Sigma logging now retains
host scalars instead of allocating two CUDA scalars on every training update.
The required finite-loss/trajectory/gradient guard remains before the optimizer.

The controlled comparison uses one RTX 5090, the frozen selected far-phantom
checkpoint, 4096 particles per direction, width 256, 32768 endpoint samples,
FP32, no derivative recomputation, and the same particles/time/crop on both
implementations. Warm measurements are medians of 50 synchronized repetitions;
compilation is excluded. No optimizer updates or recovery occur in this harness.

| Objective plus backward | Separate middle passes | Batched middle pass |
|---|---:|---:|
| Registration and LK | 16.23 ms | 14.56 ms |
| Including endpoint reconstruction | 17.43 ms | 15.75 ms |
| True CUDA kernels, combined profile | 1409 | 1189 |
| Combined peak allocated memory | 2043 MiB | 2043 MiB |

Combined warm wall time decreases 9.6%, and kernel launches decrease 15.6%.
Registration-only peak memory rises from 1575 to 1618 MiB. After batching, matrix
operations still account for approximately 58.2% of summed kernel time, sine/cosine
kernels 19.5%, and Gaussian averaging 9.6%. These sums do not measure normal
end-to-end GPU utilization; the profiler adds overhead.

CPU FP64 parity checks cover both trajectories, loss values, every model and
particle gradient, unequal particle counts, separate Gaussian neighborhoods,
brightness/differential ablations, native-mm anisotropy and recomputation. At the
trained FP32 checkpoint, three sigma stages give objective relative errors at
most 1.84e-6 and full parameter/particle VJP relative L2 differences at most
0.001044. Individual particle-array VJP differences reach 0.001867; all gradient
cosines exceed 0.9999994. Larger matrix batches change FP32 rounding, amplified
by sensitive local solves, so a subsequent fit need not be bitwise identical.

[Controlled timing, kernel counts and CUDA gradient evidence](../results/particle_batch_speed_20261115/comparison.json)
· [Reproducible harness](../results/particle_batch_speed_20261115/measure.py).

One matched seed-42, 5000-update far-phantom integration fit completed using the
selected sigma 0.4→0.1 schedule, both LK terms, endpoint/registration NCC and
independent shared-crop starts. Median steady update time, estimated from
20-update log intervals after step 200, decreased from 18.04 to 17.21 ms
(4.6%). This includes sampling, backward, clipping, finite checks, Adam and
logging. First cold-fit training time increased from 112.89 to 117.79 seconds:
the first 20 updates took 22.91 versus 32.50 seconds because the batched shape
adds compilation work. Warm throughput and cold one-case latency are therefore
different outcomes of this round.

Recovered forward/reverse EPE changed from 2.905/2.625 to 2.966/2.684 vox, and
Dice from 0.97291 to 0.97345. Peak logged pre-clipping gradient norm was 10.68
versus 9.57. Both fits had zero sampled-map folding and all 16 recovery intervals
converged. Recovery scratch was cleaned. The small accuracy differences are
single-seed numerical trajectory changes, not evidence of a new accuracy method.
[Optimized fit](../results/particle_batched_training_20261115T090410Z/summary.json).

## Whole-objective replay and fused Adam

A second execution round tested three candidates. Derivative-only CUDA graph
capture regressed the frozen combined objective from 15.75 to 20.54 ms and was
rejected. Capturing the full endpoint/particle objective and backward instead
measured 12.15 ms with identical fixed-input values and gradients. This avoids
repeated graph boundaries across derivative subcalls.

Fused Adam separately reduced actual 400-update steady timing from
16.747/16.926 to 16.515/16.146 ms across two runs per implementation on GPU2,
about 3% averaged across run medians. Identical synthetic gradients produced
maximum parameter difference 5.96e-8 after 100 updates. This is an execution
change with small FP32 optimizer rounding differences, not a new optimization
algorithm. [Optimizer evidence](../results/particle_optimizer_speed_20261115/comparison.json).

The selected preset now enables `capture_objective: true` and fused Adam. The
particle graph copies fresh endpoint coordinates/intensities, both start clouds,
stochastic split, bandwidth and effective loss coefficients into persistent
buffers. Its replay contains the objective, backward, gradient clipping and
finite-status calculation. Crop/mask-center and point/time sampling remain
ordinary, as do the pre-update finite guard, learning-rate schedule and optimizer.
Internal trajectory/solve/smoothing gradients remain live. Zero effective outer
weights use the ordinary inactive-loss-skipping path; captured gradient buffers
are reattached when returning to replay after an eager step. The graph is only
for the documented fixed-size independent shared-crop NCC execution. Other
particle ablations can disable it. The retained shared-grid default is unchanged.

Two matched seed42, 5000-step far-phantom fits used fused Adam, 4096 particles
per direction, width256 and sigma0.4→0.1. Both used the same existing warm
Inductor/Triton cache; these times are not cold compilation benchmarks.

| Actual training | Fused Adam, ordinary objective | Fused Adam, graph objective |
|---|---:|---:|
| Steady median step (20-update intervals after step200) | 16.31 ms | 12.87 ms |
| Complete training call | 88.72 s | 72.57 s |
| Forward/reverse EPE | 2.962/2.671 vox | 2.962/2.671 vox |
| Dice | 0.97188 | 0.97188 |

Graph replay reduces steady time another21.1% and complete training time18.2%
in this matched comparison. Relative to the previous round's ordinary-Adam
17.21 ms steady steps, the combined execution changes reduce time25.2%; that
comparison uses different runs. The two new complete checkpoints are bitwise
identical to each other. Both have zero sampled-map folding, all recovery
intervals converged, and recovery scratch was cleaned.

The opt-in CUDA test varies endpoint/particle inputs, split/sigma/weights and
live parameters, compares every clipped parameter gradient/preclip norm to
ordinary execution, verifies restoration after cleared gradient buffers, and
checks NaN status/recovery. CPU tests also verify that zero outer weights bypass
both capture and particle solves. Compilation cache reuse is worthwhile: the
previous experiment runner deliberately isolated per-arm caches; production
launchers do not force those empty caches. Reuse a compatible cache across fits
when measuring throughput, and report startup separately.

[Full-fit comparison and checkpoint equality](../results/particle_graph_training_20261115T091853Z/analysis.json)
· [Frozen whole-objective capture](../results/particle_batch_speed_20261115/whole_graph.json).
