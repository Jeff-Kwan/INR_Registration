# Cached enhanced-PCG timing (2026-11-11)

User-authorized frozen LPBA S02 full-grid interval, t=0.53125, shape160×192×160,
RTX5090. Both implementations used c=4, s=.01, ridge1e-6, 500-iteration cap and
2e-5 tolerance. No adaptive threshold, detailed diagnostics, training, medical
image loading, label scoring or map integration. Checkpoint hash was unchanged.

The reference reconstructs the immediately preceding unassembled operator from
the current source by bypassing block assembly and applying the original
`g(g^T v)+cH^T(Hv)+sLv+ridge*v` expression inside each iteration. All other
solve settings and reporting code are shared. Sources and runner are retained
in the ignored run folder; the JSON records their source/checkpoint provenance.

| Quantity | Previous operator | Cached blocks |
|---|---:|---:|
| Median solve, three alternating repeats | 2.73598 s | 1.60269 s |
| Iterations | 350 | 350 |
| True relative residual | 1.7693e-5 | 1.7842e-5 |

Solve speedup1.707×, wall time reduced41.42%. Timings synchronize CUDA and include
block construction, validation, and basic scalar diagnostics. Each solver had one
untimed warmup. Cache construction retains about169 MiB more at this grid size.
Peak allocated memory in raw rows also depends on whether the comparison output
is retained, so those peaks are not an isolated memory-cost comparison.

Velocity relative L2 difference4.20e-6, maximum absolute difference0.001245 native
voxels per normalized time, cosine0.99999994. The two reported residuals use the
original unassembled equation, not just recursive CG residuals.

SIREN derivatives took3.9449 s on first call and1.2057 s on the warmed call.
Adding the warmed derivative time to the solve medians gives3.9417→2.8084 s
(~1.40×), but that is a sum of measured components, not an integrated recovery
benchmark; RK2, scratch I/O and scoring are excluded. No claim is made for other
times, datasets, adaptive settings, or final registration metrics.

The basic and detailed diagnostics compute reductions on GPU. Conversion of many
individual scalars to Python floats triggers GPU/CPU synchronization; batching
those scalars before a single transfer remains a separate optimization and was
not included in this timing comparison. Detailed diagnostics were disabled, so
they do not explain the measured gain here.
