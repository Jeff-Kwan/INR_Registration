# OASIS recovery profile

Measured 2026-11-12 on one idle RTX 5090 using /venv/main PyTorch 2.14.1+cu132. Saved case: `0395_to_0396` from `results/oasis_ours_20261112_123205/ours/`. No training or parameter changes.

## Measurement

Loaded the fitted checkpoint and ran the existing recovery once as a reference, then ran a copy of that exact function instrumented via AST with synchronized stage timers. The equations, order, warm starts, tolerances, 16 intervals, 500-iteration cap, four RK2 substeps and 65536-point derivative chunk were preserved. Forward and inverse maps matched exactly (maximum absolute difference zero); all diagnostics matched. Recovery scratch was verified empty afterward. No maps or reconstructed volumes were saved.

- First/reference recovery: **26.02 s**, including first-use compilation/setup.
- Warm stage-timed recovery: **23.17 s**. Synchronization can add measurement overhead; this is one case, not a cohort benchmark.
- Original case recovery **plus evaluation** was 28.99 s; the profile below measures recovery alone.

## Recovery breakdown

| Stage | Seconds | Share |
| --- | ---: | ---: |
| PCG solves | 16.536 | 71.37% |
| Dense SIREN derivatives | 4.261 | 18.39% |
| GPU→CPU transfer and scratch writes | 1.504 | 6.49% |
| Scratch flush | 0.411 | 1.78% |
| Scratch reads and CPU→GPU transfer | 0.212 | 0.91% |
| Forward RK2 | 0.096 | 0.41% |
| Inverse RK2 | 0.096 | 0.41% |

Other setup/cleanup took 0.054 s. The derivative stage includes parameter preparation, coordinate generation, chunk dispatch and writing derivative grids. Transfer/write and read/transfer stages are combined measurements; this profile does not claim to isolate disk bandwidth from transfers or memory copies.

## PCG bottlenecks

A separate CPU/CUDA operator trace covered the first interval using the unchanged solver/settings. It stopped at 420 iterations (cap 500): approximately 13715 CUDA kernels, or 33 kernels/iteration; 1.466 s summed kernel execution.

- Device-time attribution: multiplication about 30%; in-place add/subtract about 37%; other add/subtract about 23%; division about 5%; reductions about 4%. These are operator-level shares, not an instruction-level roofline measurement.
- The matrix-free operator repeatedly applies component products and a finite-difference Laplacian, with full-volume temporary arrays and separate kernels for slice differences, additions and vector updates. The trace is consistent with a high cost from memory traffic; bandwidth saturation itself was not measured.
- CUDA launch API time was about 49 ms across 13715 launches. CPU stream-synchronization time was about 1.29 s, mostly waiting for GPU work. Do not add that waiting time to GPU time: they overlap. Convergence checks every ten iterations and final scalar diagnostics force host/device synchronization.
- Across all 16 intervals there were 4700 iterations. Only 1/16 intervals passed the final true-residual check. Residuals ranged from 1.996e-5 to 2.421e-5 against tolerance 2e-5. The solver can stop when its recursively updated residual passes tolerance, then report failure when the separately recomputed true residual is slightly larger. These intervals did not all exhaust the iteration cap.

## Priorities

1. First target PCG: fuse the Laplacian/operator and vector updates to reduce full-volume passes and temporary allocations, while preserving the same system and checking true-residual/solution parity.
2. Investigate recursive versus true-residual stopping before changing iteration budgets or tolerances. Faster code must not mask convergence differences.
3. Dense derivative construction is the next-largest stage (18%). Optimize it only against the already compiled, prepared-parameter path.
4. Scratch/transfers account for roughly 9%; bounded buffering or overlap may help, while preserving scratch cleanup and the prohibition on persistent maps.
5. RK2 is below 1% combined and is not the present optimization priority. Running all 16 solves concurrently is not justified by this profile alone; each solve already performs large parallel GPU operations and concurrency would multiply working buffers.

Raw timing JSON, one-interval trace and operator table: `results/oasis_recovery_profile/` (ignored). The standalone instrumentation driver is `/tmp/profile_oasis_recovery.py`.

## Implemented PCG optimization (2026-11-12)

The shared solver now verifies the independent true residual before accepting
convergence. If only the recursive residual passes, it replaces the residual
and restarts the preconditioned search direction within the original budget.
Zero right-hand sides return zero velocity; initial guesses are checked before
iteration. Diagnostics additionally report recursive residual, restart count
and termination reason. The objective, FP32 fields, scalar Jacobi preconditioner,
2e-5 tolerance, 500-iteration cap, temporal warm starts and RK2 are unchanged.

A tensor-only iteration fuses the clamped-neighbor positive Laplacian, operator,
reductions and vector updates with torch.compile(fullgraph=True). The Python loop
and ten-iteration convergence checks remain outside compilation. The independent
eager edge-difference operator supplies true-residual checks. CPU remains eager. Pass compile_solver=False
to either entrypoint for the corrected eager reference; no checkpoint or config
migration is needed. No external-baseline solvers are changed.

### Acceptance profiles

Saved first/last OASIS pairs were recovered with identical existing settings,
without training. Each execution used temporary scratch, verified clean afterward.
Comparison below is corrected eager versus corrected compiled, not historical
recursive-only stopping. Registration evaluation is outside these recovery times.

| Pair | Eager PCG | Compiled PCG | Eager recovery | Compiled recovery | Incremental peak MiB, eager / compiled |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0395_to_0396 | 17.115 s | 4.148 s | 23.532 s | 10.581 s | 1470.0 / 1311.7 |
| 0413_to_0414 | 17.023 s | 4.161 s | 23.467 s | 10.590 s | 1470.0 / 1311.7 |

PCG is approximately 76% faster and incremental peak allocation approximately
11% lower, passing the 25% speed/10% memory-increase gates. Full warm recovery is
approximately 55% faster. These are two-case profiles, not a repeated cohort-wide
benchmark. The compiled operator can change floating-point operation ordering;
bitwise parity is not expected.

Both compiled runs converged in all 16 intervals. First pair used 4850 iterations
and 16 restarts in both modes; last pair used 4830 iterations/8 restarts eager and
4860/11 compiled. Largest map RMS difference was 1.35e-5 voxel. First-pair DSC and
HD95 were identical; last-pair DSC differed by 1.31e-6 and HD95 was identical.
The old first-pair recovery used 4700 iterations and reported only 1/16 converged:
that historical stopping behavior is superseded, and its cohort scores have not
been silently replaced with these new two-case measurements.

An equal 20-iteration OASIS trace reduced CUDA kernel count from 712 to 212
(including setup/diagnostics), and summed kernel time from 73.16 to 20.15 ms.
A fresh temporary compiler cache gave a 4.70 s first call versus 0.0212 s warm
for that 20-iteration solve; compilation is excluded from warm recovery timings.

Validation includes independent assembled FP64 2D/3D direct solves, all boundary
impulses, initial guesses, zero RHS, iteration-limit reporting, injected optimistic
recursive residuals, compiled FP32 fixed-iteration agreement below 1e-3 relative
L2 error, and scratch cleanup on exceptions. The focused solver suite passed all
five checks with CUDA enabled. The medical suite passed 38 CPU checks, with six
opt-in CUDA checks skipped.

Timing/diagnostic/metric JSON and equal-length kernel traces are under ignored
results/pcg_optimization/. Temporary profiling drivers are
/tmp/profile_pcg_updated.py, /tmp/pcg_trace_new.py and /tmp/pcg_cold.py.
