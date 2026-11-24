# SIREN pipeline timing on four RTX 5090 GPUs

The four registrations finished in 95.4 seconds, down from 121.6 seconds with the original code. This is 1.28× the throughput, or 21.6% less elapsed time. Each GPU handled one independent registration from loading the input and fitting a fresh SIREN pair through recovery, scoring, and checkpoint serialization.

| Case | Original (s) | Optimized (s) | Speedup | Training, before → after (s) | Recovery, before → after (s) | Scoring, before → after (s) |
|---|---:|---:|---:|---:|---:|---:|
| OASIS pair 1 | 102.4 | 92.5 | 1.11× | 80.68 → 80.50 | 11.03 → 8.33 | 7.71 → 0.36 |

Each column is the mean of two complete timed runs. The four-GPU elapsed time includes interpreter startup and exit. The case timings include imports, image loading, training, applicable endpoint reconstruction, recovery, scoring and result writing. The stage columns omit loading and serialization. The matched run order was original, optimized, optimized, original.

The original batch took 123.11 and 120.18 seconds. The optimized batch took 94.82 and 95.97 seconds. Its scoring took 3.71 and 19.12 seconds across the two runs, compared with 0.18 seconds in both optimized runs. The observed per-case speedup therefore varied from 1.05× to 1.26× on that pair.

## Workload

All four inputs belong to their dataset’s frozen quick4 selection. Both versions used the same device assignment, 5,000 updates, seed 42, FP32 precision, particles, masks, loss weights, PCG tolerances and RK2 substeps. Each run started a fresh interpreter per GPU with warmed filesystem and compiler caches, four CPU threads per process and two Inductor compiler threads. The original and optimized source trees were copied from this workspace before and after editing. Checkpoint tensor names, shapes, dtypes, finite values, model and config were checked on all timed jobs.

## Implementation

Ordinary fixed-crop FP32 CUDA training replays the native box-NCC particle decoder and its backward pass in a case-local CUDA graph. The graph receives the current fields, particle starts and crop origin. It retains the finite checks and gradients through the native interpolation kernels. Sampling and optimizer updates use the existing code. The existing captured GW-NCC preset already replays its full update. Training also reuses grid extents and avoids constructing crop-exit diagnostics that the fitting loop discards.

Frozen 3D FP32 CUDA recovery uses a planar PCG stencil and block reductions for the same global linear system. It still checks the true residual and restarts when required. A Triton RK2 kernel follows a voxel’s trajectory through both border-extended trilinear samples for each midpoint step. Modest velocity histories stay in GPU memory when they fit within both 2 GiB and one twelfth of free device memory. Larger histories use transient mapped scratch. CPU, 2D and other fallback paths remain available.

Native Learn2Reg scoring performs nearest-neighbor label pullback, label overlap, central-difference Jacobians, and separable Euclidean surface distances on CUDA. Host code finds label boxes and applies the published area-weighted HD95 percentile to compact surfel records. The optimized CPU evaluator processes Jacobians in slabs and crops each label comparison to its support. Other label formats, larger grids, and working-grid point-callable evaluations can use that CPU path.

## Numerical checks

The OASIS SDlogJ differed by 1.39e-17. The scoring checks included half-voxel ties, half-voxel perturbations, missing labels, signed Jacobians and points outside the image.

The native training decoder produced identical objective values on five changing crops. The maximum relative L2 parameter-gradient difference was 2.31e-8. The fused RK2 output matched the original CUDA interpolation bit for bit on the three synthetic trials. OASIS Dice changed by −1.01e-7 and one OASIS interval took ten additional PCG iterations.

Fresh CUDA fits show run-to-run variation even at seed 42. The table below reports mean scores for the two timed fits of each version. The accuracy comparison covers these four cases. A 74-pair accuracy comparison remains unmeasured for this implementation.

| Case | Measure | Original | Optimized |
|---|---|---:|---:|
| OASIS pair 1 | Dice | 0.82894 | 0.82874 |
| OASIS pair 1 | HD95 (voxels) | 1.63528 | 1.64002 |

The largest peak PyTorch allocation among the four measured cases was 1.90 GiB originally and 2.52 GiB after optimization. All 192 PCG intervals in the timed runs met the configured true-residual tolerance.

Local measurements and provenance are in [comparison.json](../results/speed_round/comparison.json). The local benchmark harness is [.local/speed_round/batch.py](../.local/speed_round/batch.py). Per-case metrics, configs, timing records and endpoint checkpoints are under the corresponding run directories.

All four workers returned success, all 64 intervals converged, and all four cases wrote metrics and endpoint checkpoints. The launcher reported 78.5–84.0 seconds per case. These jobs are separate from the timed comparison. [Launcher summary](../results/speed_round/launcher_quick4/summary.json)
