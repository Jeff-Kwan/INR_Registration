# Discarded recovery and probe experiments (2026-11-12)

Historical record only; these alternatives are unsupported and their experimental
code, configs, checkpoints and generated results were removed at the user's request.
The retained method uses random two-pair probes and global PCG+RK2 recovery.
No defaults were changed based on these discarded experiments.

## Deterministic three-axis probes

OASIS frozen quick4, single seed 42, 5000 steps, 1 mm working grid and 80 mm
crop, PD-LK 0.1 + box NCC 0.01. Six probes at ±2 mm on XYZ axes replaced four
probes along two random directions.

| Mean metric | Random two pairs | Three axis pairs |
| --- | ---: | ---: |
| DSC | 0.770455 | 0.768211 |
| HD95 | 2.329283 | 2.344937 |
| SDlogJ | 0.050633 | 0.050460 |
| Moving/fixed PSNR dB | 34.060/34.378 | 34.083/34.423 |
| End-to-end seconds per case | 119.83 | 127.12 |

Both had zero measured folding and 64/64 converged PCG intervals. Axis probes
had lower DSC on all four pairs. A disposable step profile measured overlapping
warm wall-time ranges (random 17.22–18.75 ms, axes 17.22–17.49 ms), but peak
allocation increased from 1011 MiB to 1205–1218 MiB. Count and arrangement changed
together, so this does not isolate the orientation effect.

## Pooled local velocity recovery plus Gaussian smoothing

OASIS 0395→0396, frozen random-two-pair physical-grid checkpoint; 1 mm recovery,
16 temporal intervals, same RK2 integration. Equal-weight 3×3×3 pooling of
Q=ggᵀ+4HᵀH and b=gq+4Hᵀm, ridge 1e-6, then Gaussian velocity smoothing.
Replicated boundaries for pooling and reflected boundaries for smoothing.

| Recovery | DSC | HD95 | SDlogJ | Folding | Recovery seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Global PCG | 0.781931 | 1.950593 | 0.042914 | 0% | 13.13 |
| Local, sigma 1 mm | 0.763600 | 2.228452 | 0.402857 | 0.454% | 26.46 |
| Local, sigma 7.5 mm | 0.758869 | 2.151688 | 0.048265 | 0% | 23.42 |

Final-map RMS differences from PCG were 4.08 and 1.46 mm, respectively.
Dense differential derivatives consumed 23.69/20.84 seconds; pooling, solves and
smoothing about 0.31/0.32 seconds. Evaluation added about 6.1 seconds each.
Timings include first-use overhead, not a controlled warm benchmark. No maps
were saved. These one-case results do not establish general performance.
