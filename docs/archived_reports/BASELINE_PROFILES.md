# Baseline one-case runtime profiles

Four RTX 5090 GPUs, one baseline per GPU. Each method ran sequentially on DIR-Lab case06, the first official AbdomenCTCT validation pair, and the first official OASIS validation pair. Full audited configs were used, with one fitted seed per profile. Our method was not run.

Timings below include fresh-process startup, fitting/registration, inference, evaluation and checkpoint writing. GPU peak is PyTorch allocated memory, excluding context/driver allocations. CPU thread limits were OMP_NUM_THREADS=4 and MKL_NUM_THREADS=4 per process.

| Baseline | DIR-Lab seconds | Abdomen seconds | OASIS seconds | Peak allocated GiB | Full cohort serial hours |
| --- | ---: | ---: | ---: | ---: | ---: |
| fireants | 27.7 | 12.6 | 12.8 | 9.41 | 0.30 |
| convexadam | 7.8 | 8.7 | 11.4 | 7.00 | 0.19 |
| ccidir | 80.1 | 63.7 | 94.6 | 1.20 | 12.42 |
| nir | 308.4 | 86.1 | 78.6 | 7.57 | 2.35 |

All 12 profile runs succeeded. Full workload: 786 fits. Counts are 10 DIR-Lab + 45 abdomen + 19 OASIS for FireANTs, ConvexAdam and NIR; ccIDIR uses 500 DIR-Lab fits (50 per case, both directions from each fit) + 45 abdomen + 19 OASIS.

Linear extrapolation: 15.26 serial GPU-hours; 3.81 hours under ideal four-GPU scheduling. Reserving one GPU permanently per method takes approximately 12.42 hours. A practical planning allowance is 4–5 hours when distributing independent fits across all four GPUs.

These are one-case estimates, not measured full-cohort times or confidence intervals. DIR-Lab case06 represents the larger scans; the five smaller scans may finish faster. ccIDIR training has fixed iteration count but its native mask sampling cost varies with foreground size. Evaluation and other shared CPU work can contend across workers.

FireANTs used its official unfused PyTorch fallback. NIR used the audited 3000-step single-field adaptation, not the paper hybrid recipe. These estimates apply to the current presets only. No complete dataset or 50-seed sweep was launched.

Compact profiles, logs, configs, scores and permitted checkpoints are in ignored results/baseline_profile_20261112/. No deformation maps or warped volumes were saved.
