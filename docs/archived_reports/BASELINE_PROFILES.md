# Baseline one-case runtime profiles

Four RTX 5090 GPUs, one baseline per GPU. Full audited configs were used, with one fitted seed per profile. Our method was not run.

Timings below include fresh-process startup, fitting/registration, inference, evaluation and checkpoint writing. GPU peak is PyTorch allocated memory, excluding context/driver allocations. CPU thread limits were OMP_NUM_THREADS=4 and MKL_NUM_THREADS=4 per process.

| Baseline | OASIS seconds | Peak allocated GiB |
| --- | ---: | ---: | ---: |
| fireants | 12.8 | 9.41 |
| convexadam | 11.4 | 7.00 |
| ccidir | 94.6 | 1.20 |
| nir | 78.6 | 7.57 |

These are one-case estimates, not measured full-cohort times or confidence intervals. Evaluation and other shared CPU work can contend across workers.

FireANTs used its official unfused PyTorch fallback. NIR used the audited 3000-step single-field adaptation, not the paper hybrid recipe. These estimates apply to the current presets only. No complete dataset or 50-seed sweep was launched.

Compact profiles, logs, configs, scores and permitted checkpoints are in ignored results/baseline_profile_20261112/. No deformation maps or warped volumes were saved.
