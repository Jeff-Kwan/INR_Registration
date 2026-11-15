# Fixed AbdomenCTCT quick4 prototype set

Selected once with `random.Random(42).sample(range(1,46),4)`, then sorted,
from the official 45 directed public-validation pairs. Selection uses no scores.
The frozen manifest is `medical/protocols/abdomenctct_quick4.json`; it records
exact pair paths and the official protocol SHA-256. Do not redraw the subset.
These are development pairs, not an independent held-out validation cohort;
subjects may overlap between pairs. Labels remain evaluation-only.

| Official pair index | Fixed image | Moving image |
| --- | --- | --- |
| 2 | AbdomenCTCT_0001_0000.nii.gz | AbdomenCTCT_0007_0000.nii.gz |
| 8 | AbdomenCTCT_0001_0000.nii.gz | AbdomenCTCT_0025_0000.nii.gz |
| 18 | AbdomenCTCT_0007_0000.nii.gz | AbdomenCTCT_0010_0000.nii.gz |
| 41 | AbdomenCTCT_0019_0000.nii.gz | AbdomenCTCT_0025_0000.nii.gz |

Preview four jobs without launching:

```bash
/venv/main/bin/python -m medical.abdomen_experiments --methods ours --subset quick4 \
  --gpus 0 1 2 3 --output results/abdomenctct_quick4
```

Use `--config medical/configs/inverse_metric.json` to select the alternative
objective. `--subset` and `--pairs` are mutually exclusive. A future explicitly
requested run can append `--run`; preparation does not authorize execution.
The shared queue assigns jobs to four persistent GPU workers. Scores, endpoint
PSNR and timings use the existing official-protocol reporting, with no saved maps.

## Six-layer PD-LK 0.05 with registration NCC weight zero (2026-11-13)

The user requested applying the OASIS PD-LK-only recipe to abdomen quick4.
The run uses `medical/configs/ablations/native_registration_6layers_pdlk005_reg0.json`,
six hidden layers, width 256, 32768 points, seed 42 and 5000 steps. Endpoint
NCC with moment weight 0.01 and PD-LK weight 0.05 supply the training objective.
Registration NCC weight is zero. Every logged total was checked against
endpoint NCC + weighted moments + 0.05 × PD-LK. Labels remain evaluation-only.

| Pair | Dice | HD95, voxels | Folding |
| --- | ---: | ---: | ---: |
| 0001_to_0007 | 0.2419 | 22.7854 | 0.0229% |
| 0001_to_0025 | 0.5194 | 11.4987 | 0.0415% |
| 0007_to_0010 | 0.3916 | 13.0415 | 0.5955% |
| 0019_to_0025 | 0.3156 | 31.5438 | 0.1344% |
| Mean | 0.3671 | 19.7174 | 0.1986% |

All four pairs completed and all 64/64 PCG intervals converged. Mean training
time is 85.58 seconds per pair, recovery/evaluation 21.26 seconds and launcher
end-to-end time 111.09 seconds.

| Saved quick4 reference | Mean Dice | Mean HD95, voxels |
| --- | ---: | ---: |
| Four-layer native full method | 0.3937 | 19.7914 |
| ConvexAdam | 0.4274 | 16.9890 |

The saved full-method reference uses four layers, PD-LK 0.1 and registration
NCC 0.01, so it differs in architecture and objective weights. Its source run
is `results/full_method_suite/06_native_grid/abdomenctct/ours/`. The ConvexAdam
reference is `baselines/results/three_baselines/abdomenctct/convexadam/`.
The new run has lower Dice on all four pairs than the saved full method.
These are single-seed development results on the frozen subset.

New scores, logs, frozen configuration, checkpoints and the paired
`comparison.json` are in
`results/abdomenctct_quick4_6layers_pdlk005_reg0_20261113/`.
