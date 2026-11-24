# Fixed OASIS quick4 prototype set

Selected once using Python random.Random(42).sample without replacement from the 19 official directed validation pairs, then sorted. Selection did not use registration scores. The four jobs can occupy four GPUs concurrently. Manifest: medical/protocols/oasis_quick4.json, including protocol hash and exact paths.

| Official pair index | Fixed → moving |
| --- | --- |
| 1 | 0395_to_0396 |
| 4 | 0398_to_0399 |
| 8 | 0402_to_0403 |
| 9 | 0403_to_0404 |

This is a fixed development subset, not four independent subject-disjoint pairs: pairs 8 and 9 share subject 0403. Keep the same pairs, seed and preparation across prototype comparisons.

## Commands

Preview the existing PD-LK baseline (no execution):

```bash
/venv/main/bin/python -m medical.oasis_experiments --methods ours --subset quick4 \
  --gpus 0 1 2 3 --output results/oasis_quick4_pdlk
```

Preview the inverse-metric alternative:

```bash
/venv/main/bin/python -m medical.oasis_experiments --methods ours --subset quick4 \
  --config medical/configs/inverse_metric.json --gpus 0 1 2 3 \
  --output results/oasis_quick4_inverse_metric
```

When a run is explicitly requested, append --run. Use a fresh output directory. --subset and --pairs are mutually exclusive. Without either, the launcher still plans all 19 pairs. Execution uses persistent GPU processes and the shared queue; registration scores, whole-volume endpoint PSNR and timings retain their existing reporting.

## Historical reference scores

These scores are extracted from completed runs, not newly generated. Our reference is the original PD-LK 0.1/NCC 0.01 OASIS run, before the PCG stopping correction. For a controlled current-code comparison, run the PD-LK preset on quick4 alongside the new alternative; do not silently treat this historical reference as a new run.

| Pair | Our DSC | Our HD95 | Moving PSNR | Fixed PSNR | Time (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0395_to_0396 | 0.7873 | 1.930 | 34.44 | 34.46 | 129.8 |
| 0398_to_0399 | 0.7098 | 3.455 | 34.09 | 34.66 | 132.4 |
| 0402_to_0403 | 0.7777 | 2.028 | 34.17 | 34.73 | 124.3 |
| 0403_to_0404 | 0.8071 | 2.031 | 34.26 | 34.15 | 122.2 |

| Method on quick4 | Mean DSC ↑ | Mean HD95 ↓ | Mean SDlogJ ↓ |
| --- | ---: | ---: | ---: |
| Our historical PD-LK | 0.7705 | 2.3608 | 0.0514 |
| fireants | 0.7508 | 2.5497 | 0.0304 |
| convexadam | 0.7717 | 2.4097 | 0.0614 |
| nir | 0.7238 | 3.0904 | 0.1859 |

HD95 follows the official voxel-unit convention, numerically mm for OASIS. All reference means cover 4/4 pairs. Source results and baseline recipe limitations: [our OASIS report](OUR_OASIS_RESULTS.md) and [external baselines](EXTERNAL_BASELINES.md).

## Inverse-metric weight-1 experiment (2026-11-12)

Source: results/oasis_quick4_inverse_metric_20261112_134634/; 4/4 completed with seed 42 and 5000 steps.
Objective: MSE + 0.05 NCC + 1.0 inverse-metric agreement, no PD-LK penalty or calibration.

| Pair | Historical PD-LK DSC | Inverse-metric DSC | Historical HD95 | Inverse-metric HD95 |
| --- | ---: | ---: | ---: | ---: |
| 0395_to_0396 | 0.7873 | 0.7598 | 1.930 | 2.086 |
| 0398_to_0399 | 0.7098 | 0.6581 | 3.455 | 3.760 |
| 0402_to_0403 | 0.7777 | 0.7431 | 2.028 | 2.235 |
| 0403_to_0404 | 0.8071 | 0.7776 | 2.031 | 2.150 |

Mean DSC: 0.7347 versus historical 0.7705; HD95: 2.5578 versus 2.3608.
Mean whole-volume moving/fixed PSNR: 30.81/31.07 dB; SDlogJ 0.04573; zero measured folding.
PCG converged in 60/64 intervals. Mean training 101.10 s and end-to-end 123.25 s
per pair; all four ran concurrently.

This changes both the regularizer and NCC weight (0.01→0.05), so it does not
isolate the regularizer effect. Historical reference also predates corrected PCG
stopping and persistent workers. Weight-1 agreement underperformed on all four
pairs; this alone does not identify its cause or establish the outcome after calibration.

## Lower-NCC comparison (2026-11-12)

Source: `results/oasis_quick4_metric_ncc001_20261112_135227/`; 4/4 completed.
Only the NCC coefficient changed from the preceding experiment, from 0.05 to
0.01. Metric weight remains 1, with seed 42 and 5000 updates.

| Pair | DSC | HD95 |
| --- | ---: | ---: |
| 0395_to_0396 | 0.7739 | 1.995 |
| 0398_to_0399 | 0.6991 | 3.506 |
| 0402_to_0403 | 0.7660 | 2.086 |
| 0403_to_0404 | 0.7918 | 2.119 |

Mean DSC increased from 0.7347 to 0.7577 and HD95 decreased from 2.5578 to
2.4265. Every pair improved. Mean moving/fixed PSNR was 32.77/33.07 dB,
SDlogJ 0.04513, with zero measured folding and 64/64 PCG intervals converged.
Mean end-to-end runtime was 113.23 seconds per pair.

Historical PD-LK still scores higher (DSC 0.7705, HD95 2.3608), subject to the
recovery and worker-version differences described above. The initial alternative
preset remains NCC 0.05; this comparison used a separate frozen run config.

## Reconstruction-only MSE and regional NCC (2026-11-13)

All four frozen pairs completed seed 42 and 5000 steps with width 256,
four-layer, omega 30 endpoint SIRENs, batch 32768, learning rate 0.0002 and final
factor 0.25. Both auxiliary weights are zero. Every logged total equals the sum
of the moving and fixed MSE terms. Native OASIS independent min/max scaling and
ordinary half-volume crop sampling are retained. Labels enter only evaluation
after fitting. There are eight endpoint fits covering seven distinct subjects.

`medical.oasis_reconstruction_method` runs fitting and exact whole-volume
reconstruction evaluation. `scripts.diagnose_oasis_reconstruction.py` evaluates
labels 1–35 using exact whole-region MSE/NCC and up to 128 random patch-center
draws with replacement per region and endpoint, seed 42. Patches span 9 mm with
10 samples per axis. Full-patch NCC uses the entire window. Within-label NCC
uses only samples whose nearest-neighbor label matches the center region.
The minimum support is 8 samples and minimum variance is 1e-6 in each image.

| Metric | Mean |
| --- | ---: |
| Whole-volume MSE | 0.00028056 |
| Whole-volume PSNR | 35.53 dB |
| Whole-region PSNR | 37.31 dB |
| Whole-region NCC | 0.9414 |
| Full-patch NCC | 0.9865 |
| Within-label patch NCC | 0.9620 |

Region scores average endpoints equally within each label, then labels equally.
Whole-volume metrics average the eight endpoint fits. All 34984 full patches
were valid. Within-label NCC retained 34956 patches, excluding 28 with low
variance or insufficient support. Every label occurs in all eight endpoints.
The lowest within-label patch means are label 6 (0.9233), 34 (0.9262), 25 (0.9302),
24 (0.9329) and 5 (0.9353). Full-region NCC is lower on tiny labels 18 (0.8191)
and 34 (0.7300), whose median sizes are 66 and 54.5 voxels respectively.

The OASIS endpoint fits retain high local correlation even when patch
evaluation is restricted to individual labels.

All checkpoints, logs, resolved configuration, per-endpoint regional details and
`per_region.csv` are in `results/oasis_quick4_reconstruction_mse_20261113/`. Evaluation streams predictions and retains scalar metrics.

## Six-layer regular registration (2026-11-13)

The user requested regular registration with six hidden layers and width 256.
`medical/configs/ablations/native_registration_6layers.json` changes only the layer count
from the current default. Endpoint NCC with moment weight 0.01, PD-LK 0.1,
registration NCC 0.01, 32768 reconstruction/registration points, seed 42 and
5000 steps are retained. Every run's logs contain the active reconstruction,
moment, PD-LK and registration NCC losses. All four frozen pairs completed
through the ordinary `medical.oasis_method` adapter and PCG+RK2 evaluation.

| Pair | Saved four-layer Dice | Six-layer Dice | Saved four-layer HD95 | Six-layer HD95 |
| --- | ---: | ---: | ---: | ---: |
| 0395_to_0396 | 0.8197 | 0.8226 | 1.7018 | 1.6919 |
| 0398_to_0399 | 0.7952 | 0.7957 | 2.8786 | 2.8663 |
| 0402_to_0403 | 0.8285 | 0.8330 | 1.4610 | 1.3940 |
| 0403_to_0404 | 0.8337 | 0.8301 | 1.7376 | 1.6695 |
| Mean | 0.8193 | 0.8203 | 1.9447 | 1.9054 |

HD95 follows the official voxel convention. The four-layer reference comes
from `results/full_method_suite/06_native_grid/oasis/ours/` and has identical
saved config except for layer count. Source versions are not recorded as
identical, so this is a saved reference comparison. Mean six-layer training
time is 131.14 seconds per pair, with 23.67 seconds for recovery and evaluation.
Mean launcher end-to-end time is 158.51 seconds. All 64/64 PCG intervals
converged. Mean interior folding fraction is 0.00003882 (0.00388%).

The initial request was interpreted as reconstruction-only fitting. Those
jobs completed before they could be stopped after the user clarified regular
registration. They are marked superseded in
`results/oasis_quick4_reconstruction_mse_6layers_20261113/`.
The corrected registration results, logs, checkpoints, frozen config and
`comparison.json` are in `results/oasis_quick4_registration_6layers_20261113/`.

## Six-layer PD-LK 0.2 with registration NCC weight zero (2026-11-13)

The user requested PD-LK weight 0.2 and registration NCC weight 0 with the
same six-layer width-256 model. The config
`medical/configs/ablations/native_registration_6layers_pdlk02_reg0.json` changes only
these two weights. Endpoint NCC and moment weight 0.01, batch 32768,
seed 42 and 5000 steps are retained. Each of the 1000 logged training records
was checked against endpoint NCC + weighted moments + 0.2 × PD-LK, with
the registration NCC term equal to zero.

| Pair | Control Dice | PD-LK 0.2 / reg 0 Dice | Control HD95 | PD-LK 0.2 / reg 0 HD95 |
| --- | ---: | ---: | ---: | ---: |
| 0395_to_0396 | 0.8226 | 0.8195 | 1.6919 | 1.6878 |
| 0398_to_0399 | 0.7957 | 0.7966 | 2.8663 | 2.8253 |
| 0402_to_0403 | 0.8330 | 0.8267 | 1.3940 | 1.6121 |
| 0403_to_0404 | 0.8301 | 0.8288 | 1.6695 | 1.7612 |
| Mean | 0.8203 | 0.8179 | 1.9054 | 1.9716 |

The control is the completed current-code six-layer regular registration run
immediately above. Mean training time falls from 131.14 to 84.53 seconds per
pair. Mean launcher end-to-end time is 110.68 seconds. Mean interior folding
falls from 0.00388% to 0.00154%. All 64/64 PCG intervals converge. These two
weight changes together yield slightly lower mean accuracy with lower runtime
and folding on the frozen single-seed development subset.

Results, logs, permitted checkpoints, frozen configuration and `comparison.json`
are in `results/oasis_quick4_6layers_pdlk02_reg0_20261113/`.

## Six-layer PD-LK 0.3 with registration NCC weight zero (2026-11-13)

The user requested increasing PD-LK weight from 0.2 to 0.3. This is the only
config change in `medical/configs/ablations/native_registration_6layers_pdlk03_reg0.json`.
All four frozen pairs completed with six layers, width 256, 32768 points,
seed 42 and 5000 steps. Every logged total equals endpoint NCC + weighted
moments + 0.3 × PD-LK, with registration NCC equal to zero.

| Pair | PD-LK 0.2 Dice | PD-LK 0.3 Dice | PD-LK 0.2 HD95 | PD-LK 0.3 HD95 |
| --- | ---: | ---: | ---: | ---: |
| 0395_to_0396 | 0.8195 | 0.8198 | 1.6878 | 1.7219 |
| 0398_to_0399 | 0.7966 | 0.7919 | 2.8253 | 2.9171 |
| 0402_to_0403 | 0.8267 | 0.8232 | 1.6121 | 1.4995 |
| 0403_to_0404 | 0.8288 | 0.8329 | 1.7612 | 1.7459 |
| Mean | 0.8179 | 0.8170 | 1.9716 | 1.9711 |

Mean training time is 83.71 seconds per pair, versus 84.53 for weight 0.2.
Mean interior folding falls from 0.00154% to 0.000616%. All 64/64 PCG
intervals converge. Mean Dice decreases slightly and HD95 is essentially
equal in this single-seed development comparison.

Results, permitted checkpoints, logs, frozen config and `comparison.json`
are in `results/oasis_quick4_6layers_pdlk03_reg0_20261113/`.

## Six-layer PD-LK 0.1 with registration NCC weight zero (2026-11-13)

The user requested the final weight comparison at PD-LK 0.1 with registration
NCC weight zero. The config
`medical/configs/ablations/native_registration_6layers_pdlk01_reg0.json` changes only
PD-LK weight relative to the 0.2 run. All four pairs completed 5000 steps,
seed 42, six layers, width 256 and 32768 points. Every logged total equals
endpoint NCC + weighted moments + 0.1 × PD-LK, with registration NCC zero.

| Pair | PD-LK 0.1 / reg 0 Dice | PD-LK 0.1 / reg 0 HD95 |
| --- | ---: | ---: |
| 0395_to_0396 | 0.8257 | 1.7256 |
| 0398_to_0399 | 0.7950 | 2.7966 |
| 0402_to_0403 | 0.8340 | 1.4104 |
| 0403_to_0404 | 0.8337 | 1.6548 |
| Mean | 0.8221 | 1.8968 |

The completed six-layer comparisons are:

| PD-LK weight | Registration NCC weight | Mean Dice | Mean HD95 | Training seconds/pair |
| --- | --- | ---: | ---: | ---: |
| 0.1 | 0 | 0.8221 | 1.8968 | 83.79 |
| 0.2 | 0 | 0.8179 | 1.9716 | 84.53 |
| 0.3 | 0 | 0.8170 | 1.9711 | 83.71 |
| 0.1 | 0.01 | 0.8203 | 1.9054 | 131.14 |

Weight 0.1 with zero registration NCC has the highest mean Dice and lowest
mean HD95 in these single-seed quick4 runs. It improves Dice over weight 0.2
on three pairs and over weight 0.3 on all four. Mean interior folding is
0.00339%, and all 64/64 PCG intervals converge. Mean recovery/evaluation time
is 21.99 seconds and launcher end-to-end time is 109.63 seconds per pair.

Results, checkpoints, logs, frozen configuration and the comparison across
all four settings are in
`results/oasis_quick4_6layers_pdlk01_reg0_20261113/comparison.json`.

## Six-layer PD-LK 0.05 with registration NCC weight zero (2026-11-13)

The user requested PD-LK weight 0.05. This is the only change from the 0.1
zero-registration config in
`medical/configs/ablations/native_registration_6layers_pdlk005_reg0.json`. All four pairs
completed six-layer width-256 training with seed 42, 5000 steps and batch 32768.
Each logged total equals endpoint NCC + weighted moments + 0.05 × PD-LK.
Registration NCC is zero in every record.

| Pair | Dice | HD95 |
| --- | ---: | ---: |
| 0395_to_0396 | 0.8247 | 1.6180 |
| 0398_to_0399 | 0.7972 | 2.7858 |
| 0402_to_0403 | 0.8328 | 1.4076 |
| 0403_to_0404 | 0.8354 | 1.6767 |
| Mean | 0.8225 | 1.8720 |

Mean Dice improves slightly from 0.8221 at weight 0.1, with HD95 improving
from 1.8968. Mean interior folding rises from 0.00339% to 0.00900%. Training
averages 82.03 seconds per pair, recovery/evaluation 21.51 seconds and launcher
end-to-end time 107.46 seconds. All 64/64 PCG intervals converge. This is a
single-seed quick4 development result.

All results and the comparison with weights 0.1, 0.2, 0.3 and the six-layer
regular registration control are in
`results/oasis_quick4_6layers_pdlk005_reg0_20261113/comparison.json`.
