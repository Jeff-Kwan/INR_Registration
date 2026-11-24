# Case runner

Use the canonical [method](METHOD.md) and [configuration](../medical/configs/default.json).
Prepare inputs according to [dataset protocols](DATASETS.md) before execution.
A run requires explicit user authorization; the commands below describe interfaces.

`python -m medical` accepts `--moving`, `--fixed`, `--config`, `--output`,
`--method` and `--device`. Inputs must be matching finite 3D NIfTI grids and affines.
Outputs must use a new directory outside Data. Use `/venv/main/bin/python` and
`--device cuda:0`; SIRENPath trains on CUDA FP32. The moving image is model time 0,
the fixed image time 1. The plugin returns scores and a compact checkpoint;
the runner retains checkpoint.pt, metrics.json and run.json with provenance.

| Benchmark | Method plugin | Cohort launcher |
| --- | --- | --- |
| Learn2Reg OASIS | `medical.oasis_method` | `python -m medical.oasis_experiments` |
| IXI / LPBA40 | `medical.brain_method` | `python -m medical.brain_experiments --plan-only` |
| ACDC / Neurite-OASIS | `medical.cohort_method` | `python -m medical.cohort_experiments --plan-only` |

The ACDC/Neurite launcher validates their prepared manifests and exact ordered
cohorts. Use `--datasets acdc` or `--datasets neurite_oasis` to select one, and
`--gpus 0 1 2 3 4 5 6 7` to select all eight GPUs on this host. Planning is the
default and `--run` starts fitting. The default output is
`results/prepared_cohorts`; choose a fresh `--output` for each run. ACDC contains
all 50 ED→ES fits followed by all 50 ES→ED fits. Neurite contains 49 consecutive
earlier-fixed/next-moving fits. Each selected pair is validated before workers
start. Progress and summary files retain failures, scored denominators, pooled
label Dice and case timing. Labels remain evaluation-only.

Launchers snapshot the selected configuration and ordered pairs. One persistent
worker per explicitly selected GPU takes cases from a shared queue; every case
gets fresh model/optimizer state and its single configured seed. Case failures
remain in progress and summary files, with success denominators and runtime scopes.
Learn2Reg/brain launchers plan by default and require `--run` for experiments.
The frozen Learn2Reg OASIS quick4 subset is listed in DATASETS.md.

Whole-volume reconstruction metrics are diagnostics; endpoint training uses the
shared crop. Dataset labels are evaluation-only. Recovered maps
and warped images stay in memory; solver scratch is removed on exceptions as
well as success. Checkpoints preserve model state and the resolved configuration.

## Standardized official comparisons

The selected official cohorts are Neurite-OASIS (49), ACDC (100), LPBA40 test
(45), and IXI test (115), for 309 fits per method. L2R-OASIS ablations use the separate [SIRENPath study](SIRENPATH_ABLATIONS.md)
and are excluded from these launchers. The canonical configuration and seed 42
apply to SIRENPath. External recipes retain their recorded source settings.

Plan SIRENPath and external baselines from the repository with

```bash
bash scripts/launch_sirenpath_comparisons.sh --output results/official_sirenpath --plan-only
bash scripts/launch_external_comparisons.sh --output baselines/results/official_external --plan-only
```

Each script prints commands and missing inputs and creates no output directory.
For an explicitly authorized run, replace `--plan-only` with `--run`. Outputs
must be fresh directories outside Data. `--datasets`, `--gpus` and
`--cpu-workers` select cohorts and resources. CPU workers use their own queue
and run concurrently with GPU workers. The defaults are eight GPU workers and
four CPU workers, with four native CPU threads per baseline process.

The external script includes ConvexAdam, FireANTs-SyN, NIR, Demons and NiftyReg.
Select the requested timing-comparison subset with

```bash
bash scripts/launch_external_comparisons.sh --methods convexadam fireants demons niftyreg --output baselines/results/official_four --plan-only
```

All methods use `standardized_v1` scoring on the preserved prepared grids.
Dice is an equal foreground-label mean, HD95 uses affine-column spacing in mm,
NPJ is 100 times the fraction of nonpositive Jacobian determinants, and SDlogJ
is the population standard deviation of log(max(detJ,1e-9)). Jacobians use the
same central differences and exclude two boundary voxels on every axis.
Nearest-neighbor zero-outside label sampling applies to every method. For IXI, the prepared affine assigns 1 mm per voxel along each axis.
The reported HD95 is in mm on this benchmark grid. Label sets are Neurite
1–35, ACDC 1–3, LPBA40's 56 codes, and IXI's 30 scored anatomical codes with
equal label weighting. Both-empty labels have null Dice; missed foreground has
Dice zero. Empty surfaces have null HD95 and cause a null case HD95 mean.
Scored/intended label and pair counts are retained, including crashes.

The plans snapshot configurations, input preparation hashes and evaluator source
hash. Progress and summaries are grouped by dataset and method, with per-case
scores and failures retained. Dense maps and warped images remain temporary.
NIR minmax-normalizes LPBA40 input for its released intensity sampler; it receives
prepared [0,1] values for the other selected cohorts. SIRENPath retains each
cohort's protocol-prepared intensities. Runtime estimates exclude NIR and ccIDIR
when explicitly requested.


The finalized neural-field set is ccIDIR, NIR, NODEO and DNVF. Inspect its
1,236-fit plan with

```bash
bash scripts/launch_neural_comparisons.sh --output baselines/results/official_neural --plan-only
```

Source fidelity, localization details and checkpoint scope are described in
[the baseline guide](../baselines/README.md#finalized-neural-field-comparison-set).
Additional neural-field methods are deferred until the first paper draft.

## SIRENPath internal ablations

`scripts/launch_sirenpath_ablations.sh --output <fresh-directory> --plan-only`
plans the seven [L2R-OASIS ablation runs](SIRENPATH_ABLATIONS.md): 95 trained fits
and 38 no-training crossfade controls. Execution requires `--run` and idle selected
GPUs. The report includes endpoint PSNR/Patch NCC, DSC, HD95, NPJ, SDlogJ and runtime.

## Frozen development tuning

The separate [tuning cohort and launcher](TUNING_COHORT.md#tuning-launcher) uses
32 frozen pairs per dataset and `medical.tuning_method`. Plan without execution:

```bash
bash scripts/launch_tuning.sh --config medical/configs/default.json --output results/tuning/canonical --plan-only
```

Use a fresh directory per candidate; `--run` requires explicit authorization.
Results are development-only and must not enter official comparison aggregates.
