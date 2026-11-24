# ACDC — KAN-IDIR protocol and acquisition

The authoritative cohort, preparation and evaluation specification is in
[Dataset protocols: ACDC](DATASETS.md#acdc--kan-idir-only). KAN-IDIR is the sole
supported ACDC protocol. [Pinned author-source audit](ACDC_KAN_IDIR_PROTOCOL.md)
records executable details and metric caveats.

Official testing patients101–150 are preserved under
`Data/ACDC/testing/patientXXX/`, with `Info.cfg`, selected ED/ES MRI volumes and
corresponding GT segmentations. Phase numbers come from each configuration.
`Data/ACDC/testing_download_manifest.json` records252 files:50 configurations,
100 MRIs,100 segmentations, original license and required citation text, with
source URLs, byte counts and SHA-256 hashes. `scripts/download_acdc_testing.py`
is the acquisition interface. Original files remain unchanged.

The source [official page](https://www.creatis.insa-lyon.fr/Challenge/acdc/databasesTesting.html)
links the Human Heart Project collection. Retain its CC BY-NC-SA4.0 license and
noncommercial scientific-research restrictions. Required citation: Bernard et al.,
IEEE TMI37(11):2514–2525, 2018,
[DOI:10.1109/TMI.2018.2837502](https://doi.org/10.1109/TMI.2018.2837502).

KAN-IDIR-matched inputs are prepared under `Data/ACDC/prepared`. Its manifest
records all 100 phase images and evaluation labels, source hashes, crop offsets,
ES spacings, dependency versions and the 100 directed fits in author order.
Every phase was compared with the retained author preprocessing functions.
Preparation uses concurrent patient workers and preserves the original float32
header arithmetic when calculating rescale factors. `--workers` controls
concurrency and defaults to four.
Labels are stored as integral author evaluation codes. The adapter preserves
prepared MRI values during fitting and excludes labels from model inputs.

For a fresh preparation, with an absent output directory, run

```bash
/venv/main/bin/python scripts/prepare_acdc.py --workers 8
```

Inspect SIRENPath's full single-seed plan with

```bash
/venv/main/bin/python -m medical.cohort_experiments --datasets acdc --plan-only
```

Use `--gpus` to select devices and a fresh `--output` directory outside Data.
An explicitly authorized experiment uses `--run` in place of `--plan-only`.
Recovery scratch is temporary. Checkpoints, metrics and provenance are retained.
Each score records the author's intersection-selected labels and count, plus
fixed-label 1–3 diagnostics. Empty surfaces have null diagnostic HD95. Folding
uses full-grid finite differences of the recovered map and is identified as an
adaptation of the author's analytical estimator. Both fraction and percentage
are reported. Pooled label Dice SD, case HD95 SD and case folding SD are
available in the cohort summary. Timings separate training, recovery and scoring.

Dependency installation uses `pip install -e '.[acdc]'` inside `/venv/main`.
Synthetic tests and mocked adapter checks cover preparation order, zero-padding,
map direction, metric spacing and cleanup on exceptions. Training performance,
CUDA graph capacity at full volume, and recovery convergence require a model run.

## Official training cohort acquired — 17 November2026

All **100 official training patients001–100** are now present under
`Data/ACDC/training/patientXXX/`, separately from test patients101–150 and the
frozen prepared test cohort. From these sources,32 patients now form the [frozen tuning cohort](TUNING_COHORT.md),
with64 phases prepared separately using the same KAN-IDIR preprocessing. No model
fitting was launched.

Downloaded all **702 released files**, totaling **1,643,087,908 bytes**:100 full
4D cine sequences,200 ED/ES MRIs,200 GT segmentations,100 Info.cfg files,
100 per-patient citation files and the two root license/citation files.
Group counts are20 each for DCM, HCM, MINF, NOR and RV.

`Data/ACDC/training_download_manifest.json` records source API URLs/item IDs,
byte counts, local SHA-256 hashes, ED/ES frame IDs, cine shapes and group counts.
All500 NIfTI gzip streams passed full decompression/CRC verification. Endpoint
image/label grids and affines agree, label codes are0–3, and phase IDs fit the
4D sequence lengths. Checksums are locally computed, not independently published.
The existing license and mandatory citation were verified against the official
source without changing their contents.

The [official training page](https://www.creatis.insa-lyon.fr/Challenge/acdc/databasesTraining.html)
links the same Human Heart Project API used for the retained test sources.
Downloader: `scripts/download_acdc_training.py --workers 8`.
Existing test inputs, prepared volumes, manifests and results remain unchanged.
