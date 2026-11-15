# Reproduce the paired-SIREN registration method

The `starting-version` Git tag records the optimized paired-SIREN source used for the initial full-dataset baseline, together with its evaluators, preparation scripts, presets, tests and documentation. Historical reports remain archived under docs/archived_reports.

## Choose the recipe

The 74-pair shared accuracy result used `medical/configs/native_siren_gw_orthogonal_sobol_5k.json` at seed 42. The repository default is `medical/configs/default.json`, a separate width-256 recipe. The full report identifies the source snapshots used for the recorded timings. [Canonical pipeline](docs/METHOD.md) · [Full results](docs/LONG_RESEARCH_REPORT.md)

## Environment and inputs

Run the commands from the checkout root. On the research host use `/venv/main/bin/python`. On another host, use a Python environment with CUDA-enabled PyTorch and install the project dependencies with `python -m pip install -e .` in that environment. The captured recipe uses CUDA graphs, compilation and fused AdamW. The environment used to verify this checkout has Python 3.12.15, PyTorch 2.14.1+cu132, Triton 3.8.0, NumPy 2.5.4, SciPy 1.18.1, NiBabel 5.5.0 and surface-distance 0.1. The GPU on this host is an RTX 5090. [Dependency declarations](pyproject.toml) · [Execution details](docs/CAPTURED_TRAINING.md)

Supply the prepared inputs externally. On this host the three directories are already under `/workspace/INR_Registration/Data`. To rebuild the inputs elsewhere, use the original releases and the included preparation scripts. DIR-Lab needs both phase masks and both landmark CSVs for each subject. Raw mask preparation additionally uses SimpleITK, lungmask 0.2.22 and the documented R231 weights. OASIS and AbdomenCTCT require the official images and evaluation labels in their released directory layout. For external preparation, set your working directory to the parent of the desired Data directory and invoke the scripts by absolute path. The Learn2Reg preparation script also reads medical/protocols/{dataset}_val.json relative to that directory, so copy the two bundled validation JSONs into medical/protocols/ there first. Place the official ZIPs in Data/_downloads as OASIS.zip and AbdomenCTCT.zip. [DIR-Lab preparation](docs/DATASETS.md) · [OASIS](docs/OASIS.md) · [AbdomenCTCT](docs/ABDOMENCTCT.md)

## Preview the full comparison

Set the input and output roots for your machine. These commands emit job plans. GPU 0 is sufficient to run the cases sequentially. To distribute them on the original four-GPU host, pass `--gpus 0 1 2 3`.

```bash
PY=/venv/main/bin/python
INPUTS=/workspace/INR_Registration/Data
OUTPUTS=/workspace/INR_Registration_Reruns
CFG=medical/configs/native_siren_gw_orthogonal_sobol_5k.json

"$PY" scripts/run_dirlab_cases.py --plan-only --config "$CFG" --gpus 0 \
  --data "$INPUTS/DIRLab4DCT_IDIR" --output "$OUTPUTS/dirlab"
"$PY" -m medical.oasis_experiments --methods ours --config "$CFG" --gpus 0 \
  --data "$INPUTS/OASIS" --output "$OUTPUTS/oasis"
"$PY" -m medical.abdomen_experiments --methods ours --config "$CFG" --gpus 0 \
  --data "$INPUTS/AbdomenCTCT" --output "$OUTPUTS/abdomen"
```

For a full run, remove `--plan-only` from the DIR-Lab command and append `--run` to each Learn2Reg command. Use fresh output directories. The launchers cover 10 DIR-Lab subjects, 19 directed OASIS pairs and 45 directed abdomen pairs. Their case outputs contain metrics, timings, resolved settings and endpoint checkpoints. Keeping the output root external keeps this checkout lightweight.

The paired method uses the evaluator source and protocol JSON bundled here. For an external baseline comparison, its pinned upstream implementation can be fetched separately using the baseline setup instructions. [External baseline setup](baselines/README.md)

## Shared baseline queue

The initial baseline uses the width-512 accuracy preset on all 74 pairs, seed 42, with one shared queue across four GPUs. Preview the plan with the command below. Add `--run` to launch an authorized experiment with a fresh output directory. The trial name is an output grouping; the explicit config sets the actual recipe.

```bash
/venv/main/bin/python scripts/run_method_suite.py \
  --trials 06_native_grid --datasets oasis abdomenctct dirlab \
  --config medical/configs/native_siren_gw_orthogonal_sobol_5k.json \
  --gpus 0 1 2 3 --output results/starting_baseline
```

For frozen endpoint diagnostics, `scripts/diagnose_endpoint_patches.py` reads a completed run's summary and checkpoints. It reuses the existing min–max PSNR and patch NCC helpers. Run one worker per GPU with `--run RUN_DIRECTORY --output FRESH_DIAGNOSTIC_DIRECTORY --gpu GPU_ID --worker WORKER_INDEX --workers 4`, using worker indices 0–3. It evaluates 512 uniform centers, 64 centers per segmentation label and all DIR-Lab landmarks, using 9³ samples spanning 8 mm. Out-of-image sites are excluded and low-variance patches are counted explicitly. The separate `scripts/diagnose_dirlab_endpoint_fidelity.py` retains its original 9-mm landmark correspondence diagnostic.
