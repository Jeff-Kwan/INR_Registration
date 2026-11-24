# Reproducing SIRENPath

Use [the canonical method](docs/METHOD.md), [default configuration](medical/configs/default.json), [dataset protocols](docs/DATASETS.md) and [case runner](docs/CASE_RUNNER.md). The official comparisons comprise 49 Neurite-OASIS, 100 ACDC, 45 LPBA40 and 115 IXI directed cases. The canonical recipe uses width 512, 5,000 updates, NCC endpoints and cosine learning-rate decay.

## Environment

Install into a CUDA-enabled Python environment using the dependency declarations in pyproject.toml and baselines/requirements.txt. On the research machine use /venv/main/bin/python. The exact current installed versions and hardware are recorded in [the reproducibility archive](docs/reproducibility/README.md). This snapshot is an observation of the environment at archival time, not a verified historical lockfile or a guarantee of bitwise determinism. Installing editable project packages and baseline sources requires the setup steps in [the baseline guide](baselines/README.md). Preserve the CUDA-enabled PyTorch build when installing dependencies.

## Inputs and execution

Obtain datasets from their documented original releases and follow each dataset's preparation instructions. Compare prepared manifests, ordered pairs, geometry and recorded checksums against the archived metadata. Prepared inputs, labels and checkpoints are intentionally excluded from Git. Labels remain evaluation-only.

The SIRENPath entry point for the four official comparison datasets is scripts/launch_sirenpath_comparisons.sh, which delegates to medical.comparison_experiments. Inspect its command-line help and use its default plan-only mode before explicitly authorizing execution with --run. Use medical/configs/default.json and fresh output directories. Dataset adapters and shared scoring are documented in docs/CASE_RUNNER.md. Baseline launchers, pinned revisions, patches and localization choices are documented in baselines/README.md and docs/BASELINE_PROTOCOL_REVIEW.md.

## Recorded results

The default official run is results/official_ours_cosine_20261116 (309 cases), including per-case checkpoint.pt, run.json and metrics.json plus root-level plans/configurations/summaries. Preserve that whole folder separately. Its file sizes and SHA-256 checksums are recorded in docs/reproducibility/default_run_inventory.jsonl. Checksum verification confirms backup integrity; reproduction also requires the matching prepared inputs.

The reproducibility archive retains compact metadata for the corrected external-baseline results, tuning and ablations. It also records dataset selection/preparation and analysis selections. It contains no checkpoint weights, image arrays, deformation fields, credentials or raw process logs. Current aggregate tables remain in docs/BASELINES_REPORT.md and docs/ABLATIONS_REPORT.md.

See the [supplementary material index](docs/SUPPLEMENT_INDEX.md) for existing tables/figures, evidence sources, generation tools and remaining artifact dependencies.
