# Paired-SIREN registration

Fit a fresh pair of coordinate SIRENs to each moving/fixed image pair. Their interpolated parameters define the intensity path used for shared-grid PD-LK and bidirectional registration. Frozen recovery solves global brightness flow and integrates the velocity into dense maps. The step-by-step specification includes crop sampling, loss weights, map directions and evaluation. [Canonical method](docs/METHOD.md)

The repository default uses width 256 and box NCC. The leading full-cohort accuracy candidate uses width 512, Gaussian-window registration NCC, orthogonal query pairs, Sobol time sampling and captured training. Across 74 pairs it reached 2.861 / 2.970 mm DIR-Lab TRE, 0.81349 OASIS Dice and 0.43685 abdomen Dice in 93.9 seconds per pair on average. Some individual pairs had more folding. The separate 55.0-second prototype trades away some OASIS accuracy. [Default preset](medical/configs/default.json) · [Accuracy preset](medical/configs/native_siren_gw_orthogonal_sobol_5k.json) · [Full comparison](docs/LONG_RESEARCH_REPORT.md)

The [ten-hour research guide](docs/LONG_RESEARCH.md) covers GPU allocation, full-cohort validation and the final report.

Promising experiments and open mechanism questions are collected in [Research ideas](docs/RESEARCH_IDEAS.md).

On this Vast instance, read `/workspace/AGENTS.md` and use `/venv/main/bin/python`.

CUDA defaults compile endpoint fitting and the analytical derivative/shared-solve
path while retaining full training gradients. Frozen evaluation uses 65536-point
chunks. See [SIREN execution](docs/SIREN_EXECUTION.md) for compiler controls,
gradient checks, profiling commands and measured speed/memory results.

An opt-in [captured native path](docs/CAPTURED_TRAINING.md) includes sampling,
backward and AdamW in a complete CUDA graph. It also documents orthogonal query
geometry and fixed recovery sampling; the default execution remains ordinary.

`python -m medical --help` exposes the dataset-independent case-runner scaffold. It expects an importable method plugin; the default adapter connects training and recovery for DIR-Lab. Inputs are prepared beforehand. See [case runner](docs/CASE_RUNNER.md) and [datasets](docs/DATASETS.md).

Historical reports are under `docs/archived_reports/`; their commands and configurations are unsupported. Data and generated artifacts are ignored and must not be committed. Training and recovery experiments require explicit authorization.

External comparisons are isolated under [baselines/](baselines/README.md): pinned
author source checkouts, native adapters, configs and setup for FireANTs SyN,
ConvexAdam–MIND, ccIDIR and NIR. The baseline launcher prints a plan by default;
it requires `--run` to execute experiments. Source imports and mocked adapter
checks are available without launching registration.

Learn2Reg AbdomenCTCT public-validation preparation and all five method adapters are connected. See [AbdomenCTCT](docs/ABDOMENCTCT.md) for the official 45-pair protocol, evaluation conventions and plan-only launcher.

Learn2Reg OASIS public-validation preparation and all five adapters are also connected: [OASIS](docs/OASIS.md). Preview its 19 directed pairs with `/venv/main/bin/python -m medical.oasis_experiments`; the launcher is single-seed and plan-only by default.

The three-method all-dataset baseline launcher is ready; see [baseline run commands](docs/BASELINE_RUNS.md). It uses four GPU workers, saves under `baselines/results/`, and requires `--run` to execute.

Completed comparison scores and failure accounting: [external baseline results](docs/EXTERNAL_BASELINES.md).

Our cropped/uncropped DIR-Lab scores and runtimes: [DIR-Lab results](docs/OUR_DIRLAB_RESULTS.md).
The current ten-case gap analysis and mechanism-led experiment plan are in
[DIR-Lab gap investigation](docs/DIRLAB_GAP_INVESTIGATION.md).
The completed 5k-versus-40k experiments, checkpoint probes and cross-dataset controls
are recorded in [SIREN convergence](docs/SIREN_CONVERGENCE.md).

Our OASIS registration, reconstruction and timing results: [OASIS results](docs/OUR_OASIS_RESULTS.md).

An optional [inverse-metric velocity-agreement regularizer](docs/INVERSE_METRIC.md)
replaces the PD-LK penalty; its initial preset uses fixed weight 1 and NCC 0.05.

Experimental [1 mm physical working grid](docs/PHYSICAL_WORKING_GRID.md): optional
preprocessing with original-grid evaluation; the default native-grid preset is unchanged.

Optional [structure-centered crop sampling](docs/STRUCTURE_SAMPLING.md) uses a
50/50 usual/structure crop mixture with equal moving/fixed choice and direct
centering. Preset: `medical/configs/native_structure_crop.json`.

Optional [structure-weighted endpoint sampling](docs/STRUCTURE_SAMPLING.md) mixes
uniform and Gaussian-gradient samples globally, with a separate experimental preset.

[Six-trial full-dataset suite](docs/FULL_METHOD_SUITE.md): one plan-only master launcher for all 444 single-seed case jobs; `--run` starts all six configurations through one shared multi-GPU queue.

Optional [native grid with physical units](docs/NATIVE_PHYSICAL_UNITS.md) keeps
original image samples while using mm derivatives/velocities and spacing-weighted
PCG. Preset: `medical/configs/native_physical_units_endpoint_ncc_32k.json`.
