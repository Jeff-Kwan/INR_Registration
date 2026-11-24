# Paired-SIREN registration

**[Dataset protocols — all cohorts, preparation, pair directions and metrics](docs/DATASETS.md)**

Fit a fresh pair of coordinate SIRENs to each moving/fixed image pair. Their interpolated parameters define the intensity path used for shared-grid PD-LK and bidirectional registration. Frozen recovery solves global brightness flow and integrates the velocity into dense maps. The step-by-step specification includes crop sampling, loss weights, map directions and evaluation. [Canonical method](docs/METHOD.md)

The repository default uses the width-512 OneCycle recipe with output initialization ×30, AdamW decay 0.001, box NCC and four Euler training steps with one stochastic field time per quarter. The leading full-cohort accuracy candidate uses width 512, Gaussian-window registration NCC, orthogonal query pairs, Sobol time sampling, two RK2 training half-steps and captured training. Some individual pairs had more folding. The separate 55.0-second prototype trades away some OASIS accuracy. [Default preset](medical/configs/default.json) · [Accuracy preset](medical/configs/native_siren_gw_orthogonal_sobol_5k.json) · [Full comparison](docs/archived_reports/LONG_RESEARCH_REPORT.md)

The [ten-hour research guide](docs/archived_reports/LONG_RESEARCH.md) records the earlier GPU allocation, full-cohort validation and final report.

The [five-cohort generalist sweep](docs/GENERALIST_SWEEP_REPORT_20261116.md)
records the newer adaptive search, expanded validation and final evaluation.
Its [selected 4k-step preset](medical/configs/native_generalist_euler4_4k.json)
uses four-quarter Euler and one shared configuration across all five cohorts.

Promising experiments and open mechanism questions are collected in [Research ideas](docs/RESEARCH_IDEAS.md).

On this Vast instance, read `/workspace/AGENTS.md` and use `/venv/main/bin/python`.

CUDA defaults compile endpoint fitting and the analytical derivative/shared-solve
path while retaining full training gradients. Frozen evaluation uses 65536-point
chunks. See [SIREN execution](docs/SIREN_EXECUTION.md) for compiler controls,
gradient checks, profiling commands and measured speed/memory results.

An opt-in [captured native path](docs/CAPTURED_TRAINING.md) includes sampling,
backward and AdamW in a complete CUDA graph. It also documents orthogonal query
geometry and fixed recovery sampling; the default execution remains ordinary.

`python -m medical --help` exposes the dataset-independent case-runner scaffold. Inputs are prepared beforehand. See [case runner](docs/CASE_RUNNER.md) and [datasets](docs/DATASETS.md).

Historical reports are under `docs/archived_reports/`; their commands and configurations are unsupported. Data and generated artifacts are ignored and must not be committed. Training and recovery experiments require explicit authorization.

External comparisons are isolated under [baselines/](baselines/README.md): pinned
author source checkouts, native adapters, configs and setup for FireANTs SyN,
ConvexAdam–MIND, ccIDIR and NIR. The baseline launcher prints a plan by default;
it requires `--run` to execute experiments. Source imports and mocked adapter
checks are available without launching registration.

Learn2Reg OASIS public-validation preparation and all five adapters are also connected: [OASIS](docs/OASIS.md). Preview its 19 directed pairs with `/venv/main/bin/python -m medical.oasis_experiments`; the launcher is single-seed and plan-only by default.

The three-method all-dataset baseline launcher is ready; see [baseline run commands](docs/BASELINE_RUNS.md). It uses four GPU workers, saves under `baselines/results/`, and requires `--run` to execute.

Completed comparison scores and failure accounting: [external baseline results](docs/EXTERNAL_BASELINES.md).

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

An opt-in [particle registration objective](docs/PARTICLE_REGISTRATION.md) solves
weighted brightness/differential LK at moving particles, blurs velocities with a
continuous Gaussian in mm and uses two Euler steps through a stochastic split.
The selected particle preset uses 4096 independent starts per direction in one
shared crop, LK and NCC in both directions, and crop-relative Gaussian sigma
decreasing linearly from 0.4 to 0.1. It has no cycle penalty or transferred starts.

The particle preset uses CUDA objective/backward graph replay and fused Adam;
sampling and the finite-gradient guard stay outside the graph.
[Particle execution measurements](docs/PARTICLE_PERFORMANCE.md).

Shared-grid training defaults to [four-quarter Euler transport](docs/TRAINING_INTEGRATORS.md),
with one stochastic field time per quarter and a fixed 1/4 step per field.

An opt-in [crop-relative grid Gaussian schedule](docs/GRID_BOX_SIGMA.md) changes
only the original shared-grid velocity smoothing, retaining its pooled solve
and selected training integrator.

Experimental [paired-mean AdamW decay](docs/PAIRED_ADAMW.md) replaces decay
toward zero with a learning-rate-scaled pull between matching endpoint parameters.

[Default SIREN recipe ablations](docs/SIREN_RECIPE_ABLATIONS.md): output initialization,
learning-rate scheduling and weight decay on all three frozen quick4 cohorts.

[Additional brain benchmarks](docs/BRAIN_TRAINING.md): IXI/TransMorph and
LPBA40/SITReg are prepared with frozen author splits and protocol-specific
scoring. The plan-only launcher sets up 115 IXI and 45 LPBA40 fits with our
512-wide default recipe. Training has not been launched; see the
[protocol audit](docs/BRAIN_BENCHMARK_PROTOCOLS.md).
