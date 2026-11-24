# Captured native training and fixed recovery sampling

`runtime.training_execution: "captured"` opts into a complete CUDA graph for the native endpoint-NCC, shared-grid PD-LK and box/GW-NCC training step. Ordinary execution remains the default. The explicit [captured box preset](../medical/configs/native_siren_captured_5k.json) retains independent query directions; the [GW/orthogonal comparison preset](../medical/configs/native_siren_gw_orthogonal_5k.json) additionally changes registration similarity and query geometry. The leading full-cohort accuracy preset also uses scrambled Sobol time draws. [Accuracy preset](../medical/configs/native_siren_gw_orthogonal_sobol_5k.json). Full-cohort evidence is in the [research report](LONG_RESEARCH_REPORT.md), [scorecard](LONG_RESEARCH_RESULTS.md) and [decision log](LONG_RESEARCH_LOG.md).

The captured path includes sampling, analytical derivatives, the local solve, smoothing, both RK2 trajectories, every loss, backward, gradient clipping and fused AdamW. Every training gradient remains live. Logging values and frozen checkpoint serialization may detach. `solve_ex(check_errors=False)` retains FP32 LU and checks its status asynchronously, avoiding the synchronous error check that prevented graph capture. Highest FP32 remains selected. Parameters, optimizer state and learning-rate tensors stay on the assigned GPU.

By default sampling uses the original per-call CUDA generator streams. A separate graph handles each mask endpoint; replay resets native generator offsets for the current step. Masks still only select crop centers. All grid cells and particles contribute, including crop-exiting trajectories, with border interpolation and the full-image boundary penalty.

The first three updates warm the compiled objective. Thereafter the complete update replays a graph. A persistent worker reuses compiled code across adjacent cases with matching geometry, dtype/device and config, while creating fresh parameters, graphs and optimizer state. A change in that signature retires the joint objective's old specializations, bounding the shape cache and avoiding TorchDynamo's specialization limit. Four fresh-model real-input checks verify reuse and a geometry change; 100-update repeat differences remain below 8.5e-6 maximum parameter error. The resolved config and checkpoint record execution mode, query geometry, similarity and optimizer settings. Checkpoints retain the ordinary paired-SIREN state and model layout and recover with frozen PCG+RK2.

## Supported scope

This path requires CUDA FP32, AdamW, compiled endpoints/derivatives/velocity/decoder, native voxel units, a fixed crop, separate endpoint/registration samples, global endpoint NCC, PD-LK and three-tap velocity smoothing. Registration can use box NCC or the retained GW-NCC. GW smoothing also requires its three-tap kernel. Zero auxiliary loss weights remain valid. Objective schedules and ordinary model sizes remain configurable.

Working-grid/mm-unit training, structure sampling, tied points, a changing crop size, parameterized first frequencies and recomputation use ordinary execution. The captured path rejects unsupported combinations explicitly. This is a bounded execution interface rather than an alternative optimizer or estimator.

`objective.registration.query_geometry: "orthogonal"` optionally draws two perpendicular isotropic directions per cell. Three orthogonal pairs are also supported but cost two extra derivative sites. Absence selects the original independent directions. The method still averages the complete shared-cell normal equations before solving. Two orthogonal pairs reduce frozen query noise at the same four-site cost; full-cohort testing determines whether the training gain transfers.

## Optional temporal quadrature

`training.temporal_sampling: "sobol"` uses a scrambled two-dimensional Sobol sequence for the early/late time pair, mapped to their original half-intervals. Spatial CUDA streams, mask choice, crop, reconstruction samples, directional particles and query geometry are unchanged. The default remains `"independent"`. Sobol requires captured execution and `objective.registration.random_timesteps: true`; other combinations are rejected. The [explicit Sobol preset](../medical/configs/native_siren_gw_orthogonal_sobol_5k.json) retains GW-NCC and orthogonal two-pair queries.

This changes temporal quadrature and is a method experiment, rather than an execution-parity claim. Frozen-gradient comparisons motivate it; the complete cohort comparison in the [research report](LONG_RESEARCH_REPORT.md) determines transfer. Checkpoint metadata records the sampler, and checkpoints retain the ordinary paired-SIREN layout. A CUDA replay check verifies the exact Sobol table and bitwise equality of every spatial sample to the native sampler.

## Recovery sampling and units

`runtime.recovery_max_axis: 256` caps each native recovery axis at 256 samples, leaving shorter axes unchanged. The SIREN still represents the original image; no training image is resampled. Derivatives and velocities retain original voxel units. Actual recovery-node spacings enter PCG, velocity integration converts to recovery coordinates, and returned temporary maps are interpolated into the original evaluation geometry. The cap is an approximation of the same brightness-flow equations and requires native voxel units. It does not combine with physical/working-grid metadata or alternate recovery solvers.

Recovery settings independently control the fixed number of time evaluations and RK2 substeps. The candidate uses eight velocity evaluations with eight spatial substeps each, compared with the reference's sixteen evaluations and four substeps. Both integrate 64 total substeps. Frozen original and GW checkpoint checks show small score differences and converged solves; the full cohort remains the final denominator. This fixed temporal quadrature is separate from changing spatial recovery sampling.

Recovery uses temporary scratch and removes it on success and exception. No persistent velocity fields, deformation maps, reconstructed volumes or warped images are written.

## Verification and reporting

The synthetic checks compare every endpoint parameter VJP through both box and GW losses, including zero-weight ablations; replay masked and unmasked native CUDA samples bitwise; and verify original-grid recovery units, shape and exception cleanup. Real-input short fits and a complete cold-worker case check precede full queues. The fresh-worker telemetry check allocates a synthetic scalar tensor without training or recovery. CUDA must initialize before peak-memory counters reset.

Case records include the imported method module, actual source path/hash, peak allocated/reserved CUDA memory and stage timings. Queue timings measure dispatch through completed metrics/checkpoint output; they include cold compilation, I/O and scoring. Stage totals need not equal queue time because loading and output live outside the adapter. Keep worker/cache history and peak memory separate from scientific scores.

The configuration API now obtains `DEFAULT` from the same `medical/configs/default.json` used by command-line runs. This fixes a stale MSE/4k-particle default in calls without a path; all marathon experiments used explicit resolved configurations and are unaffected by that correction.
