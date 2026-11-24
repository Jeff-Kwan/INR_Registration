# Development

Use /venv/main on Vast; read /workspace/AGENTS.md first. The dataset-independent case runner is described in CASE_RUNNER.md.

The training entry point is `medical.registration.training.fit_pair`. It dispatches to ordinary training or the captured CUDA path using the resolved configuration. Endpoint and registration similarity are separate choices. The default and the explicit full-cohort accuracy preset are specified side by side in the [method](METHOD.md). Captured execution has narrower supported combinations. [Captured training](CAPTURED_TRAINING.md)

Training uses one summed backward pass and optional global clipping. Derivative coordinates, shared solves, filtered fields and trajectories retain gradients. Detached scalars are used only for reporting. Inference runs on fixed checkpoints.

See [SIREN execution](SIREN_EXECUTION.md) for the functional parameter bundles,
compiled endpoint/derivative/solve boundaries and opt-in synthetic profiling.
Parameter bundles are evaluation-local live graphs and must never be cached
across optimizer updates. CUDA compiled defaults and the 65536-point inference
chunk were selected from correctness and performance checks; CPU remains eager.

Endpoint fitting uses one functional kernel for eager and compiled execution.
Training batches contain only endpoint coordinates and intensities. Gradient,
chunking, ablation and phase-mask checks share the SIREN/registration test module;
CUDA parity checks remain separately opt-in.

Retired dataset orchestration, pointwise losses, experimental branches, old presets and their dedicated tests have been removed. Historical specifications and reports live in archived_reports and must not be treated as supported commands. Audit synthetic checks before running them; do not run optimizer steps or real-image smoke tests without explicit research authorization.

All cells contribute to PD-LK and both directions use all their particles. Use the single configured seed. [Dataset protocols](DATASETS.md)

Recovery uses true-residual-verified PCG and compiled CUDA iterations by default
for all datasets. CPU stays eager; recover(..., compile_solver=False) selects the
corrected eager solver. See [measured PCG optimization](OASIS_RECOVERY_PROFILE.md).
