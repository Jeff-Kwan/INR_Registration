# Development

Read [METHOD.md](METHOD.md) and [DATASETS.md](DATASETS.md). Use
`/venv/main/bin/python` on this instance after reading `/workspace/AGENTS.md`.
The sole maintained settings are [default.json](../medical/configs/default.json).
Training enters through `medical.registration.training.fit_pair`, which exposes
the captured CUDA trainer directly. There is no alternate eager training loop.

The objective is endpoint NCC with moment matching, the full-gradient local transport
loss (LT) on a shared grid, the bidirectional trajectory-based registration loss
and the endpoint parameter penalty. `captured_objective.joint_loss`
is the tensor-only implementation used by compilation/capture. Eager tensor
functions and the shared-grid reference class verify the same values and VJPs.
Prepared parameter bundles are short-lived live graphs: never cache them across
optimizer updates. Keep all coordinate, derivative, solve, smoothing and trajectory
gradients live; detach reporting scalars only.

Loss coefficients remain adjustable, including zero and an endpoint coefficient
independent of moment matching and the endpoint parameter penalty. Brightness and
spatial-gradient constancy residual coefficients can also be disabled independently. A zero registration coefficient overrides its
cosine schedule. The positive numerical ridge and NCC epsilon are required.
Retired configuration keys are rejected; do not reintroduce compatibility branches.

The test suite checks analytical derivatives against independent autograd,
loss/gradient parity, orthogonal spatial queries, exact CUDA spatial RNG replay,
Sobol timing, RK2 transport, mask crop placement, evaluator parity, protocol
manifests and mocked case/queue execution. Inspect tests before running them:
training, optimizer steps, benchmarks and saved-model recovery require explicit
user authorization. Synthetic checks can run after auditing their side effects.

```bash
/venv/main/bin/python -m unittest discover -s medical/tests -t .
```

CUDA compilation parity checks use `INR_TEST_CUDA=1`; they validate synthetic
values/gradients and do not optimize a model. Full graph capture with an optimizer
step is training, even if it uses a tiny example, and needs separate authorization.

The [case runner](CASE_RUNNER.md) saves compact checkpoints, configs, provenance,
metrics and timings. Recovery uses frozen checkpoints and temporary scratch,
with CUDA kernels and eager numerical references for the same PCG/RK2 equations.
No dense deformation or warped-image outputs may persist. Preserve existing Data
and results. Dataset acquisition/preparation code remains separate from training.
Historical descriptions/reports under archived_reports do not define current code.
