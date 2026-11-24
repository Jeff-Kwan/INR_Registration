# SIREN execution and profiling

The registered source/target SIRENs, initialization, checkpoint keys and method
equations are unchanged. Execution uses FP32 with the existing `highest` matmul
precision and TF32 disabled. No mixed precision, approximate derivatives,
detached training quantities or custom backward kernels were introduced.

## Parameter preparation and analytical derivatives

`pair.prepare()` validates the endpoint layouts and returns an evaluation-local
bundle. It computes the parameter direction once; `.at(time)` interpolates the
parameters once for that time. Chunks share these tensors. Training shares the
direction across both sampled times. Frozen grid evaluation and decoding prepare
once per interval/time, outside the chunk loop. Interpolated image evaluation
uses a functional layer loop instead of rebinding the module with dictionaries.

Bundles carry live graphs. Discard them after backward and prepare fresh bundles
after every parameter update or checkpoint load. Never retain them on a module
or cache them across optimizer steps. Compiled callables cache code, not bundles.

The analytical kernels still use an exact reverse chain for `g/q` and packed
spatial/time tangents with reverse adjoints for `g/q/H/m`. The first affine layer
uses its known coordinate tangents directly. The final scalar output avoids zero
spatial adjoint tangents, and unused final hidden spatial-state lanes are omitted.
All parameter, coordinate and time dependencies stay live. Derivatives retain
the reversed-axis, native-voxel scaling `2/(axis_size-1)`.

## Configuration

- `runtime.compile_endpoints: true`: compile endpoint forward and loss/backward,
  with separate MSE and global NCC/moment variants.
- `runtime.compile_derivatives: true`: compile analytical derivatives in training
  and frozen recovery, including parameter-direction/interpolation operations.
- `objective.registration.compile_velocity: true`: compile the larger analytical
  derivative, shared solve and optional residual-energy computation together.
  This replaces the inner derivative compiler for that call. This flag can
  explicitly enable the larger boundary even if `compile_derivatives` is false.
- `objective.registration.recompute: false`: ordinary backward is the default.
  The existing non-reentrant checkpoint option remains available to save memory.
- `runtime.inference_chunk: 65536`: the measured dense-evaluation memory default.
  The training derivative chunk remains 4096 points.

CUDA uses the selected compiled boundaries; CPU uses eager tensor code. For a
eager SIREN CUDA reference, set all three compile flags to `false`.
Recovery PCG has its own `compile_solver=False` keyword for an eager reference. Compilation
errors surface normally; there is no silent fallback. Time is a scalar tensor,
so changing its value does not specialize the graph per timestep. Architecture,
grad-mode, objective and chunk-shape changes can require additional variants.

Zero-weight ablations omit unused work: PD-LK-only skips particle sampling,
field smoothing and registration trajectories; registration-only skips residual
energy while retaining derivative-based velocities and their full gradients.
The shared field is still evaluated over all lattice cells needed by smoothing.

## Measured results, 2026-11-12

Synthetic single-GPU profiles used the installed `/venv/main` PyTorch
2.14.1+cu132 on RTX 5090, seed 42, width 256, four hidden layers and omega 30.
Endpoint batches contain 32768 points per network. The registration profile uses
an 8-cubed grid, two symmetric pairs per cell, two sampled times and 4096 particles.
Every backward is a fresh forward; no optimizer is constructed or stepped.

| Work | Before optimization | Compiled implementation |
| --- | ---: | ---: |
| Endpoint MSE and backward | 3.01 ms | 2.35 ms |
| Differential derivatives and backward, 2048 points | 6.39 ms | about 5.5 ms |
| Endpoint + PD-LK + registration loss/backward | 21.16 ms | about 15.0 ms |
| Frozen `g/q`, 262144 points | 16.81 ms | 10.4 ms |
| Combined incremental peak allocation | 1057.5 MiB | 868.8 MiB |

The pre-edit source was preserved temporarily for the reference measurement.
Reference timing used three warmups and ten repetitions. Confirmation runs used
10–20 warmups and 30–100 repetitions. Combined compiled medians ranged from
12.7–15.6 ms across runs; an isolated 100-repeat comparison gave 19.43 ms for the
refactored eager path and 15.01 ms for the compiled path. These are synthetic
loss/backward timings, not complete case-training or registration runtimes.

A fresh Inductor cache cost 17.66 seconds for the first combined call and
2.68 seconds for the first frozen basic-derivative call. Compile cost is excluded
from warm timings. Logs showed expected grad-mode/objective/shape specializations,
without per-value timestep recompilation or graph breaks in the compiled kernels.
The trace contains ordinary FP32 GEMMs and fused trigonometric/pointwise kernels;
no custom kernel is needed at this stage.

Recomputation reduced combined peak allocation from about 940 to 586 MiB but
increased its median from about 15.0 to 16.5 ms, so it remains opt-in. A second
architecture, width 128/depth 3/omega 60, reduced combined loss/backward from
16.79 to 14.31 ms; individual small endpoint timings were noisier. Reprofile new
architectures before drawing conclusions about their speed.

Dense evaluation of a synthetic 128×128×64 grid, with one preparation per time:

| Chunk | Median | Incremental peak allocation |
| --- | ---: | ---: |
| 65536 | 41.16 ms | 596.5 MiB |
| 131072 | 42.25 ms | 1174.5 MiB |
| 262144 | 40.68 ms | 2330.5 MiB |

The 65536 default trades about 1.2% runtime for about 74% less temporary memory
in this check. PCG parameters, recovery intervals and RK2 are unchanged.

## Validation and reproduction

CPU tests compare against an independent nested-autograd module forward in FP64,
including both endpoint parameter VJPs, coordinate/time gradients, Hessian
symmetry, gradcheck/gradgradcheck, chunk reuse, recomputation, ablations and
checkpoint layout. Tests cover 2D/3D, varied depths/frequencies, identical and
unequal endpoints, anisotropic scaling, endpoint/intermediate times and tails.
FP64 tolerances are `rtol=1e-8, atol=1e-10` for the analytical comparisons.

Opt-in CUDA tests check compiled FP32 outputs and VJPs against FP64, MSE/NCC
parity, large/default chunk agreement, and live checkpointed registration paths.
The relative L2 threshold is `1e-3`. Near-zero NCC bias-gradient cancellation
also uses a `1e-6` absolute allowance, with a strict whole-VJP relative check:
the investigated eager/compiled difference was only `1.2e-7`, below their FP32
error against FP64. Higher-order finite-difference checks exercise the eager
analytical cores; compiled paths are checked for the complete training VJP.

```bash
OMP_NUM_THREADS=1 /venv/main/bin/python -m unittest discover -s medical/tests -v
INR_TEST_CUDA=1 OMP_NUM_THREADS=1 /venv/main/bin/python -m unittest medical.tests.test_siren_cuda -v

# Prints the profile configuration without doing GPU work:
/venv/main/bin/python -m medical.profile_siren

# Execute only when profiling is authorized. Paths must be fresh:
OMP_NUM_THREADS=1 /venv/main/bin/python -m medical.profile_siren --run \
  --modes eager compiled fused --warmup 10 --repeats 50 \
  --output results/siren_profile/profile.json

# Frozen million-voxel grid; choose another --inference-chunk for comparison:
OMP_NUM_THREADS=1 /venv/main/bin/python -m medical.profile_siren --run \
  --modes fused --stages dense_grid --inference-chunk 65536
```

`compiled` selects compiled endpoints and standalone analytical derivatives;
`fused` adds the derivative-plus-solve boundary. Optional `--trace` writes a
profiler trace. Timing JSON and traces belong under ignored `results/`; the
profiler writes no checkpoints, volumes, deformation maps or warped images.
The implementation run's local reports are in `results/siren_optimization/`.

Complete real-input step profiling, including sampling and optimizer costs:
[OASIS training profile](OASIS_TRAINING_PROFILE.md).

CUDA `compile_velocity` also enables compiled three-tap smoothing and box-NCC masked statistics. RK2 coordinate arithmetic and sampling intentionally remain native because the fully compiled trajectory failed stress-case gradient parity. See the accepted-boundary section of OASIS_TRAINING_PROFILE.md.

User-authorized rollout, 2026-11-13: `compile_velocity` now also compiles the full box-NCC RK2 decoder on CUDA. The earlier native trajectory boundary was superseded after disposable tests showed good parity for constant/linear and ordinary random fields, with error amplification in extreme random fields. Finite trajectory checks remain outside compilation; the eager reference remains available with compile_velocity=false. OASIS quick4 comparison: results/oasis_quick4_fullcompiled_decoder_ncc_20261113/.

Full RK2 decoder compilation was subsequently disabled by default at user request. Optional `objective.registration.compile_decoder: true` explicitly enables it on CUDA; absence means false. Existing compile_velocity continues to control compiled derivative/solve, smoothing and masked box-NCC statistics, with native RK2 and sampling. No new training run was launched.

The native CUDA implementation also accelerates ordinary box-NCC training, frozen recovery, and scoring. [Four-GPU pipeline timings and numerical checks](PIPELINE_SPEED_ROUND.md).
