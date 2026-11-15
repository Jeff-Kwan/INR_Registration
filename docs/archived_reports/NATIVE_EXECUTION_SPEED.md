# Native execution speed audit

The six-trial suite's native slowdown is reproducible as a compiler
specialization/fusion problem in PCG. Training is equally fast on native and
physical grids. This audit changes no production code or configuration.

## Completed-cohort evidence

Mean seconds per successful case from `results/full_method_suite`:

| Stage | OASIS physical | OASIS native | Abdomen physical | Abdomen native |
| --- | ---: | ---: | ---: | ---: |
| Training | 89.63 | 89.50 | 89.15 | 89.58 |
| Reconstruction | 7.18 | 0.19 | 10.50 | 0.21 |
| Recovery + evaluation | 14.67 | 334.64 | 13.80 | 443.05 |
| Mean PCG iterations/interval | 173.88 | 151.74 | 199.03 | 178.44 |

Physical reconstruction includes working- and original-grid measurements;
native measures only its original grid. Native adapters did not save separate
recovery/evaluation timing fields. Native DIR-Lab saved neither stage timing;
the existing 1862.1 s successful-case mean cannot be decomposed retrospectively.
Native PCG requires fewer iterations on these cohorts, so iteration counts do
not explain the slowdown. OASIS recovery dimensions are identical across paths;
abdomen physical recovery also uses the original grid dimensions.

## Reproduction and cause

`flow._compiled_step()` uses `torch.compile(fullgraph=True)` with automatic
shape handling. Physical recovery supplies a tensor of spacing weights;
native recovery passes `axis_weights=None`. After varied shapes have been seen,
the latter can select a catastrophically slow compiled graph.

A bounded FP32 PCG-step benchmark on a 192×224×160 grid, RTX 5090, four CPU
threads, measured 30 warm calls. It first compared explicit static/dynamic
compilation, then primed the cached default callable using three small differing
shapes with tensor weights before calling the large native problem:

| Compiled path | Warm ms/iteration | Kernels/iteration |
| --- | ---: | ---: |
| Explicit `dynamic=False` | 0.806 | 7 |
| Explicit `dynamic=True` | 4.873 | not traced |
| Default after shape history, weights `None` | 134.676 | 1 |
| Same cached callable/history, unit tensor weights | 0.894 | 7 |

The slow kernel is `triton_red_fused_add_cat_clamp_min_div_mul_slice_sub…`;
the CUDA trace measures 135.037 ms inside that single kernel. This is steady
execution cost, not repeated compilation or eager fallback. A roughly 0.135 s
iteration multiplied by 2270 iterations explains about 306 s of native OASIS
recovery. This reproduces the magnitude of the suite slowdown; the original
workers were not traced, so their exact generated graphs are unavailable.

The one-step tensor outputs agree with the static reference: relative L2
differences are below 6e-7 for velocity/residual/direction and 1.9e-6 for the
scalar preconditioned residual product. Unit tensor weights give differences
below 2.2e-7 for all four outputs. Unit weights preserve the native mathematical
operator; generated FP32 operation order can still differ.

## Saved-checkpoint replays

No training or optimizer updates. Existing recovery settings, 16 intervals,
temporal warm starts, convergence checks, and RK2 were retained. Each replay ran
in a fresh process with synchronized stage timers and exception-safe temporary
scratch. No maps or warped images were saved.

| Native checkpoint | Suite recovery + eval | Replay recovery | Replay eval | Derivatives | PCG | Integration |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OASIS 0395_to_0396 | 319.81 | 12.85 | 5.90 | 6.80 | 3.87 | 0.20 |
| Abdomen 0001_to_0004 | 409.21 | 14.50 | 2.69 | 7.35 | 4.65 | 0.23 |

Times include first-use setup/compilation with the existing disk cache, not a
clean-cache cold-start benchmark. Remaining recovery time includes transfers,
scratch I/O and setup. A matching physical OASIS replay took 13.60 s recovery
and 6.16 s evaluation; its saved metrics and diagnostics matched exactly.

Native replays converged in all 16 intervals. OASIS total iterations changed
2270→2260 and DSC 0.819717160→0.819717233; abdomen iterations changed 2620→2610
and DSC 0.389808185→0.389798604. These are small numerical changes, not bitwise
equivalence. Scratch was empty after every successful replay; existing cleanup
implementation was inspected, not modified.

## Proposed follow-up

Canonicalize absent spacing to a tensor of unit axis weights, so native and
physical solves use the same compiler interface. Validate this in long-lived
workers with realistic mixed grid sizes, true residuals and map/metric
comparisons. Explicit static specialization is another option, but needs
compiler cache-limit handling when many DIR-Lab shapes share a worker. Do not
disable the shared queue or retrain models to address this issue.

Also record preprocessing, recovery and evaluation timings for native adapters;
currently several fields are conditional on working-grid metadata.

Raw summaries and operator tables: `results/native_speed_audit/` (ignored).
Disposable drivers: `/tmp/audit_native_speed.py` and
`/tmp/audit_native_compiler.py`. No DIR-Lab checkpoint replay or accuracy change
was performed. No changes committed.
