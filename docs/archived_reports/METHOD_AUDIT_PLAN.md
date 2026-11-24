# Correctness investigations for registration research

Prioritize discrepancies that could explain a measured failure. Save the smallest reproducing input, expected result, observed result, affected code path and production consequence. Use the existing tests as entry points and inspect their coverage before extending them. [Test suite](../medical/tests) · [Session decision rules](LONG_RESEARCH.md)

| Question and motivation | Discriminating check | Consequence |
| --- | --- | --- |
| Can the fast path silently change the training signal? | On identical samples and endpoint weights compare eager/autograd, analytical and compiled values and parameter VJPs through derivatives, the shared solve, smoothing, interpolation and trajectories. Include nonzero temporal change, nearly flat cells, crop exits and chunk tails. | A matching scalar loss can conceal a wrong gradient. Localize the first gradient discrepancy before comparing optimizer trajectories. [Derivative checks](SIREN_EXECUTION.md) |
| Does sampling change which loss terms contribute? | Trace a deterministic tiny crop from its endpoint-selected mask centre to the four sample streams and the shared grid. Account for every cell and particle, including those leaving the crop. Check border extension and the full-image boundary penalty against a direct evaluation. | Quantify the affected fraction on a real case if a discrepancy appears. Preserve independent directional sampling and evaluation-only labels. |

## Findings from the ten-hour session

The no-path configuration API selected obsolete MSE/4k-particle settings while the explicit default file selected NCC/32k. The API now loads the file as its single source of truth; all session experiments used explicit configs. The captured full objective passes independent value and every-parameter VJP checks for box/GW-NCC and zero auxiliary weights. Masked/unmasked CUDA spatial sampling matches the ordinary sampler bitwise; optional Sobol times preserve those spatial streams and match the recorded temporal table.

Capped recovery retains original-voxel derivative/velocity units and spacing-weighted PCG, then evaluates on the unchanged original grids. Synthetic coordinate/translation and scratch-cleanup checks pass. Full-cohort aggregation uses official pair identity, preserves failed attempts and does not silently drop nonfinite metrics. All completed final cohort PCG intervals converge and pair HD95 means are finite. Actual stage synchronization and imported source/hash/peak-memory telemetry are recorded. The remaining folding problem survives finer integration; sparse positive particle Jacobians do not override official sampled-map folds. See the [report](LONG_RESEARCH_REPORT.md) for the complete evidence and limits.

Its saved-metric parity guard remains active.

The closing audit removed unused gradient-augmented recovery code and its two tests. The brightness-only recovery implementation passed 108 CPU-suite checks, with 13 CUDA-dependent skips.
