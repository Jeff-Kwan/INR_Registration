# Full-gradient covariant residual screen across five cases

Seed-42, 5,000-step fits use MSE endpoints, shared-grid PD-LK and bidirectional
box NCC. For each selected case, only `covariant_residual_alpha` changes. Final
recovery is the unchanged PCG+RK2 pipeline. The endpoint PSNR gate was explicitly
skipped in every run. S02/S03 include alpha 0, .5 and 1; the additional cohorts
include alpha 0 and 1. These are the middle sorted cases from the existing
benchmarks: OASIS_0006_0000, HippocampusMR_0011_0000 and AbdomenCTCT_0006_0000.

| Cohort / case | Alpha | Dice | HD95 (mm) | Folded voxels (%) | PCG converged | Training (s) |
|---|---:|---:|---:|---:|---:|---:|
| LPBA S02 | 0 | .710539 | 6.2394 | 0 | 13/16 | 96.19 |
| LPBA S02 | .5 | .712256 | 6.2509 | 0 | 16/16 | 95.05 |
| LPBA S02 | 1 | .713283 | 6.2059 | 0 | 15/16 | 95.38 |
| LPBA S03 | 0 | .716784 | 6.3716 | 0 | 16/16 | 95.40 |
| LPBA S03 | .5 | .716333 | 6.4114 | 0 | 16/16 | 94.90 |
| LPBA S03 | 1 | .717358 | 6.3528 | 0 | 15/16 | 94.06 |
| OASIS_0006_0000 | 0 | .753711 | 3.0827 | .000727 | 13/16 | 96.85 |
| OASIS_0006_0000 | 1 | .759022 | 3.0679 | .000552 | 15/16 | 95.06 |
| HippocampusMR_0011_0000 | 0 | .781638 | 2.7247 | .006104 | 16/16 | 96.82 |
| HippocampusMR_0011_0000 | 1 | .778227 | 2.8953 | 0 | 16/16 | 95.39 |
| AbdomenCTCT_0006_0000 | 0 | .515445 | 22.1908 | .002162 | 13/16 | 97.63 |
| AbdomenCTCT_0006_0000 | 1 | .469430 | 23.7103 | .003115 | 15/16 | 95.39 |

Alpha 1 improved Dice and HD95 slightly on OASIS, reduced folding on the selected
Hippocampus case while worsening its Dice/HD95, and worsened both scores and
folding on the selected Abdomen CT case. On S02 and S03 alpha 1 led within each
case, though the score changes were small. Across the three additional cases,
mean Dice changed from .68360 to .66889 and mean HD95 from 9.3327 to 9.8912 mm
when moving from alpha 0 to 1. Training took about 94–98 seconds for every run;
the endpoint gate skip and near-tolerance PCG misses remain part of the evidence.

The findings are mixed and come from one case per added cohort and one seed.
They do not establish a cross-cohort benefit or select alpha 1 as a new default.
The detailed S02/S03 comparison is also available in
[covariant_residual_s02_s03_20261112.md](covariant_residual_s02_s03_20261112.md).
Run checkpoints and scalar logs remain in ignored fresh `results/covariant_residual_*`
directories.
