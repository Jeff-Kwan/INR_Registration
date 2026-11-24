# Official runs

Active recipe restored by user instruction on17 November2026: registration
cosine0.03→0.01. The constant0.02 trial below is historical; both runs are preserved.
[Original-default case advantages](CASE_ADVANTAGE.md#completed-original-default-analysis-17-november2026).

## SIRENPath — manually promoted constant registration0.02,17 November2026

Launched **11:40 HKT on17 November2026** in tmux session
`sirenpath_reg002_20261117`, using all eight RTX5090 GPUs.
Run: `results/official_ours_reg002_20261117`;309 fits across Neurite-OASIS49,
ACDC100,LPBA4045 and IXI115. Data preparation and evaluation cohorts remain frozen.

The user manually promoted registration constant0.02 after inspecting the tuning
sweeps; this did not pass the earlier automated promotion criterion. The local transport loss (LT) weight remains
constant0.1,c=4,moment0.005,endpoint parameter penalty weight 0.01,cosine LR1e-4→2e-5,no warmup,
5000 updates,seed42. Other canonical settings and standardized_v1 metrics remain.
[Promotion evidence](TUNING_PROMOTION.md). Previous results and tuning evidence
are retained; the older results below used registration cosine0.03→0.01.

Expected duration50–60min; initial finish estimate **12:30–12:40 HKT**.
[Progress](../results/official_ours_reg002_20261117/progress.json) and
[summary](../results/official_ours_reg002_20261117/summary.json).
Controller log: `.local/manual_reg002_20261117/run.log`; exit_code records completion.

## SIRENPath — cosine schedule, 2026-11-16

Launched at **18:39 HKT on 16 November 2026** under supervisor service
`inr-ours-comparisons`, using all eight RTX 5090 GPUs and one persistent worker
per GPU. Run directory: `results/official_ours_cosine_20261116`.

| Dataset | Fits |
|---|---:|
| Neurite-OASIS | 49 |
| ACDC | 100 |
| LPBA40 test | 45 |
| IXI test | 115 |
| Total | 309 |

Canonical paired SIRENs, 5,000 updates, AdamW decay 0.001, seed 42,
cosine learning rate **1e-4 → 2e-5**, no warmup. The four resolved configurations
are snapshotted in the run's `configs/` directory. Dataset directions and prepared
grids follow [DATASETS.md](DATASETS.md). Shared standardized_v1 scoring reports
Dice, HD95, NPJ (%) and SDlogJ. L2R-OASIS ablations are excluded.

Completed successfully; final results and runtime are below.

Live progress and failures are recorded after every completed case in
[progress.json](../results/official_ours_cosine_20261116/progress.json).
Final cohort aggregates and failures are recorded in
[summary.json](../results/official_ours_cosine_20261116/summary.json).
The [plan](../results/official_ours_cosine_20261116/plan.json) records all commands,
configuration hashes, preparation hashes and the scoring-source hash.
Individual case directories retain checkpoints, metrics, provenance and timings;
dense maps and warped images are not retained.

Supervisor logs: `.local/ours-comparisons.log` and
`.local/ours-comparisons-error.log`. Per-case logs sit alongside each case directory.
Supervisor automatic restart is disabled to prevent accidental reruns into an
existing output directory. The supervised job exited after completing all cases.

## Completed results

All **309/309 fits succeeded**, zero process failures. Finished at approximately
**19:30 HKT on 16 November 2026**, about **51.5 minutes** after launch.

Values are mean ± population SD. Runtime is worker wall time per case, including input handling, training, recovery and scoring.

| Dataset | Dice ↑ | HD95 ↓ | NPJ (%) ↓ | SDlogJ ↓ | HD95 pairs | Runtime (s/case) |
|---|---:|---:|---:|---:|---:|---:|
| Neurite-OASIS | 0.812 ± 0.023 | 1.797 ± 0.384 | 0.000959 ± 0.001290 | 0.341 ± 0.048 | 44/49 | 87.5 ± 18.7 |
| ACDC | 0.811 ± 0.076 | 6.806 ± 3.041 | 0.001105 ± 0.002266 | 0.340 ± 0.109 | 100/100 | 73.2 ± 7.4 |
| LPBA40 | 0.719 ± 0.017 | 5.879 ± 0.421 | 0.000043 ± 0.000099 | 0.280 ± 0.037 | 45/45 | 77.8 ± 5.5 |
| IXI | 0.739 ± 0.032 | 3.365 ± 0.774 | 0.001840 ± 0.002052 | 0.557 ± 0.109 | 115/115 | 80.3 ± 2.0 |

Neurite-OASIS HD95 is defined for 44/49 cases: pairs 2, 3, 11, 12 and 15 have undefined required surface distances under the strict shared evaluator. These are metric omissions, not registration process failures. The other metrics include all intended cases. Classical baselines have 45/49 HD95 cases, so their Neurite HD95 averages use a different denominator; use a common valid-pair subset for direct HD95 ranking. IXI distances use the released unit-voxel benchmark frame.

## Neural external baselines — launched 2026-11-16

Launched **20:00 HKT, 16 November 2026**, in tmux session
`inr-neural-baselines`.
All eight RTX 5090 GPUs run one fit per GPU from a shared queue.
Methods: **ccIDIR, NIR, NODEO, DNVF**, 309 fits each, **1,236 fits total**.
Each method covers Neurite-OASIS 49, ACDC 100, LPBA40 test 45 and IXI test 115.
Single seed 42; pinned author protocols and recorded localization adapters.
Dataset preparation is frozen: no preparation or resampling was invoked.
The run plan records preparation/configuration/evaluator hashes and the exact
ordered inputs. Resolved method configurations are copied into `configs/`.
All 1,236 planned jobs passed input-existence checks before launch.

Run directory: `baselines/results/official_neural_20261116`.
[Plan](../baselines/results/official_neural_20261116/plan.json) ·
[Live progress](../baselines/results/official_neural_20261116/progress.json) ·
[Final summary](../baselines/results/official_neural_20261116/summary.json).
Per-case logs, metrics and checkpoints are retained. No dense maps or warped
images are persisted. Standardized Dice, HD95, NPJ (%) and SDlogJ aggregates,
scored denominators and failures are written by the launcher after every case.

## Neural baseline completion

All **1,236/1,236** fits completed successfully at **22:16 HKT on 16 November 2026**,
with zero process failures; wall duration approximately **2 h 16 min**.
Final scores, runtime and HD95 coverage are consolidated in
[BASELINES_REPORT.md](BASELINES_REPORT.md); per-case evidence and exclusions are in
[EXTERNAL_BASELINES.md](EXTERNAL_BASELINES.md). The final machine-readable summary is
`baselines/results/official_neural_20261116/summary.json`.

## Ablation launch — 2026-11-16

Launched all seven SIRENPath ablations at **22:40 HKT on 16 November**, in tmux
session `inr-sirenpath-ablations`, using all eight previously idle RTX 5090 GPUs.
133 evaluations: 95 fresh trained fits and 38 no-training controls. Run 7 is
scheduled after run 5 and reuses its exact per-pair checkpoints. Initial estimate:
**20–35 minutes**, approximately **23:00–23:15 HKT**, subject to first-time
compilation and diagnostic/recovery costs. Baseline investigation remains CPU-only.
Dataset preparation remains frozen.

[Live report](../results/sirenpath_l2r_oasis_ablations_20261116/ABLATIONS_REPORT.md) ·
[Progress](../results/sirenpath_l2r_oasis_ablations_20261116/progress.json) ·
[Plan](../results/sirenpath_l2r_oasis_ablations_20261116/plan.json) ·
[Final summary](../results/sirenpath_l2r_oasis_ablations_20261116/summary.json).

Completion update: **133/133 ablation evaluations completed, zero process failures**, at **22:55 HKT on 16 November 2026**. Final results are in the linked live report and summary.

Completed aggregate scores: [ABLATIONS_REPORT.md](ABLATIONS_REPORT.md).

The corrected NIR and DNVF runs completed all 309 fits per method. [Rerun protocol](NEURAL_RERUNS.md).

## Thin-volume NIR completion

The four failed ACDC/NIR cases11,47,61,97 were rerun successfully with the
deterministic adaptive NCC support rule: largest available odd extent up to9
per sampled axis, giving9×9×7 for these cases. The same kernel is used for
all NCC moments, normalization and auxiliary background-mask operations.
Supported9×9×9 losses and gradients are bitwise identical to the preceding
source implementation. No sampler, input preparation or other case was changed.

ACDC/NIR now has **100/100 successful fits**. Its row was directly replaced
in the separate corrected-rerun table in [BASELINES_REPORT.md](BASELINES_REPORT.md).
All **618/618 corrected NIR/DNVF fits now succeed**, zero remaining process failures.
[Supplementary four-case plan](../baselines/results/official_neural_20261116/nir_thin_volume_plan.json) ·
[Four-case completion](../baselines/results/official_neural_20261116/nir_thin_retry_summary.json).
