# SIRENPath automated tuning search

The user authorized20 complete one-dimensional sweep configurations, one assembled
combination, and conditional promotion/full official comparison execution.
All21 configurations receive the same128 frozen development pairs, seed42,
5000 updates, canonical architecture and frozen recovery. No screening subsets.

## Search artifacts

Output: `results/tuning_search_20261117`.

- `catalog.json`, `configs/`:20 resolved scalar/schedule candidates and baseline hash.
- `status.json`, `pipeline.log`: current stage and controller log.
- `runs/<ID>/progress.json`, `summary.json`:128-case progress and metrics per sweep.
- `ranking.json`, `RANKING.md`: independently ranked categories and full diagnostics.
- `combined.json`, `combined/`: assembled configuration and its full128-case run.
- `decision.json`: final comparison against default, per-dataset deltas and eligibility.
- `selected.json`, `previous_default.json`: promoted configuration and prior default,
  created only when a candidate qualifies.
- `official_promoted/`: fresh309-fit official results, created only after promotion.

All original dataset preparations, old results and checkpoints are retained.
No external baseline is rerun by this workflow. The full official run uses our
method on Neurite-OASIS49, ACDC100, LPBA4045 and IXI115 pairs. It is entirely
separate from the tuning cohort; test metrics never select a configuration.

## Independent ranking and assembly

Categories: local transport loss (LT) weight schedule, registration weight schedule, endpoint parameter penalty weight,
LT c, moment weight and LR schedule. Constants belong to schedule categories.
Rank within each category by equal-weight mean of the four dataset mean Dice
scores. The reference provides the already-tested default value for each category,
including c=4; exact ties prefer that value. There is no per-dataset improvement
requirement at this stage. Every category provides a winner and the combination
is assembled/tested regardless of its predicted improvement.

Copy only the winning category's keys into a fresh copy of the original default.
Run this combination once on all128 pairs, even if it equals an existing candidate.
Independent optima can interact; the actual combination run supplies its evidence.

## Final selection and promotion

Compare each of the six category winners and the assembled combination to the
original default, reusing the completed sweep results for individual winners.
Deduplicate category winners when multiple categories select the default.
A candidate qualifies only if:

1. Its dataset mean Dice is **strictly higher on at least3 of4 datasets**.
2. Its dataset mean HD95 is **strictly lower on at least3 of4 datasets**.

The two sets of three datasets may differ. Ties do not count as improvements.
HD95 counts only when both candidate and default have all32 valid case distances
in that dataset; no missing-surface exclusion can improve the comparison.
Every sweep must complete all128 cases with valid Dice before ranking can proceed.
Complete configuration and cohort provenance must match. Failures stop the pipeline
and remain recorded; no selection over a partially completed search.

Among qualifying finalists, maximize equal-dataset mean Dice; exact ties use
stable candidate ID order. NPJ, SDlogJ, HD95 and runtime remain in the report;
no additional topology or runtime promotion gate was requested. This rule need
not imply an improvement in overall mean Dice: one dataset can regress enough
to outweigh three smaller improvements. The user's3/4 rule remains the criterion.

If no finalist qualifies, retain the current default and stop. Otherwise save the
previous default, replace `medical/configs/default.json`, update METHOD/README/agent
recipe references and promotion evidence, then launch the309 official SIRENPath
fits using a snapshot of the selected configuration. All outputs use fresh paths.

## Commands and execution

```bash
# Preparation only; output must be absent:
bash scripts/launch_tuning_search.sh prepare --output results/tuning_search_20261117

# Execute the authorized workflow:
bash scripts/launch_tuning_search.sh run --output results/tuning_search_20261117 --gpus 0 1 2 3 4 5 6 7

# Recompute rankings after all20 sweeps have completed:
bash scripts/launch_tuning_search.sh rank --output results/tuning_search_20261117
```

A process lock prevents duplicate controllers. A rerun reuses only complete runs
with matching input/configuration provenance; it refuses to overwrite partial runs.
The default is checked against its pre-search hash before any promotion.

Initial budget:21×25–30min≈8.75–10.5h, plus≈50min for the official run if promoted.
Update the ETA using the first full default run. GPU0–7, one persistent worker per
GPU within each configuration, fresh model/optimizer per pair. Prior completed
experiments are never reused as tuning evidence.

Launched in tmux session `sirenpath_tuning_20261117` at18:00 UTC on16 November2026
(02:00 HKT on17 November). Controller log and exit_code are retained under the
search output; status.json records each stage. Initial completion estimate for
tuning:10:45–12:30 HKT; official run, if promoted, adds approximately50 minutes.
