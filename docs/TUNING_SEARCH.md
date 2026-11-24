# SIRENPath automated tuning search

The original search evaluated 20 one-dimensional sweep configurations and an
assembled combination on the frozen 128-pair development cohort, with seed 42,
5,000 updates and canonical recovery. The retired sweep controller and generated
presets are no longer distributed. The [reproducibility archive](reproducibility/README.md)
preserves the settings and results.

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

The recorded decision was to retain the canonical configuration. The search
did not promote a finalist. The decision and ranking are preserved in the
[tuning records](reproducibility/README.md).
