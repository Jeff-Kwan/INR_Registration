# Status: original default restored

On17 November2026 the user reversed the manual promotion. The active recipe is
again registration cosine0.03→0.01; constant0.02 results remain preserved.

The original default official results are used for case-advantage analysis.

# Manual promotion: constant registration0.02

On17 November2026 the user explicitly promoted sweep configuration20 despite its
failure to meet the earlier automated3/4 criterion. Selection used development
results and the user's judgment. This is a manual override, not an automated win.

Only registration changes: cosine0.03→0.01 becomes constant0.02. PD-LK remains
constant0.1,c=4,moment0.005,parameterL2=0.01,and cosine LR1e-4→2e-5 without warmup.
All other method, preparation, recovery and metric settings remain unchanged.

The prior default is retained at `.local/manual_reg002_20261117/previous_default.json`.
Resolved promotion record: [TUNING_PROMOTION.json](TUNING_PROMOTION.json).
Fresh309-fit official run: `results/official_ours_reg002_20261117`.
Previous tuning rankings and official results are preserved as historical results.
