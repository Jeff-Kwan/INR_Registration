# Eight-hour generalist sweep

Historical experiment from 16 November 2026. The [results report](GENERALIST_SWEEP_REPORT_20261116.md) contains the retained OASIS, IXI, and LPBA40 results.

The shared scientific starting point is width 512, 5,000 steps, seed 42,
four-quarter stochastic Euler, shared crops, 32k independent particles per
direction, native voxel units and full-gradient LK/NCC. The initial control
uses existing complete captured execution to reduce dispatch overhead, with
independent temporal and query sampling. Each cohort used the same configuration and its documented intensity preparation.

The frozen, score-blind selection is in ignored
`results/generalist_sweep_20261116/selection.json`, seed 20261116. Screening:

| Cohort | Two screening indices | Further validation indices |
| --- | --- | --- |
| OASIS | 12, 15 | 3, 5, 7, 10, 11, 16 |
| IXI | validation 35, 46 | validation 3, 9, 23, 38, 42, 53 |
| LPBA40 | validation 9, 10 | validation 1, 3, 4, 5, 6, 8 |

Old quick4 pairs and any reverse equivalents are excluded from tuning. Screening
and further validation pair indices are disjoint. This is pair-level separation;
subjects may recur between pairs within the published protocols. Brain tuning uses validation subjects separate from the final test subjects.

Priorities are c, endpoint moment matching, LK weighting/schedules, registration
weighting/warmup, smoothing/schedules, box versus GW-NCC, and inter-SIREN L2.
First inspect the control runtime and errors. The first prepared c alternatives
are 0, 0.25, 1, 16 and 64 against control c=4. Follow-ups should refine promising
regions and test interactions, rather than selecting each coordinate in isolation.
The lower-priority L2 term means an explicitly defined differentiable penalty
between endpoint weights, not an unreported replacement with paired AdamW decay.

Favor low mean normalized error while checking the
worst cohort, per-case regressions, folding, HD95, failures and PCG convergence. No failed cases may be dropped to make a candidate win. Promote contenders to
the additional frozen pairs before locking one shared recipe. No dataset-specific
hyperparameters, rerandomized case sets or additional training seeds are allowed.

Time records include the worker's full case invocation, with phase timings also
retained. Report cold compilation separately when estimating steady-state cost;
do not silently subtract it from actual measured wall time. The final observed
mean must be below 120 seconds per case, with 90 seconds preferred. Reserve
approximately three hours of the eight-hour wall budget for the final evaluation
and reporting. `budget.json` records the original goal deadline.

OASIS completed 19/19 pairs with Dice 0.808155 ± 0.021043, HD95 1.89779 ± 0.44069 voxels, shifted SDlogJ 0.06718 ± 0.00412 and mean runtime 87.12 s. All 152 PCG intervals converged. Whole-grid folding had mean fraction 9.63e-7 and maximum 3.56e-6, with 12 of 19 pairs nonzero. The same-seed repeats differed by 0.00086–0.00167 Dice.

IXI completed 115/115 test fits with 17-group Dice 0.760293 ± 0.029075 and reverse Dice 0.773209 ± 0.020037. Mean runtime was 89.300 s, and all 920 PCG intervals converged.

LPBA40 completed 45/45 pairs with forward Dice 0.718451 ± 0.016208, reverse Dice 0.719836 ± 0.016995 and bidirectional pair mean Dice 0.719143 ± 0.016404. Mean runtime was 87.035 s, and all 360 PCG intervals converged.
