# DIR-Lab quick4: four smallest prepared cases

The frozen selection is **case01, case03, case04, case05**. It uses the four
smallest prepared pairs by summed T00/T50 voxel count, with case-number tie
breaking. It is not random and uses no registration scores. Execution is in
case-number order; size order is 01, 04, 03, 05.

| Case | Native shape XYZ (each endpoint) | Voxels per endpoint |
| --- | --- | ---: |
| 01 | 256 × 256 × 94 | 6,160,384 |
| 03 | 256 × 256 × 104 | 6,815,744 |
| 04 | 256 × 256 × 99 | 6,488,064 |
| 05 | 256 × 256 × 106 | 6,946,816 |

`medical/protocols/dirlab_quick4.json` records both endpoint shapes, spacing,
voxel counts and the complete ten-case size inventory. This selection is frozen
against prepared native grids, not reranked after working-grid resampling.
It is a development subset biased toward small cases, not a representative
cohort result. Keep the existing ccIDIR conventions and single seed.

Preview without creating outputs or starting workers:

```bash
/venv/main/bin/python scripts/run_dirlab_cases.py --subset quick4 --plan-only \
  --config medical/configs/physical_working_grid.json \
  --output results/dirlab_quick4_physical --gpus 0 1 2 3
```

A separately authorized experiment can omit `--plan-only`. The DIR-Lab launcher
executes by default, unlike the Learn2Reg launchers: do not omit that flag merely
to inspect the plan. Without `--subset`, it still selects all ten cases. Execution
uses persistent GPU workers and the shared queue. No experiment was launched
while preparing this subset. Three-axis probes are not enabled by this selection;
our ordinary random two-pair estimator remains the default.

## Six-layer default registration comparison (2026-11-13)

The user requested six hidden layers on quick4 after returning to defaults.
The run uses `medical/configs/ablations/native_registration_6layers.json`, which changes
only layer count from the default. Width is 256, seed 42, steps 5000 and
batch size 32768. Endpoint NCC with moment weight 0.01, PD-LK weight 0.1 and
registration NCC weight 0.01 are active. Every logged total was checked against
these terms. All four cases completed and all 64/64 PCG intervals converged.

The four-layer reference is the current-policy run in
`results/dirlab_quick4_native_cropcenter_20261113/`. Its saved config matches
the new run except for layer count. Forward IDIR-rounded TRE in mm is:

| Case | Four layers | Six layers |
| --- | ---: | ---: |
| 01 | 1.2093 | 1.5879 |
| 03 | 2.6613 | 3.4573 |
| 04 | 3.3495 | 3.4669 |
| 05 | 3.2048 | 4.5000 |
| Mean | 2.6062 | 3.2530 |

Reverse rounded TRE also increases, from 2.6248 to 3.3679 mm. Mean endpoint
reconstruction NCC in the same 9 mm landmark patches falls from 0.7694 to
0.5486. Thus the fidelity gain observed under reconstruction-only MSE does
not carry into this joint-training recipe. In the last 25 logged crops per
case, mean endpoint NCC loss rises from 0.01964 to 0.03501, PD-LK from
0.02130 to 0.07163 and registration NCC loss from 0.02334 to 0.04194.
All three training terms finish higher in the deeper model under this budget.

Mean end-to-end runtime is 145.07 seconds per case versus 109.81 for the saved
control. The native DIR-Lab adapter does not record a separate training timer.
Mean lung folding is 0.000104% for the six-layer run. These are single-seed
quick4 development results.

Scores, checkpoints, logs, frozen configuration, `patch_fidelity.json` and the
full `comparison.json` are in `results/dirlab_quick4_registration_6layers_20261113/`.

## Frozen six-layer checkpoint diagnostics

The user requested displacement and feature-fidelity diagnostics for the trained
six-layer checkpoints. The existing frozen replay recovered both directions,
reproduced saved continuous and rounded TRE within 0.01 mm, cleaned scratch,
and replayed 256 ordinary crop draws per case. Outputs are in
`results/dirlab_gap_6layers_20261113/`. The four-layer reference uses
`results/dirlab_gap_diagnostic_20261113/`.

| Observed motion | Landmarks | Four-layer continuous TRE | Six-layer continuous TRE | Four-layer endpoint patch NCC | Six-layer endpoint patch NCC |
| --- | ---: | ---: | ---: | ---: | ---: |
| <2 mm | 106 | 0.976 | 1.400 | 0.727 | 0.431 |
| 2–5 mm | 379 | 1.340 | 1.927 | 0.754 | 0.483 |
| 5–10 mm | 334 | 1.912 | 2.907 | 0.786 | 0.581 |
| ≥10 mm | 381 | 5.020 | 5.293 | 0.782 | 0.618 |

TRE is forward continuous error in mm. Endpoint patch NCC averages the moving
and fixed model-versus-observed 9 mm patch score for each landmark, then pools
landmarks in the motion bin. Of the extra total TRE under six layers, 85.2%
comes from landmarks moving below 10 mm, with 47.2% from the 5–10 mm group.
The large-motion group still under-travels, with signed along-motion error
changing from −4.556 to −4.707 mm and perpendicular error from 1.400 to
1.587 mm. Its true-minus-predicted observed-image patch NCC gap increases
from 0.285 to 0.441. Reverse diagnostics show the same broad deterioration.

The modeled moving/fixed patches at true anatomical correspondences correlate
less well under six layers, with mean NCC about 0.609 versus 0.800 in the
four-layer model. The observed CT correspondence patches are identical between
the two model evaluations. Crop exposure is also identical, with 1196/1200
forward landmarks exposed in 256 draws. This result localizes the regression
to degraded learned endpoint structure and trajectory fitting across the
motion range. The higher final reconstruction, PD-LK and registration losses
also indicate a poorer joint fit under the default 5000-step recipe.

`comparison.json` records weighted displacement-bin metrics and contributions
to the added error. `feature_motion.json` retains scalar per-landmark feature
NCC and the corresponding motion bins. `transport.json` contains the analytic
image-trajectory derivative probe. No fitting was performed in these replays.

## Original default architecture trained for 10000 steps

The user requested 10k steps on the original setup and checkpoint probes.
This was interpreted as the default four-layer width-256 architecture, trained
from initialization. `medical/configs/ablations/native_registration_10k.json` changes
only `training.steps` from 5000 to 10000. The same exponential learning-rate
schedule spans the longer run. Seed 42, 32768 points, endpoint NCC with
moment matching, PD-LK 0.1 and registration NCC 0.01 are retained.
All four cases completed 10000 steps and every logged objective was verified.

| Case | Default 5k forward rounded TRE | Default 10k forward rounded TRE |
| --- | ---: | ---: |
| 01 | 1.2093 | 1.2136 |
| 03 | 2.6613 | 2.4249 |
| 04 | 3.3495 | 2.8274 |
| 05 | 3.2048 | 3.0208 |
| Mean | 2.6062 | 2.3717 |

TRE is in mm. Reverse rounded TRE improves from 2.6248 to 2.4295 mm.
All 64/64 PCG intervals converge. Mean end-to-end runtime rises from 109.81
to 219.75 seconds per case. Mean lung folding is 0.0000271% at 10k.
Checkpoints and logs are in `results/dirlab_quick4_default_10k_20261113/`.

The requested frozen recovery replay reproduced continuous and rounded TRE
within 0.01 mm in both directions, cleaned scratch, and replayed 256 crop
draws per case. Feature fidelity and analytic trajectory derivatives were
also probed at the same landmark windows and 33 temporal samples.

| Observed motion | 5k continuous TRE | 10k continuous TRE | 5k endpoint patch NCC | 10k endpoint patch NCC |
| --- | ---: | ---: | ---: | ---: |
| <2 mm | 0.976 | 1.006 | 0.727 | 0.771 |
| 2–5 mm | 1.340 | 1.293 | 0.754 | 0.795 |
| 5–10 mm | 1.912 | 1.804 | 0.786 | 0.823 |
| ≥10 mm | 5.020 | 4.433 | 0.782 | 0.816 |

Overall endpoint feature NCC improves from 0.7694 to 0.8072. For the ≥10 mm
group, forward along-motion shortfall falls from 4.556 to 3.919 mm,
perpendicular error from 1.400 to 1.347 mm, and the true-minus-predicted
observed-image NCC gap from 0.285 to 0.231. Reverse results agree.
The final 25 logged crops per case average endpoint NCC loss 0.01471,
PD-LK 0.01904 and registration NCC loss 0.02143, versus 0.01964, 0.02130
and 0.02334 at 5k. The analytic transport slopes on high-motion landmarks
move closer to 1 in all cases, although absolute residual RMS is mixed.

Probe outputs, per-landmark scalar scores and the aggregate comparison are
in `results/dirlab_gap_default_10k_20261113/`. The combined report is
`comparison.json`, feature scores are in `feature_motion.json`, and derivative
probes are in `transport.json`. These are single-seed quick4 results.
