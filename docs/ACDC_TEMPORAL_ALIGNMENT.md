# ACDC monotone temporal alignment

Patient101,pair1,the same original-default lower-median ACDC case and middle
slice16 used in [the trajectory comparison](ACDC_TRAJECTORY.md).

## Sampling and score

Decode53 equally spaced model times,including both endpoints: delta t=1/52.
Compare against all14 **acquired** cine frames1–14 from ED to ES: normalized
cine spacing1/13. There are four model intervals per acquired cine interval.
No temporal interpolation of cine images is used in this analysis. These are
normalized-time sampling intervals,not an asserted physical acquisition rate.

For every model/cine pair compute the existing signed9³ whole-volume Patch NCC,
valid windows,stride1,excluding constant reference windows. Build a53×14 matrix
separately for SIREN parameter interpolation and endpoint-reconstruction crossfade.
Both methods use identical preprocessing,metrics,grid and alignment constraints.

## Constrained alignment

Dynamic programming chooses one strictly increasing model index per acquired
cine frame. Anchor the first/last pair at ED/ES. Permit model-index increments
1–16,corresponding to local cine-time/model-time slopes0.25–4; constrain normalized
time deviation to0.25. Maximize summed PNCC minus0.01×log(step/4)² for speed
stretch and0.01×log(step/previous_step)² for speed changes. Divide the objective
by14 for reporting. These constraints were fixed before examining results.
The resulting time warp is piecewise linear through its14 selected correspondences.

Unaligned scores use model indices0,4,…,52 at the same14 cine frames. The aligned
and unaligned means therefore have identical reference frames and counts.
An independent best-model-time score per frame is also reported as an optimistic
ceiling without order or endpoint constraints. It is not a physical trajectory.

## Results

| Method | Unaligned mean PNCC | Aligned mean PNCC | Gain | Independent-frame ceiling |
|---|---:|---:|---:|---:|
| SIREN weight path |0.917636|0.919325|+0.001689|0.920546|
| Reconstruction crossfade |0.918011|0.918527|+0.000516|0.918882|

Excluding the two endpoints,means are0.909549→0.911520 for SIREN and
0.909987→0.910589 for crossfade. Both selected warps deviate at most0.057692
from linear normalized time. Aligned SIREN exceeds aligned crossfade by0.000798.

These are small,post-hoc improvements selected and measured on the same cine
sequence. Whole-volume similarity includes stationary surrounding anatomy.
This is a temporal-coherence diagnostic,not independent evidence that learned
velocity or intermediate anatomy is physically correct.

## Outputs

The PNG-only directory `docs/media/acdc_trajectory/aligned/` contains:

- [Time warp](media/acdc_trajectory/aligned/time_warp.png): horizontal model time,
  vertical normalized cine time,diagonal linear reference,both constrained curves.
- [PNCC over time](media/acdc_trajectory/aligned/pncc_over_time.png): aligned and
  unaligned SIREN/crossfade on the same cine-time axis,with means in the legend.
- Individual [SIREN](media/acdc_trajectory/aligned/siren_pncc_over_time.png) and
  [crossfade](media/acdc_trajectory/aligned/crossfade_pncc_over_time.png) curves.
- [PNCC gain](media/acdc_trajectory/aligned/pncc_alignment_gain.png).
- Separate similarity-matrix heatmaps with alignment curves.
- Each method/frame subdirectory has separate aligned model,unaligned model and
  actual cine PNG panels. All use middle slice16. Images have no axes or titles.

Scores,matrix arrays,constraints,temporal correspondences and per-frame values
are outside media at `results/acdc_trajectory_alignment_20261117/`:
[Report](../results/acdc_trajectory_alignment_20261117/REPORT.md) ·
[Full metadata](../results/acdc_trajectory_alignment_20261117/report.json).

## Automation

`scripts/align_acdc_trajectory.py` is plan-only by default. `--run` performs frozen
inference and the scan; it does not train or run registration optimizers.
`--plots-only` regenerates PNCC curves from cached scores without inference.
Three synthetic tests validate linear/known warps,strict ordering and bounds,
feasibility rejection,and parity with the existing Patch NCC implementation.
