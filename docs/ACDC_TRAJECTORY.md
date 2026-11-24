# Median-Dice ACDC contraction trajectory

Patient101,pair1 of the original-default official run. This is the lower-median
Dice case among all100 ACDC directed pairs (Dice0.824868), selected before looking
at trajectory scores. The upper middle case is pair75,Dice0.824903.
Moving is ED frame1; fixed is ES frame14. Normalized trajectory time0 means ED,
time1 means ES. Display is native array axis2, middle slice16 of32 (zero-based).

All27 exported files under `docs/media/acdc_trajectory/` are separate PNGs:
fixed/moving images, pure coloured labels, label overlays, a colour legend, and
SIRENPath/Fitted crossfade/cine-linear/cine-nearest images at five times.
No volumes, reports or metadata are placed in media.

The SIRENPath images use linear **parameter interpolation** of the saved endpoint
networks, without training or flow recovery. Fitted crossfade is linear interpolation
of those decoded endpoint reconstructions, as distinct from Raw crossfade. Both share
one fixed display window from their endpoint reconstructions; each intermediate
is not independently contrast-normalized. Reported NCC uses raw predictions.

The official full cine was downloaded for patient101 into the results artifact
directory. Every selected frame uses the same KAN-IDIR rescale/minmax/crop routine
as the frozen benchmark, with ES header spacing. Prepared cine endpoints were
checked to match the saved ED/ES input arrays within1e-6. The normalized queries
map to one-based frame coordinates1,4.25,7.5,10.75,14. `real_t*.png` interpolates
adjacent **preprocessed** acquired frames at these fractional times; it is a cine
reference approximation, not an acquired frame at every exact query time.
`real_nearest_t*.png` shows the actual nearest acquired frames1,4,8,11,14.
Intermediate segmentation labels are not available and were not synthesized.

Patch NCC is signed Pearson correlation averaged over valid9×9 windows on the
shown slice and9³ windows over the whole prepared volume, stride1. Constant
reference windows are excluded, using the existing endpoint diagnostic conventions.
NCC is computed against both interpolated and nearest-acquired cine references.

[Scores and report](../results/acdc_trajectory_original_20261117/REPORT.md) ·
[All NCC values CSV](../results/acdc_trajectory_original_20261117/patch_ncc.csv) ·
[Provenance/temporal mapping](../results/acdc_trajectory_original_20261117/report.json).
Saved cine/reconstruction/reference volumes are in that results directory.

Automation: `scripts/create_acdc_trajectory.py`, plan-only unless `--run` is set.
Temporal-indexing and signed-NCC synthetic tests passed. No optimizer steps ran.
