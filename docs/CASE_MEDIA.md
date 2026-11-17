# Median-Dice comparison media

The earlier best/median/worst advantage PNG sets have been replaced by one
**median SIRENPath Dice case per dataset**. Lower median is used for even cohorts,
with pair-index tie breaking. Original-default official SIRENPath results and
corrected NIR/DNVF results are used. The separate `acdc_trajectory` subtree is
unchanged. `docs/media` contains PNG panels/legends, with the user-requested
`REPORT.md` inside each dataset’s centroid-view subdirectory as documentation exceptions.

| Dataset | Pair | Classical methods | Neural methods | Middle slice |
|---|---:|---|---|---:|
| Neurite-OASIS |10|FireANTs,NiftyReg|DNVF,NODEO|112|
| ACDC |1|NiftyReg,ConvexAdam|NODEO,DNVF|16|
| LPBA40 |40|ConvexAdam,NiftyReg|DNVF,ccIDIR|80|
| IXI |79|FireANTs,ConvexAdam|DNVF,NIR|112|

External methods are the top two classical and top two neural by **dataset-wide
mean Dice**, not selected using performance on this particular case. SIRENPath
is included alongside these four external methods. Native array axis2 is used;
these slice indices are zero-based and do not assert anatomical orientation.

Each `docs/media/<dataset>/median_dice_pairNNN/` contains shared fixed/moving
reference images and label overlays. Each method folder contains only:

- `intensity_error_overlay.png`
- `deformation_grid.png`
- `warped_labels_overlay.png`

A separate `intensity_error_colorbar.png` uses the shared0–1 normalized error
scale: absolute fixed-minus-warped intensity divided by the fixed image's1st–99th
percentile span,clipped at1. All methods use the same intensity window and magma
colormap with fixed0.65 opacity. No raw warped-image or label-error panels remain.
Class-colour legends are separate PNGs in each dataset directory.

Deformation grids are thick (1.8pt) cyan iso-contours,8voxels apart,of the moving
X/Y coordinate components of the **fixed-to-moving pull map**, sampled on the
fixed middle slice. The saved map includes3D motion; this planar visualization
does not separately represent its out-of-plane component.

## Targeted structure check

Each case has `[dataset]_target_structure_win.png` and `[dataset]_target_structure_loss.png` at
case level. Select the largest positive and most negative original per-class
Dice gaps against the strongest dataset-wide baseline. Each available figure
has a four-contour PNG and a separate anatomy/per-class-Dice legend PNG.
ACDC has no losing label; by user request, its second figure shows the smallest
positive win, named `acdc_target_structure_smallest_win.png`, not a loss.

Target-specific slices always maximize fixed target-label voxel count; ties
prefer volume centre then lowest index. Crops use the union of all four masks,
12pixel padding and1.0 crop scale (no extra enlargement). General comparison panels retain
their middle slices. Superseded target figures and separate legends were removed
from media and archived under results.
[Targets, scores and full selection details](MEDIAN_DICE_CONTOUR_INSPECTION.md).

## Automation and retained artifacts

`scripts/create_median_dice_media.py` plans by default; `--run` performs the
requested recoveries/replays and installs new PNGs only when all four cases finish.
`--render-only` uses saved data without training/recovery. Two synthetic tests
cover selection and common cropping. All20 selected method recoveries completed.

Recovery volumes,selection,logs,inspection.json and INSPECTION.md are outside media:
`results/median_dice_media_20261117/recovery/` and its parent. The old PNGs are
archived at `results/median_dice_media_20261117/previous_media_pngs/`.
Original recovered volumes remain at `results/case_media_recovery_original_20261117`.

Current contour styling:100% base crop; black footer; anatomy bottom-left29.75pt,
method/Dice legend bottom-right26.25pt (1.75× original fonts), condensed font
for long names, compact2.1-inch footer. Dataset-prefixed filenames; no stale
unprefixed structure figures remain in media.

Current output layout: each dataset-prefixed structure PNG contains only the
contour plot. Its corresponding `_legend.png` contains the anatomy name and
method/Dice legend on black, preserving1.75× fonts. Crop scale remains100%.

## Additional centroid-plane panels

All use the median original SIRENPath Dice case and best **case-specific**
classical/neural baselines. Separate source images/labels, intensity-error maps,
red label-error overlays, thin light-blue grids and warped-label/grid combinations,
plus legends and a colocated metrics report:

- [Neurite-OASIS pair10](media/neurite_oasis_pair010_centroid/REPORT.md): FireANTs,DNVF;
  axial array axis1 slice88 and coronal array axis2 slice101 (see report.json for exact centroid).
- [ACDC pair1](media/acdc_pair001_centroid/REPORT.md): FireANTs,DNVF;
  native short-axis and orthogonal long-axis reformat,not standard clinical2CH/4CH.
- [LPBA40 pair40](media/lpba40_pair040_centroid/REPORT.md): ConvexAdam,DNVF;
  axial axis2 and coronal axis1,with display directions from the retained LPS geometry.
- [IXI pair79](media/ixi_pair079_centroid/REPORT.md): ConvexAdam,DNVF;
  axial axis1 and coronal axis2.

ACDC panels preserve its1.5×1.5×3.15mm spacing aspect using2D display-only
nearest-neighbor scaling. Native low through-plane resolution limits the long-axis
reformat’s detail. This does not resample any registration inputs or3D metrics.
The one missing ACDC FireANTs map was replayed and matched recorded Dice exactly.
All other maps were reused. Data/JSON remain under the corresponding
`results/<dataset>_pairNNN_centroid_20261117` directory. Shared automation:
`scripts/create_ixi_centroid_media.py --dataset <dataset> --run`.
