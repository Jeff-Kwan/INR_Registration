# Signed SIRENPath win/loss structure contours

For each original-default median-Dice case, select the largest strictly positive and most negative full-volume per-label Dice gaps between SIRENPath and the strongest dataset-wide baseline. Keep the best classical and best neural contours as additional comparisons. This remains explicitly post-hoc extreme-structure selection.

Each figure uses the slice with the **most fixed target-label voxels**, independently for each target; ties prefer volume centre, then lowest index. Crops contain all four contours plus12pixels and a1.0 crop scale (no extra enlargement). The anatomy/per-class-Dice legend is saved separately.

| Dataset | View | Anatomy | Comparator | Original SIRENPath Dice | Original baseline Dice | Slice |
|---|---|---|---|---:|---:|---:|
| neurite_oasis | win | Right Amygdala | dnvf | 0.883939 | 0.853283 | 117 |
| neurite_oasis | loss | Right Inferior Lateral Ventricle | dnvf | 0.475073 | 0.529968 | 112 |
| acdc | win | Right ventricle | nodeo | 0.773737 | 0.701374 | 9 |
| acdc | smallest win | Left ventricle | nodeo | 0.905631 | 0.880435 | 9 |
| lpba40 | win | Left cuneus | dnvf | 0.713069 | 0.580361 | 81 |
| lpba40 | loss | Right angular gyrus | dnvf | 0.677470 | 0.739629 | 109 |
| ixi | win | Left choroid plexus | dnvf | 0.398431 | 0.308797 | 91 |
| ixi | loss | Right Cerebellum White Matter | dnvf | 0.657712 | 0.747684 | 73 |

ACDC has no negative per-label gap against NODEO for this case. By user request, its second panel now shows the smallest strictly positive win (left ventricle), saved as `acdc_target_structure_smallest_win.png`. It must be captioned as a smallest win, not a loss.

Files at case level: `[dataset]_target_structure_win.png` and `[dataset]_target_structure_loss.png`. ACDC uses `acdc_target_structure_smallest_win.png` in place of a loss panel. Each anatomy figure combines the four contours and its legend; old contour-only PNGs, separate legends and method-specific target panels are removed from media and archived outside it.

Legend scores are recovered **full-volume per-class Dice** for the actual displayed predictions. Selection uses original recorded metrics; both original scores and shown-slice scores are retained in `results/median_dice_media_20261117/four_contour_summary.json` and per-case `signed_contour_inspection.json`. NODEO/classical replay differences remain auditable.

The fixed image is the shared background. Colours: green ground truth, cyan dashed classical, orange dash-dot neural, magenta SIRENPath. Anatomy names use the released Neurite lookup table or frozen ACDC/LPBA/IXI protocols.

General comparison panels still use middle slices. Only these target-specific views use maximum target occupancy. `acdc_trajectory` is unchanged. No model recovery or fitting ran for this regeneration.

Current contour styling:100% base crop; black footer; anatomy bottom-left29.75pt,
method/Dice legend bottom-right26.25pt (1.75× original fonts), condensed font
for long names, compact2.1-inch footer. Dataset-prefixed filenames; no stale
unprefixed structure figures remain in media.

Current output layout: each dataset-prefixed structure PNG contains only the
contour plot. Its corresponding `_legend.png` contains the anatomy name and
method/Dice legend on black, preserving1.75× fonts. Crop scale remains100%.
