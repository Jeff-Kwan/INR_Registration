# acdc pair1: foreground-centroid comparisons

Fixed: patient101_ES.nii.gz; moving: patient101_ED.nii.gz. Binary fixed-foreground centroid: [52.16581029598515, 63.58165478167432, 14.634033408225067].

Views: short_axis:array axis2,slice15, long_axis_reformat:array axis1,slice64 (zero-based).

Baselines selected by original **case1** Dice: fireants (classical), dnvf (neural). SIRENPath included.

All methods share each view’s slice/crop and one fixed-image1st–99th percentile intensity window across both views. PNGs are standalone panels and legends; this requested report is included alongside them. Volumes and machine-readable metadata remain outside media.

## Metrics recomputed from the saved maps used for plotting

| Method | Dice ↑ | HD95 ↓ | NPJ (%) ↓ | SDlogJ ↓ |
|---|---:|---:|---:|---:|
| Identity | 0.696193 | 9.013839 | 0.000000 | 0.000000 |
| SIRENPath | 0.824868 | 6.751882 | 0.000232 | 0.290495 |
| FireANTs | 0.809959 | 7.523669 | 0.000000 | 0.244633 |
| DNVF | 0.792725 | 9.901882 | 0.000000 | 0.227795 |

Metrics are whole-volume standardized_v1 with the dataset-specific scored labels; identity is the unwarped moving segmentation. HD95 uses prepared-grid spacing; IXI is a nominal unit-voxel benchmark frame,not recovered patient-world geometry. NPJ is the percentage of nonpositive interior Jacobians.

## Original recorded case metrics

| Method | Dice | HD95 | NPJ (%) | SDlogJ |
|---|---:|---:|---:|---:|
| SIRENPath | 0.824868 | 6.751882 | 0.000232 | 0.290495 |
| FireANTs | 0.809959 | 7.523669 | 0.000000 | 0.244633 |
| DNVF | 0.792725 | 9.901882 | 0.000000 | 0.227795 |

Any replay difference is visible by comparing these tables. This rendering/metric script reused existing maps with no fitting or recovery. ACDC FireANTs required one separately documented native replay before rendering because that map had not been retained. Orientation follows dataset-specific source geometry and image checks; artificial benchmark affines are not treated as anatomical. Left/right markers are intentionally omitted. Full score definitions,slice/crop coordinates and intensity window are in report.json.


## Plot styling

Intensity error panels are standalone black-to-bright-red maps, with a common **linear0–1 scale** across both views and all methods. Error=absolute original-image residual divided by the shared fixed-image display span, clipped at1. The colour bar is labelled **Intensity error**. Linear scaling keeps magnitude comparisons direct and does not amplify small background residuals as a log scale would.

Each method has `intensity_error.png`, `label_error_overlay.png`, `deformation_grid.png`, `warped_labels_overlay.png`, and `warped_labels_grid.png`. Display resampling preserves each plane’s physical spacing aspect (notably ACDC’s through-plane spacing); this is2D display only. Both grids use thin0.7pt bright light-blue lines (#66e5ff),8voxel spacing. `label_error_overlay.png` paints every unequal warped/fixed label voxel opaque red over the fixed image, including missed foreground and spurious foreground; equal labels retain grayscale. Its legend is separate. All other source panels and legends remain separate.
