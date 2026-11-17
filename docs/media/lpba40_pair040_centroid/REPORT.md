# lpba40 pair40: foreground-centroid comparisons

Fixed: S32.nii; moving: S29.nii. Binary fixed-foreground centroid: [80.68987017888965, 81.52374851651163, 75.32732195417609].

Views: axial:array axis2,slice75, coronal:array axis1,slice82 (zero-based).

Baselines selected by original **case40** Dice: convexadam (classical), dnvf (neural). SIRENPath included.

All methods share each view’s slice/crop and one fixed-image1st–99th percentile intensity window across both views. PNGs are standalone panels and legends; this requested report is included alongside them. Volumes and machine-readable metadata remain outside media.

## Metrics recomputed from the saved maps used for plotting

| Method | Dice ↑ | HD95 ↓ | NPJ (%) ↓ | SDlogJ ↓ |
|---|---:|---:|---:|---:|
| Identity | 0.538954 | 8.180707 | 0.000000 | 0.000000 |
| SIRENPath | 0.719953 | 5.618652 | 0.000000 | 0.318626 |
| ConvexAdam | 0.713809 | 5.581815 | 0.001574 | 0.244589 |
| DNVF | 0.723059 | 5.907359 | 0.000000 | 0.290101 |

Metrics are whole-volume standardized_v1 with the dataset-specific scored labels; identity is the unwarped moving segmentation. HD95 uses prepared-grid spacing; IXI is a nominal unit-voxel benchmark frame,not recovered patient-world geometry. NPJ is the percentage of nonpositive interior Jacobians.

## Original recorded case metrics

| Method | Dice | HD95 | NPJ (%) | SDlogJ |
|---|---:|---:|---:|---:|
| SIRENPath | 0.719953 | 5.618652 | 0.000000 | 0.318626 |
| ConvexAdam | 0.713802 | 5.582367 | 0.001574 | 0.244532 |
| DNVF | 0.723059 | 5.907359 | 0.000000 | 0.290101 |

Any replay difference is visible by comparing these tables. This rendering/metric script reused existing maps with no fitting or recovery. ACDC FireANTs required one separately documented native replay before rendering because that map had not been retained. Orientation follows dataset-specific source geometry and image checks; artificial benchmark affines are not treated as anatomical. Left/right markers are intentionally omitted. Full score definitions,slice/crop coordinates and intensity window are in report.json.


## Plot styling

Intensity error panels are standalone black-to-bright-red maps, with a common **linear0–1 scale** across both views and all methods. Error=absolute original-image residual divided by the shared fixed-image display span, clipped at1. The colour bar is labelled **Intensity error**. Linear scaling keeps magnitude comparisons direct and does not amplify small background residuals as a log scale would.

Each method has `intensity_error.png`, `label_error_overlay.png`, `deformation_grid.png`, `warped_labels_overlay.png`, and `warped_labels_grid.png`. Display resampling preserves each plane’s physical spacing aspect (notably ACDC’s through-plane spacing); this is2D display only. Both grids use thin0.7pt bright light-blue lines (#66e5ff),8voxel spacing. `label_error_overlay.png` paints every unequal warped/fixed label voxel opaque red over the fixed image, including missed foreground and spurious foreground; equal labels retain grayscale. Its legend is separate. All other source panels and legends remain separate.
