# IXI pair79: foreground-centroid axial and coronal comparisons

Fixed subject471,moving atlas. Binary fixed-foreground centroid: [81.69400269346929, 91.39896075643544, 106.10408247011682]. Axial axis1 slice91; coronal axis2 slice106 (zero-based).

Baselines selected by original **case79** Dice: convexadam (classical), dnvf (neural). SIRENPath included.

All methods share each view’s slice/crop and one fixed-image1st–99th percentile intensity window across both views. PNGs are standalone panels and legends; this requested report is included alongside them. Volumes and machine-readable metadata remain outside media.

## Metrics recomputed from the saved maps used for plotting

| Method | Dice ↑ | HD95 ↓ | NPJ (%) ↓ | SDlogJ ↓ |
|---|---:|---:|---:|---:|
| Identity | 0.433966 | 5.804761 | 0.000000 | 0.000000 |
| SIRENPath | 0.746923 | 3.133072 | 0.001472 | 0.584654 |
| ConvexAdam | 0.733520 | 3.625039 | 0.111746 | 0.780927 |
| DNVF | 0.746591 | 3.072734 | 0.000000 | 0.359765 |

Metrics are whole-volume standardized_v1,30 scored IXI labels; identity is the unwarped moving segmentation. HD95 uses IXI’s released unit-voxel benchmark frame (nominal mm),not recovered patient-world geometry. NPJ is the percentage of nonpositive interior Jacobians.

## Original recorded case metrics

| Method | Dice | HD95 | NPJ (%) | SDlogJ |
|---|---:|---:|---:|---:|
| SIRENPath | 0.746920 | 3.133072 | 0.001472 | 0.584655 |
| ConvexAdam | 0.733280 | 3.632908 | 0.111730 | 0.780859 |
| DNVF | 0.746591 | 3.072734 | 0.000000 | 0.359765 |

Any replay difference is visible by comparing these tables. Existing saved maps were reused; no training,registration or checkpoint recovery ran. Orientation was checked from image anatomy because the NIfTI affine is an artificial unit benchmark frame. Left/right markers are intentionally omitted. Full score definitions,slice/crop coordinates and intensity window are in report.json.


## Plot styling

Intensity error panels are standalone black-to-bright-red maps, with a common **linear0–1 scale** across both views and all methods. Error=absolute original-image residual divided by the shared fixed-image display span, clipped at1. The colour bar is labelled **Intensity error**. Linear scaling keeps magnitude comparisons direct and does not amplify small background residuals as a log scale would.

Each method has `intensity_error.png`, `label_error_overlay.png`, `deformation_grid.png`, `warped_labels_overlay.png`, and `warped_labels_grid.png`. Both grids use thin0.7pt bright light-blue lines (#66e5ff),8voxel spacing. `label_error_overlay.png` paints every unequal warped/fixed label voxel opaque red over the fixed image, including missed foreground and spurious foreground; equal labels retain grayscale. Its legend is separate. All other source panels and legends remain separate.
