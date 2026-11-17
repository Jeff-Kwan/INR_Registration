# neurite_oasis pair10: foreground-centroid comparisons

Fixed: aligned_norm.nii.gz; moving: aligned_norm.nii.gz. Binary fixed-foreground centroid: [79.81596757284872, 88.2275936270743, 101.1351878193629].

Views: axial:array axis1,slice88, coronal:array axis2,slice101 (zero-based).

Baselines selected by original **case10** Dice: fireants (classical), dnvf (neural). SIRENPath included.

All methods share each view’s slice/crop and one fixed-image1st–99th percentile intensity window across both views. PNGs are standalone panels and legends; this requested report is included alongside them. Volumes and machine-readable metadata remain outside media.

## Metrics recomputed from the saved maps used for plotting

| Method | Dice ↑ | HD95 ↓ | NPJ (%) ↓ | SDlogJ ↓ |
|---|---:|---:|---:|---:|
| Identity | 0.611631 | 2.999561 | 0.000000 | 0.000000 |
| SIRENPath | 0.815422 | 1.496505 | 0.001178 | 0.319301 |
| FireANTs | 0.800926 | 1.634414 | 0.000000 | 0.214851 |
| DNVF | 0.812220 | 1.612847 | 0.000000 | 0.317129 |

Metrics are whole-volume standardized_v1 with the dataset-specific scored labels; identity is the unwarped moving segmentation. HD95 uses prepared-grid spacing; IXI is a nominal unit-voxel benchmark frame,not recovered patient-world geometry. NPJ is the percentage of nonpositive interior Jacobians.

## Original recorded case metrics

| Method | Dice | HD95 | NPJ (%) | SDlogJ |
|---|---:|---:|---:|---:|
| SIRENPath | 0.815422 | 1.496505 | 0.001178 | 0.319301 |
| FireANTs | 0.800926 | 1.634414 | 0.000000 | 0.214851 |
| DNVF | 0.812220 | 1.612847 | 0.000000 | 0.317129 |

Any replay difference is visible by comparing these tables. This rendering/metric script reused existing maps with no fitting or recovery. ACDC FireANTs required one separately documented native replay before rendering because that map had not been retained. Orientation follows dataset-specific source geometry and image checks; artificial benchmark affines are not treated as anatomical. Left/right markers are intentionally omitted. Full score definitions,slice/crop coordinates and intensity window are in report.json.


## Plot styling

Intensity error panels are standalone black-to-bright-red maps, with a common **linear0–1 scale** across both views and all methods. Error=absolute original-image residual divided by the shared fixed-image display span, clipped at1. The colour bar is labelled **Intensity error**. Linear scaling keeps magnitude comparisons direct and does not amplify small background residuals as a log scale would.

Each method has `intensity_error.png`, `label_error_overlay.png`, `deformation_grid.png`, `warped_labels_overlay.png`, and `warped_labels_grid.png`. Display resampling preserves each plane’s physical spacing aspect (notably ACDC’s through-plane spacing); this is2D display only. Both grids use thin0.7pt bright light-blue lines (#66e5ff),8voxel spacing. `label_error_overlay.png` paints every unequal warped/fixed label voxel opaque red over the fixed image, including missed foreground and spurious foreground; equal labels retain grayscale. Its legend is separate. All other source panels and legends remain separate.
