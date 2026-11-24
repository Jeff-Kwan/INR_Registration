# Data preparation

Shapes and voxel spacings below are in saved array-axis order. Anatomical segmentation labels are used only for evaluation. We apply no percentile clipping across these datasets.

## Learn2Reg OASIS

Extract `OASIS.zip` and use the affine-aligned, cropped MRIs in `imagesTr` and their 35-label segmentations in `labelsTr` unchanged. The 19 public-validation pairs use fixed subjects 0395–0413, each paired with the next subject as moving. Keep the released 160 × 224 × 192 grid and 1 mm isotropic spacing. At load time, scale each MRI independently to [0, 1] using its full-volume minimum and maximum.

## Neurite-OASIS

From `neurite-oasis.v1.0.tar`, sort the 414 subject directory names lexicographically and take the last 50. Form 49 consecutive pairs with the earlier subject fixed. Copy `aligned_norm.nii.gz` and `aligned_seg35.nii.gz` byte for byte. Keep the released 160 × 192 × 224 grid, 1 mm isotropic spacing, and already normalized image values in [0, 1]. Apply no further spatial or intensity processing.

## IXI

Use the preprocessed `IXI_data.zip` release with its 403/58/115 train/validation/test split and `atlas.pkl`. Convert each pickle’s MRI and segmentation arrays to NIfTI, storing the label codes as uint8. Preserve the 160 × 192 × 224 array order and the released MRI values in [0, 1] without additional normalization. Write an identity affine to record unit spacing along all three axes in a benchmark frame. The pickles supply no patient-space affine. Pair the moving atlas with each of the 115 fixed test subjects.

## LPBA40

Use the LONI native-space MRI and masks and delineation-space MRI and label archives. For each subject, use `delineation.to.native.air` to resample `native.brain.bfc` and `native.tissue` onto the delineation grid with inverse AIR resampling, float64 linear interpolation, and border extension. Take `delineation.structure.label` directly on that grid. Divide the resampled MRI by its mean over voxels with `abs(tissue - 3) < 1e-5` (white matter), then set MRI voxels with tissue values ≤ 0 to zero. Apply the tissue mask only to the MRI. Keep all 56 anatomical label codes.

Crop the MRI and labels by removing 11/10, 20/5, and 0/21 voxels from the low/high ends of the three axes, respectively. Save the 160 × 192 × 160 arrays at 1 mm isotropic spacing with negative affine scales on the first two axes and origin (−11, −20, 0) mm. Keep the white-matter-normalized intensities without further scaling or clipping. The 45 test pairs are all combinations of S11, S12, S13, S15, S24, S28, S29, S32, S34, and S38.

## ACDC

Use official test patients 101–150. Read the end-diastolic (ED) and end-systolic (ES) frame IDs from each `Info.cfg` and load the corresponding MRIs and ground-truth segmentations as float32. Each patient contributes both ED→ES and ES→ED.

Rescale both phases using the ES header spacing divided elementwise by (1.5, 1.5, 3.15) mm, preserving float32 spacing arithmetic. Use scikit-image 0.26.0 `rescale` with `mode="edge"`, cubic MRI interpolation (`order=3`), and nearest-neighbor label interpolation (`order=0`). Retain the default anti-aliasing for float32 arrays when downsampling and the default clipping to each input array’s value range (`clip=True`).

Independently min–max scale each full resampled MRI to [0, 1]. Then crop each MRI and its labels to 128 × 128 × min(resampled depth, 32), starting at `floor((resampled shape - crop shape) / 2)` on each axis. Depth is 19–32 slices with no padding. Save the MRI as float32 and the labels as uint8. Use the target spacing and crop offset for the benchmark affine, without another intensity transform at load time.
