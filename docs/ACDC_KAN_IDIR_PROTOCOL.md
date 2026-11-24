# KAN-IDIR ACDC protocol audit

Checked 2026-11-16 (Hong Kong). Author repository revision:
`3ac12e200159811eafb6192f3f8db3afef2a9d1a` (2026-07-13).
This is a protocol/code audit; no registration experiment was run.

## Published cohort and executable preparation

The [paper, section 2.5](https://www.mdpi.com/2504-4990/8/7/184)
uses 50 test subjects and both ED→ES and ES→ED directions. The executable
[runner](https://github.com/anac0der/KAN-IDIR/blob/3ac12e200159811eafb6192f3f8db3afef2a9d1a/run_acdc.py)
specifies patients 101–150: 100 directed optimizations per seed. It visits all
ED→ES pairs first, then all ES→ED pairs.

The dataset README still says preparation instructions are forthcoming, but
`utils/load_utils.py` and `utils/general.py` implement the actual procedure.

| Step | Public author code |
| --- | --- |
| Frames | Read the first two `Info.cfg` lines as ED and ES; use their frame numbers |
| Loading | `nibabel.get_fdata()`, then float32 for MRI and segmentations; expects uncompressed `.nii` paths |
| Source spacing | ES header `get_zooms()`, reused for both phases |
| Resampling | `skimage.transform.rescale` with factors `source_spacing / (1.5,1.5,3.15)` |
| MRI interpolation | Cubic (`order=3`), edge extension (`mode='edge'`) |
| Label interpolation | Nearest (`order=0`), edge extension |
| Normalization | Independent min–max over each entire resampled MRI, before cropping |
| Crop | Integer centered `128×128×C`, where `C=min(resampled_z,32)` |
| Padding | No explicit padding; depth is not padded to 32 |
| Geometry | Array rescaling; source world affines are not used to create a shared physical reference grid |

[Preparation source](https://github.com/anac0der/KAN-IDIR/blob/3ac12e200159811eafb6192f3f8db3afef2a9d1a/utils/general.py#L89-L105)
and [loader](https://github.com/anac0der/KAN-IDIR/blob/3ac12e200159811eafb6192f3f8db3afef2a9d1a/utils/load_utils.py#L107-L155).

`rescale` defaults, including anti-aliasing for downsampling, remain in effect.
The repository does not pin scikit-image, SciPy or nibabel versions, so code-level
reproduction needs a recorded environment; historical bitwise array identity
cannot be inferred from the public dependency list alone. Normalizing a constant
image would divide by zero in the author code.

The official files just acquired here have matching ED/ES shapes, header spacings
and raw affines for all 50 subjects. Based on the author size rule, their cropped
depths are 19,24,25,27,29,30 or32; every resampled XY extent exceeds128. Thus the
negative-index small-XY crop edge case does not apply to this cohort.

## Evaluation conventions and caveats

The [evaluation helper](https://github.com/anac0der/KAN-IDIR/blob/3ac12e200159811eafb6192f3f8db3afef2a9d1a/utils/eval_utils.py)
warps moving labels onto the fixed grid with nearest-neighbor PyTorch grid sampling,
`align_corners=True` and default zero padding. Physical HD95 uses MedPy with
spacing `(1.5,1.5,3.15)` in native array-axis order. Fold percentage uses the
analytical Jacobian of the author's learned displacement mapping at all grid
coordinates, counts `detJ<0` and multiplies by100. Dense finite-difference
Jacobian scoring of our recovered maps is a numerical adaptation, not the same
Jacobian estimator.

The [label metric code](https://github.com/anac0der/KAN-IDIR/blob/3ac12e200159811eafb6192f3f8db3afef2a9d1a/utils/metrics.py)
intersects predicted and reference label sets, converts the intersection to a
list and skips its first element, intending to drop background. Therefore a
foreground class that disappears from a warped prediction can be omitted from
both Dice and HD95. A synthetic check with truth `[0,1,2,3]` and prediction
`[0,1,0,3]` gives author Dice1.0 versus fixed-three-label Dice2/3. Any reproduction
should expose the scored labels/counts and retain a fixed-label1–3 diagnostic;
do not silently describe the author rule as always averaging all three classes.

The final runner's Dice SD pools label/direction/run scores; it is not simply SD
of the100 registration-level means. HD95 and folding SD use registration/run
scalars. The timer encloses `fit()` and excludes preprocessing and subsequent
evaluation. Our end-to-end timings must remain separately identified.

The paper describes ten random seeds; the README recommends three to five for
ACDC, and the runner defaults to one with a time-derived seed. These statements
do not alter SIRENPath's single-seed policy. No multiseed run was launched.
ACDC training samples all image voxels; labels do not define an anatomical mask.

## Selected protocol and current readiness

KAN-IDIR is now the **sole supported ACDC protocol**. The canonical specification
is [Dataset protocols: ACDC](DATASETS.md#acdc--kan-idir-only).

KAN-IDIR-matched preparation and our cohort adapter/launcher are connected.
The 100 prepared phase outputs were compared against the retained author
functions. See [ACDC readiness and commands](ACDC.md). The inspected upstream files, SHA-256
hashes and synthetic metric example remain under ignored
`results/acdc_protocol_audit_20261116/`.
