# IXI/TransMorph and LPBA40/SITReg benchmark protocols

Canonical cohort, preparation and metric choices: [Dataset protocols](DATASETS.md).

Audit and preparation: 2026-11-16 (Hong Kong). Both datasets are prepared and
connected to our paired-SIREN method. No training or saved-model recovery was
launched. The NIR/Mindboggle reproduction is no longer being pursued.

| Benchmark | Frozen cohort | Local status |
| --- | --- | --- |
| IXI/TransMorph | 403 training, 58 validation, 115 test subjects; supplied atlas | All 577 volumes converted without changing array values; 115 test jobs ready. |
| LPBA40/SITReg | 25 training, 5 validation, 10 test subjects | All 40 subjects prepared using author AIR interpolation, normalization and crop; 45 test jobs ready. |

The [machine-readable protocol](../medical/protocols/brain_benchmarks.json)
pins source revisions, exact subjects/pairs, label tables, metric conventions
and reference-helper hashes. See [training setup](BRAIN_TRAINING.md) for the
plan-only launcher and validation evidence. The published cohort and scoring
protocols are retained while fitting our method independently to each pair.

## 1. IXI: TransMorph release

### Sources and acquisition

Author repository revision: `6357a1d7fc44c36db9b1d1ccaa372409253142cf`.

- [Dataset, weights and reproduction instructions](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/6357a1d7fc44c36db9b1d1ccaa372409253142cf/IXI/TransMorph_on_IXI.md).
- [Released IXI archive](https://drive.google.com/uc?export=download&id=1-VQewCVNj5eTtc3eQGhTM2yXBQmgm8Ol), Google Drive file ID `1-VQewCVNj5eTtc3eQGhTM2yXBQmgm8Ol`, named `IXI_data.zip`.
- [Preprocessing outline](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/6357a1d7fc44c36db9b1d1ccaa372409253142cf/PreprocessingMRI.md).

Use the preprocessed release, not a fresh preprocessing of the original IXI
images. It packages FreeSurfer skull stripping, affine alignment, intensity
normalization and automated segmentations. The outline is useful provenance,
but it does not pin every processing version or reconstruct the released split.
Its commands use FreeSurfer's Talairach transform; do not replace this with a
new MNI152 alignment and claim identical preparation.

The archive has `IXI_data/Train/*.pkl`, `Val/*.pkl`, `Test/*.pkl`, and
`atlas.pkl`. Each pickle holds `(image, segmentation)` arrays, shape
`160×192×224`; images are already normalized to [0,1]. The archive's existing
memberships define the 403/58/115 split. Preserve their names and record their
hashes after extraction. A new random 7:1:2 split is not the same benchmark.
The supplied atlas comes from CycleMorph. The pickle interface does not supply
a NIfTI world affine; preserve the released array geometry instead of guessing
orientation from an unrelated NIfTI conversion.

### Pair direction and map convention

The [inference dataset loader](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/6357a1d7fc44c36db9b1d1ccaa372409253142cf/IXI/TransMorph/data/datasets.py)
always loads `x = atlas`, `y = subject`. It warps the atlas image/labels into
the subject grid and compares against the subject. Thus the 115 primary scores
are **atlas-as-moving, test-subject-as-fixed**. The sampling map goes from
subject coordinates to atlas coordinates. A reverse-direction score is an
additional result, not a replacement for this primary direction.

The released [inference script](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/6357a1d7fc44c36db9b1d1ccaa372409253142cf/IXI/TransMorph/infer_TransMorph.py)
currently defaults to `Val/`, despite its general inference name. Set it to
`Test/` for the 115-case benchmark. Its raw `glob` order is not a stable subject
identifier: output rows are called `p_0`, `p_1`, etc. A local adapter should save
the actual subject filename with each score and freeze a sorted manifest.

### Exact label warping and scoring

The [label transform](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/6357a1d7fc44c36db9b1d1ccaa372409253142cf/IXI/TransMorph/data/trans.py)
maps the released FreeSurfer IDs to 46 consecutive classes including background.
Unknown IDs map to background. Keep all 46 classes during label warping, even
though only 30 anatomical labels contribute to the reported accuracy.

For each class, make a binary channel, warp it with **trilinear interpolation**,
then take the channel argmax. The implementation uses `align_corners=True`
and default zero padding. This is not nearest-neighbor interpolation of the
integer label volume, nor linear interpolation of integer label IDs. Outside
the image all channels are zero, so argmax selects background. Flow channels
follow array-axis order; the sampler reverses channels when constructing the
PyTorch grid. See [spatial transformer and Dice utilities](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/6357a1d7fc44c36db9b1d1ccaa372409253142cf/IXI/TransMorph/utils.py).

The [anatomical-label correction](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/6357a1d7fc44c36db9b1d1ccaa372409253142cf/IXI/Anatomical_Structures.md)
specifies 30 labels, correcting the paper's count of 29. However, the published
[analysis script](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/6357a1d7fc44c36db9b1d1ccaa372409253142cf/IXI/analysis.py)
first averages the left/right Dice for 13 bilateral structures, retains four
midline structures (brainstem, third ventricle, fourth ventricle and CSF), then
averages these **17 group scores** equally. Do not silently replace that with
an equally weighted 30-label mean. For each label, Dice is
`2*intersection/(predicted_count + target_count + 1e-5)`; two absent labels
therefore score zero in this implementation.

The paper-style analysis computes its standard deviation over all 17×115
group/case values. That differs from the standard deviation of 115 per-case
means. Save both, naming them explicitly. The inference script's printed
46-class Dice is also not the final 17-group benchmark score.

Folding is the whole-grid proportion with determinant **≤0**, calculated with
`jacobian_determinant_vxm`: add the identity grid to the voxel displacement,
use NumPy spatial gradients and take the determinant. The table expresses this
fraction as a percentage. No foreground masking is applied. HD95 is not part
of this IXI release's main reproduction table; adding it would be supplementary.

As a read-only check, aggregating the released `IXI/Results/` CSV files with this
grouping gives affine Dice **0.38580969** and TransMorph Dice **0.75268277**.
For the latter, group/case SD is **0.12572224**, SD across per-case means is
**0.03048542**, and mean folding is **1.50208025%**. These reproduce the inspected
CSV contents, not new model evaluations. The README's rounded results are not
byte-for-byte identical to these CSV aggregates; retain the pinned reference.

### If reproducing the TransMorph model itself

The benchmark can evaluate our method without retraining TransMorph. For an
author-model reproduction, start with the released weights and
`IXI/TransMorph/infer_TransMorph.py`. Explicitly freeze the checkpoint filename
and SHA-256 instead of relying on its last naturally sorted directory entry.

The released [training loop](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/6357a1d7fc44c36db9b1d1ccaa372409253142cf/IXI/TransMorph/train_TransMorph.py)
uses batch size 1, 500 epochs, Adam with AMSGrad, LR 1e-4 and zero decay; the
LR follows `(1 - epoch/500)^0.9`, rounded to eight decimal places. Its objective
adds 9³-window squared LNCC and L2 displacement-gradient regularization with
weights 1 and 1. Training jointly flips both images randomly across the three
axes. Validation selects checkpoints using a 30-label Dice helper with
nearest-neighbor label warping, while the released final test script uses the
one-hot/trilinear/argmax convention above. Keep this selection/scoring
distinction. The [model configuration](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/6357a1d7fc44c36db9b1d1ccaa372409253142cf/IXI/TransMorph/models/configs_TransMorph.py)
is authoritative; preserve it instead of reconstructing architecture defaults
from another TransMorph directory.

## 2. LPBA40: SITReg's 45-pair protocol

Author repository revision: `a665a46a3553b26ade6ff5f57815e5e8ec43a948`.
[Paper](https://www.melba-journal.org/papers/2024:026.html),
[data preparation and exact split](https://github.com/honkamj/SITReg/blob/a665a46a3553b26ade6ff5f57815e5e8ec43a948/src/data/lpba40/data.py),
[selected configuration](https://github.com/honkamj/SITReg/blob/a665a46a3553b26ade6ff5f57815e5e8ec43a948/src/scripts/configs/sitreg/lpba40/cc_grad_1.0_very_very_deep.json).

### Cohort and pairs

Test subjects, in code order:
`S11 S12 S13 S15 S24 S28 S29 S32 S34 S38`.

Validation subjects:
`S14 S16 S19 S26 S39`.

The other 25 are training subjects. The loader uses
`itertools.combinations(subjects, 2)` because
`iterate_inference_pairs_in_both_directions` is false: **45 test entries and
10 validation entries**. Exact ordered input tuples are recorded in the JSON
companion. The first image is warped toward the second for the forward
evaluation; the inverse map is evaluated against the first image.

The selected configuration additionally has `do_reverse_inference: true`:
the author implementation calls the model in both input orders, and records
forward and inverse metrics under separate `input_order_0_*` and
`input_order_1_*` names. These are not 45 independent one-direction scores.
Retain the direction/order fields, plus the equal-direction mean as an
explicit supplementary summary. Do not label a new pooling convention as the
paper's exact table aggregation without verifying it against author outputs.

### Required source data and preparation

SITReg obtains **two** official LONI releases:

- [Delineation-space MRI and labels](https://resource.loni.usc.edu/atlasfiles/LPBA40_Subjects_Delineation_Space_MRI_and_label_files.zip).
- [Native-space MRI and masks](https://resource.loni.usc.edu/atlasfiles/LPBA40_Subjects_Native_Space_MRI_data_and_masks.zip).

The loader does not simply use the delineation-space skull-stripped image.
It transforms native `native.brain.bfc` images and tissue masks into delineation
space using each subject's `delineation.to.native.air` with the author's
[inverse AIR implementation](https://github.com/honkamj/SITReg/blob/a665a46a3553b26ade6ff5f57815e5e8ec43a948/src/data/lpba40/air.py).
Both are sampled linearly. Labels come from the delineation-space manual
segmentation and are converted without that image interpolation step.

It divides intensities by the mean over tissue values within 1e-5 of class 3
(white matter), then zeroes image voxels outside tissue>0. The misleadingly
named `_apply_mask_to_label` is passed the **image** filename here; it does not
mask the anatomical evaluation labels. The conversion assigns negative
affine scales on the first two array axes. Preserve that geometry.

Apply the exact crop `[[11,10],[20,5],[0,21]]` to images and labels, producing
`160×192×160`. Downsampling factors are `[1,1,1]`; min-max normalization is
disabled in this SITReg config. Use the same cropped evaluation geometry.
The author preparation reorganizes and removes intermediate source files;
if reused locally, operate on fresh copies and retain our original archives.
Local preparation now completes these operations in a separate ignored
`prepared/` directory, retaining every original input. The cropped NIfTI
affine includes the crop-origin shift; array values match the author loader.

### Scoring and model reproduction

The [evaluator's label table](https://github.com/honkamj/SITReg/blob/a665a46a3553b26ade6ff5f57815e5e8ec43a948/src/data/lpba40/evaluation.py)
contains **all 56 anatomical labels**, including cerebellum and brainstem.
Do not substitute a 54-label or seven-region LPBA40 protocol.

The [shared segmentation evaluator](https://github.com/honkamj/SITReg/blob/a665a46a3553b26ade6ff5f57815e5e8ec43a948/src/data/base.py)
warps integer labels using nearest-neighbor interpolation and clears invalid
samples. Per-label Dice has no epsilon; two empty masks give NaN, excluded by
the masked-invalid mean. HD95 uses the DeepMind surface-distance functions
with spacing `[1,1,1]`, hence voxel units in the evaluation grid. Invalid
per-label HD95 values are excluded by that implementation; record excluded
counts and failure denominators when adapting it.

The configuration computes Jacobians at one million seeded random points,
in float64, using perturbations of 1e-7 voxels. It reports determinant `<0`
and determinant standard deviation, **not** Learn2Reg's shifted SDlogJ.
Seed is `8524561973 XOR author_pair_index`, where the author index is zero-based
(our launcher uses `pair_index - 1`). Matching only a dense-grid folding
fraction does not reproduce this sampled metric. Also retain the author's
inverse-consistency diagnostics and mapping representation.

For an actual SITReg baseline, the selected model has seven resolution levels,
NCC window 7, gradient penalty weight 1, Adam LR 1e-4 and betas (0.9,0.999),
batch size 1, and 4 epochs × 5000 sampled training pairs. Select the best epoch
on the separate validation subjects. The standard inference config resamples
when composing; do not substitute its complete-composition variant.
Its environment pins `composable-mapping==0.1.2` and specifies
`deformation-inversion-layer>=1.1.2`; freeze the resolved versions for a run.

The author CLI supports metrics without persistent map outputs:

```bash
python -m scripts.inference --training-root TRAINING_ROOT --data-root DATA_ROOT \
  --division test --model-name MODEL_NAME --devices cuda:0 \
  --evaluate --do-not-save-outputs --epoch SELECTED_EPOCH
```

This command is documentation, not an executed experiment. Our method can use
the cohort/preparation/evaluator while retaining its own fitting recipe; that
would reproduce the benchmark protocol, not the SITReg algorithm.

## Local acquisition and checks

`/venv/main/bin/python scripts/download_brain_benchmarks.py --dataset all` downloads source ZIPs into
ignored `Data/IXI_TransMorph/source/` and `Data/LPBA40/source/`, checks ZIP paths,
verifies member CRCs while extracting, retains nested source archives, and
records SHA-256 provenance. It refuses existing dataset destinations. No
preprocessing, affine fitting, optimization or model recovery is invoked.

See each dataset's ignored `acquisition.json` for archive hashes and unchanged
member inventories. Protocol-specific preparation and our method adapters are now complete.
The separate [training guide](BRAIN_TRAINING.md) describes inputs, checks and
launch guards.

Acquisition completed during this audit. IXI's ZIP is 1,541,951,071 bytes,
SHA-256 `ec980b436c7cad6ae9cdcbd779829457184d7bb459ef4e9c2f40095335dbe734`.
The LPBA40 delineation ZIP is 79,496,443 bytes,
SHA-256 `fed208e6b6c41202857a0877a0b24d6660ba6ce600b180cd1c7454f26905b8d8`;
the native ZIP is 431,028,873 bytes,
SHA-256 `1e9f20f6f70c0bdfd945f6e12738b7430546866d2002e7aa1f824988592d3a98`.
These are computed provenance hashes, not comparisons with publisher-provided
checksums. ZIP member CRC validation succeeded. All 577 IXI tuples and all 40
LPBA40 subjects were read and checked; details are in each ignored
`verification.json`. No original input file was changed.


## Completed preparation and local evaluator

`scripts/prepare_brain_benchmarks.py` writes separate `prepared/images/*.nii`
and `prepared/labels/*.nii`, plus `protocol.json` and `preparation.json` for
each dataset. It refuses existing preparations and removes incomplete staging
on exceptions. Every written volume was reloaded and compared exactly with its
conversion array. A subsequent full audit checked SHA-256 hashes of all 617
prepared image/label pairs, original inputs and all three source archives,
and checked each prepared affine/shape.

For LPBA40 subjects S01 and S11, the original author's five preprocessing
methods were executed on temporary copies. Every cropped image and label voxel
matched our preparation exactly. The affine preserves the author's negative
first two axes and applies the crop-origin shift. No source was rewritten.
Validation records are kept in each ignored `prepared/validation.json`.

Our method uses these prepared intensity values directly, with evaluation
labels excluded from fitting and sampling. `medical.brain_evaluation` implements
the IXI streamed 46-channel warp and 17-group score, and uses pinned SITReg
reference method bodies for LPBA40 scoring. The reference files are attributed
under [SITReg's MIT license](../medical/protocols/LICENSE.SITReg.txt).

The launcher plans 115 IXI and 45 LPBA40 test fits, one seed per pair. It records
primary and inverse-map scores separately. LPBA40's author also runs the model
with swapped input order; our one paired fit yields two maps and does not add
a second optimization or claim that extra input-order run. Runtime and PCG
convergence are recorded with the scores. No persistent deformation map or
warped-image output is written.
