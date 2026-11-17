# Dataset protocols — canonical reference

This is the single source of truth for our dataset cohorts, preparation, pair
directions and evaluation. The README and agent guide link here. Dataset-specific
documents provide implementation details; historical reports do not select protocols.
Method architecture, losses, optimizers and training integrators are specified
separately in [METHOD.md](METHOD.md) and resolved experiment configurations.

## Current comparison selection

Official comparisons use Neurite-OASIS, ACDC, LPBA40 and IXI, with the frozen
cohorts and pair directions below. All reruns use the shared standardized_v1
Dice, physical HD95, NPJ percentage and SDlogJ evaluator specified in
[Case runner](CASE_RUNNER.md#standardized-official-comparisons). The original
author metric definitions below serve as protocol reference. L2R-OASIS is
reserved for the seven-run [SIRENPath internal ablations](SIRENPATH_ABLATIONS.md). The current comparison
launchers exclude it, DIR-Lab and AbdomenCTCT. Prepared data are preserved.

## Selected benchmarks

| Benchmark | Selected protocol | Full evaluation fits | Prepared input | Status |
| --- | --- | ---: | --- | --- |
| DIR-Lab | ccIDIR/IDIR native T00–T50 | 10 | `Data/DIRLab4DCT_IDIR` | Prepared; runner connected |
| OASIS / Learn2Reg | Official directed public validation | 19 | `Data/OASIS` | Prepared; runner connected |
| AbdomenCTCT | Official Learn2Reg directed public validation | 45 | `Data/AbdomenCTCT` | Prepared; runner connected |
| IXI | TransMorph author atlas-to-subject test split | 115 | `Data/IXI_TransMorph/prepared` | Prepared; runner connected |
| LPBA40 | SITReg frozen test combinations | 45 | `Data/LPBA40/prepared` | Prepared; runner connected |
| ACDC | KAN-IDIR test patients 101–150, both phase directions | 100 | `Data/ACDC/prepared` | Prepared; runner connected |
| Neurite-OASIS | Jena et al. last-50/consecutive-49 cohort | 49 | `Data/NeuriteOASIS/prepared` | Prepared; runner connected |

These are seven distinct benchmarks. In particular, Neurite-OASIS does not replace
or extend the official 19-pair Learn2Reg OASIS protocol. The five established
non-ACDC/non-Neurite cohorts total 234 fits. Adding both new cohorts gives 383
fits with all seven runners connected. A recovered inverse map is an additional
score, not automatically an additional optimization or reverse input pair.

## Development-data availability and overlap

See [the verified development-data inventory](TUNING_DATA.md). In particular,
L2R-OASIS IDs0365–0414 are the same50 subjects as our Neurite evaluation cohort,
confirmed by voxelwise label-array comparison after axis reorientation. Existing
L2R ablation pairs0395–0414 are therefore **not patient-disjoint** from Neurite
comparison data. The [frozen tuning cohort](TUNING_COHORT.md) now defines32 pairs
per dataset (128 total), excluding these subjects and all other test subjects,
with the protocol-shared IXI atlas exception. IXI/LPBA author validation splits
remain reserved.

## Shared rules

Preserve original releases, archives, access terms, required citations and prepared
inputs. Record source revisions, ordered pairs, geometry and SHA-256 checksums.
Labels and landmarks are evaluation-only: never optimize, select crops, stop or
tune individual cases using them. DIR-Lab's independently generated lung masks
have the crop-center-only role described below. LPBA40's author preparation mask
is an explicit intensity-preprocessing step, not an anatomical supervision loss.

Our method fits independently per pair with a single seed (42 unless explicitly
specified). Author multiseed protocols do not authorize multiseed runs of our method.
Preparation and plan inspection never authorize training or checkpoint recovery.
Keep validation and test results separate; if public-validation scores guide recipe
selection, report them as development evidence. Report all intended cases and
failures, per-case metrics, dataset aggregation and runtime scope. Do not pool
Dice and TRE. Save checkpoints, configurations and metrics; do not persist dense
registration maps or warped volumes. Recovery scratch must be cleaned on exceptions.

Intensity handling is dataset-specific: our DIR-Lab, Learn2Reg OASIS and abdomen
adapters independently min–max normalize images; IXI and LPBA40 preserve their
protocol-prepared values. Neurite preparation preserves released intensities
byte-for-byte. ACDC normalizes at the exact author stage specified below.

## DIR-Lab — ccIDIR / IDIR

**Cohort and directions.** All ten official 4DCT cases, T00 inspiration fixed and
T50 expiration moving, 300 corresponding landmarks per case. The primary point
map is T00 coordinates → T50 coordinates, which samples the moving T50 image
onto the fixed T00 grid. Report reverse TRE separately.

**Preparation.** Native voxel grids and subject-specific spacings; official raw
index/axis conversion to NIfTI and zero-based XYZ landmarks, without spatial
resampling. Prepared affines use a local arbitrary-origin frame. Source/landmark
checksums are in the local preparation manifest. R231 masks exist for both phases;
lungmask 0.2.22 weights SHA-256
`d5d2fc3df432115933491f115b283e94d1f5c841c0e82f6e7bd2d65d3ccde69f`.
Exact equivalence to the authors' original masks is not claimed.

**Training mask policy.** Choose an endpoint equally, sample a mask voxel and
center the shared crop there, shifting it to fit the image. Reconstruction and
registration use that crop; direction samples remain independent. No occupancy
rejection or mask-based loss gating. PD-LK uses every cell; NCC includes every
particle, including crop exits. Border interpolation and a live full-image boundary
penalty apply.

**Evaluation.** IDIR-compatible rounded physical TRE in mm is primary;
continuous-coordinate TRE is supplemental. Retain all 300 landmarks and report
outside-domain counts. Report whole-grid and T00-lung folding separately:
DIR-Lab's `folding_fraction` alias is the lung-ROI value, not whole-grid folding.
`medical.dirlab_evaluation.evaluate_case` accepts in-memory 300×3 T50 XYZ positions
in T00 landmark order. External ccIDIR baselines may use their published 50-seed
plan; our method remains single-seed.

[IDIR paper](https://proceedings.mlr.press/v172/wolterink22a.html),
[IDIR pinned metric source](https://github.com/MIAGroupUT/IDIR/blob/9ab3ccfbbf09dc8fc65fa848332e7d9a0b3b81b3/utils/general.py).
Frozen optional quick4: cases **01,03,04,05**, the four smallest prepared pairs;
[selection and launcher](DIRLAB_QUICK4.md). Preview with `--plan-only` because
this launcher executes by default.

## OASIS — official Learn2Reg public validation

**Cohort.** Exactly 19 directed official pairs among 20 subjects: fixed IDs
0395–0413 paired respectively with moving IDs 0396–0414. This is public validation,
not hidden-test evaluation and not the Neurite-OASIS/Jena et al. cohort.

**Preparation.** Unchanged released affine-aligned/cropped images and labels,
160×224×192 in release array-axis order, 1 mm spacing. Do not repeat alignment or
resampling. Independent image min–max normalization occurs in our adapter.

**Evaluation.** Anatomical labels 1–35, excluding background; evaluation-only.
Use the pinned Learn2Reg Dice, HD95 in **voxels** (unit spacing), and shifted
SDlogJ `std(log(clip(detJ+3,1e-9,1e9)))`. True unshifted interior folding is a
separate diagnostic. Preserve official missing-label and interpolation conventions.
The evaluator/configuration revision is
`88475095e35442087a24439ff34ec09f54bc7dac`.

[Official pair/metric configuration](https://github.com/MDL-UzL/L2R/blob/88475095e35442087a24439ff34ec09f54bc7dac/evaluation/evaluation_configs/OASIS_VAL_evaluation_config.json).
Implementation: `medical.oasis_method`; [runner and details](OASIS.md).
Plan-only by default. Frozen optional quick4: official pair indices **1,4,8,9**,
selection seed 42; [quick4](OASIS_QUICK4.md).

## AbdomenCTCT — official Learn2Reg public validation

**Cohort.** Exactly 45 directed pairs from the official list, using subjects
0001,0004,0007,0010,0013,0016,0019,0022,0025,0028. Copy that ordered list;
do not generate 90 bidirectional combinations or choose an atlas.

**Preparation.** Unchanged official affine-aligned CT release, 192×160×256,
2 mm isotropic. Our adapter independently min–max normalizes each image.

**Evaluation.** All 13 released organ labels are evaluation-only, never training
masks. Same pinned Learn2Reg Dice, voxel HD95, shifted SDlogJ and separately
reported true folding conventions as OASIS. These are public-validation results.

[Official pair/metric configuration](https://github.com/MDL-UzL/L2R/blob/88475095e35442087a24439ff34ec09f54bc7dac/evaluation/evaluation_configs/AbdomenCTCT_VAL_evaluation_config.json).
Implementation: `medical.abdomen_method`; [details and plan-only runner](ABDOMENCTCT.md).
Frozen optional quick4: official pair indices **2,8,18,41**, selection seed 42;
[quick4](ABDOMENCTCT_QUICK4.md).

## IXI — TransMorph author protocol

**Cohort.** Preserve the released 403 training / 58 validation / 115 test subjects
and one atlas (577 image/label tuples). Primary evaluation is 115 test fits, atlas
moving and held-out subject fixed. The 58 validation fits remain separate.
Our per-pair optimization does not train a shared model on the training split.

**Preparation.** Preserve author-preprocessed arrays exactly, 160×192×224, with
no extra alignment, resampling or intensity normalization. The source pickles
have no original patient-world affine: NIfTI conversion records a unit-voxel
benchmark frame, not reconstructed patient geometry.

**Evaluation.** Author 46-class one-hot encoding, trilinear label-channel warp
then argmax; 17 equally weighted anatomical groups built from 30 scored labels.
Report per-case SD and group/case SD (115×17 observations). Whole-grid NumPy
Jacobian `detJ<=0` folding; HD95 is not an author primary metric here. Reverse-map
scores are supplemental. Labels never select training crops.

Pinned TransMorph revision `6357a1d7fc44c36db9b1d1ccaa372409253142cf`.
[Full author-code audit](BRAIN_BENCHMARK_PROTOCOLS.md),
[frozen splits and metric definitions](../medical/protocols/brain_benchmarks.json),
[prepared layout and plan-only runner](BRAIN_TRAINING.md).

## LPBA40 — SITReg 45-pair protocol

**Cohort.** All 40 official subjects, frozen 25 training / 5 validation / 10 test
split. Ten validation combinations and **45 test unordered combinations**, in
author order. Our method makes one fit per pair and scores both recovered maps;
the author's additional swapped-input model call is not reproduced by a second
fit. Qualify this distinction when comparing bidirectional scores.

**Preparation.** Original native and delineation packages; author's inverse AIR
resampling, white-matter mean normalization, intensity-only tissue masking and
exact 160×192×160 crop. All 56 manual labels remain intact and unmasked. The
affine retains crop-origin shifts and negative first two axes. No adapter min–max
normalization. Pinned mapping dependencies: composable-mapping 0.1.2 and
deformation-inversion-layer 1.1.2; vendored helpers are hash-checked.

**Evaluation.** SITReg nearest-neighbor/mask convention over 56 labels; HD95 in
**voxels**, with author invalid-label handling recorded. Sample float64 Jacobians
at 1,000,000 seeded points, perturbation 1e-7 voxels; report `detJ<0` fraction and
determinant SD, not shifted SDlogJ. Seed is `8524561973 XOR (one_based_pair_index-1)`.
Report both directions and inverse consistency, including componentwise MAE in
voxels, with the one-fit qualification above.

Pinned SITReg revision `a665a46a3553b26ade6ff5f57815e5e8ec43a948`.
[Full protocol audit](BRAIN_BENCHMARK_PROTOCOLS.md),
[frozen split/metrics](../medical/protocols/brain_benchmarks.json),
[prepared inputs and plan-only runner](BRAIN_TRAINING.md).

## ACDC — KAN-IDIR only

This is the sole supported ACDC protocol. The evaluation cohort
below remains frozen. Separately, all100 official training patients001–100 have
been acquired as raw development sources under `Data/ACDC/training`; they are
not added to evaluation or the prepared test manifest. A separate32-patient development selection is now prepared using the identical
KAN-IDIR recipe; see [the frozen tuning protocol](TUNING_COHORT.md). [Acquisition details](ACDC.md).


**Cohort and directions.** Official testing patients **101–150**, read ED and ES
frames from each `Info.cfg`. Exactly **100 directed optimizations**: all 50 ED→ES,
then all 50 ES→ED. Labels are 0 background, 1 RV, 2 myocardium, 3 LV, evaluation-only;
sample the image domain without anatomical label masks.

**Preparation.** Follow KAN-IDIR revision
`3ac12e200159811eafb6192f3f8db3afef2a9d1a`:

1. Load selected MRI and GT as float32. Use the **ES header spacing for both phases**.
2. Array-rescale by `source_spacing/(1.5,1.5,3.15)` using scikit-image. MRI uses
   cubic order 3, edge extension and the author defaults for anti-aliasing;
   labels use nearest order 0 and edge extension.
3. Independently min–max normalize each **full resampled MRI before cropping**.
4. Integer-centered crop to **128×128×min(resampled_Z,32)**. Do not pad depth to32.
   These source patients produce depths 19,24,25,27,29,30 or32.

This reproduces array rescaling: do not substitute physical-affine repair,
a shared ED physical grid, linear MRI interpolation, zero padding, isotropic
spacing or full-FOV retention. Record dependency versions; upstream does not pin
the relevant numerical libraries, so historical bitwise parity is not established.

**Evaluation.** Author nearest-neighbor label warp, `align_corners=True`, zero
padding; MedPy HD95 in **mm**, spacing (1.5,1.5,3.15). Author metrics intersect
predicted/reference labels and drop the first, intended-background entry: missing
foreground classes can disappear from the average. Record scored labels/counts
and additionally report fixed-label 1–3 diagnostics. Dice SD pools label/direction/
run observations. Author folding is analytical `detJ<0` percentage; finite-difference
Jacobians of our recovered maps must be identified as an adaptation. Author fit-only
timing excludes preparation/evaluation; report end-to-end time separately. Our
single-seed policy remains in force despite author repetition counts.

**Availability.** Official originals, license, mandatory citation and 252-file
acquisition manifest are under `Data/ACDC`. KAN-IDIR-matched inputs are under
`Data/ACDC/prepared`, with 100 phase images, 100 evaluation segmentations and
a manifest of 100 ordered directed fits. `scripts/prepare_acdc.py` records
source hashes, ES spacings, crop offsets and dependency versions. Integral
evaluation labels are stored after the author's scoring cast, which commutes
with nearest-neighbor sampling. `medical.cohort_method` and the plan-only
`medical.cohort_experiments --datasets acdc` connect our model. Preparation was
checked against the retained author functions for every phase. Model execution
requires separate authorization.
[Acquisition and status](ACDC.md), [pinned source audit and caveats](ACDC_KAN_IDIR_PROTOCOL.md).
Required dataset citation: Bernard et al., IEEE TMI 37(11):2514–2525, 2018,
[DOI](https://doi.org/10.1109/TMI.2018.2837502). Retain the source CC BY-NC-SA4.0
and noncommercial scientific-research restrictions.

## Neurite-OASIS — Jena et al. last-50 / consecutive-49

**Cohort.** Neurite-OASIS v1.0, all 414 subject directories sorted
lexicographically; select the final50, from **OASIS_OAS1_0404_MR1** through
**OASIS_OAS1_0457_MR1**, retaining gaps. Consecutive subjects produce exactly49
directed registrations: earlier fixed, next moving. No atlas or reverse expansion.
We follow **Jena et al., *Deep Learning in Medical Image Registration: Magic
or Mirage?* (NeurIPS 2024)** for the evaluation-cohort protocol.
[Primary protocol reference](https://papers.nips.cc/paper_files/paper/2024/file/c3fe2a07ec47b89c50e89706d2e23358-Paper-Conference.pdf).
[GaussianDIR §2.3.1](https://pmc.ncbi.nlm.nih.gov/articles/PMC12361785/) explicitly
credits Jena et al. for the last-50 selection and describes the 49-pair construction;
it was the intermediate reference used when preparing this cohort. FireANTs'
consecutive `(fixed,moving)` code agrees with our direction, but its pinned script
contains an active five-pair development override and is not itself an executable
49-pair specification. This attribution is to the *Magic or Mirage?* study,
not a claim that its separate FireANTs method paper defines the cohort.

Only the cohort/pairing protocol is adopted here. Our official comparisons use
the explicitly documented standardized_v1 evaluation; shared subjects and pairs
do not establish metric parity with Jena et al. or GaussianDIR. The frozen
preparation manifest's legacy `neurite_oasis_gaussiandir_last50_consecutive49_v1`
identifier remains unchanged: this is a citation clarification, not a new cohort.

**Preparation.** Obtain `neurite-oasis.v1.0.tar` from
[DalcaLab](https://huggingface.co/datasets/DalcaLab/neurite-oasis), pinned release
`d3702232241b81a716f1db3dfa6c8502ad6d8027`. Published LFS SHA-256:
`86dd117dda17f736ade8a4088d7e98e066e1181950fe8b406f1a35f7fb743e78`.
Verify the full archive and all414 subject directories. Copy `aligned_norm.nii.gz`
and `aligned_seg35.nii.gz` byte-for-byte for the selected50. Preserve the released
160×192×224, 1 mm isotropic grids, alignment, orientation and intensities; no
spatial preprocessing or extra normalization. Retain release metadata and archive
documents/license if supplied. Acquisition provenance does not replace original
OASIS/Neurite attribution or access terms. The retained release README requests
Hoopes et al., *HyperMorph: Amortized Hyperparameter Learning for Image
Registration* (2021), and Marcus et al., *Open Access Series of Imaging Studies
(OASIS): Cross-Sectional MRI Data in Young, Middle Aged, Nondemented, and Demented
Older Adults* (Journal of Cognitive Neuroscience19:1498–1507), and refers users
to the OASIS Data Use Agreement. These texts are preserved in
`prepared/source_metadata/`; the archive does not supply a standalone license.

**Validation and evaluation.** Validate image/label shapes, affine agreement,
spacing, finite nonconstant MRI values and integer label codes 0–35. Manifest records
all subject IDs, selected subjects, ordered49 pairs, archive hash, volume hashes,
geometries and preparation versions. Labels **1–35**, excluding background, are
evaluation-only. Use our repository's Learn2Reg-style Dice and voxel HD95; cohort
and direction agreement **does not establish Jena et al./GaussianDIR metric parity**.
Preparation completed: all414 archive subjects inventoried; the selected50 MRIs
and50 segmentations passed geometry, intensity and label checks. All49 directed
pairs and100 output SHA-256 hashes are recorded in
`Data/NeuriteOASIS/prepared/manifest.json`. The original archive is retained.
`medical.cohort_method` preserves these intensities and validates the frozen
cohort before fitting. `python -m medical.cohort_experiments --datasets
neurite_oasis --plan-only` plans all 49 fits. Evaluation reuses the pinned
Learn2Reg-style array scorer on the preserved release grid, with shifted
SDlogJ and separately reported true folding. Model execution requires explicit
authorization.

Preparation (output must not exist):

```bash
/venv/main/bin/python scripts/prepare_neurite_oasis.py \
  --archive Data/NeuriteOASIS/neurite-oasis.v1.0.tar \
  --output Data/NeuriteOASIS/prepared
```

## Reference implementation identities

The Learn2Reg evaluator/configuration revision is
`88475095e35442087a24439ff34ec09f54bc7dac`:
[evaluation source](https://github.com/MDL-UzL/L2R/tree/88475095e35442087a24439ff34ec09f54bc7dac/evaluation).
Cite Hering et al., IEEE TMI42(3):697–712, 2023,
[DOI](https://doi.org/10.1109/TMI.2022.3213983), in addition to required original
dataset citations. Benchmark papers do not replace source-dataset acknowledgements.
