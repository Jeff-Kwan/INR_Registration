# Dataset and evaluation protocols

All three cohorts have completed full registration runs. The table defines the evaluation populations. The method specification records training settings separately. [Full results](LONG_RESEARCH_REPORT.md) · [Method](METHOD.md)

## Selected published protocols

| Dataset | Protocol followed | Starting evaluation set | Main accuracy measure |
|---|---|---|---|
| DIR-Lab 4DCT | van Harten et al., ccIDIR, IEEE TMI 2024; preparation inherited from IDIR MIDL 2022 | All 10 subjects; T00/T50; 300 expert landmarks per subject | TRE in physical millimetres |
| Learn2Reg OASIS | Hering et al., Learn2Reg, IEEE TMI 2023; official OASIS public-validation configuration | 19 directed pairs involving 20 subjects | Dice and HD95 |
| Learn2Reg abdomen CT–CT | Hering et al., Learn2Reg, IEEE TMI 2023; official AbdomenCTCT public-validation configuration | 45 directed pairs involving 10 subjects | Dice and HD95 |

### DIR-Lab 4DCT

Follow [ccIDIR, sections II-A and IV](https://arxiv.org/pdf/2310.01934), which reuses [IDIR preparation](https://proceedings.mlr.press/v172/wolterink22a.html) for the ten extreme-phase cases, native image resolution, phase-specific lung-mask sampling, and landmark evaluation. Obtain images and reference landmarks through the [official DIR-Lab source](https://med.emory.edu/departments/radiation-oncology/research-laboratories/deformable-image-registration/downloads-and-reference-data/4dct.html); respect its access conditions. Do not substitute Learn2Reg LungCT or COPDgene.

Use the following explicit map convention, matching the [IDIR example](https://github.com/MIAGroupUT/IDIR/blob/9ab3ccfbbf09dc8fc65fa848332e7d9a0b3b81b3/run.py): fixed=T00 inspiration, moving=T50 expiration. The image pullback samples the moving image at the map from fixed coordinates to moving coordinates; evaluate that map on T00 landmarks against T50 landmarks. This avoids ambiguity in phrases such as “inspiration-to-expiration registration.” Reverse-direction results are additional diagnostics.

Preparation preserves subject-specific spacing and converts raw images to native-grid NIfTI. Index and axis conversion follow the official format. The primary comparison reports IDIR-compatible rounded physical TRE. The adapter also reports continuous TRE and per-axis errors. [DIR-Lab format](https://med.emory.edu/departments/radiation-oncology/research-laboratories/deformable-image-registration/_files/format-document.pdf) · [IDIR metric implementation](https://github.com/MIAGroupUT/IDIR/blob/9ab3ccfbbf09dc8fc65fa848332e7d9a0b3b81b3/utils/general.py)

### Learn2Reg OASIS

Use the official preprocessed Learn2Reg release and its public validation list, rather than a newly chosen atlas or a TransMorph redistribution/split. The pinned [OASIS validation configuration](https://github.com/MDL-UzL/L2R/blob/88475095e35442087a24439ff34ec09f54bc7dac/evaluation/evaluation_configs/OASIS_VAL_evaluation_config.json) specifies fixed IDs 0395–0413 paired respectively with moving IDs 0396–0414: 19 pairs, 20 subjects. It scores labels 1–35, HD95, and SDlogJ, without masked evaluation. Its expected array-axis shape is 160 × 224 × 192; do not confuse this with differently ordered dimension listings.

Retain released alignment, grid and labels; avoid repeating spatial preprocessing. These are public-validation results, not the hidden-test leaderboard or an arbitrary “public labeled subset” test. Preserve the official label definitions and pair directions.

### Learn2Reg abdomen CT–CT

Use the canonical, affinely pre-aligned Learn2Reg release, not raw abdominal CT or another dataset called ABD50. The [challenge dataset description](https://learn2reg.grand-challenge.org/Datasets/) describes 2 mm isotropic volumes on a 192 × 160 × 256 grid with 13 organ labels.

Use all 45 pairs, including their directions, in the pinned [AbdomenCTCT validation configuration](https://github.com/MDL-UzL/L2R/blob/88475095e35442087a24439ff34ec09f54bc7dac/evaluation/evaluation_configs/AbdomenCTCT_VAL_evaluation_config.json). The ten subject IDs are 0001, 0004, 0007, 0010, 0013, 0016, 0019, 0022, 0025, 0028. Copy the actual list; do not generate 90 bidirectional combinations or register everything to 0001. Preserve the provided label set and use the configuration's metrics. This is public validation within the released training data, not hidden-test evaluation.

## Common preparation and execution pipeline

1. Acquire fresh official releases separately from research execution. Record source, release, archive checksums and access terms. Keep source files immutable.
2. Produce an explicit case manifest with ordered moving/fixed images, affines, masks, evaluation labels/landmarks, and preparation provenance. Validate geometry and landmark coordinates without optimizing a model.
3. Our paired-SIREN adapters independently min–max scale moving and fixed image intensities to [0, 1] and record the normalization in the case metrics. Baseline adapters use their documented preprocessing.
4. Fit each case independently using frozen method settings. Keep evaluation labels and landmarks out of optimization, sampling, stopping and case-specific parameter selection. Develop settings on separate subjects; if evaluation scores guide later changes, describe the public-validation results as development evidence.
5. Recover and evaluate in memory. Use the pinned Learn2Reg evaluator's interpolation, missing-label handling, surface distances and Jacobian conventions for comparable scores. An in-memory adapter must reproduce its results before replacing its file-based interface.
6. Save endpoint checkpoints, resolved configuration, provenance, scores, runtime and convergence/failure summaries. Do not retain image maps, warped volumes or velocity artifacts. Launch independent cases with explicit GPUs and unique output directories.

Report per-case scores and dataset-level summaries separately; include all intended cases and failure denominators. Supplement accuracy with folding, recovery convergence, outside-domain sampling and end-to-end time. Do not pool anatomical Dice and lung TRE into one score. Published numbers are contextual until pair selection, direction, preparation and metric implementations agree.

## References and pinned code

- Wolterink JM, Zwienenberg JC, Brune C. **Implicit Neural Representations for Deformable Image Registration.** MIDL 2022, PMLR 172:1349–1359. [Paper](https://proceedings.mlr.press/v172/wolterink22a.html). IDIR revision `9ab3ccfbbf09dc8fc65fa848332e7d9a0b3b81b3`.
- Hering A et al. **Learn2Reg: Comprehensive Multi-Task Medical Image Registration Challenge, Dataset and Evaluation in the Era of Deep Learning.** IEEE Transactions on Medical Imaging 42(3):697–712, 2023. [DOI](https://doi.org/10.1109/TMI.2022.3213983), [author manuscript](https://arxiv.org/abs/2112.04489). Official L2R evaluator/configuration revision `88475095e35442087a24439ff34ec09f54bc7dac`; [evaluation documentation](https://github.com/MDL-UzL/L2R/tree/88475095e35442087a24439ff34ec09f54bc7dac/evaluation).

At acquisition, also retain the original dataset citations and acknowledgements required by each release. Benchmark protocol citations do not replace dataset attribution.

## DIR-Lab preparation, 2026-11-12

The ten official 4DCT case packets were acquired and prepared locally with `scripts/prepare_dirlab_idir.py` into ignored `Data/DIRLab4DCT_IDIR/`. Each case has T00 and T50 native-grid NIfTI images, zero-based voxel-coordinate CSV landmarks, and a binary T00 lung mask. Original raw intensities are retained; mask inference uses raw values minus 1024 HU. The NIfTI affine describes a local, arbitrary-origin frame, not an original patient-world frame. Source and landmark SHA-256 checksums and conversion details are in the local manifest.

Masks were generated using `lungmask` 0.2.22, R231 weights SHA-256 `d5d2fc3df432115933491f115b283e94d1f5c841c0e82f6e7bd2d65d3ccde69f`. IDIR cites this segmentation approach but does not establish the exact mask software or weights, so exact mask equivalence is not claimed. All 20 image and mask grids agree; all 300 T00 landmarks in each case lie inside the computed lung mask. Initial 300-landmark mean TRE by case was 3.892, 4.338, 6.943, 9.830, 7.477, 10.891, 11.026, 14.995, 7.918 and 7.301 mm. The published DIR-Lab case statistics use the full landmark sets, so they need not exactly equal these 300-point subset values. No training or recovery was run.

## DIR-Lab landmark evaluation interface

`medical.dirlab_evaluation.evaluate_case(case_dir, predicted_xyz)` evaluates a method's in-memory output. `predicted_xyz` contains 300 zero-based T50 voxel coordinates in x,y,z order, corresponding row-for-row to the prepared T00 landmark CSV. It must represent the T00-to-T50 point map; warped images alone cannot determine landmark TRE. The standalone CLI accepts the same coordinates in a CSV headed `x,y,z`: `python -m medical.dirlab_evaluation --case-dir Data/DIRLab4DCT_IDIR/case01 --predicted /path/to/predictions.csv`.

The evaluator reports initial TRE, continuous-coordinate TRE, and IDIR-compatible rounded TRE and per-axis statistics. It restores one-based coordinates before NumPy rounding, as in IDIR's utility, and uses population standard deviations. The reference prints values rounded to two decimal places; this evaluator retains precision for aggregation. A count of predicted landmarks outside the moving grid is included. A caller may aggregate per-case means across the ten cases; report all 300 landmarks and failures, never silently drop out-of-bounds points.

The landmark evaluator consumes point coordinates. The DIR-Lab adapter also evaluates Jacobian folding from the recovered primary map over the whole grid and inside the T00 mask.

## DIR-Lab crop sampling

Training requires separate nonempty binary R231 masks on the native T00 and T50 grids. Select either endpoint equally, draw a mask voxel and center the shared crop there, shifting it to fit the image. Endpoint reconstruction uses that crop, registration draws independent starts in each direction, and PD-LK uses every grid cell. The mask influences training through crop placement alone. [Canonical method](METHOD.md)

Our method remains single-seed. External baselines may follow published repetition protocols; the ccIDIR DIR-Lab plan now uses 50 seeds and both directions, without launching experiments. R231 preparation is reproducible but exact equality with the authors' original masks is not claimed. Run `scripts/prepare_dirlab_ccidir_masks.py` after initial preparation to add T50 masks without replacing existing inputs. The original preparation record above describes the historical T00-only stage.

Preparation completed: all ten T50 masks have been added using the pinned R231 model. All twenty masks match their image grids and contain all 300 phase-specific reference landmarks. Existing images, landmark CSVs and T00 masks were preserved. Per-phase checksums are in the ignored ccidir_masks.json manifest. No registration experiment was launched during this update.

## AbdomenCTCT preparation and runner status

The official release is extracted unchanged and all ten public-validation subjects and their labels are verified. All 45 directed pairs are connected to our method and four native baselines. See [ABDOMENCTCT.md](ABDOMENCTCT.md) for preprocessing, exact official evaluation conventions, source hashes, single-seed configs and the plan-only multi-GPU launcher.

## OASIS preparation and runner status

The completed official release is extracted unchanged. All twenty validation subjects and matching labels are verified at 160×224×192, 1 mm spacing. Exactly 19 directed pairs and labels 1–35 are connected to our method and all four native baselines. See [OASIS.md](OASIS.md) for the single-seed plan-only launcher and evaluation details.
