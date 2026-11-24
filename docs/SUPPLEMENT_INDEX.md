# Supplementary material index

This index maps candidate supplement items to existing evidence and generation tools. Table/figure numbers below are provisional. It does not launch experiments or imply that every proposed composite figure has already been assembled.

## Backup coverage

The user reports downloading the default official results, ablation results and ACDC trajectory artifacts. Git contains the scientific code, canonical configuration, baseline patches, protocol documentation, compact result metadata and exported PNGs. Download integrity has not been independently verified here. The default-run checksum inventory is [available here](reproducibility/default_run_inventory.jsonl).

Exact prepared datasets are intentionally not backed up by the user. Rebuilding them requires the original releases, preparation scripts and dependencies, then comparison with [archived preparation manifests](reproducibility/dataset_manifests.jsonl). Continued source availability is an external dependency. Baseline/tuning checkpoint weights and visualization recovery caches remain outside Git and are not part of the user's stated backup set. Existing tables and PNGs can be reused without these weights; new baseline visualizations may require rerunning the relevant fits. New SIRENPath visualizations require the backed-up checkpoint and matching prepared inputs.

## Tables

| Item | Existing sources | Status and assembly |
|---|---|---|
| S1. Dataset cohorts, directions, labels and evaluation | [Dataset protocols](DATASETS.md), [runner](CASE_RUNNER.md), [baseline report and citations](BASELINES_REPORT.md), [preparation/selection manifests](reproducibility/dataset_manifests.jsonl) | Information exists. Assemble a concise four-dataset table; distinguish the L2R-OASIS ablation/development cohort from Neurite-OASIS testing. |
| S2. Full quantitative comparison, including runtime and CPU/GPU execution | [Aggregate report](BASELINES_REPORT.md), [detailed report](EXTERNAL_BASELINES.md), [metadata archive index](reproducibility/index.json), backed-up default run | Aggregate tables exist. Use the original default, and corrected NIR/DNVF records. Archive groups classical and neural_corrected retain per-case metrics/settings; default scores are in the downloaded official run. Preserve dataset denominators and metric units. |
| S3. Baseline recipes and implementation localization | [Protocol review](BASELINE_PROTOCOL_REVIEW.md), [baseline guide](../baselines/README.md), [source revisions](../baselines/sources.json), [patches](../baselines/patches), [checkout audit](reproducibility/upstream_audit.json) | Evidence exists. Separate source-defined choices, localization under ambiguity, deliberate departures and corrected errors. Include precision overrides, sampler behavior and CPU execution. Do not describe preserved upstream behavior as a local correction. |
| S4. Original and supplementary ablations | [Design](SIRENPATH_ABLATIONS.md), [aggregate results](ABLATIONS_REPORT.md), [time-conditioned architecture ablations](SIRENPATH_ARCHITECTURE_ABLATIONS.md), ablations archive group and downloaded ablation folder | The aggregate report covers 15 scalar conditions (285 evaluations), including eight supplementary training conditions. The architecture report adds two NCC conditions (38 fits). Include both the Raw crossfade and Fitted crossfade controls. Identify reused training cost and the NCC-output intensity-calibration caveat. |
| S5. Hyperparameter search | [Cohort](TUNING_COHORT.md), [selection strategy](TUNING_SEARCH.md), [promotion history](TUNING_PROMOTION.md), tuning archive group | Twenty complete sweep configurations, including default, plus one combined candidate. Settings, per-case scores and decisions are archived; checkpoints are optional and not in Git. Report independent category winners and combination outcome; restored canonical default is authoritative. |
| S6. Reproducibility environment | [Environment snapshot](reproducibility/environment.json), [package versions](reproducibility/packages.txt), [reproduction guide](../REPRODUCE.md) | Available. Snapshot is current, not proof of every historical run's exact environment or a tested clean-install lockfile. Container image digest was not established. |

## Figures

| Item | Existing assets and scripts | Selection and interpretation |
|---|---|---|
| S1. Median-Dice qualitative comparisons | [Media guide](CASE_MEDIA.md); centroid folders listed below; [generator](../scripts/create_ixi_centroid_media.py) | PNG panels already exist. One original-default SIRENPath median-Dice case per dataset; best classical and neural methods selected for that case. Shared fixed-foreground-centroid slices, crop and display window. ACDC views are short-axis and a long-axis reformat, not claimed standard chamber planes. |
| S2. Target-structure boundary comparisons | [Contour inspection](MEDIAN_DICE_CONTOUR_INSPECTION.md), [generator](../scripts/create_structure_contours.py); dataset median-case media folders | PNG plots and separate legends exist. Fixed label plus best dataset-level classical/neural and SIRENPath contours. Targets use largest signed per-label gaps against the strongest baseline; ACDC's second example is the smallest win because no loss exists. Slice maximizes fixed target-label voxel count. Say this explicitly in captions. |
| S3. ACDC continuous intensity trajectory | [Trajectory description](ACDC_TRAJECTORY.md), [PNG assets](media/acdc_trajectory), [generator](../scripts/create_acdc_trajectory.py), downloaded trajectory folder | Existing five-timepoint SIRENPath parameter interpolation, Fitted crossfade and cine-reference panels. Lower-median-Dice directed case: patient101/pair1, ED1 to ES14. Fractional cine references interpolate preprocessed acquired frames; they are not additional acquired images. |
| S4. Monotone temporal alignment diagnostic | [Aligned PNGs](media/acdc_trajectory/aligned), [alignment script](../scripts/align_acdc_trajectory.py), analysis archive group | Existing time-warp and PNCC curves. Uses 53 model samples and 14 acquired cine frames, anchored monotone alignment, a 0.25 time band, step bounds 1–16 and penalties documented in the script/archived records. Whole-volume trajectory mean PNCC: SIRENPath 0.917636 to 0.919325; Fitted crossfade 0.918011 to 0.918527. Post-hoc optimization of PNCC is a diagnostic, not independent anatomical-motion validation. |
| S5. Per-case metric distributions and paired differences | Downloaded default-run metrics; classical/neural_corrected archive groups | Proposed additional figure, not generated. Match dataset and directed pair; plot Dice/HD95/NPJ/runtime distributions and paired differences. Preserve missing/undefined values and report counts. No new fits needed. A dedicated plotting script remains to be written. |

### Current centroid panels

- [Neurite-OASIS pair010](media/neurite_oasis_pair010_centroid/REPORT.md): SIRENPath, FireANTs, DNVF.
- [ACDC pair001](media/acdc_pair001_centroid/REPORT.md): SIRENPath, FireANTs, DNVF.
- [LPBA40 pair040](media/lpba40_pair040_centroid/REPORT.md): SIRENPath, ConvexAdam, DNVF.
- [IXI pair079](media/ixi_pair079_centroid/REPORT.md): SIRENPath, ConvexAdam, DNVF.

Each folder contains separate source images/labels, warped-label overlays, deformation grids, grids over labels, intensity error, label mismatch and legends for two views. Intensity error is normalized absolute intensity difference on a common linear black-to-red scale; label mismatch is a separate red overlay. Pair-level reports include identity and method metrics. Distinguish original scoring from any replayed-map differences documented in those reports.

## Generation and data dependencies

Run tools from the repository root with /venv/main/bin/python on this instance, or the corresponding environment on another host. These are references for future authorized work, not commands executed by this index.

| Operation | Entry point | Required artifacts |
|---|---|---|
| Compute case advantages | scripts/rank_case_advantage.py --sirenpath results/official_ours_cosine_20261116 --output NEW_OUTPUT | Default and baseline result trees. Archived JSONL content can reconstruct the recorded JSON paths; avoid overwriting current results. |
| Generate median-case panels/recovery | scripts/create_median_dice_media.py; plan-only by default, --run executes | Prepared inputs, checkpoints or native baseline reruns; writes scratch/results outside media. |
| Generate centroid panels | scripts/create_ixi_centroid_media.py --dataset DATASET --run | Matching prepared inputs and selected median-case recovery artifacts; see CASE_MEDIA.md. |
| Generate contour panels | scripts/create_structure_contours.py | Existing recovered labels/maps and fixed inputs. This script writes figures directly; it has no plan-only switch. |
| Generate ACDC trajectory | scripts/create_acdc_trajectory.py; plan-only by default, --run executes | Backed-up default checkpoint, prepared ED/ES inputs and full cine in trajectory artifacts. |
| Generate aligned trajectory | scripts/align_acdc_trajectory.py; plan-only by default, --run executes | Trajectory artifacts and checkpoint/input dependencies used by the script. --plots-only reuses its cached alignment artifacts, if retained. |
| Inspect recorded tuning ranks | [Tuning archive](reproducibility/README.md) | The tuning JSONL records preserve the ranking and promotion decision. |

JSONL archives wrap each original record as path, sha256 and content. Restore content to the original relative path under an isolated reproduction root; serialize JSON content or write text content as appropriate. The hash describes the original file bytes; reparsing/reserializing JSON need not reproduce those bytes. The archive is scientific evidence, not a drop-in replacement directory until restored. Some original records contain nonfinite JSON numeric values; handle those explicitly.

## Caption checks

Use the canonical default and corrected baseline versions consistently. Identify whether a displayed baseline is best by dataset mean or by that case. State case, slice and target-selection rules. Distinguish intensity error from label mismatch, original metrics from replay metrics, incremental control runtime from reused training, and PNCC alignment diagnostics from independent validation. No extra experiments are required to assemble the existing tables and figures.
