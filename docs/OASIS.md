# Learn2Reg OASIS public validation

Canonical cohort, preparation and metric choices: [Dataset protocols](DATASETS.md).

The complete official OASIS v1.1 archive is extracted unchanged into ignored
`Data/OASIS/`. Archive CRC checks passed. The preparation manifest records archive,
image and label SHA-256 checksums and affines. All twenty validation subjects have
matching image/label grids at 160×224×192 voxels with 1 mm spacing. Released
alignment, intensities and label IDs are retained: no resampling, reorientation,
new affine fitting, cropping or alternative atlas selection is performed.

The pinned [official validation configuration](https://github.com/MDL-UzL/L2R/blob/88475095e35442087a24439ff34ec09f54bc7dac/evaluation/evaluation_configs/OASIS_VAL_evaluation_config.json)
defines exactly 19 directed pairs: fixed subjects 0395–0413 to their respective
moving subjects 0396–0414. It evaluates labels 1–35, without masked evaluation.
The configuration is retained verbatim with its hash under `medical/protocols/`.
Other released subjects remain available but are not added to validation.

## Preparation and planning

```bash
/venv/main/bin/python scripts/prepare_oasis.py
/venv/main/bin/python -m medical.oasis_experiments
```

Preparation has already completed here and refuses an existing destination.
The launcher prints a plan only: 19 pairs × SIRENPath plus six baselines =
133 jobs. `--methods sirenpath fireants convexadam ccidir nir` and `--pairs 1 2` select
subsets. One worker per requested GPU runs independent cases, exposing each GPU
as logical cuda:0 to preserve native bare `.cuda()` calls. A later explicitly
requested experiment can add `--run --output results/oasis_new`. Training uses
one configured seed; multiseed validation is not enabled.

The single-case runner uses `--method medical.oasis_method --config
medical/configs/default.json` for SIRENPath. For example, ConvexAdam uses `--method
baselines.convexadam --config baselines/configs/convexadam_oasis.json`. Supply the official
moving/fixed image paths and a new output directory. Use `/venv/main/bin/python`.

## Training and evaluation

The OASIS entry points use the official pair list and 35 labels. SIRENPath
uses independent in-memory min/max scaling to [0,1], endpoint
sampling inside the current shared registration crop, and the configured objective
with PCG+RK2 recovery. Whole-volume endpoint reconstruction metrics are evaluation
diagnostics; they do not change the training sampling domain.
Labels are read only by validation/scoring and are never provided to fitting,
sampling, masks, stopping or model selection.

FireANTs-SyN and ConvexAdam–MIND retain native raw-intensity input behavior.
ccIDIR retains native paired-INR training and dense second-order Taylor consensus,
with all-ones domain masks and its dense normalized-coordinate convention.
NIR retains its explicit Gaussian/SIREN field, spatial sampler and min/max scaling.
These are initial, declared OASIS adaptations of the existing native presets,
not tuned reproduction claims. Source implementation pins and compatibility
patches remain unchanged.

Metrics use the pinned official evaluator with OASIS's 35-label configuration.
Synthetic tests compare results to the complete original evaluator
with its displacement loader mocked. Nearest-neighbor label pullback, absent-label
exclusion, unit-spacing HD95, two-voxel Jacobian border exclusion and the pinned
`detJ+3` SDlogJ convention are preserved. At the released 1 mm spacing the HD95
voxel values also correspond numerically to millimetres. Unshifted folding is
reported separately from the official shifted/clipped count. Nonfinite values
are explicit JSON nulls and score coverage is retained in summaries.

Maps and warped images remain in memory. Only scores, resolved config and permitted
network checkpoints persist; dense methods retain provenance-only checkpoints.
Native logging/checkpoint scratch and recovery scratch are cleaned. No training,
optimizer step, benchmark or real-image recovery was run during preparation.

## Attribution

Follow the OASIS Data Use Agreement and retain the citations in the released
`OASIS_dataset.json`. The release attributes this preparation to Andrew Hoopes
and Adrian V. Dalca for HyperMorph. Cite the
[Learn2Reg benchmark paper](https://doi.org/10.1109/TMI.2022.3213983) and the original
dataset/preparation references; benchmark citations do not replace dataset credit.

See the [baseline source audit](../baselines/AUDIT.md) for recipe limitations and corrected native inference coordinate conventions. Native NIR single-field settings are an explicit adaptation, not a reproduction of the paper’s hybrid experiment. External seed policies are separate from SIRENPath’s single-seed restriction.

## Queue and reconstruction reporting

The shared Learn2Reg launcher uses one shared job queue across all configured
GPUs, frozen config snapshots and atomic progress updates. It records per-case
end-to-end seconds. SIRENPath additionally reports whole-volume endpoint MSE
and PSNR (peak 1 after independent min/max normalization) from final fitted
weights, without saving reconstructed images. Training and reconstruction times
are separate from recovery/evaluation time. Summary PSNR is the arithmetic mean
of pair-level dB scores, with scored/intended denominators.

Our-method jobs now use one persistent spawned Python process per configured GPU.
The shared queue still assigns the next case to whichever worker finishes first.
Each case reloads its frozen config, resets Python/NumPy/PyTorch seeds, and creates
fresh models and optimizer state. Case tensors and unused CUDA allocations are
released afterward while compiled code and library caches stay in the process.
Worker PIDs and post-case allocated bytes are recorded in job summaries for
reuse/memory auditing. Ordinary case exceptions are logged and the queue continues;
an exited or unusable CUDA worker is replaced for later cases. External baselines
continue using isolated fresh subprocesses. Plan-only execution starts no workers.

The frozen [quick4 prototype subset](OASIS_QUICK4.md) selects official pairs
1, 4, 8 and 9. Use --subset quick4 to plan those four jobs on four GPUs.
