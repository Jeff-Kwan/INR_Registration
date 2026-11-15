# Learn2Reg AbdomenCTCT public validation

The official release v1.1 is extracted unchanged into ignored
`Data/AbdomenCTCT/`. All archive CRCs passed. The preparation manifest records
archive/image/label SHA-256 hashes and affines. Ten validation subjects have
matching image/label grids, shape 192×160×256 and spacing 2 mm. No resampling,
reorientation, cropping, added affine fitting or label-derived masks are applied.
The release also contains other training/test images; they are not added to the
public-validation cohort.

Pair selection and scoring follow the official [45-pair configuration](https://github.com/MDL-UzL/L2R/blob/88475095e35442087a24439ff34ec09f54bc7dac/evaluation/evaluation_configs/AbdomenCTCT_VAL_evaluation_config.json)
at revision `88475095e35442087a24439ff34ec09f54bc7dac`. Its ordered fixed/moving
paths are used directly. Labels 1–13 are evaluation-only, never supplied to
fitting, masking, sampling or early stopping. Retain the source dataset citations
in `AbdomenCTCT_dataset.json` and cite the [Learn2Reg paper](https://doi.org/10.1109/TMI.2022.3213983).

## Preparation and planning

```bash
/venv/main/bin/python scripts/prepare_abdomenctct.py
/venv/main/bin/python -m medical.abdomen_experiments
```

Preparation refuses an existing destination. It has already completed here.
The second command only prints a plan: 45 pairs × five methods = 225 jobs.
`--methods ours fireants convexadam ccidir nir` and `--pairs 1 2` select subsets.
A future explicitly requested experiment can add `--run --output results/abdomen_new`.
One worker runs per GPU; each child sees its physical GPU as logical cuda:0.
Training remains single-seed. No multiseed validation is configured.

For one case, use the existing `python -m medical` runner with
`--method medical.abdomen_method --config medical/configs/default.json`, supplying
an official fixed/moving pair and a new output directory. Baselines use
`--method baselines.NAME --config baselines/configs/NAME_abdomenctct.json`.
All use `/venv/main`; no separate environment or replacement PyTorch is needed.

## Method settings

Our method keeps the existing paired SIRENs, endpoint MSE/NCC, full-gradient
shared-grid PD-LK, box/GW-NCC and PCG+RK2. Both endpoint images use the whole grid;
lung-mask logic is confined to DIR-Lab. Independent min/max scaling to [0,1]
occurs in memory and is recorded per case. This is our method's intensity choice,
not a claim that Learn2Reg requires this normalization.

FireANTs uses native SyN/CC; ConvexAdam uses native MIND-SSC; ccIDIR uses its native
paired-INR fit and second-order Taylor consensus on dense normalized coordinates;
NIR uses its released Gaussian/SIREN field and spatial sampler. Their original
loops/losses remain in the pinned upstream checkouts. ccIDIR uses all-ones domain
masks for abdomen, without anatomical labels or lung segmentation. Its dense
coordinate conversion follows its native slice inference convention. NIR retains
its explicit independent-min/max input scaling. These are declared abdomen
adaptations of the existing presets, not tuned abdomen results or published-score
reproductions. Native settings can be inspected in each abdomen JSON.

## Evaluation

All adapters produce a fixed-to-moving absolute voxel map in memory. The evaluator
uses nearest-neighbor label pullback with SciPy's official constant-zero boundary
convention. Network transforms are queried in chunks, with frozen network weights;
ccIDIR retains coordinate derivatives needed for consensus. No label map,
deformation field or warped image is saved.

The exact upstream `utils.py`, evaluator and configuration are retained under
`medical/protocols/`, with revision and hashes in `provenance.json`. Only the three
metric function definitions are loaded; upstream file-based deformation loaders
and writers are not invoked. A local NumPy namespace supplies the legacy `NAN`
spelling for NumPy 2 compatibility without changing the retained source files.

The native SIREN adapter scores the recovered map on its CUDA device. A Triton
kernel computes the exact separable squared Euclidean distance transform in
each label's bounding box. The HD95 percentile uses the published surfel areas
and FP64 cumulative ordering on CPU. The CUDA path also computes the label
pullback, Dice counts, interior Jacobian and folding fraction. CPU arrays,
working-grid point predictors, and images outside the native CUDA scorer's
uint8 and maximum 256-voxel axis bounds use the CPU evaluator. The same dispatch
serves OASIS. The opt-in parity tests compare CUDA scores against the pinned
functions on strided maps, folds, missing labels, and empty predictions.

The pinned evaluator has conventions that must remain explicit:

- `DSC`: mean across labels present in both original images; absent-label details
  are null in JSON, corresponding to the official NaN exclusion.
- `HD95`: surface-distance library 0.1, with unit spacing as in the official
  utility. Values are **voxels**, not millimetres. No silent factor-of-two conversion.
- `LogJacDetStd`: `std(log(clip(detJ + 3, 1e-9, 1e9)))`, with the official two-voxel
  border exclusion. The `+3` offset is retained exactly.
- `num_foldings`: the official shifted/clipped count is retained for parity.
  `folding_fraction` separately reports unshifted `detJ <= 0` on the same interior.
- Nonfinite metric values are explicit JSON nulls, never replaced by zero. Summary
  entries include scored/intended pair counts; failed jobs remain in the summary.

Dataset summaries retain mean, population standard deviation and the official
`30` field (70th percentile for LogJacDetStd; 30th for DSC/HD95), alongside
coverage counts. Reverse pairs are not added automatically to the 45-pair list.

Outputs are compact scores, resolved configs and checkpoints. ccIDIR/NIR/our
method save network weights. FireANTs/ConvexAdam save provenance only because
persisting their optimized dense fields would violate the artifact policy.
Native temporary checkpoints/logs and our recovery scratch are cleaned.

Synthetic checks compare the in-memory results to the complete pinned official
evaluator with its displacement-file loader mocked. They also check transform
axes/direction, missing labels, every baseline route, our runner outputs, and
plan-only execution. Registration accuracy, convergence and memory use have not
been benchmarked; no training, optimizer step or real-image recovery was launched.

See the [baseline source audit](../baselines/AUDIT.md) for recipe limitations and corrected native inference coordinate conventions. Native NIR single-field settings are an explicit adaptation, not a reproduction of the paper’s hybrid experiment. External seed policies are separate from our method’s single-seed restriction.

## Queue and reconstruction reporting

The shared Learn2Reg launcher uses one shared job queue across all configured
GPUs, frozen config snapshots and atomic progress updates. It records per-case
end-to-end seconds. Our method additionally reports whole-volume endpoint MSE
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

The frozen [quick4 development subset](ABDOMENCTCT_QUICK4.md) selects official
pairs 2, 8, 18 and 41. Use `--subset quick4` to preview four jobs.
