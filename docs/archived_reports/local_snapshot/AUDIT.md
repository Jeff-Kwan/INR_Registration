# Comparison implementation audit

The adapters invoke pinned native implementations. That establishes source
alignment of their algorithmic paths; it does not establish reproduction of every
published experiment or its runtime. Each result now records the recipe and
limitations from `audit.json`, plus its actual seed and pinned source revision.
No training, optimizer step, benchmark or saved-model recovery was run for this
audit. Synthetic source-math checks and mocked adapter checks were used.

| Comparison | Source alignment | Experiment alignment |
| --- | --- | --- |
| FireANTs SyN | Native `SyNRegistration.optimize` and composed normalized coordinates | Constructor defaults with an explicit generic multiresolution schedule; not a dataset-tuned paper recipe or the quickstart's affine + Greedy variant |
| ConvexAdam–MIND | Native `convex_adam_pt`; MIND-SSC, coupled convex, inverse consistency, Adam; native XYZ voxel displacement retained | API defaults, without the paper's self-configuration search; not a leaderboard-reproduction claim |
| NIR single field | Native Gaussian/SIREN field, spatial sampler, AMP loop, objective and released output conversion | Explicit released-code adaptation, not the paper's hybrid recipe or reported brain-MRI scores |

## Corrections made

**ccIDIR inference:** the general source CLI defaults to second order and the
[paper, §III-B](https://arxiv.org/pdf/2310.01934) uses second-order Taylor
consensus. Our old adapter
copied that helper default. All three presets now explicitly select
`inference_order=2`; order 1 remains an explicitly selectable source variant. Training remains 2500 epochs, static cycle weight, native balanced Jacobian
regularization and native optimizer settings. Dense Learn2Reg coordinates retain the native slice inference's (shape-1)/2 scale.

**NIR output coordinates:** the released test script uses
`util.warp_3d_volume`, which converts normalized output with
`normalized/2 * maxlen + (shape-1)/2`. Our old adapter instead used
`dataio.unnormalize_coords`, which scales by `maxlen-1`. The adapter now follows
the actual test script. This retains an inconsistency within the upstream code;
it does not silently substitute a more convenient identity-preserving conversion.
Synthetic samples match the actual native warp helper. Native input coordinate
normalization, interpolation's align-corners behavior and the model are unchanged.
The logging interval now also applies the CLI's batch-size divisor, and dataset
maxlen metadata is set as in the CLI.

**Seed propagation:** baseline subprocesses now use `baselines.case_runner`,
which applies an explicit seed override to the resolved config before invoking
the shared runner. Saved config, metrics and each planned job record that seed.
Python, NumPy and PyTorch RNGs are seeded. Our method has no new seed override
and remains single-seed.

## External repetitions

The plan now schedules 500 fits for
its ten cases, yielding 1000 directional results. Seed IDs 0–49 are our recorded
reproducible choice; the paper does not specify its numeric IDs. Each fit has a
unique `caseXX/seed_XXXX` output path. Per-case means and medians across seeds are retained. No best-seed selection is
performed and no fits were launched by changing this plan.

No matching repeated-initialization requirement was identified in the selected
FireANTs/ConvexAdam source recipes or NIR's implementation/setup sections. Baseline repetition policies are in
`protocols.py` and are separate from our method's single-seed rule.

## Remaining recipe differences

**FireANTs:** its [official SyN documentation](https://github.com/rohitrango/FireANTs/blob/f8a9cc259ab9113fc43bd99797000fa8fc34eb59/docs/docs/reference/syn.md)
supports the selected symmetric compositive optimizer. Native composition and
numerical inversion are used. The selected [constructor](https://github.com/rohitrango/FireANTs/blob/f8a9cc259ab9113fc43bd99797000fa8fc34eb59/fireants/registration/syn.py)
defaults match our explicit options; the scale/iteration schedule is our declared
generic schedule. No extra affine stage is silently added. The official PyTorch
fallback is used because fused operations are unavailable; published speed claims
are not applicable to an unmeasured fallback run.

**ConvexAdam:** our [native API](https://github.com/multimodallearning/convexAdam/blob/b229e52e44b114e2040a503334c92269750c16b2/src/convexAdam/convex_adam_MIND.py)
options match its defaults apart from logging/device selection. The returned
components already follow the input XYZ array axes and are in voxels; no extra
axis flip, spacing factor or normalized-grid scale is applied. Its source offers
separate self-configuration workflows; we do not claim to have run them or to
have obtained task-selected hyperparameters. The selected path uses raw images,
not nnUNet semantic features.

**NIR:** the [released training CLI](https://github.com/uci-cbcl/NIR/blob/45c0d5ccbe22c72603e4782560f487ff1e20bcfd/scripts/train_efficient_3d_ct_reg_pairwise.py)
has inconsistent defaults (`sine` encoding is absent in EfficientMLP; random
point tensors lack the spatial layout required by its forward method). The valid
Gaussian/SIREN and downsize paths are explicitly selected. Width/layer structure,
sine frequency, sampling, AMP, loss and learning-rate scheduling are native.
Only import compatibility is patched, with the exact diff recorded.

Our preset remains MSE + Jacobian weight 10, LR 2e-4 and 3000 steps. The
[MedIA manuscript, §IV-B](https://arxiv.org/html/2206.03111v3#S4.SS2), describes a
different single-field optimization recipe (LR 1e-4, 900 iterations and displacement
regularization 1000), and a two-phase hybrid recipe. Its OASIS cohort, crop and
evaluation labels also differ from Learn2Reg public validation. We therefore
label the current comparison **NIR single-field, released-code adaptation**;
its settings cannot be presented as a reproduction of the paper's headline
numbers. The native loop completes eight-batch passes; our 3000-step configuration
is divisible by eight. We do not rewrite that loop to pretend other budgets have
exact native step counts.

## Verification scope

- Exact upstream revisions and recorded compatibility diffs are checked on import.
- Config keys and native API signatures/defaults are inspected without fitting.
- Native ccIDIR nonlinear first-/second-order inference is checked on synthetic
  CPU fields, including a nonzero higher-order correction and both directions.
- NIR output conversion is compared with the actual native inference warp helper.
- Seed overrides, 50 distinct ccIDIR paths and missing-fit denominators are tested.

These checks establish integration behavior and source-math parity on synthetic
inputs. They do not establish numerical convergence, accuracy, reproducibility
across CUDA kernels, or agreement with published scores on real datasets.

## Evaluation dependency compatibility (2026-11-12)

The PyPI surface-distance 0.1 package uses NumPy aliases removed in NumPy 2.
Pin the official google-deepmind/surface-distance revision
`1f805cea446805e559badc80dd27f34175e4173d` in /venv/main and both dependency manifests.
Its metrics.py changes from the installed PyPI source replace only np.Inf/np.NaN
with np.inf/np.nan; the HD95 algorithm is unchanged. No monkey patch, replacement
metric or NIR method modification is applied. Empty predicted surfaces retain
infinite HD95; existing JSON serialization represents nonfinite metrics as null.
These cases must remain visible in reporting rather than being described as
successful finite HD95 scores.

Original run
artifacts are preserved; the retry directory records its own plan and summary,
and combined_summary.json selects successful retries in the original 222-job
cohort. No saved weights survived the original evaluation errors, requiring
repeat fitting rather than evaluation-only recovery.
