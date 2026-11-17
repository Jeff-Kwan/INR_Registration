# Baseline protocol review — classified findings

Updated17 November2026. These categories separate what is unknown, what is a
valid but debatable choice, and what demonstrably failed or was described
incorrectly. A low score alone is not a bug. No new experiment or recipe change
was performed for this classification.

## 1. Localization with insufficient author guidance

These are necessary or potentially necessary benchmark adaptations. Their
particular implementation may need validation, but ambiguity is not evidence of
an error.

| Method | Localization / uncertainty | What is established |
|---|---|---|
| NIR | Mapping the paper's Gaussian/Fourier settings into released normalized-coordinate code, including π versus2π | Paper and code express frequency differently; the coordinate-domain equivalence has not been established. The current spectral patch is explicit, not a proven correction to an upstream bug. |
| NIR | Cardiac sampling and NCC-window support for thin, anisotropic ACDC volumes | The brain recipe does not establish an ACDC prescription. The implemented9×9×7 thin-case window is a documented localization; an alternative3×3×1 sampler would need matching Jacobian scaling. |
| NODEO | Encoder/decoder dimensions for grids other than the original supported size | Shape changes are necessary. Original-size initialization/output parity was tested; other sizes are localized architectures. |
| Neural baselines | Whole-image fitting and preservation of the frozen MRI preparations on new cohorts | Author cohorts/preprocessing differ. Labels remain evaluation-only, and original arrays/grids are preserved. These are declared transfers, not exact reproduction claims. |
| NODEO | Interpreting the released Euler solver's temporal convention | Current code executes one step of size0.001 for times[0,1]. This is confirmed behavior; whether it contradicts the intended method is unresolved. We did not introduce or alter it. |

## 2. Valid but debatable protocol / experimental choices

These choices are defined and executable. They affect what comparison can be
claimed, but they do not by themselves make scores erroneous.

| Method / study | Choice | Qualification |
|---|---|---|
| ccIDIR | Keep native `voxel_size=[1,1,1]` for every cohort | Active cycle weighting is in unit-voxel spacing, including ACDC. This is a declared voxel-space adaptation; whether to use actual physical spacing is a protocol decision. Calling it physically calibrated would be incorrect. |
| FireANTs / NODEO | Reuse brain-oriented window and smoothing presets on ACDC | Reasonable source defaults, with unvalidated cardiac suitability. |
| NIR | Change objective, spectral settings, precision and intensity adapter together | A complete preset comparison is valid, but it cannot identify which change caused an accuracy difference. This was insufficient for causal diagnosis. |
| NIR | Use single-field downsize NIR-D | An author-defined variant; name it explicitly rather than implying every NIR variant was evaluated. |
| DNVF | Use the released single-scale200-update recipe | An author-defined variant; the separate multiscale implementation was not evaluated. |
| Demons | Official50-iteration, single-resolution example | A reproducible classic-Demons baseline, not a claim to the strongest tuned multiresolution configuration. |

ConvexAdam public defaults without self-configuration search are an explicitly
accepted user decision. GPU float16 is its native API default. These are settled
choices, not newly discovered errors. No new NiftyReg adapter defect was found.

## 3. Actual bugs, numerical failures or reporting errors

| Finding | Location / responsibility | Evidence and current status |
|---|---|---|
| NIR NCC kernel larger than thin sampled ACDC depth | Our integration applied an unsupported native input shape | Four cases crashed before an optimizer update. Fixed with deterministic odd per-axis support up to9; all four reran successfully. Original supported-case losses/gradients are bitwise unchanged. |
| Pathological NIR LPBA intensity adaptation | Our extra independent minmax step | It destroyed existing comparable white-matter scales: S38 became much darker, and frozen-field queries confirmed background collapse. Removed; corrected runs preserve prepared intensities. This is a demonstrated harmful adaptation, not a claim that minmax is always mathematically wrong. |
| Calling the initial NIR run paper-faithful MRI | Our recipe description / fidelity claim | Initial objective was MSE whereas the paper specifies local NCC; other settings also differed. MSE is a valid released option, but it did not support our paper-recipe claim. Invalid competitive outputs were replaced; current settings are explicit. |
| Native NCC precision failures | Numerical execution in the released loss paths | Native NIR NCC under CPU FP16 autocast produced nonfinite gradients, while FP32 was finite. DNVF's original convolution-precision setting showed a controlled IXI Dice change0.0576→0.7403 with TF32 disabled. Corrected runs use IEEE FP32. This does not mean every native NCC implementation has the same problem. |
| NODEO never updates `BEST_loss_sim_loss_J` | Confirmed upstream implementation defect relative to its stated best-map intent | Code repeatedly compares with constant1000. It returns the last qualifying final-window map, not genuine best-score selection. We preserve this source behavior; changing it would be a separate protocol correction. |
| Describing NODEO output as the best final-window map | Our metadata/documentation error | Earlier wording overstated what source implements. The code behavior above is the accurate description. |

The four thin-case errors and the invalid original competitive NIR/DNVF outputs
are no longer part of current aggregate scores. The NODEO upstream behavior is
still part of the recorded source-native run; it has not been silently corrected.

## 4. Checks that did not establish a defect

- NIR's released inference shape scaling matches training
  `grid_sample(align_corners=False)`; sparse CPU queries agreed to approximately
  1e-5 voxel. The unused inverse helper's different formula was not evidence of
  a real train/inference mismatch.
- NODEO calculates NCC moments using FP64 cumulative sums. It does not use
  DNVF's raw FP32 box-convolution NCC arithmetic.
- FireANTs floors local variances and clamps correlations. This addresses the
  particular invalid-denominator/unbounded-correlation mechanism, though no claim
  is made that precision can never affect its results.
- Shape/map-direction and physical-coordinate adapter tests establish specific
  integration properties; they do not establish convergence or published-score
  reproduction.

## Implications

Resolve ambiguous localization against author conventions; document legitimate
protocol choices; fix demonstrated integration/numerical failures and incorrect
claims. Do not relabel every transfer choice as a bug or every low score as a
failed implementation. Any change to upstream NODEO behavior or ccIDIR spacing
must be explicit and separately evaluated, not folded silently into current scores.

Sources: `baselines/patches/{nir-mri,nodeo-shapes}.patch`, cohort configs,
`baselines/upstream/nir/src/{dataio,loss_functions,modules,efficient_modules}.py`,
`baselines/upstream/ccidir/models/models.py`,
`baselines/upstream/nodeo/{Registration,NeuralODE,Loss}.py`, and
`baselines/upstream/fireants/fireants/losses/cc.py`.
Numerical evidence: [diagnostic audit](NEURAL_BASELINE_DIAGNOSTICS.md) and
[corrected reruns](NEURAL_RERUNS.md).
