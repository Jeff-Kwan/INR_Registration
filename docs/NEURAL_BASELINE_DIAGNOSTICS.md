# Neural baseline diagnostic audit — 2026-11-16

Read-only inspection of completed logs, prepared intensity/label arrays and pinned
source code. No registration reruns, training, optimizer steps, dense checkpoint recovery,
preparation changes or hyperparameter changes were performed. The observations below describe the original runs. The benchmark report uses the [completed corrected reruns](BASELINES_REPORT.md#corrected-nir-and-dnvf-reruns).

## NIR on LPBA40: strong evidence of an adapter intensity confound

All nine near-zero-Dice cases have S38 fixed: pairs 9,17,24,30,35,39,42,44,45.
Their mean Dice is **0.003772**; the other 36 cases average **0.636714**.
S38's prepared maximum is **3.623926**, although its foreground median is about
**0.7741**. Other sampled subjects have maxima around 1.1–1.2 despite similar
foreground intensities. NIR's independent per-image minmax adapter compresses
S38's foreground median to roughly **0.214**, versus roughly **0.694** for moving
S15 in pair 30. This is an adaptation of the frozen prepared intensities, not a
change to dataset preparation.

Native NIR minimizes global image MSE plus a negative-Jacobian penalty, with
zero-outside image sampling. The penalty does not constrain global translation
or drift. Matching a much darker target can favor sampling background instead
of correct anatomy; this mechanism is a strong explanation, not yet established
by an intervention. Pair 30 has unregistered Dice **0.5556**, but fitted Dice
**0.0000894**. Its logged loss falls from 0.241078 to 0.052834, showing that
improving this objective did not ensure anatomical alignment.

Source: `baselines/nir.py:prepare_array`,
`baselines/upstream/nir/src/loss_functions.py:image_mse,discrete_jacobian_det`,
`baselines/upstream/nir/src/deform_net.py:DeformedField`.

## DNVF on IXI: widespread failure of the released single-scale optimization

**76/115 IXI cases have Dice <0.3**. Final native image-similarity loss is worse
than its epoch-0 value in **40/115** fits, versus **1/49** on Neurite-OASIS.
For worst IXI pair 66, Dice falls from unregistered **0.3924** to **0.0576**;
image loss changes from **−0.044776 to −0.030175** (worse), and regularization
from **0.044083 to 0.073512**. Good IXI pair 93 reaches Dice **0.7597** and
image loss **−0.213777**. These are actual fitting failures, not just stricter
label metrics or process failures.

The adapter runs the author's single-scale `run_diff.sh` recipe: 200 updates,
Adam AMSGrad lr1e-4, downsample factor2, width512/five hidden layers, native NCC,
flow-gradient regularization and negative-Jacobian weight50. It has no native
best-objective checkpoint selection; the final pre-update map is evaluated.
The source also supplies a separate multiscale 8→4→2 recipe, not used in this run.
A difficult initialization basin / unstable or insufficient single-scale fit is
consistent with the logs. Exact causal attribution is unresolved: no learning-rate,
precision, initialization or multiscale intervention was run.

IXI sample identity Dice is approximately 0.35–0.41, versus 0.53–0.58 in sampled
Neurite pairs. This supports a harder atlas-to-subject alignment task, but is
not a complete cohort-level explanation. Input ranges in sampled IXI/Neurite
cases are both [0,<1]; the simple S38 minmax outlier mechanism does not explain
DNVF's IXI collapse.

Sources: `baselines/upstream/dnvf/opt_nf_diff.py`, `run_diff.sh`,
`utils/losses.py`, saved native per-case logs.

## NIR on IXI: incomplete explanation, likely objective/optimization transfer

Dice ranges **0.2716–0.6754**, with mean **0.5086**. This is a broad performance
shortfall, distinct from the nine catastrophic S38 cases. Worst sampled pair57
falls from unregistered Dice **0.3533** to **0.2716**, while pair111 improves
from **0.4124** to **0.5036**. Pair57's native loss still decreases from 0.044283
to 0.013866. The current MSE/negative-Jacobian objective therefore admits weak
anatomical solutions on these atlas-to-subject pairs.

The native single-field recipe uses 904 actual steps for 900 requested updates,
stride-three jittered whole-volume sampling, AMP, image MSE and folding penalty.
It has no explicit displacement-magnitude or boundary-restoring penalty. Photometric
mismatch, background-heavy sampling and limited capture range are plausible
contributors. The exact IXI cause remains unproven. Neurite and IXI share the same
160×192×224 array shape, weakening a shape-only explanation.

The known released training/inference coordinate mismatch is retained: training
normalizes with shape−1; released inference scales with shape. For in-domain
normalized points this component differs by at most half a voxel per axis;
it alone is not a convincing explanation for catastrophic anatomy displacement.
No new axis-order or map-direction defect was established in this inspection.

## Interpretation and next diagnostic work

Treat the affected scores as **provisional adaptation results** before paper claims.
The LPBA NIR adapter normalization needs a controlled correction preserving the
frozen dataset arrays. DNVF deserves an author-recipe optimization audit, including
its released multiscale path, before interpreting the IXI result as a method limit.
NIR/IXI still needs controlled objective/initialization/inference checks. Any reruns
should be separately named and documented; do not silently replace these results
or choose fixes using label scores.

## Additional CPU loss check during ablation training

No network evaluation or fitting was needed for this analytical check. Compute
the native MSE scale `4*mean((moving-fixed)^2)` and the hypothetical all-background
warp loss `4*mean(fixed^2)` on the existing adapter-normalized arrays:

| Dataset/pair | Identity MSE | All-background MSE | Logged final total loss |
|---|---:|---:|---:|
| LPBA40/30 | 0.244983 | 0.053284 | 0.052834 |
| LPBA40/9 | 0.178637 | 0.053284 | 0.052218 |
| LPBA40/44 | 0.144972 | 0.053284 | 0.055654 |
| LPBA40/10 (good) | 0.050236 | 0.515847 | 0.008176 |
| IXI/57 | 0.045575 | 0.147195 | 0.013866 |
| IXI/111 | 0.028199 | 0.122257 | 0.009417 |

The S38 final losses closely match the all-background solution. Small differences
are expected because native loss logging uses a jittered subsampled minibatch and
includes its regularizer. This strongly supports background collapse induced by
normalization/MSE imbalance. It is not a displacement-map inspection, so it does
not establish that every voxel was mapped out of bounds. The same explanation
does not fit the sampled IXI cases: their background-only losses are much higher
than the achieved losses. The IXI cause remains unresolved.

## Deeper CPU audit: protocol and numerical mechanisms

### NIR MRI recipe discrepancy (correction to earlier audit claims)

The [paper, sections 3.3–3.4](https://arxiv.org/html/2206.03111v3) specifies local
NCC with window9, Gaussian scale3 and sine frequency30. Completed runs instead
used the released-example MSE objective, Gaussian scale1 and native sine10.
The earlier statement that our preset was paper-faithful was incomplete and is
corrected in the source audit. The 900 requested steps, Adam lr1e-4 and Jacobian
weight1000 match, but that does not make the entire MRI recipe faithful.
This is a concrete protocol confound for IXI, not merely a low-score observation.
Do not treat current NIR scores as a validated reproduction of paper MRI results.

Frozen NIR CPU queries (32,768 deterministic spatial points per selected case)
confirm that the trained field is what the loss optimizes. For IXI pair57,
foreground MSE drops from 0.02919 to 0.01023, while foreground median/p95 displacement
is 16.27/35.71 voxels and foreground Dice falls. In sampled tissue labels,
atlas hippocampus intensity median is approximately0.273, versus0.349 in this
target, illustrating the cross-subject photometric mismatch that MSE does not
ignore. Tissue statistics use labels only for this retrospective diagnostic.
This supports objective mismatch; a corrected fit is still needed for causal
confirmation of its effect on Dice.

**Coordinate correction:** the released inference shape scale actually matches
training `grid_sample(align_corners=False)`. Algebraically both use
`normalized*shape/2 + (shape-1)/2`. CPU queries agree to ~1e-5 voxel. The
shape-versus-shape−1 difference against the unused inverse helper does **not**
establish an actual training/inference mismatch. Earlier suspicion on this point
is superseded. No axis-order or map-direction error was identified.

The raw LPBA pair30 input already has matched foreground medians0.7809/0.7741.
Preserving its prepared white-matter normalization gives identity/native-scale MSE
0.08379 versus background-only0.69977, removing the minmax-induced incentive.
Native NIR's `Single_CT_Brain` applies `Identity()`: preserving prepared values
is more source-consistent than introducing per-image minmax. The wrapper's affine
intensity transform remains mathematically defined for values above1.
An alternative label-free, fixed upper99.5th-percentile-of-positive-intensities
clipping/scaling yields medians0.7444/0.7194 and identity/background MSE
0.07242/0.60307. This is an explicit adaptation option, not an author default or
an optimized threshold. No preparation bytes or baseline config were changed.

### Native NCC precision problem

The source NCC computes local variances by subtracting raw second moments.
This is cancellation-prone in constant/near-constant brain tissue. Installed
PyTorch is2.14.1+cu132; fresh-process cuDNN TF32 permission is True, matmul TF32
permission False. The DNVF execution context does not override either flag.
[PyTorch's numerical accuracy guidance](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html)
explains TF32 convolution precision and the IEEE FP32 alternative.

For central80³ IXI patches, a **CPU TF32-input-rounding emulation**, followed by
FP32 moment arithmetic, produces nonpositive NCC denominators in approximately
12.75%,8.95%,9.37% of valid windows for pairs66,102,93. Corresponding max |cc|
is951,209,1803, although squared correlations should be in[0,1]. The FP32
reference has no invalid denominators and max|cc|<1 on these patches. Sampled
Neurite patches are much less affected (0.0035% and0.135% nonpositive denominators).
This gives a concrete numerical mechanism capable of corrupting gradients and
explaining dataset-dependent instability. **It does not prove the original
cuDNN algorithm actually selected TF32.** A controlled actual-GPU forward/backward
comparison with TF32 enabled/disabled remains necessary after the reserved run.
The CPU emulation rounds convolution inputs; it is not an exact cuDNN simulator.

The proposed native NIR NCC option has a more directly reproduced precision
problem. On an IXI stride-three24³ patch, same source NCC and dropout seed:

| CPU calculation | NCC loss | Gradient norm | Finite gradient |
|---|---:|---:|---|
| FP64 reference | −0.121654 |0.022381| yes |
| FP32 | −0.121651 |0.022375| yes |
| source-style FP16 autocast | −5.836353 | undefined | no |

A dtype-only compatibility adjustment to the source's boolean-mask transpose
convolution was used for the FP64 reference; spatial support and objective were
unchanged. This demonstrates that blindly switching the native AMP loop to NCC
is unsafe. A narrowly scoped precision correction should evaluate NCC moments
outside autocast and use IEEE FP32 convolutions. It changes numerical execution,
not the intended loss, and requires a GPU check before a corrected run.

### Evidence and scope

[Sparse frozen NIR queries](../baselines/results/neural_audit_cpu_20261116/nir_frozen_queries.json),
[DNVF NCC emulation and sparse velocity queries](../baselines/results/neural_audit_cpu_20261116/dnvf_numerics_and_velocity.json),
[NIR native NCC precision check](../baselines/results/neural_audit_cpu_20261116/nir_native_ncc_precision.json).
Reproduction scripts are alongside these artifacts in `scripts/`. All queries
ran with CUDA hidden and one CPU thread. No optimizer steps, full recovered maps,
registration reruns or GPU diagnostics were performed. The active SIRENPath
ablation run and frozen preparations were not changed.

### Recommended correction policy

Prefer the author's MRI objective and existing dataset intensity normalization.
If an input transformation is still necessary, allow a separately documented,
fixed, label-free in-memory method adapter for the entire affected cohort, not
per-subject exceptions. The corrected reruns later replaced the original scores in the official result tree.
Do not change the canonical SIRENPath normalization or the active ablations.
Percentile clipping is a declared alternative, not evidence of author fidelity.

## Controlled GPU follow-up

The explicitly requested DNVF IXI pair66 rerun with TF32 disabled improved Dice
from0.057563 to0.740307 and HD95 from23.658882 to3.680531, with NPJ decreasing
from3.413493% to0%. Both corrected methods subsequently completed all 309 cases. See the
[rerun protocol](NEURAL_RERUNS.md) and [corrected results](BASELINES_REPORT.md#corrected-nir-and-dnvf-reruns).
