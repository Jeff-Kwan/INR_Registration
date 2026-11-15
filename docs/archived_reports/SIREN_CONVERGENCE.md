# SIREN convergence investigation

The latest complete-cohort comparisons and current candidate are in the [ten-hour research report](LONG_RESEARCH_REPORT.md). This document retains the preceding quick4 convergence investigation, whose shared reference is the [uniform-30 AdamW/OneCycle preset](../medical/configs/native_siren_onecycle_5k.json). Its quick4 ranking below is not the newer full-cohort ranking.

## Objective and comparison protocol

Reach the registration performance of the default 40,000-step DIR-Lab quick4 run
using 5,000 optimizer steps. Use checkpoint diagnostics to assess endpoint fitting,
large-motion recovery and transport consistency. Validate the final candidate on
the frozen OASIS and AbdomenCTCT quick4 selections.

Status: complete. All 65 planned DIR-Lab recipes are complete. Two 5k stacks pass
both DIR-Lab 40k directional targets and improve OASIS/abdomen mean DSC and HD95.
Final software verification passed all 41 checks. The completion audit passes.
Isolated timing shows 5.38× faster training and 4.85× faster training plus recovery
for the common preset, with the broader scope exclusions recorded below.

The initial screen selected `medical/configs/native_siren_width512_output30_5k.json`:
four layers, width 512, original omega 30, Adam LR 1e-4 and output initialization
scale 30. It retains the default PD-LK 0.1 and registration NCC 0.01. Its DIR-Lab
TRE is 1.80454/1.87067 mm versus 1.89463/1.93786 mm at default 40k. OASIS DSC
is 0.82364 versus default-5k 0.81833; abdomen DSC is 0.41690 versus 0.39645.
Every OASIS/abdomen quick4 pair improves DSC. No default configuration is changed.

For the lowest DIR-Lab TRE, `medical/configs/native_siren_width512_5k.json` instead
uses output scale 1 and registration NCC 0.05 at the same width and LR. It reaches
1.71776/1.76619 mm and improves every DIR-Lab case in both directions. Its
OASIS/abdomen means are 0.81902/0.40182, with pair-level tradeoffs. Full 1k–5k
probes are complete for this stack; its 4k checkpoint already beats the 40k TRE
targets. Both final stacks have complete displacement, fidelity and transport
probes, all final PCG intervals converge, and DIR-Lab lung folding is zero.

The coarse-frequency stacks had already beaten DIR-Lab 40k but regressed OASIS.
The selected combinations retain the original frequency support. The factorial
capacity/frequency comparisons and initial Adam-update probe below explain why
wider representation and smaller updates work together. Eightfold fewer steps
yield a measured 5.38× training speedup for the common preset.
Machine-readable final candidate evidence is in
`results/siren_convergence_20261113/final_results.json`.

The user explicitly requested thorough testing while the win condition remains
unmet and encouraged stacking complementary strategies and exceeding the target.
After identifying useful individual changes, test combined recipes against their
components, then carry strong combinations into both cross-dataset validations.

All experiments use seed 42 and the frozen dataset protocols. DIR-Lab cases are
01, 03, 04 and 05. Default width is 256 with four hidden layers, 32,768 endpoint
samples and separate registration particles. Native voxel units, crop-center-only
lung masks, live full-gradient PD-LK and registration NCC, and frozen PCG+RK2
evaluation are retained. Every new 5k run saves checkpoints at 1k, 2k, 3k, 4k, 5k.
Scores are development results from one seed.

The 40k reference has mean forward rounded TRE **1.894626 mm**, reverse TRE
**1.937862 mm**, and pooled continuous TRE **3.137135 mm** on the 381 landmarks
with true motion >=10 mm. Mean endpoint-window reconstruction NCC is **0.850034**.
Primary success requires both directional mean TREs to reach the reference at 5k,
with case-level and probe results reported alongside the averages. Generalization
uses matching 5k default-versus-candidate comparisons on both other quick4 datasets.
The established 40k performance target is the DIR-Lab reference. Higher-budget
references on the other datasets can be added if needed to resolve a convergence
question, rather than being assumed to be a prerequisite for this investigation.

## Unified current ranking

Both directional targets must be met. The target column refers to DIR-Lab only;
cross-dataset validation is reported separately. Full configurations, per-case
scores, solver/folding checks and pending coverage are retained in
`results/siren_convergence_20261113/all_trials.json` and `all_trials.csv`.

<!-- ranking-start -->
65 completed recipes, 0 pending recipes.

| Recipe | Forward mm | Reverse mm | Large-motion mm | Endpoint NCC | DIR-Lab target |
| --- | ---: | ---: | ---: | ---: | --- |
| width512_first30_reg005 | 1.718 | 1.766 | 2.779 | 0.843 | met |
| width512_first20_reg005_lr1e4 | 1.779 | 1.833 | 2.659 | 0.773 | met |
| width384_first20_reg005_lr1e4 | 1.782 | 1.862 | 2.677 | 0.753 | met |
| width512_first30_reg001_lr1e4_output30 | 1.805 | 1.871 | 3.044 | 0.825 | met |
| width384_balanced10_40_reg005 | 1.813 | 1.882 | 2.948 | 0.794 | met |
| width384_first30_reg005_output30 | 1.823 | 1.877 | 2.912 | 0.807 | met |
| width384_first20_reg005_output30 | 1.823 | 1.893 | 2.699 | 0.738 | met |
| width384_first30_reg_curriculum | 1.837 | 1.891 | 3.124 | 0.822 | met |
| width384_first30_reg005 | 1.834 | 1.916 | 3.147 | 0.825 | met |
| first20_reg01_lr1e4_output30 | 1.903 | 1.953 | 2.906 | 0.650 | unmet |
| balanced10_40_reg01_lr1e4 | 1.894 | 1.962 | 3.000 | 0.714 | unmet |
| first20_reg01_lr1e4 | 1.913 | 1.972 | 2.892 | 0.678 | unmet |
<!-- ranking-end -->

## Current generalization findings

The coarse-frequency winners improve abdomen but regress OASIS. The
original-frequency width-512 combination resolves that mean tradeoff: OASIS
DSC/HD95 are 0.81902/1.91033 versus 0.81833/1.95185 at default 5k; abdomen
scores are 0.40182/19.63410 versus 0.39645/19.79874. Both endpoint fitting and
true-versus-identity correspondence probes are reported below. These results
support a common recipe across the three frozen development subsets, with
per-pair regressions retained in the saved evidence.

## Evidence before this goal

| 5k recipe | Forward TRE mm | Reverse TRE mm | >=10 mm TRE mm | Endpoint NCC |
| --- | ---: | ---: | ---: | ---: |
| alpha 1 control | 2.588743 | 2.623870 | 4.938159 | 0.769764 |
| alpha 2 all hidden weight matrices | 4.140197 | 4.271653 | 8.916338 | 0.790849 |
| alpha 0.5 all hidden weight matrices | 2.883067 | 3.014468 | 4.179152 | 0.569363 |
| First-layer 10–30 linear ladder, alpha 1 | 2.315843 | 2.397091 | 3.759853 | 0.696335 |

Artifacts are in `results/dirlab_quick4_init_scale_5k_20261113/` and
`results/dirlab_init_scale_probes_20261113/`. The reference and its checkpoint
probes are in `results/dirlab_quick4_default_40k_20261113/` and
`results/dirlab_milestone_probes_20261113/`.

Increasing all-layer initial scales improved reconstruction NCC while impairing
large-motion transport. Reducing scales reversed that tradeoff. Restricting the
change to a first-layer ladder improved registration on every case. Uniform
first-layer controls are needed to distinguish frequency diversity from a lower
average scale. These results motivate the hypotheses below, without establishing
a unique cause of slow convergence.

## Experiments and hypotheses

1. **First-layer spectrum.** Compare uniform effective omega 20 and 15 with
   ladders 5–30 and 10–50. The existing 10–30 ladder is the reference candidate.
   Row scaling changes both initial first-layer weights and biases. The activation
   multiplier remains 30 during training, so this tests initialization distribution.
2. **Learning rate.** On the existing ladder, test Adam starting at 4e-4 and 6e-4
   with the existing exponential factor 0.25. Then assess cosine and OneCycle
   schedules on promising spectra. Compare actual rates and gradient clipping,
   not just configured peak rates.
3. **Optimizer.** Compare AdamW with explicit small weight decay, Adam beta changes,
   momentum SGD, and Muon for hidden matrices with Adam for biases and boundary
   layers. Use appropriate rate scales for each optimizer. Weight decay may alter
   frequency content, so reconstruction and derivative probes are essential.
4. **Objective timing.** Assess short reconstruction warm-up and gradual activation
   of PD-LK/registration, against full joint training. Initial hypotheses include
   cleaner early derivatives versus delayed motion learning. Preserve all live
   gradients once the terms are active.
5. **Representation extensions if needed.** Consider explicit neuron frequencies,
   split frequency groups, and trainable sinusoidal mixtures. Audit functional
   evaluation, temporal interpolation, derivatives, and frozen checkpoint restoration
   before experiments using architectural changes.

Frequency and rate screening begins with final checkpoint probes. Full milestone
probes follow for competitive or mechanistically informative candidates. Runs are
never selected by labels used in training. No multiseed experiments are planned.

Initial observations from the live screen:

- Uniform first-layer scale 20 reached 2.2605 mm forward TRE and 3.5851 mm
  large-motion TRE, versus 2.3158 and 3.7599 mm for the earlier 10–30 ladder.
  Frequency diversity is therefore not yet established as the source of the gain.
- Uniform scale 15 reached 2.4633 mm forward TRE with endpoint NCC 0.6118.
  Moving to still lower scales sacrifices local fidelity at this step budget.
- Logged gradient norms exceed the default clipping threshold 1 on about
  97–100% of sampled steps for scale 15 and ladder 5–30. The recorded pre-clip
  norms average roughly 2.4–3.4. This motivates a controlled clipping comparison
  alongside learning rates. Adam's adaptive normalization means clipping frequency
  alone does not establish that the update magnitude is limited proportionally.

Implementation checks completed for the new controls: default Adam parameter
updates match the previous implementation exactly on a synthetic step; optimizer
parameter groups cover every parameter once; AdamW, SGD, Muon and default Adam
pass four-step synthetic training with objective warm-up and LR-only OneCycle.
All 16 existing synthetic SIREN derivative and RK2 training tests passed.
The OneCycle option uses cosine rise/fall with fixed optimizer momentum. Its
configured learning rate is the peak, and `lr_final_factor` is relative to it.

The second prepared screen uses first-layer scale 20 as a common reference and
tests Adam 4e-4, cosine peak 6e-4, OneCycle peak 8e-4, AdamW decay 0.01,
Adam beta2 0.99, hybrid Muon/AdamW, momentum SGD at 0.01 and 0.03,
250-step reconstruction warm-up plus 500-step motion-loss ramp, clipping threshold
5, and registration NCC weight 0.05. A global scalar omega 20 control distinguishes
the initialized spectrum from the frequency-dependent parameterization. Resolved
configs in `results/siren_convergence_20261113/` define every parameter.

Adaptive additions to the second screen compare Adam 1e-4, a lower OneCycle
peak 4e-4, and a first-layer LR multiplier 0.25 at both base LR 2e-4 and 4e-4.
These were motivated by the 10–30 ladder degrading at 4e-4 and 6e-4.
First-layer rate groups were checked for full, non-overlapping parameter coverage.

## Literature and implementation considerations

- Weight scaling changes initial spectral content and convergence. The paper's
  scale selection is workload-dependent. [Yeom et al., ICLR 2025](https://arxiv.org/abs/2410.04779)
- A per-neuron omega canceled by inverse-omega weight initialization preserves the
  initial effective weights. First-layer frequency scaling avoids that cancellation.
  Forward variance and spatial-derivative variance need independent checks.
- OneCycle changes learning rate each optimizer step and can also cycle momentum.
  Initial experiments should keep momentum fixed to identify the LR effect.
  [PyTorch OneCycleLR](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html)
- Native Muon supports matrix parameters, requiring a second optimizer for biases.
  Its matrix update scale differs from Adam.
  [PyTorch Muon](https://docs.pytorch.org/docs/main/generated/torch.optim.Muon.html)
- Richer sinusoidal activations are another option with dedicated variance-aware
  initialization. [STAF](https://github.com/AlirezaMorsali/STAF)

## Live experiment results

### Parameter-level transport diagnostic

The new frozen diagnostic `scripts/diagnose_dirlab_parameter_transport.py` splits
the exact temporal derivative into each layer's weight and bias contributions.
The sum is checked against the existing analytical temporal derivative at all 33
times on every observed landmark path. No fitting or dense-map recovery is used.

For >=10 mm landmarks, the landmark-count-weighted mean of case-level alignment
slopes gives the following additive contributions. These are projections onto
the required `-gradient dot displacement`, rather than variance fractions.

| Model | First layer, weight+bias | First hidden weight matrix | Total alignment slope |
| --- | ---: | ---: | ---: |
| alpha 1, 5k | 0.209 | 0.509 | 0.823 |
| alpha 2, 5k | 0.196 | 0.156 | 0.393 |
| alpha 0.5, 5k | 0.085 | 0.684 | 0.895 |
| First-layer scale 20, 5k | 0.128 | 0.628 | 0.924 |
| Default, 40k | 0.273 | 0.585 | 0.926 |

The alpha 2 alignment deficit is particularly large in the first hidden matrix.
This supports targeted hidden-matrix optimization and separate first-layer rate
tests. Scale 20 already approaches the 40k total slope, while its TRE and endpoint
NCC remain worse, so total slope alone cannot establish convergence. The probe
uses straight observed endpoint paths and measures consistency along those paths.
Artifacts are in `results/siren_convergence_20261113/mechanisms/`.

<!-- convergence-results-start -->
| Recipe | Trained/probed | Forward TRE mm | Large-motion TRE mm | Endpoint NCC |
| --- | ---: | ---: | ---: | ---: |
| first20 | 4/4, 4/4 | 2.2605 | 3.5851 | 0.6839 |
| first15 | 4/4, 4/4 | 2.4633 | 3.5826 | 0.6118 |
| ladder5_30 | 4/4, 4/4 | 2.3538 | 3.5452 | 0.6629 |
| ladder10_50 | 4/4, 4/4 | 2.8490 | 5.4849 | 0.7693 |
| ladder10_30_lr4e4 | 4/4, 4/4 | 2.9013 | 4.4298 | 0.5928 |
| ladder10_30_lr6e4 | 4/4, 4/4 | 3.4778 | 5.2196 | 0.4923 |
| first20_lr4e4 | 4/4, 4/4 | 3.1211 | 4.7069 | 0.5775 |
| first20_cosine6e4 | 4/4, 4/4 | 3.6315 | 5.5399 | 0.5975 |
| first20_onecycle8e4 | 4/4, 4/4 | 4.0488 | 5.4569 | 0.6294 |
| first20_adamw | 4/4, 4/4 | 2.8923 | 4.2496 | 0.5811 |
| first20_beta099 | 4/4, 4/4 | 2.9877 | 4.5628 | 0.5828 |
| first20_muon | 4/4, 4/4 | 8.5459 | 15.2442 | -0.0007 |
| first20_sgd01 | 4/4, 4/4 | 2.2524 | 3.7123 | 0.6654 |
| first20_sgd03 | 4/4, 4/4 | 3.7927 | 5.0364 | 0.5136 |
| first20_warmup250 | 4/4, 4/4 | 3.0937 | 4.6618 | 0.5664 |
| first20_clip5 | 4/4, 4/4 | 2.9733 | 4.5072 | 0.5953 |
| globalomega20 | 4/4, 4/4 | 2.2909 | 3.8431 | 0.7066 |
| first20_reg005 | 4/4, 4/4 | 2.6603 | 3.9090 | 0.5556 |
| first20_lr1e4 | 4/4, 4/4 | 2.2855 | 3.9184 | 0.7001 |
| first20_onecycle4e4 | 4/4, 4/4 | 2.7016 | 4.0848 | 0.6382 |
| first20_firstlr025 | 4/4, 4/4 | 2.3001 | 3.6928 | 0.6825 |
| first20_firstlr025_lr4e4 | 4/4, 4/4 | 3.1077 | 4.6786 | 0.5773 |
<!-- convergence-results-end -->

## Stacked spectrum and motion-supervision trials

The earlier native-grid NCC-weight 0.03 trial improved quick4 forward TRE from
2.606 to 2.256 mm with all four cases improving. Combine this independently
supported change with uniform first-layer scale 20 and the 10–30 ladder, at the
default 2e-4 learning rate. This directly tests whether stronger motion supervision
and a lower initial spectrum provide complementary gains.

These trials also measure two-process-per-GPU throughput. The initial screen used
roughly 25–35% GPU utilization and 2–4 GB of the available 32 GB, with substantial
host capacity free. Shared-GPU per-case wall times are reported separately from
the single-process timing baseline.

Observed shared-GPU per-case times were roughly 155–170 seconds, versus
120–130 seconds for earlier single-process cases. Two processes raised GPU
utilization to roughly 95–100% on most devices. These are informal throughput
observations across the experiment queue; per-case speed comparisons must account
for sharing and recovery differences.

Uniform scale 20 plus NCC weight 0.03 reached 2.1189 mm forward TRE,
2.1872 mm reverse TRE, 3.1914 mm large-motion TRE and 0.6825 endpoint NCC.
Its large-motion score approaches the 40k reference, while overall TRE and local
fidelity remain short of the target. The measured first combination supports
stacking the spectrum and motion-supervision changes.

Follow-up combinations target the parameter-level diagnostic: halve only the
first hidden matrix's initial weight scale, or apply 0.8 to all hidden matrices
after the input layer. The input's effective scale stays 20. A separate NCC-weight
0.05 comparison keeps the default learning rate. Per-layer initialization retains
the existing model and exact derivative path, which passed an independent
autograd check including Hessians and mixed derivatives.

<!-- stacked-results-start -->
| Recipe | Trained/probed | Forward TRE mm | Large-motion TRE mm | Endpoint NCC |
| --- | ---: | ---: | ---: | ---: |
| first20_reg003 | 4/4, 4/4 | 2.1189 | 3.1914 | 0.6825 |
| ladder10_30_reg003 | 4/4, 4/4 | 2.1219 | 3.2038 | 0.6952 |
| first20_reg003_hidden1half | 4/4, 4/4 | 2.2110 | 3.1437 | 0.6392 |
| first20_reg003_hidden08 | 4/4, 4/4 | 2.2154 | 3.2986 | 0.6588 |
| first20_reg005 | 4/4, 4/4 | 2.0403 | 3.0267 | 0.6757 |
| first20_reg003_firstlr025 | 4/4, 4/4 | 2.1307 | 3.2496 | 0.6815 |
| first20_reg003_lr1e4 | 4/4, 4/4 | 2.0381 | 3.2990 | 0.6995 |
<!-- stacked-results-end -->

## Stronger motion-supervision combinations

At first-layer scale 20 and default LR, increasing NCC weight from 0.03 to 0.05
improved forward TRE from 2.1189 to 2.0403 mm and reverse TRE from 2.1872 to
2.0968 mm. Follow this direction with weights 0.1 and 0.2 before selecting the
cross-dataset candidate. These are full 5k runs with the same milestones and probes.
The objective and inference algorithms remain the retained method.

Lowering LR from 2e-4 to 1e-4 at NCC weight 0.03 independently improved forward
TRE from 2.1189 to 2.0381 mm and reverse TRE from 2.1872 to 2.1304 mm.
The stronger-supervision screen therefore also tests weights 0.1 and 0.2 with
LR 1e-4, alongside their 2e-4 controls. The initial validation process had already
loaded its original two-trial plan and selected the current completed best before
these additions ran. The four added lower-rate/mixed-omega trials are queued to
resume after that validation pass. Any improved recipe will require its own
cross-dataset comparison.

Two matched-spectrum parameterization trials retain first-layer effective scale
20 and NCC weight 0.05, while using hidden-layer omega 15 or 10 with inverse-omega
hidden weight/bias initialization. Input omega and output initialization stay 30.
For omega 15, the initial function was checked to match the omega 30 control.
This isolates smaller effective hidden-layer parameter updates from changes to
the initial spectrum. Independent autograd checks passed for spatial, temporal,
Hessian and mixed derivatives with different omega values across layers.

Native DIR-Lab evaluations now retain preprocessing, training, recovery and
evaluation timings, so future comparisons can distinguish training cost from
recovery cost and GPU sharing. Earlier screen records retain end-to-end times.

<!-- strong-results-start -->
| Recipe | Trained/probed | Forward TRE mm | Large-motion TRE mm | Endpoint NCC |
| --- | ---: | ---: | ---: | ---: |
| first20_reg01 | 4/4, 4/4 | 2.0601 | 2.9549 | 0.6436 |
| first20_reg02 | 4/4, 4/4 | 2.2537 | 3.2170 | 0.5469 |
| first20_reg01_lr1e4 | 4/4, 4/4 | 1.9130 | 2.8921 | 0.6783 |
| first20_reg02_lr1e4 | 4/4, 4/4 | 1.9052 | 2.7461 | 0.6233 |
| first20_reg005_hiddenomega15 | 4/4, 4/4 | 1.9737 | 3.0541 | 0.6944 |
| first20_reg005_hiddenomega10 | 4/4, 4/4 | 2.0183 | 3.0913 | 0.6633 |
<!-- strong-results-end -->

## Broad frequency support with derivative-aware initialization

After the optimizer queue, compare first-layer ladders 10–40 and 5–80, each with
and without outgoing-column balancing. All four use NCC weight 0.05, default LR,
four layers and width 256. The outgoing column gain for neuron j is proportional
to `1 / omega_j`, normalized so the mean squared gain is 1. This preserves expected
outgoing affine variance while making `omega_j * gain_j` constant across neurons.
Both intervals have geometric mean 20, so their balanced initial derivative
scales approximately match the successful uniform-20 setup while providing wider
frequency support. The weights subsequently train freely.

This is an initialization experiment with the existing SIREN computation and
analytical derivatives. Unit-RMS gains, equal frequency-times-gain products, and
the independent spatial/temporal/Hessian/mixed-derivative oracle were checked.
Balanced versus unbalanced comparisons are needed to establish whether this
actually improves convergence and transport.

The synthetic initialization check used 4096 independent uniform coordinates.
Normalized-coordinate gradient RMS was 0.1551 for uniform scale 20, 0.2174/0.1612
for plain/balanced 10–40, and 0.3812/0.1537 for plain/balanced 5–80. These checks
verify first-derivative scale at initialization; the training probes determine
whether wider frequency support actually helps. The explicit frequency mode also
passed four full compiled-CUDA synthetic training steps with all gradients live.

<!-- frequency-balance-results-start -->
| Recipe | Trained/probed | Forward TRE mm | Large-motion TRE mm | Endpoint NCC |
| --- | ---: | ---: | ---: | ---: |
| ladder10_40_plain | 4/4, 4/4 | 2.0633 | 3.3249 | 0.7407 |
| ladder10_40_balanced | 4/4, 4/4 | 2.0388 | 3.1458 | 0.7287 |
| ladder5_80_plain | 4/4, 4/4 | 2.7294 | 5.1278 | 0.7825 |
| ladder5_80_balanced | 4/4, 4/4 | 2.3839 | 4.1703 | 0.7702 |
<!-- frequency-balance-results-end -->

## Frequency parameterization and capacity combinations

Two explicit-frequency comparisons precede the wider-model trials. They use the
same balanced 10–40 and 5–80 initial functions, but parameterize each first-layer
row with a fixed omega-dependent factor. Effective weights remain linear in raw
weights, so endpoint parameter interpolation and all analytical derivatives retain
their existing form. This changes optimizer geometry while matching the initial
functions to the initialization-only controls.

With outgoing gains proportional to 1/omega, explicit per-neuron frequency
parameterization also balances the initial gradient scale to raw input weights.
Checks cover initial-function agreement, spatial/temporal/Hessian/mixed derivatives
and parameter VJPs against autograd, optimizer coverage, and a strict frozen
checkpoint roundtrip. This is implemented through fixed row parametrizations,
with all gradients live and the same retained SIREN computation.

The initial cross-dataset candidate regressed OASIS DSC from 0.81833 to 0.80248,
with HD95 increasing from 1.95185 to 2.04202 voxels. It has not passed the
generalization requirement. This reinforces the need to recover local detail
while retaining the large-motion gain.

Earlier pure-MSE reconstruction trials reached landmark patch NCC 0.8645 at width
512 versus 0.7593 at width 256 after 5k steps. These were reconstruction-only
results, so registration gains require new evidence. Queue widths 384 and 512
with first-layer scale 20, NCC registration weight 0.05 and LR 2e-4 or 1e-4.
An additional width-512 default-frequency/default-loss run separates increased
capacity from the combined recipe. All remain four-layer, 5k-step runs with 32k
samples and the same milestone schedule. Training time and parameter count must
be reported because each wider step costs more.

<!-- capacity-results-start -->
| Recipe | Trained/probed | Forward TRE mm | Large-motion TRE mm | Endpoint NCC |
| --- | ---: | ---: | ---: | ---: |
| parameterized_ladder10_40 | 4/4, 4/4 | 2.0463 | 3.2289 | 0.7276 |
| parameterized_ladder5_80 | 4/4, 4/4 | 2.2999 | 4.0383 | 0.7696 |
| width384_first20_reg005_lr2e4 | 4/4, 4/4 | 2.1432 | 3.0556 | 0.6904 |
| width384_first20_reg005_lr1e4 | 4/4, 4/4 | 1.7818 | 2.6774 | 0.7533 |
| width512_first20_reg005_lr2e4 | 4/4, 4/4 | 2.4642 | 3.5902 | 0.6580 |
| width512_first20_reg005_lr1e4 | 4/4, 4/4 | 1.7790 | 2.6592 | 0.7729 |
| width512_default | 4/4, 4/4 | 2.1637 | 3.4360 | 0.7568 |
<!-- capacity-results-end -->

## NCC gauge and output initialization

The inherited SIREN output initialization gives roughly 0.03 output standard
deviation on synthetic normalized coordinates, while our NCC Gaussian-moment
prior targets standard deviation 1. Test whether removing this initial gauge
mismatch improves early coupled optimization. Output scaling changes PD-LK
residual scale and early conditioning as well as the head's relative update size,
so this remains a hypothesis requiring the full registration probes.

Queue output scale 30 on the default recipe, output scales 10 and 30 on the
first-20/NCC-0.05 recipe, and output scale 30 on the near-target LR-1e-4/NCC-0.1
combination. Output scaling by 30 was checked directly against the initialized
module function. These trials follow cross-dataset component controls and retain
all other settings, training gradients, milestone saves, and frozen inference.

<!-- output-gauge-results-start -->
| Recipe | Trained/probed | Forward TRE mm | Large-motion TRE mm | Endpoint NCC |
| --- | ---: | ---: | ---: | ---: |
| default_output30 | 4/4, 4/4 | 2.2140 | 3.8478 | 0.7790 |
| first20_reg005_output10 | 4/4, 4/4 | 2.0688 | 3.0391 | 0.6856 |
| first20_reg005_output30 | 4/4, 4/4 | 1.9773 | 3.0007 | 0.7031 |
| first20_reg01_lr1e4_output30 | 4/4, 4/4 | 1.9032 | 2.9059 | 0.6501 |
<!-- output-gauge-results-end -->

## Early-strong registration curriculum

An exact width-256 control for the successful width-384 recipe is queued with
the capacity/output-scale combinations,
changing only width and retaining first-layer scale 20, LR 1e-4 and registration
weight 0.05. This fills the missing cell in the width-by-learning-rate comparison
and tests whether the same convergence can be obtained with the cheaper model.

Two combinations run before the schedule trials: balanced first-layer 10–40 with
LR 1e-4 and constant NCC weight 0.05 or 0.1. At LR 2e-4/weight 0.05, balancing
retained the uniform-20 TRE while improving endpoint NCC from 0.6757 to 0.7287.
The lower-rate/weight-0.1 uniform-20 recipe nearly met the 40k target. Combining
these changes tests whether extra local detail can close its remaining gap.

The OASIS component controls show DSC 0.81078 for first-layer scale 20 alone,
0.81407 for registration weight 0.05 alone, and 0.80248 for both, versus default
0.81833. Both changes contribute to the regression. Test whether stronger motion
supervision can be concentrated early while the final objective returns to the
default registration weight 0.01.

Four recipes use starting registration weight 0.1, cosine decay to 0.01 by step
2500 or 5000, LR 1e-4, and first-layer scale 20 or default 30. PD-LK remains 0.1.
The schedules are predefined constants per step, and every derivative, solve and
trajectory gradient stays live. Schedule endpoint checks and four-step synthetic
training verified the actual logged weights. These trials follow output-gauge
experiments and retain 1k milestone checkpoints.

<!-- curriculum-results-start -->
| Recipe | Trained/probed | Forward TRE mm | Large-motion TRE mm | Endpoint NCC |
| --- | ---: | ---: | ---: | ---: |
| balanced10_40_reg005_lr1e4 | 4/4, 4/4 | 2.0017 | 3.2423 | 0.7280 |
| balanced10_40_reg01_lr1e4 | 4/4, 4/4 | 1.8937 | 3.0004 | 0.7140 |
| first20_reg01_to001_2500 | 4/4, 4/4 | 2.0656 | 3.2304 | 0.6948 |
| first20_reg01_to001_5000 | 4/4, 4/4 | 1.9782 | 2.9578 | 0.6914 |
| first30_reg01_to001_2500 | 4/4, 4/4 | 2.2016 | 3.7842 | 0.7616 |
| first30_reg01_to001_5000 | 4/4, 4/4 | 2.0819 | 3.5326 | 0.7601 |
<!-- curriculum-results-end -->

## Generalization-oriented combinations

The width-384 winner improved abdomen DSC to 0.41089 from 0.39645, with HD95
19.57859 versus 19.79874 voxels, but OASIS remained worse. Its OASIS full-patch
reconstruction NCC is 0.97893 versus default 0.97884, so endpoint-image correlation
alone does not explain the registration regression. This motivates checking the
learned interpolation and registration balance while preserving capacity.

After the capacity queue, test widths 384 and 512 with original first-layer
frequency 30, LR 1e-4 and NCC registration weight 0.05. Also test width 384 with
balanced 10–40 frequencies, and width 384/frequency 30 with registration weight
0.1 decaying to the default 0.01 by step 2500. These isolate whether the original
frequency support or a final default objective recovers MRI behavior while the
larger representation retains the DIR-Lab improvement.

<!-- generalization-refinement-results-start -->
| Recipe | Trained/probed | Forward TRE mm | Large-motion TRE mm | Endpoint NCC |
| --- | ---: | ---: | ---: | ---: |
| width384_first30_reg005 | 4/4, 4/4 | 1.8341 | 3.1469 | 0.8248 |
| width512_first30_reg005 | 4/4, 4/4 | 1.7178 | 2.7792 | 0.8435 |
| width384_balanced10_40_reg005 | 4/4, 4/4 | 1.8129 | 2.9482 | 0.7937 |
| width384_first30_reg_curriculum | 4/4, 4/4 | 1.8375 | 3.1244 | 0.8219 |
<!-- generalization-refinement-results-end -->

## Cross-dataset validation

Results below use the same frozen protocol selections. OASIS official quick4 pairs are 1, 4, 8, 9. AbdomenCTCT official quick4
pairs are 2, 8, 18, 41. Use the official evaluation conventions, evaluate labels
only after fitting, report DSC/HD95/folding and endpoint reconstruction fidelity,
and compare matching baseline and candidate runs at the same budget.

The first native 5k baseline-versus-candidate comparison completed through
`results/siren_convergence_20261113/validate.py`. Candidate selection uses completed
DIR-Lab mean bidirectional TRE, before either validation dataset is evaluated.
Both datasets use their exact frozen protocol pairs and official metric adapter.
Reconstruction probes reuse the same 9-mm windows with 128 samples per label,
using labels 1–35 for OASIS and 1–13 for abdomen only after training.
The initial selected candidate is uniform first-layer scale 20 with registration
NCC weight 0.05, chosen at a completed DIR-Lab mean bidirectional TRE of 2.0685 mm.

<!-- validation-results-start -->
| Dataset | Recipe | Completed/probed | DSC | HD95 voxels | Endpoint patch NCC |
| --- | --- | ---: | ---: | ---: | ---: |
| oasis | validation_default | 4/4, 4/4 | 0.81833 | 1.952 | 0.97884 |
| oasis | validation_candidate | 4/4, 4/4 | 0.80248 | 2.042 | 0.97012 |
| abdomenctct | validation_default | 4/4, 4/4 | 0.39645 | 19.799 | 0.74480 |
| abdomenctct | validation_candidate | 4/4, 4/4 | 0.39002 | 19.475 | 0.66853 |
<!-- validation-results-end -->

The initial candidate also decreased abdomen DSC from 0.39645 to 0.39002,
while HD95 improved from 19.7987 to 19.4749 voxels. Endpoint full-patch NCC
decreased from 0.74480 to 0.66853. OASIS patch NCC decreased from 0.97884 to
0.97012. This candidate has not passed either dataset's DSC comparison.

### Cross-dataset component controls

After the broad-frequency screen, independently test first-layer scale 20 with
default registration weight 0.01, and default frequency with registration weight
0.05, on both frozen quick4 datasets. Together with the existing default and
combined candidate, these form a two-by-two comparison of frequency and loss
weight. The anatomical probes and official evaluation remain identical.

Abdomen weight-only DSC is 0.39845 versus default 0.39645; frequency-only DSC is
0.39328. Thus stronger NCC supervision alone improved abdomen, while the lower
frequency initialization regressed it. OASIS showed regressions from each factor.
The evidence favors testing broader representation support and loss schedules
instead of promoting the coarse first-layer recipe across datasets.

<!-- component-validation-results-start -->
| Dataset | Recipe | Completed/probed | DSC | HD95 voxels | Endpoint patch NCC |
| --- | --- | ---: | ---: | ---: | ---: |
| oasis | validation_frequency_only | 4/4, 4/4 | 0.81078 | 1.959 | 0.97172 |
| oasis | validation_weight_only | 4/4, 4/4 | 0.81407 | 1.983 | 0.97758 |
| abdomenctct | validation_frequency_only | 4/4, 4/4 | 0.39328 | 19.298 | 0.71508 |
| abdomenctct | validation_weight_only | 4/4, 4/4 | 0.39845 | 19.523 | 0.69868 |
<!-- component-validation-results-end -->

### Width-384 candidate validation

The first recipe to beat both directional DIR-Lab targets is validated next on
all official quick4 pairs, with the same anatomical probes and frozen default-5k
baselines above. It uses four layers, 445441 parameters per endpoint, first-layer
scale 20, Adam LR 1e-4 with the default factor-0.25 schedule, PD-LK 0.1 and
registration NCC 0.05. Validation timing may include sharing with other trials.

<!-- width384-validation-results-start -->
| Dataset | Recipe | Completed/probed | DSC | HD95 voxels | Endpoint patch NCC |
| --- | --- | ---: | ---: | ---: | ---: |
| oasis | validation_width384 | 4/4, 4/4 | 0.81055 | 1.985 | 0.97893 |
| abdomenctct | validation_width384 | 4/4, 4/4 | 0.41089 | 19.579 | 0.71534 |
<!-- width384-validation-results-end -->

All 64 final width-384 PCG intervals converged, with zero measured lung folding.
Training took 147.7–156.3 seconds per case under the recorded GPU-sharing workload.
Full 1k/2k/3k/4k/5k probes are complete under
`results/siren_width384_milestone_probes_20261113/`. Unscored intermediate
checkpoints use one recovery and explicitly record that no previous saved metric
was available for parity checking; the final checkpoint retains its saved-score
parity check.

### Width-512 candidate validation

The width-512 version of the same first-20/LR-1e-4/NCC-0.05 recipe reached
1.779 mm forward and 1.833 mm reverse TRE at 5k, with large-motion TRE 2.659 mm
and endpoint NCC 0.773. Its completed official OASIS/abdomen quick4
validation is below; the original-frequency version is evaluated separately.

<!-- width512-validation-results-start -->
| Dataset | Recipe | Completed/probed | DSC | HD95 voxels | Endpoint patch NCC |
| --- | --- | ---: | ---: | ---: | ---: |
| oasis | validation_width512 | 4/4, 4/4 | 0.81286 | 1.987 | 0.98175 |
| abdomenctct | validation_width512 | 4/4, 4/4 | 0.40945 | 19.416 | 0.72805 |
<!-- width512-validation-results-end -->

## Output scale and capacity combinations

Output scaling alone improved the default DIR-Lab forward TRE from 2.5887 to
2.2140 mm, while increasing width alone reached 2.1637 mm at width 512. Test
whether these gains combine while retaining original first-layer frequency 30
and the default registration weight 0.01. Both original LR 2e-4 and the successful
wide-model LR 1e-4 are included. Also stack output scale 30 with the width-384
winner and its original-frequency counterpart. These are predefined 5k trials,
with the same milestones and final probes, queued after width-512 validation.

<!-- combination-results-start -->
| Recipe | Trained/probed | Forward TRE mm | Large-motion TRE mm | Endpoint NCC |
| --- | ---: | ---: | ---: | ---: |
| width256_matched_winner_control | 4/4, 4/4 | 1.9812 | 3.1117 | 0.6939 |
| width384_first20_reg005_output30 | 4/4, 4/4 | 1.8230 | 2.6992 | 0.7375 |
| width384_first30_reg005_output30 | 4/4, 4/4 | 1.8233 | 2.9117 | 0.8066 |
| width512_default_output30 | 4/4, 4/4 | 2.2087 | 3.3767 | 0.7595 |
| width512_first30_reg001_lr1e4_output30 | 4/4, 4/4 | 1.8045 | 3.0441 | 0.8250 |
<!-- combination-results-end -->

## Width-384 convergence trajectory

| Step | Forward TRE mm | Reverse TRE mm | Large-motion TRE mm | Endpoint NCC |
| --- | ---: | ---: | ---: | ---: |
| 1000 | 2.63279 | 2.77567 | 3.56476 | 0.50066 |
| 2000 | 2.28003 | 2.36736 | 3.26901 | 0.61429 |
| 3000 | 2.02789 | 2.07710 | 2.93765 | 0.68205 |
| 4000 | 1.92988 | 2.00173 | 2.82643 | 0.72439 |
| 5000 | 1.78180 | 1.86219 | 2.67744 | 0.75329 |
| Default 40000 | 1.89463 | 1.93786 | 3.13713 | 0.85003 |

All 320 forward recovery intervals across the five checkpoints converged. Final
checkpoint scores passed the saved-metric parity check. Intermediate checkpoints
had no previously saved scores; their probes explicitly record that parity was
not available. Full scalars are in
`results/siren_width384_milestone_probes_20261113/summary.json`.

![Width-384 checkpoint convergence](../results/siren_width384_milestone_probes_20261113/convergence.png)

The original-frequency refinement queue was interrupted by SIGTERM before its
first case completed; no metric was produced. It was restarted using the same
frozen configurations. These incomplete attempts are excluded from scores.

## Capacity and temporal-transport decomposition

A frozen derivative probe evaluates 33 times along each observed landmark path
and decomposes temporal change into per-layer weight and bias contributions.
The components sum to the existing analytic temporal derivative within the
probe tolerance; no training or recovery is performed.

For landmarks moving at least 10 mm, mean case-level target-alignment slopes
are 0.9354 at width-384 step 3k, 0.9391 at step 5k and 0.9477 for width-512 at
5k. The first hidden weight matrix contributes 0.7086, 0.7034 and 0.7177,
respectively. Most useful temporal transport is therefore still carried by the
hidden representation rather than a larger first-layer frequency range. Width
384 improves TRE substantially from 3k to 5k while this alignment is already
similar, supporting a later spatial-accuracy bottleneck. These slopes do not
uniquely identify the optimizer's mechanism and are not substitutes for TRE.
Raw decompositions are in `results/siren_convergence_20261113/mechanisms/capacity/`.

### validation_width512_first30

Frozen source: `results/siren_generalization_refinement_20261113/width512_first30_reg005`. Same 5k budgets and official quick4 metrics.

<!-- validation_width512_first30-start -->
| Dataset | Trained/probed | DSC | HD95 voxels | Endpoint patch NCC |
| --- | ---: | ---: | ---: | ---: |
| oasis | 4/4, 4/4 | 0.81902 | 1.910 | 0.98621 |
| abdomenctct | 4/4, 4/4 | 0.40182 | 19.634 | 0.76279 |
<!-- validation_width512_first30-end -->

## Why width changes the useful Adam rate

A one-step diagnostic used the actual case-01 images, the same native crop-center
mask policy, omega 30, registration weight 0.05 and matched sampling streams.
Only width and LR changed. It wrote scalar summaries and performed no recovery.
On 4096 evaluation-only lung voxels:

| Width | LR | Hidden-1 relative weight update RMS | Output-function update RMS | Temporal derivative RMS |
| --- | ---: | ---: | ---: | ---: |
| 256 | 2e-4 | 0.06788 | 0.08236 | 0.01395 |
| 384 | 2e-4 | 0.08316 | 0.11743 | 0.03216 |
| 512 | 2e-4 | 0.09607 | 0.17657 | 0.03262 |
| 256 | 1e-4 | 0.03394 | 0.04400 | 0.00712 |
| 384 | 1e-4 | 0.04158 | 0.06594 | 0.01788 |
| 512 | 1e-4 | 0.04804 | 0.10409 | 0.01918 |

SIREN hidden weights shrink with inverse square-root width, while the first Adam
step is approximately a per-coordinate sign step. The measured relative updates
follow that width dependence. Global gradient clipping does not remove Adam's
coordinate normalization. This supports a concrete step-size explanation for
why widening at the old rate impaired convergence; one step does not establish
all causes of the final score. Raw setup and values are in
`results/siren_convergence_20261113/mechanisms/initial_update.py` and `.json`.

## Where the coarse-frequency OASIS regression occurs

Across 140 pair-by-label observations, width-384/first-20 improved only 47 label
Dice scores; width-512/first-20 improved 50. Grouping observations into quartiles
by fixed-label voxel count, the largest quartile improved by 0.00135/0.00301,
while each smaller quartile regressed. The second-smallest quartile lost
0.01434/0.01254 Dice. Reconstruction patch NCC was stable or higher in most
quartiles. These observations support a fine-anatomy alignment tradeoff that
whole-image reconstruction correlation misses. They motivate retaining original
frequency support rather than treating higher endpoint NCC as sufficient.
Per-region evidence is in `results/siren_convergence_20261113/mechanisms/oasis_region_comparison.json`.

Five persistent synthetic checks now cover unchanged default initialization and
Adam updates, all combined frequency controls against independent autograd
(including parameter VJPs), strict checkpoint restoration with fixed-frequency
buffers, optimizer parameter coverage, and registration schedule endpoints.
All passed. Earlier 36 audited mathematical/dataset-adapter checks also passed.

## Validation and timing protocol

The original-frequency width-512 recipe passes the OASIS mean comparison:
DSC 0.81902 versus 0.81833, HD95 1.91033 versus 1.95185 voxels. Abdomen DSC
is 0.40182 versus 0.39645, HD95 19.63410 versus 19.79874. The remaining planned
training recipes are complete; follow-up validation continues.
A predefined follow-up validates the qualified width-384 balanced-frequency
recipe, the cheaper original-frequency width-384 model if it qualifies, and
promising output-scale combinations selected by DIR-Lab alone. This does not
use labels in training. These remain development quick4 comparisons.

After the experiment, probe and validation queues finish, an isolated timing
repeat runs each frozen DIR-Lab quick4 case once per GPU, with the default 40k
recipe followed by the original-frequency width-512 5k recipe on the same GPU.
It measures `fit_pair` including optimizer setup and process-local compiler-cache
lookup, with identical sampling and logging cadence. Data loading, checkpoint
callbacks, reconstruction and evaluation are excluded. Frozen PCG+RK2 recovery
is timed separately to expose the wider model’s inference cost; its scratch
is temporary and cleaned on exceptions. No extra seed or persistent deformation
map or warped image is produced. Eightfold fewer steps must not be described as
eightfold faster without this wall-clock comparison. The authorized timing
scripts and scalar records are under `results/siren_training_timing_20261113/`.

## Correspondence discrimination of the learned endpoints

The same 9-mm landmark windows provide a registration-relevant comparison
beyond endpoint-to-image NCC. Compare the learned moving/fixed windows at true
landmark correspondences and at unchanged coordinates. All values below are
means across the four cases; transport slopes are also means of case slopes.

| Recipe | True-correspondence model NCC | Identity model NCC | True-minus-identity gap | Large-motion transport slope |
| --- | ---: | ---: | ---: | ---: |
| Default 40k | 0.80711 | 0.24402 | 0.56308 | 0.91425 |
| Width 384, first 20, 5k | 0.81844 | 0.27150 | 0.54694 | 0.93906 |
| Width 512, first 20, 5k | 0.82224 | 0.27560 | 0.54664 | 0.94772 |
| Width 512, first 30, 5k | 0.83610 | 0.25358 | 0.58251 | 0.94372 |

The original-frequency winner improves correlation at true correspondences
while retaining better separation from identity than the coarse-frequency
models. This supports the inference that the coarse initialization helped motion
learning but sacrificed local correspondence specificity. Extra capacity plus
smaller updates permits stronger motion supervision while retaining that detail.
The best recipe needs no altered sinusoidal frequency: four layers, width 512,
omega 30, Adam LR 1e-4 and registration NCC 0.05. The probes support this account
without uniquely proving its causal decomposition.

## First combination passing all dataset means

| Dataset and metric | Comparator | Width-512 original-frequency 5k |
| --- | ---: | ---: |
| DIR-Lab forward TRE mm | 1.89463 (40k) | 1.71776 |
| DIR-Lab reverse TRE mm | 1.93786 (40k) | 1.76619 |
| DIR-Lab large-motion TRE mm | 3.13713 (40k) | 2.77919 |
| DIR-Lab endpoint NCC | 0.85003 (40k) | 0.84345 |
| OASIS DSC | 0.81833 (5k) | 0.81902 |
| OASIS HD95 voxels | 1.95185 (5k) | 1.91033 |
| Abdomen DSC | 0.39645 (5k) | 0.40182 |
| Abdomen HD95 voxels | 19.79874 (5k) | 19.63410 |

All final DIR-Lab PCG intervals converge and lung folding is zero. OASIS and
abdomen mean folding fractions are 0.0000303 and 0.0008308, versus defaults
0.0000334 and 0.0014446. The mean gains do not imply every pair improves: OASIS
DSC improves in two of four pairs and abdomen in two of four. The exact pair
scores remain in each saved evaluation directory.

### validation_width384_balanced

Frozen source: `results/siren_generalization_refinement_20261113/width384_balanced10_40_reg005`. Same 5k budgets and official quick4 metrics.

<!-- validation_width384_balanced-start -->
| Dataset | Trained/probed | DSC | HD95 voxels | Endpoint patch NCC |
| --- | ---: | ---: | ---: | ---: |
| oasis | 4/4, 4/4 | 0.81738 | 1.948 | 0.98225 |
| abdomenctct | 4/4, 4/4 | 0.41023 | 19.546 | 0.73404 |
<!-- validation_width384_balanced-end -->

## Original-frequency width-512 checkpoint convergence

| Step | Forward TRE mm | Reverse TRE mm | Large-motion TRE mm | Endpoint NCC |
| --- | ---: | ---: | ---: | ---: |
| 1000 | 2.43490 | 2.52436 | 3.84391 | 0.65356 |
| 2000 | 2.11395 | 2.12534 | 3.29755 | 0.74017 |
| 3000 | 1.92240 | 2.00044 | 3.03793 | 0.78903 |
| 4000 | 1.82328 | 1.85631 | 2.95672 | 0.82252 |
| 5000 | 1.71776 | 1.76619 | 2.77919 | 0.84345 |

The 4k checkpoint already passes both 40k directional targets. It belongs to the
predefined 5k learning-rate schedule; this does not imply that changing total
steps to 4k gives identical training. All 320 PCG intervals converge. Final saved
scores pass parity checking; intermediate probes explicitly record that no
previous score was available for comparison. The curve comparator uses the
40k run's own schedule and checkpoints, rather than the independently scheduled
5k control.

![Original-frequency width-512 versus default 40k convergence](../results/siren_width512_first30_milestone_probes_20261113/convergence.png)

Standalone PNG/PDF and full scalars are saved alongside the probe summary.

## Frequency-by-capacity interaction at matched optimization

All four comparisons below use LR 1e-4, registration NCC 0.05, PD-LK 0.1,
four layers and the same 5k schedule. Only width and first-layer initialization
frequency differ; hidden layers retain omega 30.

| Width | First frequency | Forward mm | Reverse mm | Mean directional mm | Endpoint NCC |
| --- | ---: | ---: | ---: | ---: | ---: |
| 384 | 20 | 1.78180 | 1.86219 | 1.82200 | 0.75329 |
| 384 | 30 | 1.83410 | 1.91561 | 1.87486 | 0.82484 |
| 512 | 20 | 1.77905 | 1.83332 | 1.80618 | 0.77294 |
| 512 | 30 | 1.71776 | 1.76619 | 1.74197 | 0.84345 |

At width 384, the coarse frequency gives better TRE. At width 512, original
frequency wins. Increasing width benefits the original-frequency model more
than the coarse-frequency model. The evidence therefore supports a capacity
interaction: lower frequencies are useful under a tighter representation budget,
whereas sufficient width permits more local detail without sacrificing motion
learning. This is more informative than treating frequency 20 as a universally
better initialization. The original-frequency width-384 model also meets the
DIR-Lab target and is being validated as a cheaper alternative.

## Case-level 40k comparison for the passing original-frequency stack

| Case | Default 40k forward/reverse mm | Width-512 5k forward/reverse mm |
| --- | ---: | ---: |
| 01 | 1.08839 / 1.07730 | 0.99509 / 1.02546 |
| 03 | 2.16105 / 2.15554 | 1.79059 / 1.83301 |
| 04 | 1.90771 / 2.01011 | 1.76492 / 1.76511 |
| 05 | 2.42135 / 2.50850 | 2.32043 / 2.44118 |

Every case improves in both reported directions; the mean gain is not driven by a single case.

### validation_width384_first30

Frozen source: `results/siren_generalization_refinement_20261113/width384_first30_reg005`. Same 5k budgets and official quick4 metrics.

<!-- validation_width384_first30-start -->
| Dataset | Trained/probed | DSC | HD95 voxels | Endpoint patch NCC |
| --- | ---: | ---: | ---: | ---: |
| oasis | 4/4, 4/4 | 0.81701 | 1.896 | 0.98430 |
| abdomenctct | 4/4, 4/4 | 0.40494 | 19.503 | 0.75137 |
<!-- validation_width384_first30-end -->

## Experiment-completeness audit

All 65 planned recipes have completed training on all four DIR-Lab cases:
260 case fits. All 1300 requested 1k/2k/3k/4k/5k checkpoint files are present.
Nine recipes pass both directional 40k targets after final probes. Follow-up cross-dataset validation and all 12 isolated timing fits/recoveries
are complete; the final audit is `completion_audit.json`.
The interrupted attempt was retained separately and contributes no scores.

To preview a normal final-checkpoint reproduction of the passing stack:

```bash
/venv/main/bin/python scripts/run_dirlab_cases.py --subset quick4 --plan-only \
  --config medical/configs/native_siren_width512_5k.json \
  --output results/dirlab_width512_5k_reproduce
```

The experiment-specific callback adapters supplied intermediate checkpoints;
the ordinary case runner saves the final checkpoint. The preview creates no
outputs and launches no jobs.

## Opt-in configuration controls

These controls were added for reproducible experiments. Defaults preserve the
existing model, objective and Adam schedule; the selected preset needs only the
three ordinary fields described at the top. Checkpoints retain their full model
and training configuration.

| Key | Default | Meaning |
| --- | --- | --- |
| `model.init_scale` | `1` | Multiply non-output weight matrices; biases unchanged. |
| `model.first_omega_range` | omitted | Initial first-layer row frequencies `[low, high]`, scaling weights and biases. |
| `model.first_omega_mode` | `initialization` | `parameterized` retains fixed row factors on raw trainable parameters; effective endpoint weights still interpolate linearly. |
| `model.layer_init_scales` | omitted | One additional weight multiplier per sine layer. |
| `model.hidden_omega` | `model.omega` | Non-first sine multiplier with corresponding inverse weight/bias initialization; first/output initialization use the original omega. |
| `model.balance_first_frequencies` | `false` | Inverse-frequency outgoing-column gains, normalized to unit RMS. Requires a first-layer range. |
| `model.output_init_scale` | `1` | Multiply initialized output weights and bias. |
| `training.first_layer_lr_factor` | `1` | First-layer learning-rate multiplier, including raw fixed-frequency parameters. |
| `training.optimizer` | Adam, betas `[0.9,0.999]`, decay `0` | Names: `adam`, `adamw`, `radam`, `sgd`, `muon`. SGD uses Nesterov momentum; Muon covers hidden matrices and AdamW covers remaining parameters. |
| `training.lr_schedule` | exponential | Alternatives: `constant`, `cosine`, `onecycle`. OneCycle changes LR only; configured LR is its peak. |
| `training.objective_warmup_steps` / `objective_ramp_steps` | `0` / `0` | Endpoint-only warm-up followed by activation of both motion terms. |
| `training.registration_weight_schedule` | omitted | `{name: linear/cosine, start_weight, decay_steps}` decays to `objective.registration.weight`, then stays there. |

Optimizer options include betas, nonnegative weight decay, SGD/Muon momentum and
a Muon hidden-rate multiplier. Cosine/OneCycle schedules accept warmup fraction
and start factor; `lr_final_factor` is relative to the configured rate. RAdam is
implemented but was not included in the completed screen; no performance claim
is made for it. Tested optimizer failures refer to the recorded rates and recipes,
not to all possible settings of those optimizers.

### validation_width512_first30_reg001_lr1e4_output30

Frozen source: `results/siren_combination_20261113/width512_first30_reg001_lr1e4_output30`. Same 5k budgets and official quick4 metrics.

<!-- validation_width512_first30_reg001_lr1e4_output30-start -->
| Dataset | Trained/probed | DSC | HD95 voxels | Endpoint patch NCC |
| --- | ---: | ---: | ---: | ---: |
| oasis | 4/4, 4/4 | 0.82364 | 1.864 | 0.98510 |
| abdomenctct | 4/4, 4/4 | 0.41690 | 19.486 | 0.78553 |
<!-- validation_width512_first30_reg001_lr1e4_output30-end -->

## Final default-loss combination

The output-scale combination retains all default objective weights and changes
only width to 512, LR to 1e-4 and initialized output scale to 30. The ordinary
first/hidden omega remains 30, with four sine layers. This is the recommended
common preset because it reaches the DIR-Lab target and improves all eight
cross-dataset pair Dice scores. The stronger-NCC stack remains preferable when
minimizing the observed DIR-Lab TRE is the priority.

| Metric | Reference | Stronger NCC 5k | Default losses + output scale 30, 5k |
| --- | ---: | ---: | ---: |
| DIR-Lab forward TRE mm | 1.89463 (40k) | 1.71776 | 1.80454 |
| DIR-Lab reverse TRE mm | 1.93786 (40k) | 1.76619 | 1.87067 |
| DIR-Lab >=10 mm TRE mm | 3.13713 (40k) | 2.77919 | 3.04406 |
| DIR-Lab endpoint NCC | 0.85003 (40k) | 0.84345 | 0.82503 |
| OASIS DSC | 0.81833 (5k) | 0.81902 | 0.82364 |
| OASIS HD95 voxels | 1.95185 (5k) | 1.91033 | 1.86353 |
| Abdomen DSC | 0.39645 (5k) | 0.40182 | 0.41690 |
| Abdomen HD95 voxels | 19.79874 (5k) | 19.63410 | 19.48622 |

The default-loss stack's mean OASIS/abdomen reconstruction patch NCC is
0.98510/0.78553 versus baseline 0.97884/0.74480. OASIS folding fraction is
0.0000367 versus baseline 0.0000334; abdomen is 0.0010134 versus 0.0014446.
The small OASIS folding increase is retained in the evidence rather than being
hidden by the DSC/HD95 improvements. DIR-Lab lung folding remains zero.

The smaller width-384 original-frequency and balanced-frequency alternatives
regressed OASIS mean DSC slightly, to 0.81701 and 0.81738, although both improved
abdomen. Their lower parameter count therefore does not reproduce every gain of
the selected width-512 stacks. Full evidence is retained in the preceding tables.
The output-scale result is a tested combination; without its complete matched
factorial controls it should not be attributed uniquely to output initialization.

## Final reproducibility and verification

`final_source_hashes.json` under the central results directory records the
current selected-model implementation and preset hashes. Both opt-in presets
validate successfully. The final integrated run of the mathematical, RK2,
Abdomen/OASIS adapter and convergence-control tests passed **41 tests**; the
complete output is `final_verification.log`. The only output warnings are
upstream SciPy namespace deprecations in the pinned metric dependency.

Both selected stacks preserve the native grid, original data, mask-center-only
DIR-Lab crops, evaluation-only Learn2Reg labels, single seed 42, live training
gradients and unchanged frozen PCG+RK2 inference. There are no persistent map or
warped-image outputs. The repository default remains unchanged; no commit or
push was performed. Generated evidence and checkpoints remain ignored.

## Isolated runtime and memory results

All four frozen DIR-Lab cases were timed with seed 42, one computing fit per GPU.
Each case used the same GPU for default 40k, the stronger-NCC 5k stack and the
common default-loss 5k stack. Training includes `fit_pair`, optimizer setup and
process-local compiler-cache lookup. Data loading, checkpoint callbacks, whole
volume reconstruction and metric evaluation are excluded. Frozen PCG+RK2 recovery
was timed separately. These are full recipe runs, not per-step extrapolations.

| Recipe | Training seconds | Recovery seconds | Sum seconds | Training speedup | Sum speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Default 40k | 779.20 | 11.93 | 791.13 | 1.00× | 1.00× |
| Width 512, stronger NCC, 5k | 153.03 | 19.38 | 172.41 | 5.09× | 4.59× |
| Width 512, default losses + head ×30, 5k | 144.87 | 18.17 | 163.04 | 5.38× | 4.85× |

The common preset reduces mean training from **779.20 to 144.87 seconds** per
case. Recovery rises from **11.93 to 18.17 seconds**. The measured training plus
recovery sum falls from **791.13 to 163.04 seconds**. This sum is not a complete
end-to-end case-runner time because the stages listed above are excluded.
The stronger-NCC stack trains in 153.03 seconds and recovers in 19.38 seconds.
All 192 timing-repeat PCG intervals converge and all recovery scratch is removed.

| Recipe | Training peak allocated MiB | Recovery peak allocated MiB |
| --- | ---: | ---: |
| Default 40k | 1046.03 | 1415.71 |
| Width 512, stronger NCC, 5k | 1953.23 | 1621.25 |
| Width 512, default losses + head ×30, 5k | 1953.23 | 1621.25 |

Memory is PyTorch peak allocated device memory, including input arrays; it is not
reserved memory or total driver/context memory. The benchmark hardware is RTX
5090 and the recorded runtime is PyTorch 2.14.1+cu132, FP32/highest precision.
Per-case measurements, configurations/hashes, solver diagnostics and runtime GPU
snapshots are in `results/siren_training_timing_20261113/`. No extra seed was used.

![Isolated training and recovery runtime](../results/siren_training_timing_20261113/timing.png)

## Mechanistic conclusions

1. **Capacity and update size interact.** Standard SIREN hidden weights shrink
   with inverse square-root width, while Adam's first coordinate updates are
   approximately magnitude-normalized. The measured relative hidden update grows
   from 6.8% at width 256 to 9.6% at width 512 at the old rate. Lowering the rate
   makes the larger representation useful rather than increasing early disruption.
2. **Frequency support needs sufficient capacity.** Coarse initialization helps
   the narrower model's motion learning but sacrifices fine correspondence detail.
   At width 512 the original frequency wins. The smaller-label OASIS losses and
   true-versus-identity correlation probes are consistent with this tradeoff.
3. **Output initialization should fit the objective's gauge.** Under the usual
   sine-activation second-moment approximation of 1/2, the standard output bound
   gives expected output standard deviation about `1/omega`, or 0.033 at omega
   30. Our NCC moment prior targets standard deviation 1. Output gain 30 brings
   initialization near that scale without increasing spatial oscillation frequency.
   The successful default-loss combination supports this conditioning hypothesis;
   its gains are not uniquely attributable to the head without further controls.

A complementary observation is that changing hidden omega from 30 to 15 while
inversely scaling its initialization preserves the initial function but halves
its phase change for a fixed raw Adam update. Its registration improvement
therefore supports an optimization-geometry explanation, independently of adding
frequency bandwidth. Neither the broad ladders nor the optimizer/schedule screen
exceeded the selected standard-frequency stacks under the recorded settings.

The practical result is two reproducible 5k choices: the output-calibrated common
preset for all three development subsets, and the stronger-NCC preset for the
lowest observed DIR-Lab TRE. These results are single-seed frozen quick4 evidence;
full-cohort or multiseed generalization is outside this experiment.

## 2.5-hour follow-up execution

Authorized follow-up uses the common preset, seed42 and frozen quick4. Results are saved in `results/siren_followup_2h_20261113/`.

<!-- followup-2h-results-start -->
| Recipe | Trained/probed | Forward TRE mm | Large-motion TRE mm | Endpoint NCC |
| --- | ---: | ---: | ---: | ---: |
| adam_onecycle | 4/4, 4/4 | 1.8237 | 2.9551 | 0.8297 |
| adamw_exp | 4/4, 4/4 | 1.8164 | 2.9293 | 0.8263 |
| adamw_onecycle | 4/4, 4/4 | 1.8095 | 2.9194 | 0.8302 |
| first25 | 4/4, 4/4 | 1.7853 | 2.8997 | 0.8108 |
| ladder10_40 | 4/4, 4/4 | 1.8509 | 3.0613 | 0.8139 |
| balanced10_40 | 4/4, 4/4 | 1.8002 | 2.8639 | 0.8125 |
| output1_control | 4/4, 4/4 | 2.1878 | 4.1938 | 0.8431 |
| parent_lr05 | 4/4, 4/4 | 2.0161 | 3.4022 | 0.7712 |
| adam_onecycle_lr05 | 4/4, 4/4 | 1.9323 | 3.2081 | 0.7799 |
| adam_onecycle_lr2 | 4/4, 4/4 | 2.8237 | 4.4572 | 0.6768 |
| adamw_exp_lr05 | 4/4, 4/4 | 2.0502 | 3.3683 | 0.7695 |
| adamw_exp_lr2 | 4/4, 4/4 | 2.1875 | 3.3789 | 0.7603 |
| adamw_onecycle_lr05 | 4/4, 4/4 | 1.9098 | 3.1260 | 0.7796 |
| adamw_onecycle_lr2 | 4/4, 4/4 | 2.7503 | 4.3272 | 0.6791 |
| ladder20_30 | 4/4, 4/4 | 1.8504 | 2.9015 | 0.8116 |
| adamw_onecycle_wd1em3 | 4/4, 4/4 | 1.7970 | 2.8601 | 0.8269 |
| stack_adamw001_onecycle_ladder20_30 | 4/4, 4/4 | 1.8205 | 2.8654 | 0.8117 |
<!-- followup-2h-results-end -->

### Follow-up hidden-omega mechanism probe

The one-step case-01 CPU diagnostic starts all three arms from identical endpoint
predictions and spatial/temporal derivatives. With ordinary hidden omega 30,
source prediction-change RMS is 3.05784 and temporal-derivative RMS is 0.55744.
Using hidden omega 15 with doubled initial hidden weights and biases gives
1.61770 and 0.29802. Keeping omega 30 and halving the LR on those hidden weights
and biases gives 1.61770 and 0.29802. Maximum final prediction/gradient/temporal
mismatches between the latter two arms are 1.29e-5, 1.68e-6 and 1.76e-5.
This supports the smaller hidden phase update explanation for the first update.
Raw scalars and code are in `results/siren_followup_2h_20261113/mechanism.json`
and `mechanism.py`. Longer-run behavior remains to be assessed from the saved
training results.

### Follow-up matched comparisons at LR 1e-4

The optimizer factorial gives forward/reverse TRE 1.82373/1.88284 for Adam
OneCycle, 1.81637/1.89831 for AdamW exponential and 1.80947/1.86602 for their
combination. The parent is 1.80454/1.87067. Large-motion TRE improves from
3.04406 to 2.95506, 2.92934 and 2.91936 respectively. The single changes improve
large-motion TRE in three cases each. Adding AdamW within OneCycle improves
both TRE directions in three cases. The observed optimizer–schedule interaction
contrast is -0.02609/-0.04447 mm at this fixed rate. The follow-up LR brackets
will assess the best tested operating point for each combination.

Uniform first frequency 25 reaches 1.78533/1.86188 mm, the plain 10–40 ladder
1.85087/1.92227, and its balanced counterpart 1.80018/1.87137. Their patch NCCs
are 0.81083, 0.81394 and 0.81249. Balancing recovers much of the ladder's lost
registration accuracy and lowers large-motion TRE to 2.86391. These comparisons
show a motion/detail tradeoff at the common rate. Cross-dataset validation is
complete; see the final review below. Current paired case data and LR profiles are in
`results/siren_followup_2h_20261113/review.json` and `dirlab_summary.csv`.

### Output-initialization control and intermediate transport

At width512, LR1e-4 and the default losses, changing output initialization from
×30 to ×1 increases forward/reverse TRE from1.80454/1.87067 to2.18781/2.27040
and large-motion TRE from3.04406 to4.19381. Endpoint reconstruction NCC improves
from0.82503 to0.84309. Learned endpoint windows at true correspondences have
almost equal NCC,0.82949 and0.82993 respectively. Their identity-window NCCs
are0.26100 and0.25734.

The mean case-level large-motion temporal-target alignment slope falls from
0.93313 to0.87694 and correlation from0.97652 to0.95068 with standard output
initialization. This localizes an observed difference to consistency along the
intermediate parameter path, while endpoint fit and endpoint correspondence
correlation remain good. It supports investigating early coupling of the
brightness-flow objective with endpoint fitting. The probe measures straight
observed landmark paths and does not uniquely identify a causal mechanism.
Raw paired summaries are in `output_control_transport.json` under the follow-up
results directory.

### Frequency transfer and image geometry

SIREN inputs use normalized full-volume coordinates. The prepared quick4 physical
extents span159–223mm across OASIS axes,232.5–293.25mm across DIR-Lab axes,
and318–510mm across abdomen axes. A fixed initial omega therefore sets different
physical spatial scales across datasets. This is context for interpreting
transfer results alongside anatomy and contrast. The present comparisons apply
identical frequency recipes across datasets. They do not isolate geometry from
image content. Per-pair geometry is saved in `frequency_geometry.json` under the
follow-up results directory.

### validation_adamw_onecycle

Frozen source: `results/siren_followup_2h_20261113/adamw_onecycle`. Same 5k budgets and official quick4 metrics.

<!-- validation_adamw_onecycle-start -->
| Dataset | Trained/probed | DSC | HD95 voxels | Endpoint patch NCC |
| --- | ---: | ---: | ---: | ---: |
| oasis | 4/4, 4/4 | 0.82511 | 1.871 | 0.98672 |
| abdomenctct | 4/4, 4/4 | 0.42319 | 19.325 | 0.80028 |
<!-- validation_adamw_onecycle-end -->

### validation_first25

Frozen source: `results/siren_followup_2h_20261113/first25`. Same 5k budgets and official quick4 metrics.

<!-- validation_first25-start -->
| Dataset | Trained/probed | DSC | HD95 voxels | Endpoint patch NCC |
| --- | ---: | ---: | ---: | ---: |
| oasis | 4/4, 4/4 | 0.82202 | 1.842 | 0.98317 |
| abdomenctct | 4/4, 4/4 | 0.42669 | 19.061 | 0.77889 |
<!-- validation_first25-end -->

### validation_ladder20_30

Frozen source: `results/siren_followup_2h_20261113/ladder20_30`. Same 5k budgets and official quick4 metrics.

<!-- validation_ladder20_30-start -->
| Dataset | Trained/probed | DSC | HD95 voxels | Endpoint patch NCC |
| --- | ---: | ---: | ---: | ---: |
| oasis | 4/4, 4/4 | 0.82311 | 1.871 | 0.98344 |
| abdomenctct | 4/4, 4/4 | 0.43098 | 19.267 | 0.77462 |
<!-- validation_ladder20_30-end -->

### validation_ladder10_40

Frozen source: `results/siren_followup_2h_20261113/ladder10_40`. Same 5k budgets and official quick4 metrics.

<!-- validation_ladder10_40-start -->
| Dataset | Trained/probed | DSC | HD95 voxels | Endpoint patch NCC |
| --- | ---: | ---: | ---: | ---: |
| oasis | 4/4, 4/4 | 0.82224 | 1.897 | 0.98383 |
| abdomenctct | 4/4, 4/4 | 0.42308 | 19.538 | 0.77568 |
<!-- validation_ladder10_40-end -->

### validation_balanced10_40

Frozen source: `results/siren_followup_2h_20261113/balanced10_40`. Same 5k budgets and official quick4 metrics.

<!-- validation_balanced10_40-start -->
| Dataset | Trained/probed | DSC | HD95 voxels | Endpoint patch NCC |
| --- | ---: | ---: | ---: | ---: |
| oasis | 4/4, 4/4 | 0.82204 | 1.884 | 0.98439 |
| abdomenctct | 4/4, 4/4 | 0.42576 | 19.497 | 0.77516 |
<!-- validation_balanced10_40-end -->

### validation_adam_onecycle

Frozen source: `results/siren_followup_2h_20261113/adam_onecycle`. Same 5k budgets and official quick4 metrics.

<!-- validation_adam_onecycle-start -->
| Dataset | Trained/probed | DSC | HD95 voxels | Endpoint patch NCC |
| --- | ---: | ---: | ---: | ---: |
| oasis | 4/4, 4/4 | 0.82619 | 1.853 | 0.98671 |
| abdomenctct | 4/4, 4/4 | 0.42409 | 19.143 | 0.79952 |
<!-- validation_adam_onecycle-end -->

### validation_adamw_onecycle_wd1em3

Frozen source: `results/siren_followup_2h_20261113/adamw_onecycle_wd1em3`. Same 5k budgets and official quick4 metrics.

<!-- validation_adamw_onecycle_wd1em3-start -->
| Dataset | Trained/probed | DSC | HD95 voxels | Endpoint patch NCC |
| --- | ---: | ---: | ---: | ---: |
| oasis | 4/4, 4/4 | 0.82602 | 1.889 | 0.98676 |
| abdomenctct | 4/4, 4/4 | 0.42142 | 19.334 | 0.80055 |
<!-- validation_adamw_onecycle_wd1em3-end -->

### validation_stack_adamw001_onecycle_ladder20_30

Frozen source: `results/siren_followup_2h_20261113/stack_adamw001_onecycle_ladder20_30`. Same 5k budgets and official quick4 metrics.

<!-- validation_stack_adamw001_onecycle_ladder20_30-start -->
| Dataset | Trained/probed | DSC | HD95 voxels | Endpoint patch NCC |
| --- | ---: | ---: | ---: | ---: |
| oasis | 4/4, 4/4 | 0.82203 | 1.891 | 0.98516 |
| abdomenctct | 4/4, 4/4 | 0.42422 | 19.068 | 0.78942 |
<!-- validation_stack_adamw001_onecycle_ladder20_30-end -->

<!-- followup-final-review-start -->
## Final 2.5-hour follow-up review — 2026-11-13

Execution completed in 108.02 minutes from the original 13:37:48 UTC launch, within the 140-minute execution and 150-minute review limits. All 132 new fits completed: 68 DIR-Lab and 64 transfer cases. All 660 milestone checkpoints (1k–5k) were loaded and checked for matching configuration, step and finite tensors. All final probes succeeded. The current default configuration is unchanged.

The common preset used width 512, four sine layers and output initialization ×30. The output-initialization control used ×1. All comparisons use the frozen quick4 sets and seed 42. The short hidden-frequency probe and isolated optimizer/precision checks supplement the fits. The 1e-4 decay duplicate was skipped at the user’s request after its float32 multiplier was verified to round to 1 throughout this LR range.

| Recipe | DIR-Lab forward / reverse TRE mm | ≥10 mm TRE mm | Endpoint NCC | OASIS Dice | Abdomen Dice |
| --- | ---: | ---: | ---: | ---: | ---: |
| Common Adam/exponential | 1.80454 / 1.87067 | 3.04406 | 0.82503 | 0.82364 | 0.41690 |
| OneCycle, zero decay | 1.82373 / 1.88284 | 2.95506 | 0.82974 | 0.82619 | 0.42409 |
| OneCycle, decay 0.01 | 1.80947 / 1.86602 | 2.91936 | 0.83021 | 0.82511 | 0.42319 |
| OneCycle, decay 0.001 | 1.79704 / 1.83221 | 2.86005 | 0.82695 | 0.82602 | 0.42142 |
| Uniform first25 | 1.78533 / 1.86188 | 2.89969 | 0.81083 | 0.82202 | 0.42669 |
| First 20–30 | 1.85041 / 1.90476 | 2.90148 | 0.81160 | 0.82311 | 0.43098 |
| First 10–40 | 1.85087 / 1.92227 | 3.06134 | 0.81394 | 0.82224 | 0.42308 |
| Balanced first 10–40 | 1.80018 / 1.87137 | 2.86391 | 0.81249 | 0.82204 | 0.42576 |
| OneCycle/0.001 + first 20–30 | 1.82051 / 1.87115 | 2.86544 | 0.81169 | 0.82203 | 0.42422 |

DIR-Lab aggregate TRE uses the retained rounded ccIDIR convention in each direction. Large-motion TRE pools 381 forward landmarks with observed motion ≥10 mm using continuous predictions. Endpoint NCC is the landmark-window reconstruction probe, not anatomical-label NCC. Parent results are reused from the sources recorded in the paired review.

### What the optimizer tests establish

The full 5e-5/1e-4/2e-4 LR brackets prefer 1e-4 in all four optimizer/schedule families. This is the best tested rate for the original decay 0.01 factorial, not a joint optimum over decay, betas, schedule shape and LR. OneCycle is LR-only, with fixed Adam betas. Its cumulative LR at peak 1e-4 is0.31246 versus 0.27054 for exponential, so this schedule comparison also changes the total nominal LR budget.

The high-rate OneCycle failures are visible in training: late unweighted PD-LK is 0.0656 at peak 2e-4 versus 0.0239 at 1e-4 for Adam (AdamW: 0.0635 versus 0.0243), with more sampled gradients above the clipping threshold. Final PCG solves converge in every arm. This supports a training problem in that regime rather than a failure arising only during recovery.

Decay 0.001 gives the best DIR-Lab optimizer-arm score and improves large-motion TRE in all four cases relative to zero-decay OneCycle. Zero decay gives higher Dice and better HD95 on both transfer datasets. At zero / 0.001 / 0.01, OASIS HD95 is 1.85311/1.88884/1.87114 voxels; abdomen HD95 is 19.14332/19.33418/19.32523. Relative to the common parent, zero-decay OneCycle improves OASIS Dice in 4/4 pairs and abdomen in 2/4; decay 0.001 improves OASIS in 4/4 and abdomen in 1/4. The positive abdomen mean for 0.001 is therefore not a uniform case-level gain.

The earlier 0.270%/0.312% decay-only shrinkage figures are real-arithmetic products for decay 0.01, not exact finite-precision shrinkage measurements. Isolated CPU and CUDA updates confirm decay 1e-4 is numerically zero at the present float32 LR range; 0.001 remains active near peak LR and rounds away near the minimum. A 10-step synthetic width 512 check gives exactly matching Adam and AdamW parameters at zero decay.

### What the frequency and combination tests establish

The 20–30 ladder is preferable to plain 10–40 in these quick4 results: it has lower DIR-Lab large-motion TRE and higher Dice on both transfer datasets. It improves abdomen Dice in 3/4 pairs, versus 1/4 for plain 10–40. Uniform 25 also improves abdomen in3/4. All four changed-frequency arms have slightly lower OASIS mean Dice than uniform 30. This does not support choosing a universal new frequency recipe from the DIR-Lab ranking.

The narrow ladder’s large-motion gain mainly reduces underestimation: average signed error along observed motion changes from −2.386 to −2.194 mm, while perpendicular error stays near 1.17–1.18 mm. At motion 0–2 mm, TRE worsens 0.9852→1.0390 mm. Balanced 10–40 retains better small-motion accuracy (0.9768 mm) and improves large motion, but its Dice gain does not transfer to OASIS.

The label-size analysis gives a related tradeoff. Uniform 25 lowers the smallest OASIS-label-quartile Dice by 0.70 percentage points; 20–30 lowers it by 0.41 points. On abdomen, 20–30 raises the smallest-quartile Dice by 3.42 points. Groups were fixed by reference fixed-image label size within each pair. These anatomical means supplement the official case-aggregated Dice and retain their own absent-label handling.

The predefined stack fails to add the single-change gains. Versus the optimizer alone, its DIR-Lab large-motion TRE is essentially unchanged (2.86544 versus 2.86005 mm); OASIS Dice falls 0.82602→0.82203. Abdomen Dice 0.42422 is below the ladder alone 0.43098. The factorial Dice interaction is −0.00347 on OASIS and −0.01128 on abdomen; the large-motion TRE interaction is +0.14797 mm. These results describe this fixed-rate stack and do not establish that all combinations are ineffective. Its lower abdomen folding (0.0002404) and HD95 19.06775 are retained in the full metrics.

### Mechanistic insight beyond hyperparameter ranking

The first-layer frequency prior persists: first-layer weights move only about 2–3% in relative Frobenius norm over 5k, with mean row-direction cosine >0.999 to initialization, while hidden matrices move about 110–130% of their initial norms. Training changes the deeper representation substantially without greatly reshaping the input frequency basis. The ladders are initialization spectra, not fixed frequency bins. Normalized full-volume coordinates also give different physical frequency scales across the three datasets; the saved affine/extent summary documents that geometry.

The controlled hidden-omega probe starts all arms from exactly matching functions and spatial/temporal derivatives. Hidden omega 15 with doubled initial hidden parameters closely matches omega 30 with half hidden LR after one update: maximum prediction/gradient/temporal differences are 1.29e-5/1.68e-6/1.76e-5. Prediction-change RMS falls 3.05784→1.61770. This directly supports a phase-update/parameterization explanation for the immediate omega effect, rather than a different initial representable function. It does not prove equivalence over a full run.

The matched output-init control remains the clearest evidence that endpoint fidelity alone is insufficient. Standard output initialization improves reconstruction NCC 0.82503→0.84309 but worsens large-motion TRE 3.04406→4.19381mm. Endpoint windows at true matches have almost identical NCC (~0.8295), while large-motion intermediate temporal-target correlation drops 0.97652→0.95068 and alignment slope 0.93313→0.87694. This localizes an observed difference to consistency along the parameter path; it does not uniquely identify its cause. Smaller decay changes show much smaller path-correlation differences.

### Decision and remaining work

For a next optimizer reference, retain uniform 30 and compare zero-decay OneCycle with decay 0.001: the former has stronger transfer means, the latter better DIR-Lab TRE. Keep 20–30 as an abdomen-favored frequency candidate. The current default remains unchanged. The most informative next mechanism test is a full-run matched hidden-layer update comparison motivated by the controlled omega probe, with phase/update statistics, before a broader frequency search. A physical-scale input-frequency probe could test whether geometry explains part of the dataset preference.

Smaller models, new full hidden-omega training, isolated timing repeats, wider optimizer/decay sweeps, additional stacks and full-protocol confirmation were deferred. No fit or final probe in the agreed expanded pilot remains pending.

Detailed [case/label/size-group review](../results/siren_followup_2h_20261113/review_summary.md), [DIR-Lab paired contrasts](../results/siren_followup_2h_20261113/dirlab_paired_contrasts.csv), [transfer paired contrasts](../results/siren_followup_2h_20261113/transfer_paired_contrasts.csv), [requirement audit](../results/siren_followup_2h_20261113/completion_audit.json) and [checkpoint audit](../results/siren_followup_2h_20261113/checkpoint_audit.json).

![LR response and frequency transfer](../results/siren_followup_2h_20261113/followup_review.png)

![Matched decay comparison](../results/siren_followup_2h_20261113/weight_decay_review.png)
<!-- followup-final-review-end -->
