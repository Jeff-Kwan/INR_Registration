# Registration research session, 2026-11-13/14

The retained results cover the 19 Learn2Reg OASIS public-validation pairs. The 4k time-budget candidate averaged 56.83 s per OASIS pair, with a small reduction in Dice compared with the original fit.

Closed at **2026-11-14 02:37:52 UTC**, after **9.94 hours** of wall-clock research.

## Complete-cohort comparison

Dice, voxel HD95 and shifted SDlogJ follow the official evaluator. Folding below always means the **unshifted nonpositive-Jacobian fraction**, not the official shifted/clipped count.

<!-- LIVE_FULL_RESULTS_START -->
| Recipe | Completed OASIS pairs | OASIS Dice / HD95 vox |
| --- | ---: | ---: |
| Original shared 5k | 19/19 | 0.81279 / 1.805 |
| Captured bulk-RNG reference | 19/19 | 0.81330 / 1.793 |
| GW + orthogonal, native RNG | 19/19 | 0.81304 / 1.806 |
| GW, recovery smoothness 0.02 | 19/19 | 0.81195 / 1.838 |
| Box + orthogonal, retained capture | 19/19 | 0.81362 / 1.804 |
| GW + orthogonal + Sobol time | 19/19 | 0.81349 / 1.796 |
| 4k time-budget candidate | 19/19 | 0.81143 / 1.813 |

- Original shared 5k: 0 unconverged PCG intervals; 0 nonfinite pair HD95 means.
- Captured bulk-RNG reference: 0 unconverged PCG intervals; 0 nonfinite pair HD95 means.
- GW + orthogonal, native RNG: 0 unconverged PCG intervals; 0 nonfinite pair HD95 means.
- GW + orthogonal + Sobol time: 0 unconverged PCG intervals; 0 nonfinite pair HD95 means.
- Box + orthogonal, retained capture: 0 unconverged PCG intervals; 0 nonfinite pair HD95 means.
- 4k time-budget candidate: 0 unconverged PCG intervals; 0 nonfinite pair HD95 means.
- GW, recovery smoothness 0.02: 0 unconverged PCG intervals; 0 nonfinite pair HD95 means.

The original reference uses the shared width-512/four-sine-layer, head-initialization ×30, OneCycle peak 1e-4, AdamW decay 0.001, 5k recipe. The accelerated bulk reference preserves its objectives but changes RNG draw order and caps large native recovery grids at 256 samples per axis. It is a method-level comparison, not a bitwise sampling claim.

The GW candidate uses original per-call spatial CUDA random streams, orthogonal two-pair queries, global/local 50/50 GW-NCC, the same endpoint NCC/moment and PD-LK weights, and fixed eight-time/eight-substep PCG+RK2 recovery. The completed box/orthogonal cohort tests the contribution of GW guidance with the retained execution and sampling. The Sobol cohort changes temporal coverage while integrating the same joint uniform early/late target and retaining the spatial streams. The stronger-regularity comparison reuses the GW trained checkpoints and changes only fixed PCG smoothness from 0.01 to 0.02; its fitting is not repeated, and any composed pipeline-time estimate must be labeled as an estimate.

Do not infer a full-cohort improvement from quick4 alone. The first GW/orthogonal quick4 brain gain was about 0.0022 Dice, but its complete OASIS gain is only 0.00026. Ten brain pairs improve and nine worsen. That is a real remaining tradeoff, not a clean shared win.

## Current selection and transfer

The leading shared accuracy candidate is the [explicit GW/orthogonal/Sobol 5k preset](../medical/configs/native_siren_gw_orthogonal_sobol_5k.json): native voxel units, width 512/four sine layers, uniform omega 30, output initialization ×30, OneCycle peak 1e-4, AdamW decay 0.001, endpoint NCC with moment 0.01, PD-LK 0.1, GW-NCC 0.01, 32k separate particles, original spatial sampling and scrambled two-dimensional temporal quadrature. Accurate captured training, cap-256 native recovery and eight times/eight RK2 substeps retain the ordinary analytical estimator and frozen brightness-flow recovery.

OASIS has 9 improving and 10 worsening pairs, so its tiny mean gain is not a broad per-pair improvement. Brain changes are +0.00166/−0.00022/+0.00066.

Folding remains a qualification. The candidate's mean/worst brain fractions are slightly below the original, but some individual pairs worsen. The absolute cohort maximum hides that within-pair increase. It is substantially below the non-Sobol GW spike of 0.010991, but neither candidate establishes a topology-safe solution. The default configuration remains unchanged; this is the strongest measured shared accuracy candidate with its limitation made explicit. The faster 4k full comparison is complete and reported separately below.

The 4k variant averaged 56.83 s per OASIS pair. It used high matmul mode during fitting, four recovery times, and 16 RK2 substeps, with highest precision restored for inference.

That speed has a measured cost: OASIS falls by 0.001353 Dice versus the original and by 0.002055 versus the leading 5k candidate; 13/19 brain pairs lose Dice. The decrease spans small/medium/large label groups (−0.00035/−0.00101/−0.00282). These are two useful operating points, not a claim that the one-minute recipe preserves every accuracy metric.

Those adapted baseline runs are documented with their limitations; these are not claims about reproducing published headline scores.

## What the mechanism probes establish

**Endpoint fidelity does not specify a useful interpolation path.** A shared reconstruction prefix improves endpoint-window NCC yet worsens registration and temporal transport alignment. The independent reconstruction-prefix control also loses, showing that delaying joint registration itself matters. Post-hoc endpoint-affine calibrations can leave NCC unchanged to about 1e-7 while worsening registration substantially. The learned path and its intensity gauge therefore contain information that endpoint correlation alone does not constrain.

Its large-motion temporal-target correlation reaches 0.9933. Small-motion error is best at 10k and worsens thereafter. The exact first-5k OneCycle trajectory is retained, then optimizer/RNG state continues at terminal LR; this is not a schedule stretched to 40k. All 18 new milestone recoveries and fidelity/path probes are saved.

**The global solver can express large coherent motion when the forcing is consistent.** A frozen oracle probe replaces the learned temporal derivative by `q = −g·u_known` for a 20 mm motion and solves the unchanged global brightness-flow system. A faster-varying mode is attenuated to 0.7796, with 0.953 mm error. The 40k checkpoint slightly improves that mode; Gaussian-only features recover the fine mode poorly (0.4820 amplitude, 5.344 mm error). All 12 solves converge, and cap-64 CPU checks agree with the distinction. They show that the large actual gap is not explained by the global operator's inability to represent a coherent displacement of that size. Learning a consistent `q` and retaining enough fine spatial information remain separate problems.


The three-stage coarse/medium/raw 5k cascade gives OASIS Dice 0.80555, with more folds and especially poor small-label transfer.

In OASIS quick4 pair 4, a frozen width-256 Gaussian prefix trained for 2k updates gave Dice 0.63996. Adding a width-512 fine residual trained for 3k updates gave Dice 0.79730 with a zero-initialized output head and 0.79028 with cancelling opposite heads.


**Spatial parameter sensitivity mixes motion and texture.** The existing temporal derivative is `q = J_theta f · delta_theta`; its spatial derivative `m` already enters PD-LK. Large `m` is not automatically useful deformation. Spatially averaging frozen raw derivatives at roughly 8 mm improves large-motion q/target alignment, but averaging derivatives before forming normal equations loses much of the local information and does not improve the local velocity diagnostic. A better temporal correlation is not sufficient evidence of a better velocity estimator.

**Gradient quality has identifiable sampling components.** With one crop held fixed, query directions produce far more registration-gradient variation than resampling particles. Two orthogonal pairs reduce frozen quadrature variation at unchanged four-site cost. Three orthogonal pairs reduce it further but cost more and do not transfer consistently. Fixed axes have zero redraw variance but retain bias. A tetrahedral four-site rule gives a modest frozen improvement without a clear shared training win. Keep the same complete-cell solve and live gradients.

On OASIS pair 1, scrambled 2D Sobol time sampling reduced the 16-draw averaged-gradient error from 0.137 to 0.045 against an 8×8 frozen reference.

**The sampled-map folding problem is often compression, rather than insufficient RK2 substeps.** Increasing spatial substeps on the same fields barely changes the measured folds. All folds in the examined brain cases are central; 38–47% are inside labels. Finer integration keeps those particle Jacobians positive. These observations support severe compression and spatially sampled map differences in those examples; they do not establish global positivity or excuse negative official Jacobians. Stronger PCG regularity reduces folding but also changes alignment, which requires the full metric comparison.

The temporal-sampling fold audit keeps 64 total RK2 substeps and varies only four/eight/sixteen velocity evaluations. The box control is nonmonotone. Additional temporal evaluations therefore do not reliably remove the sampled-map fold tail. Reverse maps also fold: at eight evaluations, GW pairs 3/17 have 0.573/0.651% reverse whole-interior folding, and 0.0296/0.2627% within evaluation-label foreground. Forward background localization cannot be assumed to describe the reverse direction. The leading Sobol candidate also completes that check: pairs 3/17 have 0.625/0.608% reverse whole-interior folds and 0.0091/0.0929% within labeled foreground. The issue remains present in the selected candidate. The fast recipe also has reverse folds on its shared pair-3 stress case and its worst primary-fold pair 11: reverse whole-interior fractions are 0.627/0.689%, with 0.0023/0.0573% within labeled foreground. These are targeted reverse-map checks, not a new full reverse-cohort score.

## Speed and model-size findings

The retained implementation captures sampling, the complete full-gradient objective, backward, clipping and fused AdamW. It uses accurate FP32 LU through `solve_ex` with asynchronous status checking, and preserves every derivative/solve/smoothing/trajectory gradient. Highest FP32 remains the retained numerical mode. Original per-call CUDA RNG replay is checked bitwise for masked/unmasked sampling. Adjacent cases with equal geometry/config reuse compiled code while creating fresh parameters and optimizer state. Four real-input 100-update fits check code reuse and a geometry change; repeated fits introduce no new Dynamo graph and retain close parameter trajectories.

The complete captured step is about 15–16 ms, versus about 25 ms in its matched earlier boundary test. The solver replacement enabled graph capture. High matmul mode accelerates warm fitting but changes gradients.

Fixed temporal recovery sampling is useful independently of spatial RK2 refinement. Eight rather than sixteen velocity evaluations, keeping 64 total substeps, gives small score changes on original and GW frozen checkpoints with converged PCG. Both are declared sampling approximations, not adaptive/enhanced recovery.

The 8k→16k→32k endpoint point curriculum introduces several compiled shapes and graph-transition costs.

A 3k width-512 fit is faster, but OASIS quick4 Dice falls to 0.81844 from about 0.827 at 5k. The high variant can run below a minute, but it is not selected as a well-rounded replacement. Frozen phase diagnostics separate direct parameter displacement from upstream feature changes; neither is cumulative optimizer path length.

The width-256/four-layer pair has 397,314 parameters, and the width-512 pair has 1,581,058.

[Execution interface and guards](CAPTURED_TRAINING.md) · [Performance evidence](PERFORMANCE_RESEARCH.md)

## Audit, failures and evidence

The final audit removed the legacy differential-normal recovery builder, its gradient-weight option and the optional normal-matrix/RHS PCG interface. Brightness-flow recovery retained the same operator and spacing paths.

The no-argument configuration API previously returned obsolete MSE/4k-particle settings while `default.json` selected NCC/32k. It now uses the preset as the single source of truth. Every marathon experiment used an explicit resolved config and is unaffected by that correction. Both adapters synchronize frozen recovery before recording stage timing. Case records include actual imported method source/hash and peak allocated/reserved memory.

Initial dynamic-shape capture failed after one case; the static-shape retry corrected an unpinned CPU shape tensor created inside capture. Newly added peak-memory telemetry also failed in cold workers before training because CUDA was not initialized; a fresh-worker synthetic check preceded its restart. A mixed-version three-frame trial was preserved and rerun from an immutable consistent snapshot. A queue-dependency typo briefly overlapped GPU0 work; the extra job was stopped and the two affected timing cases were repeated. Failures are retained, not silently dropped from the record.

All scientific trials remain single seed 42. No data, masks, volumes, deformation maps or warped images were written as research outputs. Recovery scratch is transient and cleaned on exceptions. The late process-handoff controller also waited on a zombie child; its two completed proof cases were preserved, the idle controller stopped, and the matched control rescheduled after the full run. Source/config/checkpoint identities and all failures are under ignored `results/long_research_20261114/`; no commit or push was made during this session.

Final verification passed 108 CPU tests and six focused CUDA checks. The CPU suite skipped 13 CUDA-dependent cases.

See the [decision log](LONG_RESEARCH_LOG.md), [scorecard](LONG_RESEARCH_RESULTS.md), [research ideas](RESEARCH_IDEAS.md), and [audit plan](METHOD_AUDIT_PLAN.md).

## Stretch targets and interpretation

The faster 4k recipe loses OASIS Dice and HD95 relative to the original reference. These were single-seed development measurements.

## Mechanisms to pursue next

1. **Protect coarse transport while learning detail.** Coarse images establish a useful capture range, while sharp residuals can change the temporal derivative of the summed representation. First measure whether a refinement preserves the coarse q/target alignment and map amplitude, then compare fine endpoint fidelity at the same full runtime. The additive proof and its matched initialization control identify whether inactive hidden gradients are actually the limiting factor; more initial gradient alone is not a success criterion.
2. **Explain and control compression in both directions.** Time refinement does not reliably repair the fold tail, and foreground/background localization differs by map direction. Use the existing direct-particle versus sampled-map Jacobian check to identify the deformation scales being compressed. A training change should reduce those tails while retaining organ alignment, rather than hiding them through masks or shifted Jacobian metrics. Simply increasing the global PCG prior already loses useful alignment.
3. **Use complete time-to-quality measurements.** Capture and recovery sampling removed large costs. Dense SIREN arithmetic and the required number of useful updates now dominate. Wider/shallow arithmetic estimates, a matched LR integral and faster isolated GEMMs did not predict full registration wins. A new numerical or scheduling change needs full-gradient checks and complete-pipeline accuracy/folding comparisons; the 4k frontier provides the practical baseline for the remaining one-minute gap.
