# PD-LK-free covariant gradient normals: LPBA S02

The requested 5,000-step S02→S01/seed42 pilot completed with finite gradients, preserved reconstruction and positive auxiliary-map Jacobians. It improved registration over matched MSE-only fitting but remained below strict PD-LK. No loss coefficient or map setting was tuned, and no cohort was run.

| Training | Dice ↑ | HD95 mm ↓ | Folding % | PCG converged |
|---|---:|---:|---:|---:|
| Identity, previously measured | .563048 | 7.83706 | 0 | N/A |
| MSE endpoint-only, existing matched control | .634323 | 7.12078 | .000102 | 16/16 |
| MSE + covariant normals, new pilot | .648737 | 6.99070 | .000061 | 16/16 |
| MSE + edge-Wasserstein, existing pilot | .655575 | 6.85013 | .000468 | 15/16 |
| MSE + strict PD-LK, existing baseline | .698229 | 6.56515 | 0 | 15/16 |

All 56 nonzero union labels were scored with the same MONAI1.6.1 macro Dice and symmetric maximum-directed HD95 in mm. No labels were omitted. New recovered-flow folding affected **3/4,915,200 voxels**. All 16 PCG intervals met the 2e-5 tolerance (maximum true residual 1.98581e-5). Mean inverse-composition error was .05146mm and outside-sampling fraction 3.840%. Existing baselines were not retrained. One case/seed cannot establish general superiority or a fundamental failure of the objective.

## Exact experiment

The paired scalar SIRENs, shared initialization, linear parameter interpolation, endpoint minibatch size 32,768, original 5,000-step Adam schedule and original-data MSE loss were retained. PD-LK weight was zero for every step, including warm-up. Runtime patches made pointwise PD-LK, local velocity solves, PCG and RK2 raise if invoked during fitting. All 250 logged rows have zero PD-LK loss, contribution and weight. No NCC, edge-Wasserstein, intensity/operator consistency or photometric fitting term was present.

The existing four bounded cubic B-spline maps were reused: six controls/axis, Lipschitz bound .25, geometry Adam LR .05 native voxels, normal/data/membrane/boundary geometry weights 1/1/.01/1. Training used 2,000 MSE-only updates followed by 3,000 alternating geometry/SIREN updates. Each block sampled 1,024 moving coordinates and one uniform continuous time with an independent RNG. The image normal weight was .1, with a separate cap of .1 times each endpoint's MSE-gradient norm. Geometry retained its norm-1 gradient clip and coefficient projection. No coefficient search followed the result.

For t, compose complete interval maps and a fractional last map, computing full spatial Jacobians analytically. Compare the current SIREN gradient at phi_t(a) against `solve(Dphi_t(a).T, g0)` after converting both covectors to physical world units. Geometry updates freeze SIREN parameters while keeping query coordinates live. SIREN updates detach the maps and Jacobians. Endpoint readout biases have zero normal gradient. Maps are auxiliary and discarded; final inference uses only the learned SIREN interpolation and the unchanged global brightness-flow PCG+RK2 pipeline.

Reliability is the detached product of `A/(A+tau²)` and `B/(B+tau²)` for the two squared physical gradient magnitudes, with tau=.01 intensity/mm. Angular normalization uses epsilon=1e-4 intensity/mm, cosine clamping and a reliability denominator floored at one. Empty or nearly flat batches remain finite and contribute negligibly. Positive monotone contrast preserves the normal direction; finite epsilon and magnitude-based reliability make the practical weighted objective approximately, not exactly, invariant to intensity rescaling. Negative contrast reversal is not an invariant.

## Synthetic validation

All **58 medical + 1 Cameraman synthetic tests passed**. New tests cover:

- A known smooth shear with an explicitly inverted scalar image and positive exponential contrast remapping; correct inverse-transpose transport gives near-zero loss and the transposed convention fails.
- Cubic-map and composed-trajectory Jacobians against spatial finite differences, including time knots and final composition.
- Finite differences for parameters of both endpoint SIRENs and the B-spline coefficients. Detached reliability is held fixed during the finite-difference comparison, matching the intended adjoint.
- Physical coordinate-basis conversion, flat/tiny-gradient stability, nonnegative loss, rejection of invalid Jacobians and incompatible objectives, freezing of maps/SIRENs, independent endpoint gradient budgets and mocked pipeline affine handoff.

The synthetic translated Gaussian experiment gave normal loss **5.80e-15** for transport and **7.29e-14** after positive monotone remapping, versus **.021218** for intensity crossfade. This strict invariance check used epsilon=1e-8. It shows discrimination for that example; some crossfades can preserve local normal direction, so the objective is not a general appearance-versus-motion classifier.

## Reconstruction, geometry and gradients

Full-volume endpoint MSE was **.00045127/.00041142** and PSNR **33.46/33.86dB**, moving/fixed. Both passed the standard 25dB gate. Minimum sampled endpoint PSNR over active logged monitors was 32.04/32.57dB. No reconstruction collapse occurred.

On 8,192 independent positions at times .25/.5/.75/1, normal loss averaged **.64623** with learned auxiliary geometry versus **.70499** with identity geometry on the same learned SIREN path. Reliability-weighted mass varied by time (.290–.367); comparisons use the same stated weighting rule, not frozen common weights. Auxiliary original-image endpoint MSE fell **.01956→.00812** relative to identity. Auxiliary displacement averaged **4.04mm**, maximum **9.46mm**. Final independent minimum composed-map determinant was **.75691**; the minimum across active training monitors was **.73220**. These auxiliary maps are not the final recovered deformation.

The coefficients reached their preserved bound (**2.20833 native voxels per coefficient**). This is a capacity/constraint flag, not an invalid map: the .25 per-interval derivative bound still guarantees invertible partial maps and compositions. No sampled nonpositive Jacobian occurred.

Weighted raw normal-gradient norm had median **4.288**, maximum **5.400**, versus median applied norm **.003741**, maximum **.005134** after budgeting. Median endpoint scale factors were **.000713/.002054**, moving/fixed; 100%/99.33% of logged endpoint structural gradients were capped. Geometry raw gradient norm peaked at .03167. All checked gradients were finite and remained below the 1e6 instability cutoff. The budget protected reconstruction, but realized structural influence was much smaller than the nominal weighted gradient.

## Cost and startup abort

The complete fit took **75.42s**, with **3.57ms/step** during settled MSE warm-up and **22.47ms/step** during settled active alternating training. Peak allocated training memory was **684.12MiB** (878MiB reserved). Active update timings logged after startup ranged **17.05–24.50ms**, excluding periodic report work and final optimizer overhead; the block-average step timing includes those costs. Final recovery took **12.89s** and the successful fit/diagnostics/recovery runner took **94.18s**. Hardware was one RTX5090, four CPU threads, PyTorch2.14.1+cu132.

There were **two attempts, one complete pilot**. The initial attempt aborted at the first active update (step2001) because its new .5s runtime guard included first-use initialization. It ran 2,000 MSE updates and one geometry update, then stopped before the first normal-regularized SIREN optimizer update; no checkpoint or recovery was produced. Reconstruction was above the gate and the maps were valid.

An audited synthetic timing diagnostic, with **no optimizer steps or medical data**, measured first-use forward/backward cost .689s, followed by .017–.021s for five repeats. The exact lazy operator/library responsible was not isolated. The runtime guard alone was corrected: first active update allows up to 2s once, all later updates retain the .5s guard, and the 180s total training guard remains. The scientific configuration was unchanged. The identical pilot was restarted once and completed; its startup update took **.5065s**, included in total timing and retained in all active log rows. The initial aborted artifact was preserved, not overwritten. This was a startup-guard correction, not loss/map coefficient tuning.

## Interpretation and evidence

The tested normal term gave **+.01441 Dice and -.13008mm HD95** versus MSE-only interpolation, but was **.04949 Dice below PD-LK**. It was also below the existing MSE edge-Wasserstein result. Normal covariance, finite gradients and auxiliary image alignment were insufficient for competitive brightness-flow recovery in this pilot. Heavy gradient budgeting, bounded map capacity, the signed normal descriptor and the mismatch between normal consistency and brightness-flow recovery remain plausible contributors; this one configuration does not isolate them. The branch remains opt-in and is not promoted.

Complete evidence is in `results/covariant_normals/20261110T153823_929227Z`; the guard-aborted attempt is `results/covariant_normals/20261110T153420_489120Z`. Both have exact runners/configs/source snapshots and status; the complete run contains one final endpoint checkpoint and compact diagnostics only. The [reviewed scalar record](covariant_normals_20261110.json) includes both attempts, synthetic timings, final metrics, references and preservation hashes.

Original images and all protected reference checkpoints matched SHA256 before/after. Existing datasets, results and checkpoints were preserved. No auxiliary coefficients, generated volumes, dense intermediate maps or persistent velocities were saved; final recovery scratch was removed. The inspected 46-file wheel excludes data, results, tests and archives. Defaults and presets are unchanged. No broad search, additional scientific configuration, cohort, commit or push was performed.
