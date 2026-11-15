# NCC + strict PD-LK with endpoint intensity gauge: S02 pilot

**Outcome: partial recovery of performance, not a new baseline.** A fixed Gaussian-moment penalty brought output variances close to one and improved unchanged trajectory-based PCG+RK2 registration over a matched NCC control: Dice **.644834 → .667706**, HD95 **6.88864 → 6.79266 mm**, and converged flow intervals **0/16 → 16/16**. It still trails the existing MSE+PD-LK reference (**.698229 / 6.56515 mm**) and introduces 88 folding voxels. No further weights or cohorts were tested.

## Objective and explicit gate policy

Keep paired SIRENs, shared initialization, linear parameter interpolation and strict PD-LK. Replace endpoint MSE with global minibatch Pearson NCC:

`L = sum_endpoints(1-corr(prediction,data)) + lambda_moment/2 * sum_endpoints[mu²+(v+eps)-log(v+eps)-1] + .1*L_PDLK`.

Here mu/v are the mean/population variance of each **predicted endpoint's intensities over the existing uniform voxel minibatch**, including background. Targets remain normalized [0,1] arrays. There are no local patches, image z-scoring, learned affine calibration, deformation network or post-hoc intensity changes. Epsilon=1e-8 stabilizes correlation and the moment term. The Gaussian-KL-inspired penalty constrains two moments, not a Gaussian histogram or variational posterior. It is soft, not an exact normalization layer.

Per the user's explicit instruction, **skip the PSNR reconstruction gate** for NCC. Raw MSE/PSNR remain diagnostic: `endpoint_quality_pass=null`, `psnr_gate_skipped=true`. No affine-fitted PSNR or replacement reconstruction gate is introduced. Non-finite losses/reconstructions still abort. MSE defaults retain their original 25 dB gate. The retired endpoint-only solver remains absent: both conditions recover motion from the interpolated path using unchanged PCG+RK2.

## Fixed controlled test

Prepared LPBA40 S02 moving to S01 fixed, seed42, 5,000 steps. Two conditions were frozen before training: moment weights **0** and **.1**. Both use 32,768 shared endpoint voxel sites, 4,096 independent continuous PD-LK sites, width256/four sine layers/omega30, Adam2e-4 decaying to .25 times its start, global gradient clip1, PD-LK c4/eta1e-6/rho0/weight.1. No architectural, optimizer or recovery change accompanies the penalty.

Initialization and RNG schedules match; all 250 logged time/LR/PD-LK-weight rows are identical. The moment term consumes no extra RNG. Annotation loading is disabled during fitting; labels are read only after both configurations are frozen and trained. Final checkpoints are evaluated without mutation. No coefficient selection, retry or expanded cohort followed. Whole pilot time: **123.79 seconds**. The MSE comparator is the existing immutable S02 run, not a third fit.

## Results

| Measure | NCC+PD-LK | NCC+moments+PD-LK (.1) | Existing MSE+PD-LK |
|---|---:|---:|---:|
| Dice | .644834 | **.667706** | .698229 |
| HD95 mm | 6.88864 | **6.79266** | 6.56515 |
| Converged flow intervals | 0/16 | **16/16** | 15/16 |
| Maximum true relative flow residual | .00103029 | **1.96501e-5** | 2.01950e-5 |
| Folding fraction | 0 | .0000179036 | 0 |
| Mean inverse-composition error mm | .08663 | .15322 | .12678 |
| Outside inverse-map samples | 6.831% | 7.968% | 7.810% |
| Endpoint global NCC moving/fixed | .997767 / .997737 | .996228 / .995999 | .996846 / .996663 |
| Endpoint output means moving/fixed | -.016194 / -.018904 | **.022996 / -.021947** | not measured here |
| Endpoint output variances moving/fixed | .00119989 / .00107877 | **.998869 / .969219** | not measured here |
| Raw endpoint PSNR moving/fixed dB | 9.621 / 10.368 | 3.000 / 2.910 | 32.225 / 32.512 |
| Reconstruction gate | **skipped** | **skipped** | pass |
| Training seconds | 39.68 | 38.21 | 43.78, historical timing |
| Settled training ms/step | 7.635 | 7.640 | about7, historical timing |
| Recovery seconds | 20.06 | 14.94 | 19.58, historical timing |

All 56 union labels are scored using MONAI1.6.1 macro Dice and maximum directed HD95 in millimetres; none are missing, failed or omitted. The regularizer gains **.022871 Dice** and reduces HD95 by **.095981 mm** versus NCC alone. It remains **.030523 Dice lower / .227508 mm worse HD95** than MSE. Folding is **88/4,915,200 voxels = .00179036%**, not zero. Low raw PSNR is not an NCC training failure or a passed gate.

Recovery uses the same 16 midpoint brightness-flow systems, smoothness.01, ridge1e-6, 500-iteration cap, true relative residual tolerance2e-5 and four RK2 substeps per interval. No intensity calibration, smoothness adjustment or map substitution occurs. Timings are observations from sequential single runs, not controlled speed estimates.

## Endpoint gauge and intermediate path

The regularized final minibatch has means **.02425 / -.02028**, variances **1.00135 / .96242**, moment loss **.00086236**, weighted contribution **.00008624**, NCC loss **.00742286** and weighted PD-LK **.00251314**. The control has zero weighted moments, NCC loss .00410903 and weighted PD-LK .00005997. Total losses across these different objectives are not comparable quality scores.

A fixed independent 65,536-site continuous-coordinate probe at nine times checks the interpolated SIRENs without dense intermediate volumes. Regularized path means range **-.0168 to .0279**, variances **.97747–1.00409**. Control variances are **.00109–.00130**, with means approximately -.019 to -.016. The penalty establishes a near-unit scale on these sampled intermediate times. The control also lacks an obvious moment explosion: this is not simply evidence that its endpoint ranges were mismatched. Sparse probes do not prove anatomical transport.

## Interpretation: established math versus causal hypotheses

Ideal NCC+PD-LK has a genuine **scale-shrinkage degeneracy**. Scaling both endpoint readouts by a common positive a and adding a common offset preserves ideal Pearson NCC and transforms the interpolated movie to `a*I+b`. All PD-LK derivatives scale by a. Choosing local velocity zero gives `min_v E <= a²*(q²+c*||m||²)`, so shrinking a can reduce PD-LK toward zero without improving motion. The numerical NCC epsilon eventually breaks exact invariance. The moment term removes this free scale/offset direction. This mathematical observation is consistent with the low-amplitude control; it does not establish the full cause of its anatomical error.

Fixed recovery regularization adds scale dependence: the flow equation becomes `(a²*g*g^T+s*L+eta*Id)v=-a²*g*q`. Holding s/eta fixed changes the velocity and conditioning. The observed 0/16 versus16/16 convergence supports the importance of scale but does not separate better conditioning, changed effective smoothness and better learned trajectories. Training PD-LK also changes with amplitude. Moment matching alone has not solved recoverability.

This is a practical partial correction on one pair/seed. It neither closes the MSE gap nor eliminates folding. Preserve both models and qualifications; do not select another coefficient from these anatomical scores without a new controlled protocol.

## Implementation and verification

`registration.ncc` supplies the pure loss; the shared objective/trainer and sole pointwise PD-LK implementation remain. Configuration requires the explicit gate policy and excludes other experimental objectives. Both endpoint gradients are live. Three synthetic tests cover affine NCC invariance, analytical moment gradients, finite gradients into both SIRENs, constant-output stability, schema/gate handling and non-finite reconstruction rejection. Default MSE values/parameter gradients match the original expression bit-for-bit. Mocked pipeline checks verify recovery with a skipped/null gate. Both suites pass: **50 medical + one Cameraman test**, without optimizer steps in tests.

Local provenance: `results/ncc_moments/20261110T130823_052551Z` preserves frozen protocol/configs, audited runner/source snapshot, final weights, training/path diagnostics and fresh recovery evidence. The [reviewed scalar summary](ncc_moments_20261110.json) includes both settings and the existing reference. All **1,540 earlier result files** passed before/after hashes. No dense image/map/trajectory/velocity is retained. No training outside S02 or commit/push occurred.

Final preservation checks confirm all **2,343 Data files / 3,405,406,449 bytes** and **830 earlier benchmark files / 83,266,043 bytes** unchanged, frozen source hashes matching, exactly two final endpoint checkpoints and empty recovery scratch. The clean wheel has 33 Python modules, six default JSON configs, one JPEG and four metadata files; no retired solver, archives, tests, data or results. Both isolated CLI help checks and current documentation file links pass. Runtime: NVIDIA GeForce RTX5090, PyTorch2.14.1+cu132, CUDA13.2, four CPU threads, original compiled analytical PD-LK/recovery derivatives. Timings have no repeated-run uncertainty estimate.
