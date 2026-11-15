# PD-LK-free edge-Wasserstein pilot: LPBA S02

The two requested 5,000-step edge-Wasserstein pilots were fast and finite. Matched endpoint-only controls show that edge-Wasserstein **substantially improves on unconstrained SIREN interpolation**, confirming that the path otherwise becomes incoherent. It still falls short of PD-LK registration and held-out path consistency. No coefficient was tuned or promoted. All results are one S02→S01 pair and seed42.

| Endpoint loss | Path regularizer | Dice ↑ | HD95 mm ↓ | Folding % | Converged PCG intervals |
|---|---|---:|---:|---:|---:|
| MSE, endpoint-only matched control | None | .634323 | 7.12078 | .000102 | 16/16 |
| MSE, new pilot | Edge Wasserstein | .655575 | 6.85013 | .000468 | 15/16 |
| MSE, existing reference | Strict PD-LK | .698229 | 6.56515 | 0 | 15/16 |
| NCC + fixed moments .01, endpoint-only matched control | None | .589612 | 7.20439 | .484395 | 16/16 |
| NCC + fixed moments .01, new pilot | Edge Wasserstein | .606637 | 6.99130 | .645610 | 16/16 |
| NCC + fixed moments .01, existing reference | Strict PD-LK | .694369 | 6.44561 | .002645 | 16/16 |

All 56 union labels were scored using MONAI1.6.1 macro Dice and maximum directed HD95 in mm; none were omitted. Folding counts were 0/23/0/23,809/31,733/130 out of 4,915,200 voxels in table order. Relative to their exactly matched endpoint-only controls, the edge term gained .02125/.01702 Dice and reduced HD95 by .27065/.21310mm for MSE/NCC respectively. The MSE recovery missed tolerance in one interval (maximum residual 2.00327e-5 versus 2e-5). The NCC recovery converged everywhere but still folded substantially. Existing PD-LK baselines were read-only references, not retrained.

## Fixed protocol and implementation

The optional `objective.edge_wasserstein` replaces PD-LK from the first update. Objective PD-LK weight is exactly zero; a patched PD-LK energy function would raise if reached, and all 250 logged rows per fit have zero PD-LK loss/contribution/weight. No auxiliary deformation, particle registration, velocity network or temporal image derivative enters training. Existing model architecture, shared initialization, endpoint batch size, optimizer/schedule and final brightness-flow PCG+RK2 settings were retained. Original images and labels are warped only during final evaluation.

Each step samples one continuous time, 2,048 uniform continuous positions and eight directions from a fixed independent bank of 64 unit world directions. Each distribution uses 32 midpoint quantiles. Spatial positions and derivatives use the real NIfTI affine and mm units. Original endpoint targets use 32,768 common independent positions, original-image trilinear finite differences and fixed detached weighted quantiles. Edge density has an epsilon floor of 1e-5 intensity/mm; a 1e-20 squared-norm stabilizer prevents undefined norm derivatives at zero. The inverse empirical CDF is linearly interpolated and piecewise differentiable through edge weights.

The frozen structural weight was .01 in both pilots, with no warm-up. Each endpoint's structural gradient was independently capped at 10% of its reconstruction-objective gradient, then combined before the existing global clipping/Adam update. Thus this is budgeted optimization, not the exact gradient of the nominal scalar sum logged as total. NCC used global minibatch correlation and fixed mean-zero/unit-variance Gaussian-moment regularization at weight .01. MSE had no moment term. MSE kept the 25dB gate; NCC explicitly skipped it and never reported a pass.

## Matched endpoint-only control

After the two pilots, the missing endpoint-only control was run for both endpoint losses: MSE alone and NCC plus fixed moments at .01, with PD-LK weight exactly zero and no path loss. Initialization, seed42, sampled endpoint voxels, 5,000-step optimizer schedule, recovery and image pair match the corresponding edge-SW run. The structural module uses its own RNG, so it consumes none of the endpoint sampler's randomness. A runtime guard forbade the pointwise PD-LK solve during fitting; all logged path-loss rows are zero. These are two added fits, not a broad sweep.

The no-path controls confirm the user's earlier observation. On the fixed 8,192-site, 16-direction held-out probe, mean interior (.25/.5/.75) edge loss was **100.43mm² for MSE-only and 70.60mm² for NCC-only**, versus **13.96 and 7.05mm²** with edge-Wasserstein. PD-LK references scored **7.56 and 3.23mm²** on the same probe. Endpoint-only training therefore produces a particularly incoherent interpolated path; edge-Wasserstein materially repairs that path statistic, but does not recover as well as PD-LK.

Registration follows the same ordering. Edge-SW improves over endpoint-only by about **.02 Dice** for both endpoint objectives, but remains .043 Dice below MSE+PD-LK and .088 below NCC+moments+PD-LK. The NCC endpoint-only control itself has .4844% folding; NCC edge-SW has .6456%. This matched result weakens the explanation that edge-SW only looked poor because it removed PD-LK: the term provides real path regularization, yet its global edge-distribution constraint remains insufficient for competitive recovery. It does not distinguish whether that remaining gap comes from descriptor limitations, MC estimator noise, the fixed .01 weight, or gradient-budget clipping.

## Synthetic validation and reconstruction

For a translated Gaussian structure at midpoint, the projected edge loss was **.0000559 mm²**, versus **14.1921** for intensity blending and **41.3738** for fading at the original location. Quantile-weight gradients passed double-precision gradcheck; a SIREN parameter adjoint agreed with central finite differences. Synthetic tests also covered affine axis/spacing conversions, original-image gradient sampling, both endpoint gradients, separate gradient budgets, schema PD-LK exclusion and real-affine pipeline handoff with mocked training/recovery. These synthetic examples do not guarantee medical anatomical correspondence. Pure amplitude fading of an already correctly transported structure is largely invisible to normalized edge mass by design.

MSE endpoint PSNR was **33.41/33.85dB** (moving/fixed), passing the gate. NCC full-volume endpoint correlation was **.997852/.997850**, mean **.000869/.002921**, variance **1.001571/.997879**. NCC raw MSE was .50900/.52340, consistent with intentionally different intensity units; no PSNR pass was claimed. Its independent post-hoc min–max PSNR was **22.92/23.76dB**, diagnostic only and never used in recovery.

## Speed, memory and stability

| New pilot | Full fit seconds | Settled milliseconds/step | Peak training allocated MiB | Maximum logged raw / applied structural gradient norm |
|---|---:|---:|---:|---:|
| NCC + moments + edge SW | 41.04 | 7.98 | 683.59 | 62.77 / .07509 |
| MSE + edge SW | 36.61 | 7.32 | 683.33 | 79.80 / .01831 |

The peak allocated measurement covers fitting, including target construction, and excludes later full-volume endpoint decoding. Allocator reserved peaks were 868/1,380MiB; reservation depends on prior allocations and is not a comparable live-memory measure. Existing MSE/NCC PD-LK logs give approximately 6.99/7.55ms per settled step, but these were earlier runs without repeated timing uncertainty or comparable peak-memory measurements. All timings are from one RTX5090, four CPU threads, PyTorch2.14.1+cu132. Both fits, held-out diagnostics and recoveries took **117.46s**; new recoveries themselves took 11.32/14.71s for NCC/MSE.

Structural gradients remained finite at every update and were capped at both endpoints. Every logged endpoint gradient required capping. Median multiplier was .00145/.00157 for NCC moving/fixed, and .000224/.000242 for MSE. This prevents an endpoint update from being dominated by a large inverse-CDF gradient, but also limits the realized structural influence. No NaN/Inf or runtime-limit failure occurred.

## Held-out structure and sampling limitations

Using the same independent 8,192 positions, 16 directions and original-image targets estimated from 65,536 positions, average loss across times .25/.5/.75 was:

| Endpoint objective | Existing PD-LK path, mm² | New edge-SW path, mm² |
|---|---:|---:|
| MSE | 7.5600 | 13.9607 |
| NCC + moments | 3.2259 | 7.0475 |

The new training losses decreased (first/last 20 logged rows: 24.71→11.82 for NCC, 40.66→15.52 for MSE), but neither new path beat its PD-LK reference on this independent structural metric. Comparing an original endpoint against a separately sampled estimate of its own fixed target yielded a sizeable Monte Carlo floor: at 2,048 samples, moving/fixed loss averaged **7.94±3.20 / 7.85±2.66 mm²** over eight repeats. At 8,192 samples it was **2.31±1.26 / 2.12±.72 mm²**. These are estimator diagnostics, not values subtracted from the reported losses.

Confirmed: the implementation is practical at this sample count, gradients are mathematically checked and bounded, and endpoint-only linear interpolation has very high intermediate edge loss. Adding edge-SW sharply reduces that loss and modestly improves registration over matched endpoint-only fits, but final registration remains worse than PD-LK. The structural descriptor loses local edge identity and does not explicitly impose recoverable brightness transport. Sampling noise and heavy gradient capping are additional plausible limitations. The remaining gap cannot be attributed among descriptor limitations, sampling, weight and gradient capping from these runs. No further coefficient search or cohort run is justified by these results alone.

## Evidence and preservation

Exact pilot configs, runner, source snapshot/hashes, final weights and recovery metrics are in `results/edge_wasserstein/20261110T144805_812809Z`. The matched endpoint-only controls and their fresh recoveries are in `results/edge_wasserstein_controls/20261110T150354_725206Z`. The [compact scalar evidence](edge_wasserstein_20261110.json) includes all four method metrics, gradient budgets and sampling diagnostics. The control runner verified both images, both PD-LK references and both edge-SW checkpoints against pre-run SHA256 hashes; source hashes matched during the experiment. No dataset content was changed; previous checkpoints/results were preserved. No generated volumes, maps, dense intermediate trajectories or persistent velocities were saved. Temporary recovery scratch was removed.

Validation: **54 medical + 1 Cameraman synthetic tests passed**. The 45-file wheel contains maintained packages, packaged configs and the JPEG, without datasets/results/tests/archives. No default preset was changed, no additional training configuration was evaluated beyond these two matched controls, and no commit or push was made.
