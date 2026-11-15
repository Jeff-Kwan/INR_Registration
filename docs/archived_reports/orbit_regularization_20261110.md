# Structured diffeomorphic orbit: bounded S02 calibration

**Outcome: numerically feasible, negative for registration.** Four authorized 5,000-step fits passed reconstruction and numerical guards. The configuration selected without labels reduced held-out orbit error but gave Dice **.698070 / HD95 6.57849 mm**, worse than the adjacent-only operator control (**.700765 / 6.52270 mm**). None of the four candidates improved both anatomical metrics over that control. All had zero measured recovered folding. Do not promote a configuration or expand to cohorts from this evidence.

**Retirement:** implementation, dedicated tests and training/configuration hooks were subsequently archived locally. `operator_orbit` is no longer supported. The report and scalar evidence below are retained unchanged as experimental history; see the [archived design and settings](../archive/methods/STRUCTURED_ORBIT.md).

## Model and safeguards

The opt-in `objective.operator_orbit` branch preserves paired SIRENs, shared initialization, weight-linear interpolation, endpoint reconstruction, strict PD-LK, Adam schedule and final PCG+RK2. Four projected cubic B-spline maps define the continuous forward orbit `phi_t=(id+tau*u_k) o psi_(k-1) o ... o psi_0`, with `Kt=k+tau`. Each partial map retains the sufficient global spatial diffeomorphism bound; the orbit is continuous and piecewise differentiable in time. Direction is moving coordinates to fixed coordinates, with native (W,H,D) vector components.

The orbit loss is the Monte Carlo expectation of `[I(phi_t(a),t)-alpha(t)*I(a,0)-beta(t)]²`. Four coefficients set `alpha=1+p1*t+p2*t*(1-t)` and `beta=b1*t+b2*t*(1-t)`, exactly anchored at t=0. The identity prior is the integrated squared gain deviation plus squared bias. A float64 4-by-4 ridge solve fits coefficients with images/maps frozen; conservative radial shrinkage bounds gain to [.8,1.2] and bias to [-.05,.05]. It is not an exact constrained-QP solve. No photometric network or differentiation through the fit is used.

Each active step fits photometry on 512 independent sites, updates geometry with frozen SIRENs, then updates SIRENs with geometry/photometry fixed. Both terms involving SIREN images remain differentiable. The geometry anchor remains the **uncorrected original-image** endpoint MSE, with unit weight, plus the original membrane/boundary regularizers. Photometry cannot reduce this anchor directly. The image orbit gradient has its own .1 baseline-relative norm budget before original global clipping. All existing coefficient projection, non-finite checks, 65,536-site reconstruction probes and final full-volume 25 dB gates remain.

There is no dense training reconstruction, dense map, derivative-derived velocity, training PCG or integration adjoint. Auxiliary geometry and photometry are discarded; only final SIRENs and compact scalar evidence are retained. [Equations and limitations](../archive/methods/STRUCTURED_ORBIT.md#experimental-structured-diffeomorphic-orbit); [configuration and hooks](../archive/methods/STRUCTURED_ORBIT.md#structured-orbit-prototype).

## Frozen, label-free calibration

Prepared LPBA40 S02 moving to S01 fixed; seed 42; four maps/six controls per axis; 1,024 spatial sites per geometry/image block; one independent uniform time per block; activation after step 2,000; 5,000 steps. Geometry learning rate .05 and all original PD-LK/recovery settings remain fixed. Maximum four fits/20,000 endpoint updates, 180 training seconds per fit, 240 seconds per fit/pair operation and 900 seconds globally. Actual whole-study time was **441.30 seconds**, including four recoveries and preservation checks. No fit failed or was retried.

The prespecified sequence was identity photometry/weight .1, ridge .1/weight .1, ridge .01/weight .1, then weight .05 for the better feasible finite-photometry ridge. Feasibility required full endpoint PSNR >=25 dB, numerical/runtime stability and >=25% reduction in the raw original-data geometry-anchor MSE relative to identity geometry. All four qualified.

For selection, refresh only four photometric coefficients on 4,096 independent sites, without endpoint/geometry updates. Fit at four times (.125, .375, .625, .875); evaluate on a shared 2,048 held-out spatial sites and eight distinct midpoint times. The common score is `heldout orbit + .1*integrated photo energy`, with the same .1 scoring prior for every candidate. Identity photometry participates in selection. **Ridge .1 / orbit weight .1 won**, frozen at **11:52:27 UTC** before any candidate recovery/annotation access. Training annotation loading was patched to raise. Anatomy did not select ridge, weight or feasibility.

All 100 logged pre-activation rows match the prior adjacent control exactly in reconstruction, PD-LK, total loss, sampling time, learning rate and PD-LK weight. The orbit is a time **mean**, whereas adjacent consistency is a four-interval **sum**. Equal nominal weights therefore do not isolate photometry alone; the new identity-orbit control isolates photometry within the orbit formulation. Geometry retains unit orbit weight in every trial. Most image gradients hit the independent budget, so halving nominal weight does not halve the actual applied gradient.

## Reconstruction, cost and unchanged recovery

| Condition: ridge / image orbit weight | PSNR moving / fixed, dB | MSE moving / fixed | Dice | HD95, mm | Converged flow intervals |
|---|---:|---:|---:|---:|---:|
| PD-LK-only reference | 32.225 / 32.512 | .0005991 / .0005608 | .698229 | 6.56515 | 15/16 |
| Adjacent operator reference | 32.171 / 32.449 | .0006065 / .0005689 | **.700765** | **6.52270** | 15/16 |
| Identity photometry / .1 | 32.082 / 32.283 | .0006192 / .0005911 | .698070 | 6.58009 | 15/16 |
| **Selected: .1 / .1** | **32.190 / 32.228** | **.0006040 / .0005987** | **.698070** | **6.57849** | **16/16** |
| .01 / .1 | 32.102 / 32.369 | .0006163 / .0005795 | .699234 | 6.52294 | 16/16 |
| .1 / .05 | 32.223 / 32.382 | .0005994 / .0005779 | .697301 | 6.62891 | 15/16 |

All six conditions pass the full gate, with **zero measured recovered folding**. MONAI 1.6.1 scores all 56 union labels; none are missing. The usual macro Dice and maximum directed 95th-percentile surface distance in physical millimetres are unchanged. No case was omitted. References are immutable earlier runs; the adjacent result was independently reproduced exactly in the preceding path-consistency study, rather than refit a fifth time here.

The selected candidate loses **.002695 Dice** and adds **.05579 mm HD95** versus adjacent-only. Relative to PD-LK-only it also loses .000158 Dice and adds .01334 mm. Relative to the identity-orbit control its Dice change is only **+0.00000068**, with HD95 lower by .00160 mm. The weaker-prior candidate has better anatomical scores than the selected candidate, but still trails adjacent-only; it is not reselected using labels.

Recovery uses the unchanged 16 brightness-flow solves, tolerance 2e-5, maximum 500 iterations, and midpoint RK2 with four substeps per interval. Maximum true residuals, in trial order, are **2.01612e-5, 1.95693e-5, 1.99334e-5, 2.00750e-5**. Identity and half-weight trials have one near-tolerance miss each. Full convergence of both finite-photometry/weight-.1 trials did not establish anatomical improvement.

| Trial | Training, s | Settled active step, ms | Held-out corrected orbit | Common selection score | Endpoint displacement RMS, native voxels | Orbit / endpoint outside samples |
|---|---:|---:|---:|---:|---:|---:|
| Identity / .1 | 77.13 | 20.39 | .00261070 | .00261070 | 5.185 | .568% / 1.221% |
| Selected .1 / .1 | 90.84 | 25.74 | .00251429 | .00251800 | 5.205 | .537% / 1.172% |
| .01 / .1 | 91.20 | 25.79 | .00262806 | .00264085 | 5.233 | .555% / 1.270% |
| .1 / .05 | 88.62 | 24.48 | .00255098 | .00255337 | 5.296 | .543% / 1.270% |

Warm-up medians are 7.05–7.14 ms/step; recovery takes 17.23–17.85 seconds per candidate. These are observations from this study, not a controlled speed benchmark. Outside fractions concern auxiliary maps, with explicit border sampling and overshoot penalties; the spatial diffeomorphism bound does not guarantee that the finite image box maps into itself. They are not recovered folding rates.

## Appearance versus suppressed geometry

At activation all trials have orbit residual **.00839144**. The selected final residual is .00253748 with its last stochastic training coefficients, or **.00251429** after the independent coefficient refresh. Its uncorrected residual is **.00252908**: photometric correction directly removes only about **0.585%** of residual on that final model. Its common selection score improves 3.55% over the trained identity-orbit control, but that reduction does not transfer to anatomy.

Selected refreshed coefficients are `p=(-.00980085,-.00266440)`, `b=(.00190988,-.00319622)`. Endpoint gain/bias are **.990199 / .001910**; conservative all-time deviation bounds are **.010467 / .002709**, well inside hard limits. Integrated photo energy is **3.71467e-5**. The weaker prior permits roughly twice the gain deviation (.021390), but has worse orbit validation. No photometric projection is needed in any logged active row.

Motion is not trivially replaced by affine intensity correction. The selected raw original-data endpoint MSE is **.00778279** versus identity geometry **.01921604**, a 59.50% reduction, with 5.205 native-voxel RMS displacement versus the identity-orbit control's 5.185. Applying photometry lowers that original-data MSE slightly to .00771592. A **separately fitted no-motion** photometric control still has orbit residual **.00765297** and original-data MSE **.01690653**, much worse than learned geometry. These controls reject complete geometry substitution in this pilot; they do not establish anatomically correct auxiliary motion.

Only two original endpoints are observed. Small changes fitted to a synthetic intermediate SIREN movie are not evidence of genuine physical appearance changes. In particular, exact orbit consistency implies a source `I_t+v·grad I=alpha_dot*I(a,0)+beta_dot`; the unchanged strict PD-LK and global brightness-flow solver omit that source. Selected refreshed source RMS is **.00370659**, versus zero for the identity limit. This is a confirmed mathematical mismatch, **not a demonstrated cause** of the negative anatomical result. Other unisolated factors include orbit-versus-adjacent normalization, single-time sampling and auxiliary/inference geometry mismatch.

## Stability, validation and decision

For the selected run's 150 logged active rows, nominal weighted orbit-gradient norms are **3.56e-6–.06102**, baseline norms **.04254–.09655**, and applied orbit norms **3.56e-6–.00765**. Median raw/applied norms are .02232/.00523. The .1 relative budget is enforced every active update, within floating-point rounding; 116/150 logged updates are capped. Geometry-gradient norms are .000278–.001315. Photometric normal-matrix condition numbers are **174.6–206.4**; all solves and gradients pass finite checks. Lowest sampled endpoint PSNR is **30.296 dB** (all trials >=30.279). These logged ranges are not asserted maxima over unlogged steps. No reconstruction collapse or safeguard failure occurs.

Both synthetic suites pass: **51 medical + 1 Cameraman tests**. Four new orbit tests cover forward composition/knot continuity, native coordinates, identity/positive bounded photometry, known affine-photo recovery/ridge shrinkage, finite differences into both SIRENs, frozen geometry/photometry, geometry gradients, schema exclusion and the inherited gradient budget. A full-width synthetic CUDA backward gives endpoint gradient norms **.030868 / .027620**, with no optimizer step. The wheel contains 33 Python modules, six JSON presets, one bundled JPEG and four metadata files; no tests, archives, datasets or results. Both isolated-wheel CLI help checks pass.

**Do not propose a broader fixed configuration from this pilot.** The implementation is practical and stable, but the central registration hypothesis is unsupported on the requested first pair. Preserve the adjacent operator comparator and PD-LK baseline. Any future mechanism test should be separately authorized and prespecified; this study did not change recovery, search more coefficients or launch cohorts.

Local provenance: `results/orbit_regularization/20261110T114635_670758Z` retains frozen protocol/selection, audited runner, source hashes, final checkpoints and compact evidence. [Reviewed scalar summary](orbit_regularization_20261110.json) includes all candidates and exact diagnostics. All earlier result files passed before/after hashes; **2,343 Data files / 3,405,406,449 bytes** and **830 earlier benchmark files / 83,266,043 bytes** remain unchanged. Checkpoints were immutable during recovery. No auxiliary geometry checkpoint was saved; four photometric coefficients appear only in compact diagnostics, not inference checkpoints. Temporary recovery velocity scratch is removed. No commit or push was made.
