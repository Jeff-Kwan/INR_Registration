# Why the particle-loss S02 reconstruction collapsed

**Finding:** the tiny registration *value* concealed enormous, numerically sensitive registration *gradients*. Differentiating repeated local-velocity queries through four RK2 steps amplifies spatial sensitivity. Those gradients produce large Adam updates, consume the global clipping budget, and suppress endpoint fitting. This is supported by an exact replay, numerical checks and short interventions; a simple coefficient error or persistent gradient accumulation is not supported.

The maintained objective, baseline, presets and inference code were **not changed**. A gradient-budget intervention below is diagnostic evidence, not a deployed fix.

## Reproduction and onset

Original case: LPBA S02, seed 42, 128 particles, four RK2 substeps, strict PD-LK, registration weight 1e-5 after step 2,000. Original runs saved only final weights, without optimizer states. Four bounded diagnostics therefore replayed the 2,000-step warm-up in memory to reconstruct Adam/RNG state. Each warm-up matched every logged loss exactly. The main 120-update trace also matched every original logged loss through step 2,120 exactly. Other branches used 13 or 60 active prefix steps and at most 47 continuation updates. No full fit, cohort, recovery or parameter search was launched.

Gradient norms below are global L2, **before clipping**. PD-LK and registration columns include their configured weights. MSE is measured after each update on the same independent 65,536-voxel probe, rather than changing training minibatches.

| Step | Reconstruction gradient | Weighted PD-LK gradient | Weighted registration gradient | Global clipping multiplier | Moving / fixed probe MSE |
|---|---:|---:|---:|---:|---|
| 2,001 | .0463 | .0361 | .00649 | 1 | .000917 / .000799 |
| 2,014 | .0573 | .0429 | **1.435** | .696 | .001372 / .000821 |
| 2,020 | .216 | .161 | .0883 | 1 | .004602 / .000983 |
| 2,060 | .753 | 1.70 | **64.5** | .0155 | .04773 / .005557 |
| 2,100 | .858 | 2.40 | **33,263** | 3.01e-5 | .09958 / .007964 |
| 2,115 | .822 | 5.02 | **246,501** | 4.06e-6 | 0.10455 / 0.008227 |
| 2,120 | .825 | 62.4 | **1,242** | .000804 | .10593 / .008325 |

At step 2,014, the weighted registration value is only **3.80e-8**. Its moving/fixed gradient norms are **1.428 / .143**, versus reconstruction **.0475 / .0321**. Moving-network weight gradient norms by layer 0–4 are .103, .987, .694, .673, .304: the effect is distributed across the network. The main trace records every parameter tensor, both networks and both Adam moments.

Clipping activates on 93/120 updates. At step 2,115 it scales the reconstruction gradient by about **1/246,500**. Across the trace, registration/reconstruction gradient cosine ranges only from -.0154 to .0135: overwhelming magnitude and sensitivity dominate, rather than a consistently opposed objective direction. All component gradients remain finite. Their sum agrees with the actual combined backward within 4.7e-8 relative error; there is no detected gradient accumulation or extra scaling factor.

The original full-run logs show continued deterioration after this onset; final full-volume moving/fixed MSE is .125544/.047616 (PSNR 9.01/13.22 dB). Intermediate PSNR inferred from the fixed probe is about 30.4/31.0 dB before activation, 23.4/30.1 at step 2,020 and 13.2/22.6 at step 2,060. Only the final values are full-volume evaluations.

## Local solve, RK2 and floating-point audit

- Axes, moving-to-fixed time direction, native-voxel derivative factors, image normalization and interpolation match the documented conventions. At frozen trained parameters, g/q/H/m agree with an independent double-precision autograd construction to **1.8e-15 relative error** on three locations at three times. Existing synthetic translation and finite-difference tests cover axes, sign, RK2 queries and original-image sampling.
- Local systems are positive definite, with minimum eigenvalues near the configured **1e-6** ridge. Maximum sampled condition number is **170,281** at activation and 56,390 at step 2,014; maximum solve residual stays below 4e-6 in the audited paths. These are sensitive systems, not observed singular solves or NaN/Inf failures.
- Eager path audits give maximum particle endpoint spatial-Jacobian amplification of **30,775** at activation, **193,137** at step 2,014 and **238 million** at step 2,060. Individual spatial velocity Jacobians reach norms of 802 at step 2,014; the RK2 time increment is .25. Local sensitivity is repeatedly multiplied along the trajectory.
- At step 2,014, compiled/eager local derivative differences at identical coordinates are about **1e-7 relative**. Yet their particle endpoints differ by up to **.769 voxel** after four steps; compiled/double endpoints differ by **1.150 voxels**. The weighted registration gradient norms are **1.435 compiled float32, .01794 eager float32, .06762 eager float64**. Compiled/double gradient cosine is **-.977**. Passing a finite-gradient check therefore does not imply a reliable gradient direction here.
- Double-precision directional finite differences converge to autograd: **.067623745 versus .067623670**, with perturbation 1e-12 along the normalized parameter-gradient direction. Ordinary 1e-6 perturbations are outside the local linear regime. This supports correct differentiation of an extremely sensitive discrete objective. Local compiled arithmetic differences are amplified by the path; this evidence does **not** establish a compiler backward bug.
- The compiled image-MSE gradient alone has norm **1.434889** at the first large spike; the weighted boundary-gradient norm is only **8.36e-7**. The boundary penalty is not its source. No moving/fixed SIREN intensity output enters the data residual.
- A frozen double-precision resolution check changes registration loss from .003744 to .004031 and weighted gradient norm from .06762 to .09970 when increasing RK2 steps from four to eight. Convergence is not established; simply doubling steps is not a demonstrated remedy.

## Adam and causal interventions

At step 2,014, from identical parameters, minibatch and Adam state:

| One update | Parameter update L2 | Moving MSE before → after |
|---|---:|---|
| Baseline gradients only | .01861 | .001179 → .001175 |
| Joint gradients | **.07393** | .001179 → **.001372** |
| Joint gradients, fresh Adam state | .07212 | .001179 → .001233 |

The existing Adam state is not necessary for the immediate damage. Adam's coordinate-wise normalization makes the registration contribution to the joint update **.07214**, compared with .00324 for reconstruction, .00268 for PD-LK and .01610 for old momentum. These additive contributions use the joint step's clipping multiplier and denominator; their vector sum reconstructs that step to 8.1e-8 relative error. Independent Adam-step predictions match executed updates within 2e-5 relative error, including parameter-rounding effects.

The joint step's first-order prediction for reconstruction is still negative, but measured reconstruction rises: this is also a nonlinear step-size/curvature problem, not merely an uphill gradient projection. The moving-network first-moment norm grows from .00939 at activation to .143 before step 2,020 and .229 before step 2,120; moment history carries earlier damage even on steps with a smaller current registration gradient.

Removing registration just before step 2,014, keeping the existing model and Adam state, yields moving/fixed probe MSE **.000973/.000845** at step 2,060, versus **.04773/.005557** with the unchanged objective. This isolates the particle branch as the trigger.

## Smallest targeted correction and limits

The smallest tested **containment** is to limit the separately computed, weighted registration gradient **before** combining it with the unchanged baseline gradient. A single diagnostic used `||g_reg|| <= .1 ||g_recon + g_PDLK||`, then applied the existing combined clipping/Adam step. Its 47-step continuation reached **.000984/.000858** MSE at step 2,060. At the first spike it reduced the registration gradient by .00485, without scaling away the baseline gradient.

This is a concrete next correction to validate, **not evidence that registration learning is useful or numerically trustworthy**. It addresses destructive optimizer influence; it does not cure the sensitive local trajectory. Lowering a fixed scalar weight, resetting Adam or disabling compilation is not established as a sufficient fix. Any production change should remain opt-in, preserve the baseline, and require gradient-budget and numerical-sensitivity checks before another full fit. More stable particle dynamics/integration would be a separate methodological investigation, not an inferred fix from this pilot.

Confirmed: gradient explosions, clipping starvation, large Adam updates, numerical path sensitivity, correct derivative formulas at audited points, and recovery after branch removal. Most likely mechanism: a feedback loop between highly sensitive local-velocity/RK2 gradients and destructive optimizer steps. Unresolved: how much sensitivity belongs to the continuous local flow versus this coarse discretization, and whether a stable version would improve final global registration.

## Evidence and preservation

[Compact scalar diagnostics](particle_collapse_20261110.json) accompany this report. Full scalar traces and audited diagnostic scripts are local and ignored under `results/particle_diagnosis/`: `20261110T092210_575342Z` (trace), `20261110T092408_589743Z` (path audit/switch-off), `20261110T092639_003914Z` (Adam attribution/budget intervention), `20261110T092823_953525Z` (components/derivative oracle/resolution). No intermediate checkpoints, image outputs, dense maps or persistent particle trajectories were written. Original inputs and earlier run artifacts were read-only.

Validation: both README synthetic suites passed (38 medical tests and one Cameraman test), current documentation links resolve and `git diff --check` passes. Research-source hashes match the diagnostic start. A fresh SHA256/path audit confirms all 2,343 dataset files (3,405,406,449 bytes) and 830 earlier benchmark files (83,266,043 bytes) unchanged. No commit or push was made.
