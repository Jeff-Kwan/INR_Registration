# Guarded LPBA S02 Euler2 registration-aware training

Completed one 5,000-step seed42 pilot. Reconstruction stayed stable, but final
registration is essentially baseline-level at substantially higher training cost.
This is not evidence to promote the Gaussian branch or launch cohorts.

**Interpretation:** this was an augmentation experiment with PD-LK active throughout,
not a replacement experiment. Its baseline-level result does not show that Euler2
can replace PD-LK. After this pilot the user directed that every future new-objective
experiment disable PD-LK entirely, including warm-up. Further experiments are stopped.
Historical configuration and measurements below remain unchanged.

## Frozen configuration

S02 moving→S01 fixed, maintained paired 4×256 SIRENs, shared initialization, MSE
endpoint reconstruction, linear weight interpolation and strict PD-LK (weight=.1,
c=4, eta=1e-6, rho=0) unchanged. Original Adam/lr schedule and global clipping remain.
Registration branch: Cartesian32³, 256 particles, two explicit Euler steps at t=0,.5,
physical Gaussian sigma4 mm, derivative chunks4096, packed analytic derivatives,
compiled local solves and checkpoint recomputation. Nominal registration weight=.01
activates after step2000, with each endpoint's added gradient capped at .1× its base
gradient norm. Original-image endpoint MSE plus normalized boundary penalty; border
sampling. No NCC, BCC, shortened Gaussian kernel or other experimental objective.

Budgets were active at all 150 active logged samples: median scales .059 source /
.174 target. Thus the nominal weighted loss is logged, while the actual update uses
separately budgeted gradients; it is not the exact gradient of that nominal sum.
Original final recovery remains the global brightness-flow PCG + RK2 pipeline.

## Final standard fixed-grid evaluation

These are the maintained inverse-map benchmark scores: warp moving onto the fixed
grid. They differ from preceding integrator probes scored on the moving grid.
MONAI1.6.1 macro Dice/symmetric HD95 over56 labels, none missing. Identity Dice=.563048,
HD95=7.837058 mm. Folding denominator=4,915,200 voxels, det(I+Du)<=0.

| Method | Dice | HD95 mm | Folding | Endpoint PSNR moving/fixed |
|---|---:|---:|---:|---:|
| Existing MSE+PD-LK baseline | .698229 | 6.56515 | 0% | 32.225 / 32.512 dB |
| MSE+PD-LK+Gaussian Euler2 | .698994 | 6.58364 | 0% | 32.040 / 32.211 dB |

Dice difference +.000765; HD95 is .01849 mm worse. This is a mixed, effectively
baseline-level outcome, not a meaningful demonstrated registration improvement.
One pair/seed only. The existing seed42 baseline predates the packed derivative
implementation and was not retrained; the chain is mathematically equivalent but
float32 operation ordering differs, so this is not an exact matched-code ablation.

Final endpoint MSE=.00062516/.00060098. Both endpoints pass the full-volume25 dB
gate. All16 PCG intervals converged; maximum relative linear residual=1.9404e-5
(tolerance2e-5). The historical baseline had a nonconverged near-tolerance interval;
its existing metrics and artifacts are preserved, not silently replaced.

## Training stability and cost

Completed training248.11 s (~4m8s); unchanged final recovery18.10 s. Historical
baseline training43.78 s, so the successful pilot costs about5.7× more in total.
Active-step median74.93 ms, registration portion67.42 ms in logged passes. First
activation/compiler cost is included in the total but excluded from steady-state
interpretation. These timings include the guarded training loop, not just frozen
backward passes. Peak allocated CUDA memory1026.4 MiB on RTX5090.

At150 active logged samples, minimum monitored PSNR30.27/30.86 dB. Final sampled
PSNR31.94/32.25 dB; the full-volume final values above are authoritative. Weighted
raw registration gradient norms were finite and nonzero at both endpoints: source
.0205–.3023, target .00837–.09498. Their separate budgets bounded applied gradient
norms, not Adam parameter updates. No reconstruction collapse was observed.
Unweighted registration loss across logged samples=.00293–.01063 (median .00549);
weighted contribution median5.49e-5. Fresh Monte Carlo samples make these snapshots
noisy; auxiliary loss reduction alone is not a registration success criterion.

## Preflight fix, initial abort and controlled retry

Preflight caught a CSV-schema issue: active Gaussian metrics introduced columns
absent during warm-up. Fixed by initializing the full evidence schema; unmeasured
PSNR is blank. Synthetic coverage writes warm-up/activation/ordinary-step rows.

The first attempt stopped shortly after logged step2060 because one registration
pass took .364 s against a strict .25 s limit. Preceding passes were about .067 s,
with monitored PSNR30.54/31.12 dB. No final checkpoint was saved; its failed status,
logs and23.43 s elapsed evidence remain under the initial run directory.

The fresh retry changed only runtime-outlier handling, not any scientific parameter,
seed, schedule or gradient budget. A bounded60 s first-use compiler allowance remains;
then three consecutive passes above .25 s or one pass above2 s abort. A normal pass
resets the consecutive counter. Numerical/PSNR guards remain independent. Synthetic
tests verify isolated-spike tolerance and sustained/hard-stall rejection. Logged
steady registration passes in the completed retry were66.74–68.72 ms; per-step
pauses that did not coincide with log steps are not individually persisted.

## Validation and preservation

68 medical + one Cameraman synthetic tests passed after the guard change. Wheel
contents passed (47 entries; no datasets/results/tests/archives), and git diff
--check passed. No dataset presets or maintained PD-LK behavior were changed.

Selected input images, annotations and original baseline checkpoint hashes are
unchanged. All datasets remain available; no preprocessing/resampling was performed.
Only one final endpoint checkpoint and scalar evidence were saved for the successful
run. No deformation/velocity/warped-volume writers were added. PCG temporary velocity
scratch was removed; scratch root was empty after recovery. No commit or push.

Successful local run: `results/gaussian_euler2_training/20261111_retry1/`.
Final checkpoint: `run/siren_pdlk_gaussian_registration/S02/endpoints.pt` within it.
Initial failed attempt: `results/gaussian_euler2_training/20261111/`.
Exact configs, source hashes, per20-step CSV, final metrics and per-label evidence
remain in these ignored directories; compact reviewed evidence accompanies this report.
