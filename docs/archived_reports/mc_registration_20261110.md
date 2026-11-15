# Experimental particle endpoint-registration pilot

The local analytic PD-LK particle loss is computationally practical but did not preserve endpoint reconstruction. Neither tested weight is validated. The original PD-LK-only presets remain the baseline.

## Protocol and decision

All fits used seed 42 and 5,000 steps. The experimental term sampled 128 continuous moving-domain particles, integrated four midpoint RK2 substeps with live space/time derivatives, and activated after step 2,000. It used original-image MSE plus normalized squared boundary overshoot, boundary weight 1. Strict PD-LK constants, paired SIRENs, linear parameter interpolation, endpoint reconstruction, optimizer settings and final global PCG/RK2 recovery were preserved. Particle sampling used an independent generator. Native voxel axes and derivative units follow [METHOD](../METHOD.md).

The initial weight .1 completed nine LPBA fits but failed every 25 dB endpoint gate. The user then authorized exactly one smaller-weight calibration: 1e-5 on S02, to be frozen for both cohorts only if it passed. It also failed. Selection used endpoint reconstruction only; no label score was used. The lower-weight full LPBA/OASIS cohorts were therefore **not launched**. No additional coefficient search was performed.

## Results

Timing is the mean synchronized logging-block time over steps 2,021–5,000, excluding initial compilation and first-active-block overhead. Cohort entries average pairs; these are observed training timings, not a separate benchmark.

| Condition | Dataset | Completed fits | Endpoint gates passed | Active/late ms per step | Original endpoint MSE, moving / fixed | Dice | HD95 mm | Folding fraction |
|---|---|---:|---:|---:|---|---:|---:|---:|
| PD-LK control | LPBA | 9/9 | 9/9 | 7.13 | .0005684 / .0005610 | .69875 | 6.294 | 0 |
| PD-LK control | OASIS | 10/10 | 10/10 | 7.09 | .0008644 / .0009481 | .79146 | 2.574 | .00000448 |
| Particle, .1 | LPBA | 9/9 | 0/9 | 23.77 | Gate failures; see per-pair table below | unavailable | unavailable | unavailable |
| Particle, 1e-5 | LPBA S02 | 1/1 | 0/1 | 23.75 | .125544 / .047616 | unavailable | unavailable | unavailable |

| Pair (.1) | Moving PSNR dB | Fixed PSNR dB | Moving MSE | Fixed MSE |
|---|---:|---:|---:|---:|
| S02 | 8.934 | 11.851 | 0.127806 | 0.065297 |
| S03 | 9.172 | 12.135 | 0.121007 | 0.061162 |
| S04 | 9.510 | 10.765 | 0.111933 | 0.083843 |
| S05 | 9.729 | 11.333 | 0.106429 | 0.073576 |
| S06 | 8.455 | 11.896 | 0.142726 | 0.064627 |
| S07 | 8.917 | 11.010 | 0.128312 | 0.079247 |
| S08 | 9.267 | 11.339 | 0.118380 | 0.073471 |
| S09 | 8.913 | 11.829 | 0.128439 | 0.065629 |
| S10 | 8.728 | 12.179 | 0.134029 | 0.060549 |

The 1e-5 calibration ended at moving/fixed PSNR **9.012 / 13.222 dB**, below the required 25 dB. Its last sampled image MSE was .067840, boundary penalty .004898, combined unweighted registration loss .072737 and weighted contribution 7.274e-7. The sampled out-of-bounds fraction was 19.53%. These are final training samples, not held-out registration estimates. Warm-up losses through step 2,000 matched its S02 control exactly at every logged step.

At first activation, registration-only gradient L2 norms into the moving/fixed networks were .006422 / .000966 at weight 1e-5: finite and nonzero. Synthetic finite-difference checks also pass. Finite gradients do not establish well-conditioned training. The cause of reconstruction collapse is unresolved; the small final scalar contribution does not bound earlier gradient effects through repeated local solves and trajectories.

Registration was skipped by the existing quality gate for all experimental completed models. Dice, HD95 and folding are **unavailable**, not zero. The OASIS experimental condition was not run. Controls use unchanged final recovery; there is no valid experimental inference comparison. Control scores use MONAI 1.6.1 label macro Dice and maximum directed HD95 in millimetres, under the existing local released-training-label protocol. These single-seed results do not support superiority claims.

## Engineering checks and provenance

The initial diagnostic implementation encountered a compiled-autograd donated-buffer error at first activation in nine LPBA attempts. These attempts produced no final endpoints and are separate from the completed scientific pilot above. The correction recomputes a separate registration-only diagnostic backward, restores the particle generator state, and uses no retained compiled graph. A full-width CUDA synthetic check verified the diagnostic followed by the combined training-loss backward, with compiled/eager gradient agreement and no optimizer step.

Local ignored evidence roots under `results/mc_registration/`:

- `20261110T052912_070532Z`: 19 valid controls; nine initial diagnostic failures.
- `20261110T055142_565574Z`: corrected .1 LPBA pilot, nine completed gate failures.
- `20261110T060840_457464Z`: authorized 1e-5 S02 calibration, gate decision, compact scalar analysis and preservation audit.

Each root retains explicit input configs, source snapshot/hashes and execution records. The calibration root's `analysis/pilot_summary.json` retains scalar evidence. No dense maps, warped volumes or persistent velocities were added. Medical data and earlier benchmark artifacts remain unchanged. The original .1 experimental implementation default is retained only as an opt-in research setting; no successful replacement weight was selected.

Final validation: `python -m unittest discover -s medical/tests -t . -v` (38 passed), `python -m unittest discover -s cameraman/tests -t . -v` (1 passed), wheel contents inspected, current documentation links resolved and `git diff --check` passed. SHA256/path audits preserved 2,343 dataset files (3,405,406,449 bytes) and 830 earlier benchmark files (83,266,043 bytes); recovery scratch is empty. Training executed only for the authorized conditions described above.

Follow-up: the [collapse diagnosis](particle_collapse_20261110.md) established gradient explosions and numerically sensitive particle trajectories with exact replay and bounded interventions. The implementation/defaults remain unchanged; its gradient-budget intervention is diagnostic only.
