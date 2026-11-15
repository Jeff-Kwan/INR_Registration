# Six-trial full-dataset results

Source: `results/full_method_suite/summary.json`. Single seed 42. 444 jobs attempted, 439 successful. OASIS 19/19 and abdomen 45/45 for every trial. DIR-Lab failures: case10 in trials 01–04; case09 in trial06. All failed with `Sampled crop contains no lung-mask cells`: endpoint particle support exists, but sampled PD-LK grid centers have no union-mask support. No retries or fixes applied during collection.

DIR-Lab table compares the common cases 01–08 only, forward rounded TRE in mm. OASIS/abdomen DSC use full official cohorts.

| Trial | OASIS DSC | Abdomen DSC | DIR-Lab matched-8 TRE (mm) |
| --- | ---: | ---: | ---: |
| Reference | 0.804023 | 0.333952 | 4.970988 |
| Target moments | 0.752492 | 0.290748 | 5.513897 |
| Registration NCC 0.03 | 0.800217 | 0.351212 | 4.163969 |
| PD-LK 0.03 | 0.801757 | 0.331531 | 5.385646 |
| 120 mm / 12³ | 0.808383 | 0.368717 | 4.729719 |
| Native grid | 0.807198 | 0.392181 | 4.318668 |

| Trial | Abdomen HD95 (voxels) | Abdomen SDlogJ | Abdomen folding (%) |
| --- | ---: | ---: | ---: |
| Reference | 18.440710 | 0.500690 | 0.406343 |
| Target moments | 19.071539 | 0.101620 | 0.000000 |
| Registration NCC 0.03 | 18.235318 | 0.498881 | 0.404740 |
| PD-LK 0.03 | 18.798665 | 0.649561 | 0.727547 |
| 120 mm / 12³ | 18.221531 | 0.502425 | 0.464542 |
| Native grid | 17.649197 | 0.332132 | 0.229539 |

120 mm context scores best OASIS DSC. Native-grid scores best abdomen DSC/HD95 and lower abdomen folding than physical reference. Registration NCC 0.03 gives the lowest matched-eight DIR-Lab TRE. Target moments improve raw reconstruction PSNR and regularity but reduce registration scores. Raw NCC PSNR is not comparable to MSE-like calibrated outputs without accounting for intensity gauge.

PCG: target-moment DIR-Lab 92/144 converged; every other successful trial/dataset has all intervals converged. Do not interpret failed/partial-cohort DIR-Lab aggregates as ten-case scores.

Runtime estimates were optimistic for native-grid execution: mean case seconds OASIS 424.8, DIR-Lab 1862.1 (9 successful), abdomen 533.5. The subsequent [native speed audit](NATIVE_EXECUTION_SPEED.md) reproduces a PCG compiler specialization/fusion slowdown after varied shape history; training itself takes the same time. Fresh-process native checkpoint replays recover in 12.85 s on OASIS and 14.50 s on abdomen. Production code is unchanged by that audit.
