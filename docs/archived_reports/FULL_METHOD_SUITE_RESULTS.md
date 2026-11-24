# Six-trial full-dataset results

Source: `results/full_method_suite/summary.json`. Single seed 42. All 114 OASIS fits completed. No retries or fixes applied during collection.

| Trial | OASIS DSC |
| --- | ---: |
| Reference | 0.804023 |
| Target moments | 0.752492 |
| Registration NCC 0.03 | 0.800217 |
| PD-LK 0.03 | 0.801757 |
| 120 mm / 12³ | 0.808383 |
| Native grid | 0.807198 |

120 mm context scores best OASIS DSC. Target moments improve raw reconstruction PSNR and regularity but reduce registration scores. Raw NCC PSNR is not comparable to MSE-like calibrated outputs without accounting for intensity gauge.

The subsequent [native speed audit](NATIVE_EXECUTION_SPEED.md) reproduces a PCG compiler specialization/fusion slowdown after varied shape history; training itself takes the same time. Production code is unchanged by that audit.
