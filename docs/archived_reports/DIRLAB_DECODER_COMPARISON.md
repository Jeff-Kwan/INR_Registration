# Frozen DIR-Lab training decoder versus PCG recovery

Checkpoints: `results/dirlab_quick4_native_cropcenter_20261113/caseNN/checkpoint.pt`, cases 01,03,04,05. No fitting or optimizer updates. Production code and coefficients unchanged.

For each checkpoint, draw 256 crops using the current equal-endpoint lung-mask center sampler, seed 42, configured native half-volume crop. Use the same random two-pair derivative probes, pooled brightness+gradient local solve, Gaussian smoothing, random early/late field times and two spatial midpoint RK2 half-steps as training. Evaluate landmarks whose starts lie in the central 80% particle-sampling region of each crop. Landmark positions and target correspondences never select crops, fields, times or weights. Do not discard trajectories leaving a crop.

Because a local decoder depends on its crop and random fields, there is no single whole-volume training map. Report both the mean error of individual crop predictions (averaged per landmark before averaging landmarks) and the error after averaging endpoint predictions per landmark. The latter is a diagnostic ensemble, not the training objective or a proposed inference method. PCG is compared on exactly the same covered landmarks. Independently regenerate the standard 16-interval global brightness PCG+RK2 maps from the same frozen checkpoints, preserving warm starts and original coefficients.

| Case | Covered / 300 | Local single-crop mean | Local averaged prediction | PCG same landmarks |
| --- | ---: | ---: | ---: | ---: |
| 01 | 299 | 1.646 | 1.632 | 1.269 |
| 03 | 299 | 3.435 | 3.418 | 2.675 |
| 04 | 300 | 4.275 | 4.247 | 3.357 |
| 05 | 298 | 3.921 | 3.903 | 3.251 |
| Case mean | 299 | 3.319 | 3.300 | 2.638 |

All table errors are continuous forward T00-to-T50 TRE in mm. Use continuous TRE to avoid mixing per-crop rounding and averaging. The four cases cover 1196/1200 forward landmarks; uncovered point indices and observation counts are in per-case metrics, not imputed. Coverage is 299,299,300,298. The original full-cohort quick4 PCG continuous TRE was 2.635 mm; this matched-subset PCG mean differs slightly.

Reverse-direction matched-subset continuous TRE is 3.439 mm for single-crop local predictions and 2.666 mm for PCG. Local forward prediction variability across crops/times/probes has case-mean RMS 0.395 mm. Local averaged endpoints differ from PCG endpoints by a case-mean 1.302 mm. All 64 PCG intervals converge; temporary scratch is clean.

## Interpretation

The local training decoder does not expose a hidden high-quality registration that global PCG discards: its landmark TRE is worse in all four cases, including the crop-averaged diagnostic ensemble. This makes the learned trajectory/training surrogate the leading area to inspect. It does not isolate a specific loss or prove that training and recovery decoders are equivalent; derivative evidence, smoothing, spatial support and temporal discretization differ together. It also does not mean that a low crop NCC guarantees low landmark TRE. The oracle recovery study shows what PCG can achieve with suitable derivative evidence, while this comparison shows the present checkpoint is already inaccurate under the decoder used to train it.

Results and reproducible disposable driver: `results/dirlab_quick4_decoder_comparison_20261113/`. Only JSON statistics and source/log text are retained. No persistent maps, warped images or reconstruction volumes. No model changes or commits.

## Particle crop-exit replay

The training logs saved losses only, not particle support diagnostics. Historical
counts over 5000 optimizer steps cannot be reconstructed from final checkpoints.
A subsequent replay used 256 configured lung-centered crops per final checkpoint,
32768 independently sampled particles per direction/crop, the original probe/time
RNG streams and the same RK2 decoder. No optimization or global recovery.

Each case/direction evaluates 8,388,608 particle trajectories. Any crop exit at an
RK2 midpoint or endpoint counts, even if a trajectory later returns.

| Case | T50→T00 crop exits | Fraction | T00→T50 crop exits |
| --- | ---: | ---: | ---: |
| 01 | 0 | 0% | 0 |
| 03 | 174 | 0.002074% | 0 |
| 04 | 554 | 0.006604% | 0 |
| 05 | 245 | 0.002921% | 0 |

Total 973/67,108,864 trajectories (0.001450%). Worst sampled direction/crop was
58/32768 exits (0.177%). No final endpoints left the full image domain. In the
current policy none of these particles is removed from the registration loss.
Exits are too rare at the final checkpoints to explain the present TRE gap; this
does not rule out more exits earlier in training. Raw results and disposable
driver: `results/dirlab_quick4_particle_exits_20261113/`.
