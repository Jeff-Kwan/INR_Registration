# Gaussian-smoothed local velocities versus PCG: LPBA S02

Gaussian smoothing explains much of the gap between local PD-LK and global PCG velocity on these checkpoints. At sigma4mm, the smoothed local fields closely agree in large-scale direction with PCG, but remain different in magnitude and local detail. This is an instantaneous-field comparison, not a registration-quality or speed benchmark.

## Measured comparison

The same MSE+PD-LK and NCC+fixed-moments(.01)+PD-LK checkpoints were frozen. Full native-grid PCG and local velocities were recomputed at three prespecified recovery midpoints, **.21875/.46875/.71875**. Both local candidates use PCG's identical g/q inputs; strict PD-LK adds analytical H/m with c4 eta1e-6 rho0. Filtering uses SciPy Gaussian convolution per velocity component, reflect boundaries and truncate4, with fixed sigma0/1/2/4mm. These are standard deviations, not kernel radii or diameters. No width was selected using labels or recovered registration performance.

Pool the same 17,322 foreground-union sites across the three times. The relative vector error is RMS(local−PCG)/RMS(PCG); field cosine is the dot product of concatenated point/time vectors normalized by their norms.

| Gaussian sigma, mm | MSE field cosine ↑ | NCC field cosine ↑ | MSE relative vector error ↓ | NCC relative vector error ↓ |
|---:|---:|---:|---:|---:|
| 0, raw local PD-LK | .392 | .365 | 2.530 | 2.830 |
| 1 | .704 | .721 | 1.066 | 1.038 |
| 2 | .851 | .870 | .615 | .565 |
| 4 | .944 | .932 | .330 | .364 |

At sigma4mm, RMS velocity is **4.705 versus PCG5.143** for MSE and **5.513 versus6.190** for NCC, measured in mm per unit interpolation time. Thus filtering is about **8.5%/10.9% slower in velocity magnitude**, not an exact field match. These are velocities, not final displacement errors. Median pointwise directional cosines (median of the three time medians) are **.968/.966**. Opposite-direction fractions fall from **16.80%/14.47%** unfiltered to **1.25%/1.04%**, counting only points where both speeds exceed .1.

Neighbor-difference RMS relative to PCG changes as follows:

| Sigma mm | MSE local/PCG neighbor-change ratio | NCC ratio |
|---:|---:|---:|
| 0 | 33.19 | 26.05 |
| 1 | 5.62 | 4.11 |
| 2 | 2.23 | 1.54 |
| 4 | .82 | .55 |

The broad filter removes much of the local fluctuation and, for NCC, produces a noticeably smoother field than PCG. The underlying large-scale local PD-LK field is therefore much closer to PCG than its raw pooled cosine suggested. The previous raw-field comparison remains valid; this follow-up qualifies its interpretation.

## Brightness-only local velocities

Smoothing the local minimum-norm brightness velocity, `-g*q/(|g|²+1e-6)`, also improves direction agreement, but strongly reduces its already limited amplitude:

| Sigma mm | MSE cosine | NCC cosine | MSE/PCG RMS speed ratio | NCC/PCG RMS speed ratio |
|---:|---:|---:|---:|---:|
| 0 | .532 | .542 | .999 | 1.006 |
| 1 | .701 | .730 | .726 | .704 |
| 2 | .819 | .826 | .570 | .553 |
| 4 | .912 | .884 | .427 | .412 |

At sigma4, relative vector errors remain .635/.664. Smoothed local PD-LK is a closer amplitude-and-direction approximation here than smoothed brightness-only flow. Its extra differential constraints provide local motion components that a minimum-norm brightness solve does not identify, although these local estimates require substantial spatial smoothing in this probe.

## Interpretation and limits

A Gaussian computes a fixed spatial average, without considering whether its output still explains the local g/q brightness equation. PCG jointly solves that equation and spatial regularization, with an image-dependent `g*g^T` term. Even in a constant-coefficient case, Gaussian convolution and the regularized normal-equation inverse are different operators. The measured approximation at sigma4 does not make them mathematically equivalent.

A sensible revised hypothesis is that Gaussian-smoothed local PD-LK can approximate much of the PCG field on this pair. It warrants a later unchanged-RK2 registration comparison if explicitly requested. This probe does not establish comparable Dice/HD95/folding, trajectory consistency or inference speed. Broad smoothing may discard local motion detail; no Gaussian width is promoted from this three-time, one-pair comparison. Dense H/m construction also has a cost, so Gaussian postprocessing is not automatically faster overall.

The original 65,536 interior sample sites and image-derived masks from seed20261111 were reused; 17,322 foreground-union and4,331 strong-original-edge sites. Nearest-neighbor roughness uses the first4,096 centres and their positive-axis neighbors, restricting centres to foreground. All velocities were finite. MSE PCG converged at **2/3** selected times (maximum true residual2.01950e-5 versus tolerance2e-5); NCC converged at **3/3** (maximum1.83310e-5). These small convergence qualifications are retained rather than silently dropping a time point. The complete probe took **32.14s**, including both candidate fields and all filtering widths; this is not the isolated cost of any one alternative.

Evidence: `results/gaussian_local_pcg_probe/20261110T161947_512004Z`, containing the exact runner, protocol, per-time scalar evidence and summary. The [compact record](gaussian_local_pcg_probe_20261111.json) preserves all widths and both local-field definitions. Both checkpoints and source images matched pre-probe SHA256 hashes. No training, optimizer steps, RK2 integration, labels, generated volumes or saved velocity/map fields were used. Maintained code and defaults are unchanged; no commit or push occurred.
