# Structure sampling

## Direct structure-centered crop mixture

Opt-in preset: `medical/configs/native_structure_crop.json`. It copies the current
native-voxel default and adds under `training`:

```json
"structure_crop_sampling": {"structure_fraction": 0.5, "sigma_mm": 2.0}
```

Each minibatch chooses its crop once:

- With probability 0.5, use the usual crop sampler, including DIR-Lab's equal
  endpoint lung-mask crop-center selection when masks are supplied.
- Otherwise choose moving or fixed with equal probability, draw a voxel from
  that observed image's Gaussian-gradient magnitude distribution, and center the
  crop directly there. Round the origin and shift it to fit the image, exactly
  as for ordinary mask-centered crops. There is no learned transport to t=0.5.

The structure distribution is pure gradient weighting inside this branch, rather
than another 50/50 point mixture. It covers the full observed endpoint image;
lung masks do not filter structure scores or draws. Constant images fall back to
uniform voxel centers. The Gaussian derivative construction and physical 2 mm
scale match the endpoint sampler described below; evaluation labels are unused.
A dedicated seeded generator keeps routing/endpoint/structure draws independent
of registration particle, probe and time generators. The changed crop naturally
changes the spatial distribution of all crop-dependent terms.

Endpoint reconstruction and both registration directions share the resulting
crop, retaining their usual separate sample sets. PD-LK uses all cells, every
particle contributes, and all loss derivatives, solves, smoothing and RK2 paths
remain live. Crop selection is discrete sampling; it is not a differentiable
transport objective. There is no inverse-probability correction. Crop size,
losses, sample counts and inference are unchanged. Global `endpoint_sampling`
cannot be combined with this option. Cropped training is required. Omitting the
option preserves the default sampler and its RNG draws.

Checkpoint training metadata records the mixture, endpoint probabilities,
whole-volume support and constant-image fallback flags. Scores/CDFs remain
transient. Synthetic tests in `medical.tests.test_structure_crop` check mixture
frequencies, endpoint balance, XYZ/ZYX conversion, reproducibility, boundary
shifts, ordinary mask sampling, fallback and training wiring before any loss.
No experiment is launched by selecting this preset.

## Existing global endpoint reconstruction option

Optional preset: `medical/configs/physical_working_grid_structure.json`. It copies
our 1 mm working-grid PD-LK/NCC preset and adds:

```json
"endpoint_sampling": {"structure_fraction": 0.5, "sigma_mm": 2.0}
```

This section belongs under `training`. This option selects global endpoint sampling; without it, reconstruction
samples the shared registration crop by default. The current implementation supports endpoint MSE, including the
inverse-metric alternative. Registration crop/probe/particle samplers and inference
are unchanged. The first OASIS quick4 experiment is recorded below.

For each endpoint independently, compute s = |gradient(G_sigma * I)| from its
normalized working image. Gaussian derivative filters use sigma 2 mm converted to
ZYX voxel scales, nearest-edge extension and four-sigma truncation. Divide each
derivative by its corresponding voxel spacing before taking the Euclidean norm.
Orthogonal rotated/reflected axes are supported; shear is rejected. Constant
images fall back to uniform sampling. The tensor-order (ZYX-column) affine supplied by the method adapters is required.

The reusable sampler supports full-volume or indexed support. Current global
endpoint training uses full-volume support and rejects crop-center masks.
Draw from p(x) = 0.5/N + 0.5*s(x)/sum(s). The global endpoint option does not use lung masks. Each sampler is built once; inverse-CDF lookup avoids a full-volume
sampling scan per step. FP64 CDFs consume eight bytes per supported voxel per
endpoint and are retained on the training device. Scores/CDFs are transient and
are not checkpointed. Checkpoint training metadata records the sampler parameters,
support and fallback status; the resolved config also retains the settings.

Targets remain the unblurred normalized images. No inverse-probability correction
is applied: this deliberately changes the expected MSE into a structure-weighted
reconstruction objective. Whole-volume PSNR remains uniformly evaluated. The
2 mm scale is our proposed experimental setting, not a validated optimum.
No labels or learned predictions construct the map. Reconstruction remains global
and independent of registration crops; its RNG cannot perturb registration streams.

## Attribution if retained

If this strategy is retained in the final method, **cite the sampling precedents**
and describe our adaptation explicitly; do not present gradient-guided sampling
as a new contribution:

- Öfverstedt, Lindblad and Sladoje. *INSPIRE: Intensity and spatial information-based
  deformable image registration*. PLOS ONE 18(3):e0282432, 2023.
  https://doi.org/10.1371/journal.pone.0282432 . Section 3.2.1, equations 11–12,
  gives Gaussian-gradient sampling mixed with uniform sampling. Our endpoint-MSE
  application, physical scale and unthresholded map are experimental choices.
- Wu et al. *Scalable High-Performance Image Registration Framework by Unsupervised
  Deep Feature Representations Learning*. IEEE TBME 63(7):1505–1516, 2016.
  https://doi.org/10.1109/TBME.2015.2496253 . Uses smoothed gradient-magnitude maps
  for unsupervised patch selection; this differs from our gradient-of-smoothed-image
  construction.
- Bhagalia, Fessler and Kim. *Accelerated Nonrigid Intensity-Based Image Registration
  Using Importance Sampling*. IEEE TMI 28(8):1208–1216, 2009.
  https://doi.org/10.1109/TMI.2009.2013136 . Earlier edge-dependent importance
  sampling for stochastic registration gradients. Our uncorrected weighted MSE
  should be distinguished from objective-preserving importance sampling.

Synthetic validation: ` /venv/main/bin/python -m unittest medical.tests.test_endpoint_sampling`.
This checks distributions, constant-image fallback, mask support, deterministic
sampling, physical derivatives, axis conventions and configuration validation.
It performs no optimizer steps or saved-model recovery.

## Deferred reconstruction support experiment

Consider making endpoint reconstruction cover the entire volume even when endpoint
lung masks exist, ignoring those masks for reconstruction only. This is deferred:
the current global endpoint option rejects masked training inputs. Registration masks
and intermediate-domain support are separate choices. Do not change this behavior
until explicitly requested.

## OASIS quick4 result

Completed all four frozen pairs, seed 42, 5000 steps, PD-LK 0.1 / NCC 0.01.
Source: `results/oasis_quick4_structure_20261112/summary.json`.

| Pair | DSC | HD95 | Original-grid PSNR moving / fixed (dB) |
| --- | ---: | ---: | ---: |
| 0395_to_0396 | 0.7720 | 2.023 | 35.53 / 35.46 |
| 0398_to_0399 | 0.7096 | 3.408 | 34.81 / 35.30 |
| 0402_to_0403 | 0.7742 | 2.070 | 35.19 / 35.59 |
| 0403_to_0404 | 0.8009 | 2.037 | 35.20 / 35.25 |

Mean DSC 0.764196 versus uniform-global 0.751125; mean HD95 2.384401 versus
2.476788. All four pairs improved DSC. Mean moving/fixed PSNR 35.180/35.398 dB
versus 33.321/33.681 dB. Mean case runtime 121.41 s versus 118.66 s.
The historical crop-coupled reconstruction run remains higher at DSC 0.770455,
HD95 2.329283. These are single-seed development results.

## Direct structure-crop OASIS quick4 result (2026-11-13)

Source: `results/oasis_quick4_native_structure_crop_20261113`. Native-voxel preset, seed 42,
5000 steps, frozen official pairs 1/4/8/9; 4/4 completed.

| Pair | DSC | HD95 (voxels) |
| --- | ---: | ---: |
| 0395_to_0396 | 0.817711 | 1.685320 |
| 0398_to_0399 | 0.791658 | 2.792525 |
| 0402_to_0403 | 0.827980 | 1.456932 |
| 0403_to_0404 | 0.835410 | 1.733195 |

Mean DSC 0.818189, HD95 1.916993, shifted SDlogJ 0.080499.
All 64/64 PCG intervals converged. Mean end-to-end time 153.72 s/case.
No matched current native-default control was launched, so this run alone does
not isolate improvement over ordinary crop sampling.
