# Baseline comparison aggregate report

Finished evaluations: **2781**; successful: **2781**; process failures: **0**. Invalid original NIR/DNVF results were removed; only their corrected reruns appear below. Single seed42; mean ± population SD. Unscored values are —.

## SIRENPath and other baselines

| Dataset | Method | Fits succeeded | Dice ↑ | HD95 ↓ | NPJ (%) ↓ | SDlogJ ↓ | HD95 pairs | Runtime (s/case) | Backend |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Neurite-OASIS | SIRENPath | 49/49 | 0.812 ± 0.023 | 1.797 ± 0.384 | 0.000959 ± 0.001290 | 0.341 ± 0.048 | 44/49 | 87.51 ± 18.67 | GPU (RTX5090) |
| Neurite-OASIS | ConvexAdam | 49/49 | 0.761 ± 0.032 | 2.294 ± 0.497 | 0.015924 ± 0.011275 | 0.362 ± 0.070 | 45/49 | 9.38 ± 1.52 | GPU (RTX5090) |
| Neurite-OASIS | FireANTs-SyN | 49/49 | 0.781 ± 0.036 | 2.188 ± 0.528 | 0.000000 ± 0.000000 | 0.230 ± 0.019 | 45/49 | 12.05 ± 0.36 | GPU (RTX5090) |
| Neurite-OASIS | Demons | 49/49 | 0.761 ± 0.047 | 2.631 ± 0.735 | 1.403663 ± 0.213901 | 2.474 ± 0.182 | 45/49 | 15.48 ± 1.15 | CPU, 4 threads |
| Neurite-OASIS | NiftyReg | 49/49 | 0.780 ± 0.028 | 2.208 ± 0.533 | 0.763889 ± 0.210997 | 1.847 ± 0.241 | 45/49 | 36.70 ± 0.62 | CPU/OpenMP, 4 threads |
| Neurite-OASIS | ccIDIR | 49/49 | 0.773 ± 0.026 | 2.360 ± 0.631 | 0.657165 ± 0.215252 | 1.691 ± 0.255 | 45/49 | 93.41 ± 2.69 | GPU (RTX5090) |
| Neurite-OASIS | NODEO | 49/49 | 0.789 ± 0.027 | 2.105 ± 0.477 | 0.016039 ± 0.016933 | 0.449 ± 0.077 | 44/49 | 54.11 ± 0.75 | GPU (RTX5090) |
| ACDC | SIRENPath | 100/100 | 0.811 ± 0.076 | 6.806 ± 3.041 | 0.001105 ± 0.002266 | 0.340 ± 0.109 | 100/100 | 73.23 ± 7.37 | GPU (RTX5090) |
| ACDC | ConvexAdam | 100/100 | 0.732 ± 0.097 | 8.667 ± 3.192 | 0.002812 ± 0.011324 | 0.202 ± 0.084 | 100/100 | 4.84 ± 0.26 | GPU (RTX5090) |
| ACDC | FireANTs-SyN | 100/100 | 0.673 ± 0.137 | 10.008 ± 3.296 | 0.000000 ± 0.000000 | 0.221 ± 0.055 | 100/100 | 5.58 ± 0.29 | GPU (RTX5090) |
| ACDC | Demons | 100/100 | 0.725 ± 0.098 | 9.090 ± 3.111 | 2.403632 ± 0.945270 | 3.170 ± 0.563 | 100/100 | 4.08 ± 0.24 | CPU, 4 threads |
| ACDC | NiftyReg | 100/100 | 0.772 ± 0.100 | 7.829 ± 3.426 | 0.060753 ± 0.120980 | 0.426 ± 0.385 | 100/100 | 5.04 ± 0.63 | CPU/OpenMP, 4 threads |
| ACDC | ccIDIR | 100/100 | 0.757 ± 0.074 | 9.695 ± 2.805 | 2.556702 ± 1.859972 | 3.095 ± 1.267 | 100/100 | 81.37 ± 2.21 | GPU (RTX5090) |
| ACDC | NODEO | 100/100 | 0.795 ± 0.064 | 7.917 ± 3.271 | 0.013810 ± 0.057407 | 0.332 ± 0.192 | 100/100 | 9.14 ± 0.35 | GPU (RTX5090) |
| LPBA40 | SIRENPath | 45/45 | 0.719 ± 0.017 | 5.879 ± 0.421 | 0.000043 ± 0.000099 | 0.280 ± 0.037 | 45/45 | 77.82 ± 5.55 | GPU (RTX5090) |
| LPBA40 | ConvexAdam | 45/45 | 0.708 ± 0.017 | 5.980 ± 0.424 | 0.021687 ± 0.032468 | 0.368 ± 0.150 | 45/45 | 6.65 ± 0.25 | GPU (RTX5090) |
| LPBA40 | FireANTs-SyN | 45/45 | 0.691 ± 0.027 | 6.583 ± 0.649 | 0.000000 ± 0.000000 | 0.265 ± 0.037 | 45/45 | 9.42 ± 0.31 | GPU (RTX5090) |
| LPBA40 | Demons | 45/45 | 0.635 ± 0.046 | 7.314 ± 0.842 | 1.252480 ± 0.280245 | 2.340 ± 0.250 | 45/45 | 12.05 ± 0.30 | CPU, 4 threads |
| LPBA40 | NiftyReg | 45/45 | 0.694 ± 0.030 | 6.529 ± 0.613 | 0.565932 ± 0.326843 | 1.571 ± 0.408 | 45/45 | 26.50 ± 0.66 | CPU/OpenMP, 4 threads |
| LPBA40 | ccIDIR | 45/45 | 0.702 ± 0.022 | 6.266 ± 0.526 | 0.012858 ± 0.014798 | 0.253 ± 0.114 | 45/45 | 87.20 ± 1.47 | GPU (RTX5090) |
| LPBA40 | NODEO | 45/45 | 0.692 ± 0.028 | 6.648 ± 0.658 | 0.001485 ± 0.003070 | 0.332 ± 0.031 | 45/45 | 39.27 ± 1.08 | GPU (RTX5090) |
| IXI | SIRENPath | 115/115 | 0.739 ± 0.032 | 3.365 ± 0.774 | 0.001840 ± 0.002052 | 0.557 ± 0.109 | 115/115 | 80.29 ± 2.01 | GPU (RTX5090) |
| IXI | ConvexAdam | 115/115 | 0.741 ± 0.026 | 3.486 ± 0.794 | 0.178308 ± 0.127113 | 0.902 ± 0.282 | 115/115 | 7.79 ± 0.29 | GPU (RTX5090) |
| IXI | FireANTs-SyN | 115/115 | 0.744 ± 0.023 | 3.047 ± 0.455 | 0.000000 ± 0.000000 | 0.280 ± 0.021 | 115/115 | 11.80 ± 0.23 | GPU (RTX5090) |
| IXI | Demons | 115/115 | 0.520 ± 0.045 | 5.652 ± 0.748 | 1.781895 ± 0.252861 | 2.779 ± 0.180 | 115/115 | 15.37 ± 0.56 | CPU, 4 threads |
| IXI | NiftyReg | 115/115 | 0.620 ± 0.053 | 7.748 ± 1.307 | 1.382104 ± 0.474935 | 2.448 ± 0.395 | 115/115 | 35.98 ± 0.70 | CPU/OpenMP, 4 threads |
| IXI | ccIDIR | 115/115 | 0.645 ± 0.060 | 7.800 ± 2.181 | 0.889751 ± 0.258626 | 1.963 ± 0.277 | 115/115 | 92.75 ± 2.02 | GPU (RTX5090) |
| IXI | NODEO | 115/115 | 0.688 ± 0.039 | 5.253 ± 1.128 | 0.041488 ± 0.035358 | 0.577 ± 0.126 | 115/115 | 54.22 ± 0.92 | GPU (RTX5090) |

## Corrected NIR and DNVF reruns

NIR uses the corrected MRI LNCC/spectral recipe in IEEE FP32; DNVF uses its original recipe with TF32 disabled. All preparations and scoring remain frozen. Each metric mean uses successful cases with a defined score. ACDC/NIR uses the largest available odd NCC extent up to9 per sampled axis:9×9×7 for four thin cases,9×9×9 otherwise.

| Dataset | Method | Fits succeeded | Dice ↑ | HD95 ↓ | NPJ (%) ↓ | SDlogJ ↓ | HD95 pairs | Runtime (s/case) | Backend |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Neurite-OASIS | NIR | 49/49 | 0.757 ± 0.030 | 2.428 ± 0.506 | 0.278743 ± 0.023533 | 1.268 ± 0.044 | 44/49 | 36.26 ± 1.97 | GPU (RTX5090) |
| Neurite-OASIS | DNVF | 49/49 | 0.803 ± 0.024 | 1.944 ± 0.408 | 0.000000 ± 0.000000 | 0.324 ± 0.024 | 45/49 | 101.44 ± 0.81 | GPU (RTX5090) |
| ACDC | NIR | 100/100 | 0.658 ± 0.108 | 10.245 ± 3.100 | 0.326434 ± 0.223622 | 1.244 ± 0.367 | 100/100 | 11.55 ± 0.43 | GPU (RTX5090) |
| ACDC | DNVF | 100/100 | 0.767 ± 0.090 | 8.604 ± 3.559 | 0.000054 ± 0.000460 | 0.277 ± 0.055 | 100/100 | 11.60 ± 0.63 | GPU (RTX5090) |
| LPBA40 | NIR | 45/45 | 0.666 ± 0.026 | 6.720 ± 0.556 | 0.197507 ± 0.021231 | 1.119 ± 0.039 | 45/45 | 27.62 ± 1.03 | GPU (RTX5090) |
| LPBA40 | DNVF | 45/45 | 0.712 ± 0.022 | 6.231 ± 0.529 | 0.000128 ± 0.000627 | 0.308 ± 0.043 | 45/45 | 71.67 ± 0.38 | GPU (RTX5090) |
| IXI | NIR | 115/115 | 0.727 ± 0.024 | 3.357 ± 0.496 | 0.318063 ± 0.025520 | 1.338 ± 0.044 | 115/115 | 34.05 ± 1.10 | GPU (RTX5090) |
| IXI | DNVF | 115/115 | 0.747 ± 0.026 | 3.214 ± 0.564 | 0.000003 ± 0.000020 | 0.355 ± 0.029 | 115/115 | 100.76 ± 0.57 | GPU (RTX5090) |

Runtime is averaged over successful cases and includes case loading, fitting, recovery/transformation evaluation, scoring and output serialization. Neural externals use fresh subprocesses; SIRENPath uses persistent workers. CPU classical baselines ran in parallel with GPU baselines. FireANTs used its PyTorch fallback.

All preparations/cohorts and standardized_v1 scoring are frozen. HD95 is prepared-grid mm (IXI uses its unit-voxel benchmark frame). Its valid-pair coverage can differ between methods; direct ranking needs a common valid subset. Dice uses nearest-neighbour zero-outside label warps. NPJ is100×mean(detJ≤0); SDlogJ isstd(log(max(detJ,1e-9))), with central differences and two-voxel border exclusion.

## Methods

- **ConvexAdam** runs on GPU and uses native MIND-SSC coupled convex optimization, inverse consistency and 80-step Adam refinement with public API defaults. [Siebert et al., TMI 2024](https://github.com/multimodallearning/convexAdam/blob/b229e52e44b114e2040a503334c92269750c16b2/README.md)
- **FireANTs-SyN** runs on GPU and uses native symmetric compositive SyN with Adam and the author OASIS preset, running its PyTorch fallback backend. [Jena et al., FireANTs](https://arxiv.org/abs/2404.01249)
- **Demons** runs on CPU and uses classic SimpleITK Demons, 50 iterations, displacement smoothing σ = 1 and MRI histogram matching. [Thirion, 1998](https://pubmed.ncbi.nlm.nih.gov/9873902/), [official recipe](https://simpleitk.readthedocs.io/en/master/link_DemonsRegistration1_docs.html)
- **NiftyReg** uses CPU/OpenMP reg_f3d v2.1.1 B-spline registration with default NMI, conjugate-gradient optimization and regularization. [Modat et al., 2010](https://discovery.ucl.ac.uk/id/eprint/17431/1/17431.pdf)

- **ccIDIR** uses native paired SIRENs with 2,500 updates, cycle consistency and order-2 cycle-consensus inference, adapted to whole-image MRI fitting. [van Harten et al., TMI](https://doi.org/10.1109/TMI.2023.3321425)
- **NIR** rerun uses native local NCC plus Jacobian penalty, paper spectral settings, 900 requested updates, preserved prepared intensities and IEEE FP32. [Sun et al., MedIA2024](https://doi.org/10.1016/j.media.2024.103249)
- **NODEO** uses the native 300-epoch neural-ODE/custom-adjoint registration loop; only encoder/decoder sizes are localized for prepared grids. [Wu et al., CVPR 2022 / author source](https://github.com/yifannnwu/NODEO-DIR/tree/fb010664d02ca6c6bca01c0f50f2de17f61d23ed)
- **DNVF** uses native single-scale stationary neural velocity optimization, 200 updates and seven-step scaling-and-squaring integration. [Author implementation](https://github.com/uci-cbcl/DNVF/tree/27cf2f6e6bffc7b00f8c2721cc48553e415bf8f9)
- **SIRENPath** uses paired SIRENs, 5,000 AdamW updates with cosine LR 1e-4→2e-5 and frozen brightness-flow PCG/RK2 recovery. [Canonical method](METHOD.md)

## Datasets and protocols

- **Neurite-OASIS** follows Jena et al., *Magic or Mirage?* (NeurIPS2024): final50 of414 lexicographically sorted released subjects,49 consecutive earlier-fixed/next-moving pairs, unchanged arrays and labels1–35; our standardized scoring is separate. [Jena et al., cohort protocol](https://papers.nips.cc/paper_files/paper/2024/file/c3fe2a07ec47b89c50e89706d2e23358-Paper-Conference.pdf), [GaussianDIR attribution](https://pmc.ncbi.nlm.nih.gov/articles/PMC12361785/), [Neurite release](https://huggingface.co/datasets/DalcaLab/neurite-oasis), [Hoopes et al., HyperMorph](https://arxiv.org/abs/2101.01035)
- **ACDC** uses testing patients 101–150, 100 directed ED↔ES fits, ES-spacing resampling to 1.5×1.5×3.15 mm, normalization before 128×128×min(Z,32) cropping and labels 1–3. [Bernard et al., 2018](https://doi.org/10.1109/TMI.2018.2837502), [KAN-IDIR protocol source](https://github.com/anac0der/KAN-IDIR/tree/3ac12e200159811eafb6192f3f8db3afef2a9d1a)
- **LPBA40** uses the frozen SITReg 25/5/10 subject split, 45 test combinations, author AIR/white-matter-normalized preparation and all 56 anatomical labels. [Shattuck et al., 2008](https://doi.org/10.1016/j.neuroimage.2007.09.031), [SITReg protocol source](https://github.com/honkamj/SITReg/tree/a665a46a3553b26ade6ff5f57815e5e8ec43a948)
- **IXI** uses the TransMorph-prepared atlas-to-subject test split, 115 fits, unchanged image arrays and equal weighting of its 30 scored anatomical labels. [IXI dataset](https://brain-development.org/ixi-dataset/), [TransMorph protocol source](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/6357a1d7fc44c36db9b1d1ccaa372409253142cf/IXI/TransMorph_on_IXI.md)

All results use standardized_v1 scoring. [Classical aggregates](../baselines/results/official_four_20261116/summary.json) · [Current neural aggregates](../baselines/results/official_neural_20261116/summary.json) · [SIRENPath aggregates](../results/official_ours_cosine_20261116/summary.json) · [Correction/precision audit](NEURAL_BASELINE_DIAGNOSTICS.md) · [Rerun plan](NEURAL_RERUNS.md)

Remaining source/transfer qualifications: [BASELINE_PROTOCOL_REVIEW.md](BASELINE_PROTOCOL_REVIEW.md). No method settings or results were changed by that review.
