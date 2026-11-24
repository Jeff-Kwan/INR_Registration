# Time-conditioned SIREN architecture ablations

This study fits a fresh scalar field to each of the 19 directed Learn2Reg OASIS validation pairs. Both variants use one SIREN with four width-512 sine layers and frequency 30. Time runs from the moving image at $t=0$ to the fixed image at $t=1$. The conditioning signal is $\tau=2t-1$.

| Run | Architecture | Time conditioning |
| ---: | --- | --- |
| 20 | XYZT input | Append $\tau$ as a fourth input coordinate, with a learned column of 512 input weights. |
| 21 | Hidden-layer FiLM | After every hidden sine, replace its activation $h_\ell$ with $(1+\tau a_\ell)\odot h_\ell+\tau b_\ell$. Each layer has its own learned channel-wise scale and shift slopes. |

The spatial weights and biases use the canonical SIREN initialization. The time input weights and FiLM slopes start at zero, so both fields start with the same spatial function for every time. XYZT has 791,041 trainable parameters, and FiLM has 794,625. A canonical pair has 1,581,058, so model capacity changes along with conditioning.

## Shared training and inference

Both variants copy the canonical settings, with the endpoint parameter penalty weight set to zero. Each fit uses seed 42, 5,000 captured CUDA FP32 updates, AdamW decay 0.001 and learning rate decaying from $10^{-4}$ to $2\times10^{-5}$. Endpoint NCC uses moment weight 0.005. The local transport loss (LT) has weight 0.1, and the trajectory-based registration weight decays from 0.03 to 0.01. The crop, reconstruction samples, Sobol times, velocity grid and particle samples follow the canonical protocol.

The implementation folds explicit time conditioning into the spatial affine operations at each query time. The fourth coordinate contributes $\tau w_t$ to the first affine bias. A FiLM transform before an affine with parameters $(W,c)$ gives

$$
W[(1+\tau a)\odot h+\tau b]+c
= W\,\operatorname{diag}(1+\tau a)\,h+c+\tau Wb.
$$

The same identity applies to the scalar output affine. The temporal derivatives of the folded weights and biases follow directly from these expressions. This lets both variants use the same analytical derivative and gradient machinery as the canonical pipeline, with one set of learned spatial weights per variant. CPU tests compare the folded calculations against explicit conditioning, including all four derivative outputs and gradients with respect to the trainable parameters.

Frozen inference uses the shared full-domain brightness-flow solve and eight RK2 substeps in each of eight intervals. The original evaluation grids and Learn2Reg label scoring are retained. Archived run 12 is the paired-SIREN comparison with the same zero endpoint parameter penalty weight. Archived run 5 is the canonical comparison with an endpoint parameter penalty weight of 0.01.

## Execution

`bash scripts/launch_architecture_ablations.sh --plan-only` prints the two variant configurations and the 38 case commands. Explicitly add `--run` for execution. Use a fresh output directory with `--output` for a repeat. Outputs include case settings, checkpoint weights, metrics, and aggregate results. Deformation maps and warped images are transient.

## Results

All 38 fits succeeded on 22 November 2026. Each architecture has 19 of 19 scored pairs for every metric. Values are case means ± population SD. Runtime is per-case wall time through recovery, scoring and checkpoint serialization, including graph setup. PSNR uses the original fitted intensities at peak 1. Patch NCC measures local correlation.

| Run | Endpoint parameter penalty weight | Endpoint PSNR (dB) | Patch NCC | Dice | HD95 (voxels) | NPJ (%) | SDlogJ | Runtime (s/case) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5. SIRENPath (canonical) | 0.01 | 6.72 ± 0.59 | 0.9429 ± 0.0043 | 0.8147 ± 0.0210 | 1.8155 ± 0.4313 | 0.001073 ± 0.000647 | 0.3425 ± 0.0258 | 80.50 ± 7.50 |
| 12. SIRENPath ($L_\theta = 0$) | 0 | 6.72 ± 0.59 | 0.9428 ± 0.0044 | 0.8146 ± 0.0213 | 1.8082 ± 0.4370 | 0.000940 ± 0.000680 | 0.3418 ± 0.0261 | 79.47 ± 6.51 |
| 20. XYZT input | 0 | 6.66 ± 0.57 | 0.9375 ± 0.0044 | 0.8145 ± 0.0209 | 1.8826 ± 0.4254 | 0.001274 ± 0.000795 | 0.3498 ± 0.0283 | 84.31 ± 18.15 |
| 21. Hidden-layer FiLM | 0 | 7.02 ± 0.55 | 0.9352 ± 0.0045 | 0.8081 ± 0.0224 | 1.9355 ± 0.4249 | 0.001782 ± 0.000976 | 0.3673 ± 0.0295 | 79.10 ± 8.68 |

- **XYZT input versus matched paired SIRENs.** Mean Dice changed by -0.0001, and mean HD95 changed by +0.0743 voxels.
- **Hidden-layer FiLM versus matched paired SIRENs.** Mean Dice changed by -0.0065, and mean HD95 changed by +0.1273 voxels.

These differences describe this cohort and seed. The single-field variants also have fewer learned parameters than the paired comparison.

[Architecture summary](../results/sirenpath_l2r_oasis_architectures/summary.json) · [Frozen plan](../results/sirenpath_l2r_oasis_architectures/plan.json) · [Scalar ablation results](ABLATIONS_REPORT.md)
