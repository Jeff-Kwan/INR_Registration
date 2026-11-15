# Differential patch-NCC: gate-bypassed LPBA S02 recovery (2026-11-11)

The three PD-LK-free patch-NCC endpoint checkpoints were run through the
maintained brightness-flow PCG and midpoint RK2 recovery despite failing the
25 dB endpoint reconstruction gate. Recovery used each checkpoint's unchanged
standard settings: 16 intervals, smoothness .01, ridge 1e-6, up to 500 PCG
iterations, relative tolerance 2e-5, four midpoint RK2 substeps, and the default
previous-interval velocity warm start. Checkpoints and data were read-only. This
was an explicit diagnostic bypass; each checkpoint's original failed gate is
retained below and in the machine-readable results.

| Training objective | Moving / fixed endpoint PSNR (dB) | Warped-image PSNR (dB) | Dice | HD95 (mm) | Folding fraction | All PCG intervals converged | Max relative residual |
|---|---:|---:|---:|---:|---:|---|---:|
| Patch-NCC, constant .1 | 22.364 / 22.290 | 20.618 | .58767 | 7.425 | .000125 | Yes | 1.976e-5 |
| Patch-NCC, constant .01 | 23.528 / 22.991 | 21.551 | **.59376** | **7.357** | .000161 | Yes | 1.977e-5 |
| Patch-NCC, .01 ramp after step 2,000 | 20.188 / 19.580 | 20.386 | .56392 | 7.998 | .000441 | Yes | 1.616e-5 |

All label scores use the repository's MONAI metric definition, with 56 nonzero
labels. The constant .01 model was best among these three, but still scored below
the existing LPBA S02 MSE+strict-PD-LK reference (Dice .70245, HD95 6.605 mm,
folding 0). That reference came from the maintained benchmark and was not rerun
in this diagnostic; it used the then-current zero initial-velocity solve and one
interval narrowly missed the configured residual tolerance. The three diagnostic
runs used the current previous-velocity warm start. The ramp run was worse than
either constant-weight run.
Thus bypassing the endpoint gate permits numerical recovery and yields measurable
label scores, but these results do not support the new training objective over
the maintained baseline. They also do not make failed endpoint reconstructions
acceptable for normal pipeline use.

Recovery used the same trajectory-based SIREN analytical derivatives, global
PCG solve and midpoint RK2 map integration as ordinary inference. No alternate
solver, endpoint recalibration or image-space refinement was used. Compact
metrics and per-interval flow diagnostics are in the ignored
`results/differential_patch_ncc/recovery_gate_bypass_20261111/` directory; no
deformation maps or warped volumes were saved.
