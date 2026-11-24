# Default SIREN recipe ablations

These results use the 2026-11-15 RK2 recipe. The default changed to four-quarter
Euler on 2026-11-16; the scores below do not evaluate that change. The original
`native_siren_onecycle_5k.json` reference explicitly retains RK2.

Completed 2026-11-15 on four RTX 5090 GPUs. The 12 exact-config default controls are reused from `results/paired_adamw_quick4_20261115T124612Z/adamw_zero_1em3`; 36 new fits took 930.4 seconds wall time. This is a single-seed prototype comparison, not full-cohort validation.

The control is `medical/configs/default.json`: width 512, four sine layers, omega 30 throughout, output initialization ×30, ordinary AdamW decay 0.001, OneCycle peak 1e-4. Endpoint NCC plus moments, shared-grid PD-LK, box registration NCC, 32k particles per direction, RK2, and three-tap sigma 0.75-cell smoothing remain fixed. No paired decay or particle objective is enabled.

| Arm | OASIS Dice ↑ | Seconds/case |
| --- | --- | --- |
| Default | 0.82450 | 100.2 |
| Output init ×1 | 0.82696 | 101.3 |
| Exponential LR | 0.82587 | 100.4 |
| No weight decay | 0.82388 | 100.7 |

| Arm | OASIS % |
| --- | --- |
| default_512 | 0.00436 |
| output_init_x1 | 0.00438 |
| no_weight_decay | 0.00397 |
| exponential_lr | 0.00439 |

Keep the current default.

Output amplitude affects the LK derivative energies and their balance with fixed ridge regularization, as well as the endpoint Gaussian moment penalty; endpoint NCC alone does not make this scale irrelevant. At initialization, changing output scale from 1 to 30 multiplies the intensity derivatives by 30 and their squared data terms by 900. During optimization this also changes hidden-parameter gradient scaling. The current experiments do not isolate which mechanism explains the final accuracy difference.

The exponential arm starts at 1e-4 and approaches 2.5e-5. The OneCycle arm starts and ends at 2.5e-5, reaching 1e-4 after 10% warmup. Their summed learning rates are 0.270543 and 0.3124625 respectively, so this comparison changes both schedule shape and cumulative update size; it is not a warmup-only ablation.

Frequency initialization remains uniform omega 30. For normalized 3D coordinates, first-layer weights and biases are uniform ±1/3. Hidden weights and biases are uniform ±sqrt(6/512)/30. Output weights have that latter range before the output multiplier; output bias is zero. No frequency ladder, frequency balancing or extra hidden initialization scale is active.

All 36 new cases succeeded and all 576 new recovery intervals converged (768/768 including reused controls). Recorded source and input SHA-256 hashes are unchanged. Runtime averages include loading, fitting, recovery and evaluation; reused control timing is historical. No dense maps or warped images are persisted.

Reproduction presets:

- `medical/configs/ablations/native_siren_output_init_x1_5k.json`
- `medical/configs/ablations/native_siren_exponential_5k.json`
- `medical/configs/ablations/native_siren_no_decay_5k.json`
