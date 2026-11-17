# Decoupled attraction between endpoint SIRENs

The paired networks interpolate their parameters to define the intensity path.
Large endpoint parameter differences can make that path complicated even when
both endpoints fit their images. A small symmetric pull can limit separation
while preserving the shared parameter mean. This experiment replaces AdamW's
zero-centered decay with that pull and asks whether registration improves while
endpoint reconstruction and folding remain acceptable.

Select `training.optimizer.name: "paired_adamw"`. Its `weight_decay` is the
attraction coefficient lambda. For each same-named source/target parameter pair,
with pre-update values a,b and the current parameter-group learning rate eta:

```text
m = (a + b) / 2
new_a = a - eta * lambda * (a - m)
new_b = b - eta * lambda * (b - m)
then apply Adam's adaptive gradient update to both parameters
```

Both contractions use the original a,b. The implementation forms one difference
and applies opposite simultaneous changes, preserving m in exact arithmetic
and contracting a-b by `1 - eta*lambda`. All parameters that ordinary AdamW
would decay are paired, including biases. This is the instantaneous pair mean,
not a historical moving average. The usual zero-centered decay is disabled.

The order follows the installed PyTorch AdamW decoupled update: parameter decay
precedes the adaptive update. Objective gradients, clipping and Adam's first/
second moments do not include an attraction/L2 gradient. The pre-optimizer
finite guard remains in the standard trainer. Parameter groups retain their
own current learning rates; endpoint partners must share a group. Both absent
gradients skip the pair; one-sided absent gradients raise before mutation.
State dictionaries retain the mean-decay coefficient. The separate complete
captured trainer and AMP gradient scaling are unsupported for this optimizer.

The current 512-neuron recipe uses output initialization x30, 5000 steps,
OneCycle peak1e-4 and AdamW decay0.001. The requested sweep replaces the optimizer
with paired_adamw at1e-4,1e-3,1e-2, retaining the original fixed sigma0.75-cell
three-tap shared-grid smoothing,32k registration particles and native16-interval
PCG/RK2 recovery. A matched ordinary-AdamW0.001 arm is included. All three frozen
quick4 subsets use one seed42; labels remain evaluation-only, and DIR-Lab masks
only place shared crops. No particle objective or grid-sigma sweep is combined.

At this recipe's summed learning rate0.3124625, isolated cumulative separation
shrinkage is approximately0.0031%,0.0312%,0.3120% for the three coefficients.
Objective gradients can increase or decrease separation at the same time, so
final model distance is a diagnostic rather than a direct coefficient check.
Small coefficients may also have changes below individual FP32 update rounding.
The study reports all cases and rates without selecting cases by their scores.

Synthetic CPU/CUDA checks cover exact mean/difference behavior, explicit
contraction-plus-AdamW0 update parity, unchanged gradients/moments, learning-rate
groups, zero-coefficient parity, inactivity guards and state restoration.
Implementation: `medical/registration/paired_adamw.py`.

## Completed quick4 comparison

All48 single-seed fits completed on four RTX5090 GPUs in1246seconds (20m46s),
with all768 PCG recovery intervals converged. The default512-neuron recipe and
all image/data/evaluation settings were held fixed; the control uses ordinary
AdamW0.001, and the three experimental arms replace it with mean-centered decay.

| Optimizer | DIR-Lab primary/reverse rounded TRE mm | OASIS Dice | Abdomen Dice | Mean total seconds/case |
|---|---:|---:|---:|---:|
| AdamW tozero,lambda0.001 | 1.7418/1.8433 | 0.82450 | 0.42435 | 100.16 |
| Pair mean,lambda0.0001 | 1.7977/1.8528 | 0.82484 | 0.42262 | 100.33 |
| Pair mean,lambda0.001 | 1.8249/1.8793 | 0.82541 | 0.42458 | 101.79 |
| Pair mean,lambda0.01 | 1.8022/1.8605 | 0.82449 | 0.43148 | 103.57 |

No coefficient consistently improves all three datasets. Lambda0.001 has the
highest OASIS Dice, while0.01 improves abdomen Dice by0.00713 but worsens lung
TRE and increases abdomen mean folding from0.0526% to0.0887% (maximum0.311%).
All DIR-Lab lung folding fractions are zero. These are four-case/single-seed
development results, not evidence of a universally optimal coefficient. The
repository default stays ordinary AdamW.

Final endpoint parameter separation is not monotonic in coefficient: the data
objective moves endpoints as well, and the isolated mean-pull strength is small.
For example, mean parameter-gap L2 over the four abdomen pairs is2.58959 for
the control and2.58973 atlambda0.01. The update-level tests, rather than final
trained distance alone, verify that the decoupled contraction is applied.
No separate mean-pull coefficient0 control was run, so score differences combine
removal of zero-centered decay and addition of the mean pull.

Presets are `medical/configs/ablations/native_siren_paired_adamw_1em4_5k.json`,
`native_siren_paired_adamw_1em3_5k.json` and
`native_siren_paired_adamw_1em2_5k.json`.

[Per-case scores, runtime, folding and parameter diagnostics](../results/paired_adamw_quick4_20261115T124612Z/analysis.json)
· [Frozen plan/configs and source/input checks](../results/paired_adamw_quick4_20261115T124612Z/plan.json)
· [Installed PyTorch AdamW reference](../results/paired_adamw_quick4_20261115T124612Z/adamw_reference.json).
