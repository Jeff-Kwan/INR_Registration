# Inverse-metric velocity agreement

This is an alternative training regularizer, selected with
`medical/configs/inverse_metric.json`. The repository default uses endpoint NCC with a 0.01 moment penalty,
0.1 PD-LK and 0.01 box NCC. The alternative uses **MSE + 0.05 box NCC + 1.0 inverse-metric
agreement from the first update**, per the requested initial experiment. There is
no warm-up or step-500 calibration; calibration is deferred.

For each probe, Q = ggᵀ + 4HᵀH and b = gq + 4Hᵀm. The original equal-weight
four-probe cell estimator remains A = mean(Q) + 1e-6 I, v = -solve(A,mean(b)). Probe forces s = Qv+b are centered within each cell: z = s-mean(s). The penalty
is mean(zᵀ solve(A,z)) over probes, cells and the two sampled times. The code uses
a multi-right-hand-side solve, never an explicit matrix inverse. Ridge belongs
to A; no extra ridge term is inserted into individual probe forces or the penalty. All crop cells contribute.

The PD-LK residual penalty is not evaluated in inverse-metric mode. Both modes
use exactly the same analytical derivatives, equal-weight velocity solve,
Gaussian smoothing, trilinear interpolation and bidirectional particle NCC.
All gradients through velocities, centered forces, A and the metric solve remain
live. No detached estimators or gradient suppression are introduced. Final
PCG+RK2 inference is unchanged.

The optional objective.regularizer selector is `pdlk` (also the omitted default)
or `inverse_metric`. objective.weight is the fixed metric coefficient in the new
preset, initially 1.0; zero remains available for ablations. The alternative uses
MSE, constant weights, c=4, eta=1e-6, two symmetric pairs/cell and a nonnegative box NCC weight
(initially 0.05; the lower-weight comparison uses 0.01). Unsupported combinations are rejected. Logs name the new term
`inverse_metric`, and checkpoints record the selected regularizer and coefficient
under `training`, as well as retaining the resolved config.

Plan an OASIS run without launching it:

```bash
/venv/main/bin/python -m medical.oasis_experiments --methods ours \
  --config medical/configs/inverse_metric.json --gpus 0 1 2 3 \
  --output results/oasis_inverse_metric
```

Execution still requires explicit experiment authorization and `--run`. All use the existing persistent workers and shared queues.

Synthetic tests check an independent literal formula and gradients, gradcheck,
zero penalty for identical probe evidence, probe-order invariance, unchanged
velocity fields, live endpoint gradients and compiled CUDA parity. These establish
implementation consistency, not empirical stability or registration improvement.
No training experiment was launched during implementation.
