# Training trajectory integrators

The shared-grid method defaults to `training.integrator: "euler4"`.
Set `training.integrator: "rk2"` to select two RK2 half-steps. Both use the same crop, shared-grid LK solve,
velocity smoothing, independent bidirectional starts and target NCC objective.
Frozen PCG/RK2 recovery is configured separately.

## Four stochastic quarters with Euler

The default configuration contains this key inside its training section:

```json
"integrator": "euler4"
```

With `objective.registration.random_timesteps: true`, each update samples
four independent times, one uniformly within each quarter:

`t_i = (i + U_i) / 4`, for `i = 0, 1, 2, 3` and `U_i ~ Uniform[0,1)`.

Each draw represents its entire quarter, as a stochastic replacement for its
midpoint. The integration step size stays **1/4**; it does not depend on gaps
between sampled times. At each time, the shared-grid SIREN derivatives, pooled
LK velocity solve and smoothing produce one field. The same spatial query
neighborhood is used at all four times within that update.

Moving→fixed transport applies `x_next = x + 0.25 v_i(x)` in time order.
Fixed→moving uses the same four fields in reverse order and step size `-0.25`.
Both directions retain independently sampled starts in the same crop.
The LK regularizer is averaged over all four times, and registration NCC plus
the full-image boundary penalty is averaged over both directions. There is no
cycle penalty. All derivatives through solves, smoothing and trajectories stay
live. Crop exits contribute to NCC with border extension, as in the default.

With `random_timesteps: false`, the four deterministic quarter midpoints are
`0.125, 0.375, 0.625, 0.875`. This is available for controlled ablations.

## Relationship to RK2 and execution

The RK2 option samples one field from each half and applies a spatial midpoint
RK2 step of size 1/2 within each frozen field. Four-quarter Euler increases
temporal sampling; it evaluates the spatial velocity only at the current
particle position. Both options perform four velocity interpolations per
direction, while Euler computes four derivative/solve/smoothing fields instead
of two. An accuracy or runtime advantage has not been measured in this change.

Ordinary training, the compiled decoder, the decoder CUDA graph and complete
captured training support both options. Independent temporal sampling uses one
scalar RNG draw per stratum. Captured `training.temporal_sampling: "sobol"`
uses a scrambled four-dimensional sequence for Euler and two dimensions for RK2,
with the same quarter/half mapping and unchanged spatial sampling streams.
Checkpoint metadata records the integrator, number of strata, time step and
within-stratum sampling convention.

The separate pointwise particle objective keeps its existing two-step stochastic
split protocol. It rejects `training.integrator` so this shared-grid option
cannot silently change that experiment.

## Ready preset and validation

[native_siren_euler4_5k.json](../medical/configs/ablations/native_siren_euler4_5k.json)
matches the current 512-wide, 5,000-step default recipe, including
`training.integrator: "euler4"`. To inspect its IXI/LPBA40 plan:

```bash
/venv/main/bin/python -m medical.brain_experiments --plan-only \
  --config medical/configs/ablations/native_siren_euler4_5k.json
```

Omitting the integrator in shared-grid training also selects Euler. The original
`native_siren_onecycle_5k.json` reference and the recorded
`native_siren_gw_orthogonal_sobol_5k.json` accuracy candidate explicitly retain
`"rk2"` for their published local results. Use resolved configs from experiment
outputs when reproducing earlier runs.

No training was launched by this default change. Synthetic tests exercise stochastic quarter sampling,
reverse field order with noncommuting affine fields, LK mean weighting, full
SIREN parameter gradients, NCC/GW-NCC equivalence across execution paths,
crop exits, nonfinite guards and CUDA graph sampling/replay. Tests do not take
optimizer steps or run saved-model recovery.
