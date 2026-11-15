# Frozen strong-baseline comparison: Euler ×2 versus midpoint RK2 ×1

Used the existing LPBA S02 MSE+strict-PD-LK benchmark checkpoint, whose historical
standard PCG+RK2 result is Dice .702452 / HD95 6.60482 mm / zero folding. No
training, optimizer step or checkpoint change occurred. Checkpoint SHA256 was
unchanged after evaluation.

Both compared integrators use identical full-image 24³ analytical local velocity
fields with c=4, eta=1e-6 and Gaussian sigma4 mm. Forward maps use fields at
t=0,.5; reverse maps independently use t=1,.5 with negative time steps. Each
direction requires two fields for either method. Reverse maps are numerical
approximations, not exact inverses. Full native-grid image/label scoring uses the
original data and repository MONAI56-label metric definitions. These are the
training surrogate's coarse local fields, not the global PCG velocity. The prior
crop24³/half-volume training setting is also different; this probe evaluates
whole-image maps to obtain full-volume Dice/HD95.

| Recovery | Dice | HD95 mm | Folding % | Mean forward/reverse composition error mm |
|---|---:|---:|---:|---:|
| Euler ×2, local Gaussian24³ | .656778 | 6.94437 | .121114 | 2.61961 |
| Midpoint RK2 ×1, same fields | .647632 | 7.22264 | 1.193441 | 1.38381 |
| Historical standard PCG+RK2 reference | .702452 | 6.60482 | 0 | .127777 |

Euler ×2 was better in Dice, HD95 and folding on this frozen checkpoint, despite
RK2 ×1 having a smaller composition error. The forward endpoint maps differed by
2.215 mm on average (95th percentile4.407 mm); reverse maps differed by2.105 mm
(95th percentile4.076 mm). LPBA spacing is1 mm isotropic. Formal RK2 order does
not ensure a better registration map for one large integration step and this
approximate velocity field. This test does not establish which integrator would
train better or isolate coarse-field error from integration error.

After compilation warm-up, per-field construction medians over three repeats
were3.74/3.72/3.72 ms at t=0/.5/1. Full-grid forward-plus-reverse integration
medians over five repeats were1.88 ms for Euler ×2 and1.85 ms for RK2 ×1, with
fields cached. Three shared fields therefore cost11.18 ms for either method,
plus about1.9 ms integration; label/image evaluation is excluded. Timings are
frozen forward inference, not differentiable training or backward timings. Both
methods were evaluated sequentially; no claim of a meaningful timing difference
is supported.

No native maps, warped volumes or persistent velocities were written. Scalar
metrics, exact observer script and provenance remain in ignored
`results/integrator_compare/baseline_s02_20261111/`.

## Midpoint-in-time Euler follow-up

On the same frozen checkpoint, a third rule evaluated velocity at interval
midtimes while retaining the current spatial position:

\[
x_{k+1}=x_k+\tfrac12v(x_k,t_k+.25),\qquad t_k=0,.5.
\]

Reverse integration used the same fields in reverse order, with step -.5:
times .75,.25. Velocity construction, grid, Gaussian, data, metric definitions
and all other settings matched the prior comparison; no training occurred.

| Integrator | Dice | HD95 mm | Folding % | Composition error mm |
|---|---:|---:|---:|---:|
| Ordinary Euler ×2 | .656778 | 6.94437 | .121114 | 2.61961 |
| Midpoint-in-time Euler ×2 | .660689 | 6.78242 | .240397 | .67062 |
| Midpoint RK2 ×1 | .647632 | 7.22264 | 1.193441 | 1.38381 |

The midpoint-in-time variant slightly improved Dice/HD95 and composition error
over ordinary Euler, but roughly doubled the folding fraction. The standard
PCG+RK2 reference remains higher at Dice .702452 and zero folding. Higher label
scores do not establish more accurate integration, and this frozen test does
not establish training performance.

Both directions shared two fields costing7.50 ms in total, plus1.90 ms for
forward/reverse native-grid integration (median of five repeats, cached fields).
Thus about9.40 ms excludes compilation and scoring, compared with about13.1 ms
for the prior methods requiring three distinct fields for both directions.
Forward-only field counts remain two for every method. No training backward
timing was measured. Forward maps differed from ordinary Euler by2.576 mm on
average (95th percentile5.095 mm); inverse maps differed by2.496/5.005 mm.

Checkpoint SHA256 remained unchanged. Scalar evidence and the exact script are
in ignored `results/integrator_compare/time_midpoint_s02_20261111/`;
[compact JSON](time_midpoint_baseline_20261111.json) is retained here.
