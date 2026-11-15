# Central-80% sampling, random boxes and random path times

LPBA S02, seed 42. The threshold screen and three requested variants each used
5,000 steps, 8³ cell-centred fields, two symmetric pairs per cell, the three-tap
Gaussian (sigma 4 mm), 4,096 particles per direction, MSE endpoint reconstruction,
registration weight .01 after step 2,000, uncapped registration gradients, global
norm-1 clipping, and PD-LK disabled throughout. Final recovery was unchanged
PCG+RK2. The fixed-box screen used a half-size box and field times .25/.75. Random
box runs independently sampled each image-axis fraction from [.25,.5] each
minibatch; the dedicated crop RNG produced depth/height/width fractions ranging
.256–.500, .255–.500 and .256–.500 respectively (means .380/.377/.378) over
5,000 steps. Random-time runs sampled one field time in [0,.5] and one in [.5,1].
All boxes remained within the original image.

The 80% particle region passed the requested <1% rejection screen: mean rejection
was **.019%** over active logged steps; the maximum logged directional rate was
.513%. This met the condition for making central 80% the symmetric-pair default
and using it in all three follow-ups.

| Variant | Endpoint PSNR moving/fixed (dB) | Dice | HD95 (mm) | Folded voxels | Mean rejection | Train/recovery (s) |
|---|---:|---:|---:|---:|---:|---:|
| Fixed half-size box, fixed times | 31.36 / 31.76 | .67161 | 6.776 | 2 | .019% | 56.09 / 9.48 |
| Random box size/aspect | 31.01 / 31.11 | .65587 | 6.920 | 10 | .240% | 57.80 / 10.06 |
| Random field times | 31.34 / 31.70 | **.67497** | **6.753** | 1 | .015% | 55.14 / 9.59 |
| Random box and times | 31.11 / 31.36 | .65832 | 6.883 | 3 | .140% | 57.32 / 10.10 |
| MSE+PD-LK reference | 32.20 / 32.48 | .70245 | 6.605 | 0 | — | 37.03 / 18.35 |

All four SIREN runs passed the 25 dB endpoint gate, had zero skipped registration
batches and converged in all 16 PCG intervals. The highest logged rejection at a
single active checkpoint was 5.05% for the random-box-only variant and 4.27% for
the combined variant; averages stayed below .25%. Random-time samples covered
[.0045,.4988] in the early half and [.5040,.9990] in the late half. A few isolated
branch timings exceeded the .25 s soft threshold in the random-time variants, but
never reached the 2 s hard limit or triggered the consecutive-slowdown guard.

Random field times gave the best of these four Dice/HD95 pairs, but every variant
remained below the maintained PD-LK reference. These are single-seed results on
one pair and do not establish general benefit. The logged rejection rate is
measured at the 150 active CSV checkpoints, not every optimizer step.

Ignored artifacts and full training logs are under
`results/pair_randomized_20261111/{fixed80,random_box80,random_time80,random_box_time80}/`.

## Follow-up: box size and registration weight

A further seed-42 S02 screen retained 8³ cells, two pairs, random times,
central-80% starts, three-tap smoothing, uncapped registration gradients and
global norm-1 clipping. A .25 crop at weight .01 gave endpoint PSNR 30.24/30.98
dB, Dice .64242 and HD95 6.880 mm. With the .5 crop, weight .05 gave endpoint
PSNR 30.75/31.18 dB, Dice .69243 and HD95 6.586 mm. Two of 4,915,200 voxels
folded; all 16 recovery intervals converged. Weights .1, .5 and 1.0 were
stopped by the 25 dB reconstruction monitor after activation (source PSNR 24.68,
20.12 and 18.89 dB respectively), so no final registration was evaluated.

The user selected the .05 half-box setup as the opt-in registration-aware
default. This is a single-pair pilot; the maintained PD-LK reference remains
Dice .70245 and HD95 6.605 mm. Local run evidence is under
`results/pair_followup_20261111/`.

## Immediate activation pilot

Using the selected .5-box, weight-.05 setup, registration loss was active from the
first optimizer update (configured activation step 0). The PSNR gate was explicitly
skipped; endpoint PSNR was 30.63/31.29 dB at completion (moving/fixed; 13.18/14.18
dB at the first step-20 diagnostic). Dice was .69730, HD95 6.515 mm, and 3 of
4,915,200 voxels folded. All 16 PCG intervals converged. Total wall time was
94.11 s: 78.64 s training and 10.60 s recovery. Compared with delayed activation
at the same weight (.69243 Dice, 6.586 mm HD95, 70.95 s total), this single run
was modestly better in label metrics and took longer; it is one seed and one pair.
Artifacts are in the ignored `results/pair_immediate_activation_20261111/` folder.

## Immediate activation with NCC and 0/1 moments

The paired registration branch was run with global minibatch NCC endpoints, the
fixed Gaussian moment penalty (weight .01, target mean 0 and variance 1),
registration weight .05 from the first optimizer step, PD-LK disabled and the
PSNR gate skipped. At completion, endpoint NCC was .99618/.99635 but source/fixed
output means were -.686/-.668 and variances .705/.585; raw endpoint PSNR was
-.06/.54 dB. The requested intensity gauge was therefore not achieved.

Unchanged PCG+RK2 recovery converged in all 16 intervals. Dice was .66328, HD95
6.553 mm, and 7,901 of 4,915,200 voxels folded (0.1607%). Runtime was 93.41 s
(80.82 s training, 8.05 s recovery). Against the immediate MSE run, label scores
were lower (.66328 vs .69730 Dice) and folding was much higher (7,901 vs 3 voxels).
This single S02 result suggests the .01 moment term did not constrain the output
gauge strongly enough alongside registration-aware training; it does not identify
why or establish a general NCC limitation. Artifacts are in the ignored
`results/pair_ncc_moments_immediate_20261111/run_complete/` folder.

## Immediate registration loss plus strict PD-LK

At the user's explicit request, an S02 seed-42 pilot combined MSE endpoints,
strict PD-LK weight .1 (`rho=0`, 4,096 full-domain points), and the default
registration loss weight .05 active from the first optimizer step. Paired
registration retained 8³ cells, two pairs, the half-size crop, central-80%
particle starts, random times, three-tap smoothing and 4,096 particles per
direction. PD-LK used its established full-domain point sampling while endpoint
reconstruction used the local box. The explicit `allow_pdlk=true` setting is
required for this combination. The PSNR gate was skipped, so endpoint quality
pass is null; measured moving/fixed PSNR was 31.26/31.52 dB.

Final PCG+RK2 recovery gave Dice .70692, HD95 6.373 mm and zero folded voxels
out of 4,915,200; all 16 PCG intervals converged. Training took 94.73 s,
recovery 10.59 s and total pair time 109.98 s. For context, immediate MSE plus
registration loss without PD-LK gave .69730 Dice, 6.515 mm HD95 and 3 folds
(training/recovery 78.64/10.60 s); the previously recorded MSE+PD-LK baseline
gave .70245 Dice, 6.605 mm HD95 and zero folds (37.03/18.35 s). These are
one-pair, one-seed results and historical recovery timings are not a controlled
solver-speed comparison. Across 250 logged checkpoints both weights were
constant, no registration batch was skipped and mean particle rejection was
.015%. Outputs are under ignored `results/pair_pdlk_reg_immediate_20261111/`.

## Shared-grid PD-LK and a single backward

One further seed-42 LPBA S02 run retained the preceding combined weights (.1
PD-LK, .05 registration), immediate activation, 8³ cells, two pairs per cell,
half-size boxes, central-80% particles, random times, three-tap smoothing, global
norm-1 clipping and 5,000 updates. It disabled derivative recomputation and used
one backward for the summed loss. PD-LK now reuses the existing raw four-point
cell optimum: average the squared brightness and differential residuals over
the four sites and add the ridge once, then average over cells and both times.
The optimum is frozen for the PD-LK envelope gradient, and remains live through
the registration loss. The nominal derivative-site budget falls from 8,192 to
4,096 per step: 512 cells × four sites × two times. There are 1,024 pooled
cell/time PD-LK observations. No independent PD-LK queries are evaluated.

The implementation-only changes preserve the summed objective, but sharing the
grid changes full-domain independent pointwise PD-LK into a crop-local pooled
velocity objective. All changes were applied together in this pilot, so the
clinical timing result does not isolate their individual contributions.

| Configuration | Dice | HD95 mm | Folded voxels | Endpoint PSNR moving/fixed dB | Training/recovery s |
|---|---:|---:|---:|---:|---:|
| Previous independent PD-LK + registration | .70692 | 6.373 | 0 | 31.26 / 31.52 | 94.73 / 10.59 |
| Shared-grid PD-LK + optimized backward | .70664 | 6.384 | 0 | 30.82 / 31.19 | 62.17 / 10.66 |
| Historical MSE + strict pointwise PD-LK | .70245 | 6.605 | 0 | 32.20 / 32.48 | 37.03 / 18.35 |

Shared-grid training was 34.4% shorter (1.52× throughput measured over the complete
training stage). Median logged step time fell from 18.09 to 11.13 ms. Total pair
time fell from 109.98 to 77.49 s. Compilation/startup is included in stage timing.
The PSNR gate was explicitly skipped (pass null); finite/runtime guards remained.
All 16 recovery intervals converged, with maximum relative residual 1.9644e-5
and minimum Jacobian .08461. All 250 logged checkpoints had finite total gradients,
no skipped registration batches and constant loss weights; mean logged particle
rejection was .0084%. Logged total loss matches reconstruction plus weighted
PD-LK plus weighted registration to absolute error 1.13e-9. Separate component
gradient norms are unmeasured/blank on the single-backward path.

The medical synthetic suite passed 105 tests and Cameraman passed one. New tests
check pooled-energy normalization and finite-difference/envelope gradients,
one derivative pass per time, gradients into both endpoints, and single-backward
agreement with the split/recompute calculation for the same sampled objective.
This is one pair and one seed, with essentially similar scores, not evidence of
accuracy equivalence across cohorts. The historical baseline recovery timing is
not a controlled solver-speed comparison. Final recovery remains PCG+RK2. Local
artifacts are ignored under `results/pair_shared_grid_pdlk_20261111/`.

## g/q-only two-pair directional approximation

A matched S02/seed-42 run changed only the shared-grid derivative mode to
`paired_difference`, retaining two random symmetric pairs per cell, 2 mm physical
radius, 8³ grid, combined weights .1/.05, immediate activation, no checkpoint
recomputation and one backward. It approximated Hd and d·m through central g/q
differences, using factor 3 directional weighting at S02's isotropic spacing.
No analytical H/m or independent PD-LK points were evaluated during training.
Two pair directions do not determine a full Hessian. Finite-radius bias and
angular sampling variance both change the objective.

| Mode | Dice | HD95 mm | Folded voxels | Endpoint PSNR moving/fixed dB | Training/recovery s |
|---|---:|---:|---:|---:|---:|
| Analytical shared-grid H/m | .70664 | 6.384 | 0 | 30.82 / 31.19 | 62.17 / 10.66 |
| g/q-only two-pair differences | .63643 | 6.948 | 0 | 24.09 / 24.48 | 55.76 / 10.33 |

Total pair time was 70.76 s. Final PCG+RK2 converged in all 16 intervals, with
maximum relative residual 1.9657e-5. The PSNR gate was explicitly skipped (pass
null), but measured endpoint PSNR was below the usual 25 dB threshold. The small
speed gain did not compensate for the substantial quality loss. This result
does not isolate angular variance from finite-difference bias or objective-scale
effects. The analytical shared-grid mode remains preferred. A subsequently
proposed three-axis/six-site deterministic approximation was tested subsequently
as recorded below.

108 medical + one Cameraman synthetic tests passed, including quadratic
directional recovery, native-unit anisotropic angular quadrature, finite-difference
parameter gradients, flat-region stability and g/q-only queries/finite gradients
into both SIRENs. Artifacts are under ignored
`results/pair_directional_difference_20261111/`.

## Deterministic three-axis finite differences

One authorized matched S02/seed-42 5,000-step run changed the preceding finite
difference configuration to `pair_layout="axes", pairs_per_cell=3`. The six
sites are ±2 mm on each native image axis. First derivatives g/q remain
analytical; only spatial changes of g/q (H/m actions) use central differences.
The three directional residuals are summed, with no stochastic angular weights.
This covers a full basis but remains a finite-radius approximation, not exact
analytical SIREN second derivatives. Registration gradients remain live; the
shared PD-LK envelope uses detached raw local velocities.

All other settings were held: 8³ half-size crop, central-80% starts, 4,096
particles/direction, random field times, two Euler steps, three-tap smoothing,
combined weights .1 PD-LK/.05 registration from step zero, no separate gradient
cap, global clip 1, one backward and no recomputation. Final global brightness
PCG+RK2 was unchanged. No additional configuration or cohort was run.

| Mode | Dice | HD95 mm | Folded voxels | Endpoint PSNR moving/fixed dB | Training/recovery s |
|---|---:|---:|---:|---:|---:|
| Analytical shared-grid H/m | .70664 | 6.384 | 0 | 30.82 / 31.19 | 62.17 / 10.66 |
| Two random pairs, finite H/m actions | .63643 | 6.948 | 0 | 24.09 / 24.48 | 55.76 / 10.33 |
| Three axis pairs, finite H/m actions | .70428 | 6.359 | 0 | 29.87 / 30.49 | 56.54 / 12.39 |

The axis run reduced measured training time by 9.1% relative to analytical H/m;
median logged step time was 10.09 ms. Total pair time was 73.56 s. Endpoint MSE
was .00102954/.00089265. The PSNR gate remained explicitly skipped/pass null,
although diagnostic PSNR exceeded 25 dB. All 16 recovery intervals converged,
maximum relative residual 1.95723e-5; minimum Jacobian was .080776 over 4,915,200
voxels. All 250 logged total-gradient norms were finite (median .3177, maximum
3.0484 before global clipping). No particle-loss step was skipped; mean rejected
fraction was .2199%. Loss-assembly discrepancy was below 6.1e-9.

The result supports deterministic full-basis differences over this two-random-pair
approximation, but changes both directional coverage and site count (four to six).
It does not isolate angular noise from sampling density or finite-radius bias.
Dice was slightly lower and HD95 slightly better than analytical H/m; do not claim
cohort equivalence or promote the default. Analytical mode remains default.

108 medical and one Cameraman synthetic tests passed, including anisotropic
quadratic exactness, deterministic layout/no offset-RNG consumption, g/q-only
query reuse and finite nonzero gradients into both endpoints. Wheel inspection
found 53 entries without tests/data/results/archives; `git diff --check` passed.
Local inputs and prior artifacts were preserved. New artifacts remain ignored
under `results/pair_axes_difference_20261111/`.

## NCC registration similarity in a local box

One S02 seed-42 pilot replaced bidirectional original-image particle MSE with one
Pearson NCC per direction over valid correspondences. It computes one correlation
across the sampled half-size box, not sliding-window NCC. Endpoint reconstruction
remained MSE, PD-LK training weight was zero, and the local velocity used
analytical g/q/H/m. Both runs used 8³ cells, two random pairs, 2 mm radius,
central-80% starts, 4,096 particles per direction, random field times, two Euler
steps, three-tap smoothing, registration weight .05, 5,000 steps, global clip 1,
seed 42 and unchanged PCG+RK2. NCC epsilon was 1e-8. NCC used one backward and
no recomputation; the historical MSE run used separate backward and recomputation,
so timing is not a metric-only speed comparison. Both explicitly skipped the
PSNR gate (pass null).

| Objective | Dice | HD95 mm | Folded voxels / 4,915,200 | Endpoint PSNR moving/fixed dB | Training / recovery s |
|---|---:|---:|---:|---:|---:|
| MSE endpoints + MSE registration | .69730 | 6.515 | 3 | 30.63 / 31.29 | 78.64 / 10.60 |
| MSE endpoints + NCC registration | .70413 | 6.325 | 2 | 29.15 / 29.64 | 60.86 / 10.65 |
| Historical MSE + strict PD-LK baseline | .70245 | 6.605 | 0 | 32.20 / 32.48 | 37.03 / 18.35 |

The NCC endpoint MSE was .00121647/.00108707; total pair time was 76.20 s and
median logged step was 11.32 ms. Four of 250 logged pre-clip total gradient norms
exceeded 1; all were finite. Mean particle rejection was .1315%, with no skipped
registration updates. Two of 16 PCG intervals narrowly missed relative tolerance
2e-5 (maximum 2.01552e-5); the MSE control converged on all intervals. MONAI
1.6.1 Dice/HD95 are macro means over 56 labels. This single case is not cohort
evidence; equal numerical weights need not give equal gradients. New evidence
is ignored under `results/pair_registration_ncc_20261111/`.

## Shared-grid PD-LK plus NCC registration

One S02 seed-42 pilot used the existing shared-grid combined-objective settings
and changed only registration similarity from MSE to NCC. Endpoint MSE, PD-LK
weight .1, registration weight .05, random two-pair 8³ sampling, activation
from step zero, one backward, global gradient clip 1 and final PCG+RK2 were held
fixed. PD-LK is the pooled strict differential energy over the same paired sites,
not independent full-domain pointwise PD-LK. The PSNR gate was explicitly
skipped (pass null).

| Objective | Dice | HD95 mm | Folded voxels | Endpoint PSNR moving/fixed dB | Train/recover s |
|---|---:|---:|---:|---:|---:|
| Shared-grid PD-LK + MSE registration | .70664 | 6.384 | 0 | 30.82 / 31.19 | 62.17 / 10.66 |
| NCC registration, no PD-LK | .70413 | 6.325 | 2 | 29.15 / 29.64 | 60.86 / 10.65 |
| Shared-grid PD-LK + NCC registration | .71441 | 6.251 | 0 | 30.16 / 30.60 | 61.90 / 11.41 |

The combined NCC run's endpoint MSE was .00096299/.00087062; total pair time was
77.84 s and median logged step was 11.35 ms. All 16 PCG intervals converged,
maximum relative residual 1.90013e-5, minimum Jacobian .07613 over 4,915,200
voxels. All 250 logged total gradient norms were finite; two exceeded the global
clip threshold. Mean particle rejection was .0555%, with no skipped updates.
The logged objective accounting discrepancy stayed below 2.4e-9. Dice and HD95
are MONAI 1.6.1 macro averages across 56 labels. These are one pair and seed;
the numerical MSE and NCC weights are not necessarily gradient matched. Local
artifacts remain ignored under `results/pair_pdlk_registration_ncc_20261111/`.

## Brightness-only patch LK and NCC registration

One S02 seed-42 pilot replaced the differential local solve with a shared
brightness-only patch solve: analytical g/q at two random symmetric pairs per
cell, eta=1e-6, patch weight .1, registration NCC weight .05, MSE endpoints,
PD-LK weight zero, and both losses active from the first update. Patch and
registration share the same local velocity; the energy uses an envelope gradient
and registration differentiates through the live solve. Other settings and
unchanged PCG+RK2 recovery match the preceding combined NCC run.

| Objective | Dice | HD95 mm | Folded voxels / 4,915,200 | Endpoint PSNR moving/fixed dB | Train/recover s |
|---|---:|---:|---:|---:|---:|
| Shared-grid PD-LK + NCC registration | .71441 | 6.251 | 0 | 30.16 / 30.60 | 61.90 / 11.41 |
| NCC registration, no PD-LK (differential solve) | .70413 | 6.325 | 2 | 29.15 / 29.64 | 60.86 / 10.65 |
| Brightness patch LK + NCC registration | .60720 | 7.171 | 7 | 19.59 / 19.91 | 55.53 / 7.80 |

The brightness run's endpoint MSE was .0110002/.0102007. The explicitly skipped
PSNR gate remains null: the measured PSNR was below the usual 25 dB threshold,
so registration is diagnostic and the run is not a reconstruction-quality pass.
All 16 final PCG intervals converged (max relative residual 1.94349e-5); minimum
Jacobian was -1.48214. Median training step was 10.13 ms, total pair 68.17 s.
All 250 logged pre-clip gradient norms were finite but exceeded the global clip
threshold 1 (median 21.95, maximum 148.17). Mean particle rejection was 5.95%,
with no skipped updates; PD-LK logs stayed zero and loss assembly discrepancy
was below 8.4e-9. MONAI 1.6.1 scores are macro averages over 56 labels.

This single pilot changes both the local velocity model and the regularizer, so
it cannot isolate the cause. Large gradients and clipping coincide with poor
reconstruction; attribution to the live solve, patch energy, or objective
interaction requires a separate diagnostic. Do not promote or infer a general
negative result about brightness-only patch LK. Artifacts remain ignored under
`results/brightness_patch_ncc_20261111/`.

### Brightness patch weight follow-up and separated velocity solve

The .01 patch-weight run retained the brightness registration solve, NCC .05,
and all other settings. It produced Dice .56329, HD95 7.474 mm, 1,716 folded
voxels, and endpoint PSNR 17.64/18.62 dB (gate skipped); all 250 logged total
gradient norms exceeded 1 (median 33.17), and all 16 final PCG intervals converged.
Its artifacts are under ignored `results/brightness_patch_ncc_w001_20261111/`.

The .001 weight diagnostic was stopped by the user after step 4800, before
checkpoint or recovery. Its first compiled attempt stopped at step 20 because
retained backward traversals are incompatible with donated buffers; the fresh
retry used uncompiled velocities. At steps 20/40/100/400 the combined endpoint
gradient norms for reconstruction were 2.897/1.157/.566/1.166, for weighted
patch energy .00019/.00020/.00109/.01399, and for weighted NCC registration
1.044/1.498/8.529/37.677, respectively. The partial log is ignored under
`results/brightness_patch_ncc_w0001_uncompiled_20261111/`; no final metrics exist.

One subsequent S02 seed-42 pilot restored the original differential velocity
for NCC while retaining a separate brightness patch energy at weight .1. The
two local optima use the same g/q/H/m query sites: the brightness optimum is
detached for its patch-energy envelope gradient, and the differential optimum
remains live for registration. Strict PD-LK training loss was zero. NCC weight
was .05, with the usual two random symmetric pairs per 8³ cell, half-size box,
5,000 steps and unchanged final PCG+RK2. For retained-graph component logging,
the velocity was uncompiled and three component-gradient traversals were made
every 20 steps; timing does not isolate the objective or match compiled runs.

| Objective / registration velocity | Dice | HD95 mm | Folded voxels | Endpoint PSNR moving/fixed dB | Train/recover s |
|---|---:|---:|---:|---:|---:|
| Brightness patch + brightness velocity | .60720 | 7.171 | 7 | 19.59 / 19.91 | 55.53 / 7.80 |
| Brightness patch + differential velocity | .70932 | 6.273 | 0 | 29.56 / 30.09 | 84.95 / 12.46 |
| Shared-grid PD-LK + differential velocity | .71441 | 6.251 | 0 | 30.16 / 30.60 | 61.90 / 11.41 |

The restored-velocity pilot had endpoint MSE .00110657/.00098047; its PSNR gate
was explicitly skipped/null, though measured PSNR exceeded 25 dB. All 16 final
PCG intervals converged (max relative residual 1.97985e-5); minimum Jacobian
was .06452 across 4,915,200 voxels. Total pair time was 102.22 s. All 250
logged gradient components were finite; combined weighted component medians
were .151 reconstruction, .078 patch and .318 registration. Three logged
total gradient norms exceeded 1 (median .375), compared with all 250 and
median 21.95 in the brightness-velocity .1 control. Mean particle rejection was
.110%, with no skipped updates. Objective accounting error was below 4.1e-9.
The uncompiled/logged numerical trajectory differs from the compiled control,
so the improvement strongly implicates the velocity choice but is not a perfect
bitwise controlled intervention. These remain one pair/seed, not cohort evidence.
See ignored `results/patch_lk_differential_ncc_20261111/` for component logs.

## Pointwise PD-LK at the paired registration sites

One S02 seed-42 pilot compared `pdlk_sampling="paired_points"` with shared-grid
PD-LK. Each of the four sites per cell obtains its own strict PD-LK velocity
and minimized energy; registration retains the original pooled differential
velocity. Both use the same g/q/H/m derivative queries at the two shared field
times. Objective weights remain .1 PD-LK/.05 registration NCC, with MSE
endpoints, 8³ grid, two symmetric pairs, a half-size local box and standard
PCG+RK2 final recovery. PD-LK is active throughout and uses envelope gradients.
The 25 dB PSNR gate was explicitly skipped (pass null) for both runs.

| PD-LK training velocity | Dice | HD95 mm | Folded voxels | Endpoint PSNR moving/fixed dB | Train/recover s |
|---|---:|---:|---:|---:|---:|
| Shared across four sites | .71441 | 6.251 | 0 | 30.16 / 30.60 | 61.90 / 11.41 |
| Independent at each site | .71430 | 6.237 | 0 | 29.72 / 30.68 | 85.84 / 12.54 |

The pointwise run logged component gradients every 20 steps using uncompiled
velocity and extra retained-graph backward traversals; the shared control used
compiled velocity without those traversals. Therefore neither runtime comparison
nor bitwise equivalence is claimed. Endpoint MSE was .00106713/.00085596. All
16 PCG intervals converged (max relative residual 1.97794e-5), minimum Jacobian
.07620 over 4,915,200 voxels. Of 250 logged pre-clip total norms, two exceeded
1 (median .254); component gradient medians were .142 reconstruction, .068
weighted PD-LK and .187 weighted registration. Mean particle rejection was .0652%,
loss accounting discrepancy below 4.4e-9. MONAI 1.6.1 label scores are macro means
across 56 labels. This near tie is one case/seed, not a general equivalence claim.
Ignored scalar logs and endpoint weights are under
`results/paired_points_pdlk_ncc_20261111/`.

## Matched compiled S03 comparison

Follow-up LPBA S03 runs compared shared-grid and independent pointwise strict
PD-LK at the same paired registration sites. Both used MSE endpoint fitting,
the local NCC registration term (.05), PD-LK weight .1, the same 8³/two-pair
central-80% symmetric-pair setup, and unchanged final PCG+RK2. Component-gradient
logging was disabled and compiled velocity was enabled for both; thus the
diagnostic traversals and compile mismatch in the earlier S02 pointwise run do
not apply. The explicit PSNR gate was skipped for these experiments (reported
as null), so PSNR is diagnostic only. No other settings were swept.

| Training PD-LK at paired sites | Dice | HD95 mm | Folded voxels | Endpoint PSNR moving/fixed dB | Train/recover s | Total s |
|---|---:|---:|---:|---:|---:|---:|
| Shared pooled velocity | .71915 | 6.315 | 0 | 31.22 / 30.81 | 62.55 / 11.56 | 78.62 |
| Independent pointwise optima | .71343 | 6.386 | 0 | 31.20 / 30.94 | 65.30 / 11.94 | 81.98 |
| Existing MSE + strict PD-LK baseline | .68592 | 6.933 | 0 | 33.12 / 32.56 | 36.92 / 17.49 | 62.92 |
| Existing ANTs SyN CC | .71810 | 6.626 | 0 | — | — / 260.90 | 267.44 |

Both new runs converged on all 16 final PCG intervals (maximum relative
residual 1.9760e-5 shared, 1.9992e-5 independent), with no folded voxels.
Minimum Jacobians were .06954 and .02369, respectively. The shared variant
scored highest on this S03 pair, 0.00105 Dice above ANTs SyN and 0.00572 above
the independent variant; these are single-seed comparisons, not cohort
evidence. Shared training was about 2.8 s faster. The old MSE+PD-LK baseline
used its passing 25 dB endpoint gate, whereas the two new runs explicitly
skipped that gate. The ANTs result used the same LPBA label metrics and had an
approximate, non-converged forward inversion recorded by its runner. Existing
artifacts are under `results/s03_shared_pdlk_ncc_20261111/` and
`results/s03_independent_pdlk_ncc_20261111/`; baseline artifacts remain under
`results/benchmark_20261110/`.

## Matched compiled S08 comparison

The same two settings were run on LPBA S08 with seed 42: MSE endpoint fitting,
strict PD-LK weight .1, local NCC registration weight .05, 8³/two-pair
central-80% starts, half-size crop and unchanged PCG+RK2 recovery. Compiled
velocity was enabled and component-gradient logging disabled in both. The PSNR
gate was explicitly skipped (null), and the observed endpoint PSNR is
diagnostic only.

| Training PD-LK at paired sites | Dice | HD95 mm | Folded voxels | Endpoint PSNR moving/fixed dB | Train/recover s | Total s |
|---|---:|---:|---:|---:|---:|---:|
| Shared pooled velocity | .69977 | 6.586 | 0 | 29.92 / 30.58 | 62.45 / 11.48 | 78.64 |
| Independent pointwise optima | .69453 | 6.642 | 0 | 29.95 / 30.60 | 62.52 / 12.05 | 79.29 |
| Existing MSE + strict PD-LK baseline | .68864 | 6.669 | 0 | 31.67 / 32.39 | 37.94 / 17.80 | 73.04 |
| Existing ANTs SyN CC | .70883 | 6.451 | 0 | — | — / 276.00 | 284.92 |

The shared and independent runs converged on all 16 final PCG intervals (maximum
relative residual 1.99584e-5 and 1.95838e-5); minimum Jacobians were .12368 and
.08224. Both beat this S08 MSE+PD-LK baseline in Dice, but remained below ANTs
SyN and had slightly higher HD95. The ANTs runner recorded an approximate,
non-converged forward inversion. The prior S03 shared-over-independent ordering
repeated, though with small single-case margins; this is still only two pairs
and one seed, not cohort evidence. Baseline metrics are existing results;
no baseline was rerun. Outputs are under `results/s08_shared_pdlk_ncc_20261111/`
and `results/s08_independent_pdlk_ncc_20261111/`.

## NCC endpoint objective on S02 and S03

Endpoint fitting used global minibatch NCC plus fixed zero-mean/unit-variance
moment matching (weight .01), with shared-grid PD-LK (.1), local NCC
registration (.05), and unchanged PCG+RK2 recovery. The comparison below
reports registration outcomes only.

| Case | Endpoint objective | Dice | HD95 mm | Folded voxels | Train/recover s | PCG |
|---|---|---:|---:|---:|---:|---|
| S02 | MSE | .71441 | 6.251 | 0 | 61.90 / 11.41 | converged |
| S02 | NCC + moments | .71441 | 6.328 | 2 | 63.28 / 8.88 | converged |
| S03 | MSE | .71915 | 6.315 | 0 | 62.55 / 11.56 | converged |
| S03 | NCC + moments | .71148 | 6.652 | 2 | 64.21 / 9.14 | converged |

The S02 Dice was unchanged; S03 Dice was lower by .00767. HD95 increased by
.077 mm on S02 and .337 mm on S03. Outputs and exact configs are under
`results/s02_shared_pdlk_endpoint_ncc_moments_20261111/` and
`results/s03_shared_pdlk_endpoint_ncc_moments_20261111/`.

## Relative pair offsets and adaptive three-tap smoothing check

After changing the pair offset to .125 of the smallest physical cell width
(symmetric sites .25 cell widths apart) and the three-tap Gaussian width to one
velocity-grid sample, the packaged LPBA config was trained on S02 (seed 42,
5,000 steps) and evaluated with unchanged PCG+RK2 recovery. The PSNR gate was
explicitly skipped; observed endpoint PSNR was 29.94/30.62 dB (moving/fixed).

| Run | Dice | HD95 mm | Folded voxels | Train/recover s | Total s | PCG intervals |
|---|---:|---:|---:|---:|---:|---|
| Relative offset + adaptive three-tap | .71003 | 6.260 | 0 | 62.06 / 12.14 | 78.78 | 15/16 converged |
| Historical S02 MSE + shared-grid PD-LK + local NCC | .71441 | 6.251 | 0 | 61.90 / 11.41 | 78.64 | 16/16 converged |

The historical .71441 run used `pair_radius_mm=2` (4 mm between points) and
`sigma_mm=4`. The new run uses a relative pair offset and `sigma_cells=1`, so
this is not a pair-distance-only comparison. The configs otherwise match on
seed 42, endpoint/objective settings, optimizer schedule and recovery settings;
the saved `dataset.subjects` differs, but both commands selected only S02 via
`--limit 1`. The current run's Dice is lower by .00438 and HD95 higher by .00849
mm. One PCG interval missed tolerance narrowly (relative residual 2.0131e-5
against 2e-5); the other 15 converged. Treat this as a scale-refactor check,
not evidence isolating either scale change. Artifacts are in the ignored
`results/relative_sampling_s02_20261111/` directory.

A follow-up changed only `pair_offset_fraction` to .25 (pair separation .5 of
the smallest cell width, 4.9375 mm here). The other training and recovery
settings and seed were unchanged.

| Offset fraction from center | Separation mm | Dice | HD95 mm | Folded voxels | Endpoint PSNR moving/fixed dB | Train/recover s | Total s | PCG intervals |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| .125 | 2.46875 | .71003 | 6.260 | 0 | 29.94 / 30.62 | 62.06 / 12.14 | 78.78 | 15/16 converged |
| .2 | 3.9500 | .71134 | 6.267 | 0 | 30.38 / 30.90 | 62.53 / 11.49 | 78.65 | 16/16 converged |
| .25 | 4.9375 | .70993 | 6.270 | 0 | 30.33 / 30.84 | 62.76 / 11.36 | 78.94 | 15/16 converged |

Increasing the pair separation by 2x made negligible difference on this S02
seed: Dice changed by -0.00010 and HD95 by +0.010 mm. In this run one interval
again missed the PCG tolerance narrowly (maximum relative residual 2.0147e-5
against 2e-5); the PSNR gate remained explicitly skipped. This does not establish
that either spacing is preferable. Artifacts are in the ignored
`results/relative_sampling_offset025_s02_20261111/` directory.

An additional .2 fraction run gave a 3.95 mm separation. It scored Dice .71134
and HD95 6.267 mm with no folding; all 16 PCG intervals converged (maximum
relative residual 1.9900e-5). Training/recovery took 62.53/11.49 s, and endpoint
PSNR was 30.38/30.90 dB with the gate still skipped. Compared with .125, Dice
was higher by .00131 and HD95 by .00768 mm; these small single-case differences
do not establish a preferred setting. Artifacts are in the ignored
`results/relative_sampling_offset020_s02_20261111/` directory.

All three new run configs share seed 42 and the same code path; the .125, .2
and .25 configs differ only in `pair_offset_fraction`. The historical run's
manifest records the same Git revision but also `dirty=true`; provenance stores
no working-tree content hash, so it cannot establish byte-identical source
between the historical and new runs.
For the .2 run, the exact difference from the historical Dice is -0.00307 and
HD95 is +0.01617 mm. Its 3.95 mm point separation nearly matches the historical
4 mm, but its smoothing config differs, so the change cannot be assigned to
pair separation alone.

## Three-tap smoothing-width screen

With `pair_offset_fraction=.2`, seed 42 and every other setting unchanged,
three-tap smoothing widths .5 and .75 were compared with the 1.0 run. These
correspond to nominal Gaussian widths of 4.9375, 7.4063 and 9.875 mm along the
crop's shortest axes, respectively.

| sigma_cells | Dice | HD95 mm | Folded voxels | Endpoint PSNR moving/fixed dB | Train/recover s | PCG intervals |
|---:|---:|---:|---:|---:|---:|---|
| 1.0 | .71134 | 6.267 | 0 | 30.38 / 30.90 | 62.53 / 11.49 | 16/16 converged |
| .75 | .71268 | 6.222 | 0 | 30.29 / 30.76 | 61.83 / 11.42 | 15/16 converged |
| .5 | .71211 | 6.276 | 0 | 30.14 / 30.64 | 61.98 / 11.39 | 15/16 converged |

The smaller kernel changed Dice by +.00077 and HD95 by +.00852 mm, with no
folding. One PCG interval narrowly missed tolerance (maximum relative residual
2.0056e-5 against 2e-5). Endpoint gates were explicitly skipped in both runs;
runtime was similar. These single-seed results do not establish a preferred
kernel width. Artifacts are in the ignored
`results/relative_sampling_sigma05_s02_20261111/` directory.

The intervening `sigma_cells=.75` run scored Dice .71268 and HD95 6.222 mm,
with zero folds and one interval narrowly outside PCG tolerance (maximum
relative residual 2.0123e-5). Compared with .5, Dice increased by .00057 and
HD95 decreased by .05373 mm. Training/recovery took 61.83/11.42 s; endpoint
PSNR was 30.29/30.76 dB with the gate skipped. This single-seed result does not
establish superiority, but the user selected .75 for the packaged default.
Artifacts are in the ignored
`results/relative_sampling_sigma075_s02_20261111/` directory.
