# DIR-Lab registration gap investigation

## Saved-run cross-dataset comparison

The native-grid full-method suite (`results/full_method_suite/06_native_grid/`) is
paired by case/pair ID with the completed external baseline run
(`baselines/results/three_baselines/`). The suite predates the current DIR-Lab
crop-center-only sampling policy. Its native config equals the current
`medical/configs/default.json`, but the implementation and sampling policy
changed afterward. Baseline methods use their audited native recipes. All entries
use seed 42. W/T/L counts compare our method against the named baseline on
exactly the scored pairs. Delta means ours minus baseline, so lower DIR-Lab TRE
and HD95 and higher Dice are favorable. The three Abdomen NIR evaluation retries
are included from `baselines/results/nir_abdomen_eval_retry/`.

| Dataset and metric | Baseline | Pairs | Ours | Baseline | Delta | W/T/L |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| DIR-Lab forward rounded TRE, mm | ConvexAdam | 9/10 | 4.2072 | 1.4102 | +2.7970 | 0/0/9 |
| DIR-Lab forward rounded TRE, mm | NIR | 9/10 | 4.2072 | 1.5486 | +2.6586 | 1/0/8 |
| DIR-Lab forward rounded TRE, mm | FireANTs | 9/10 | 4.2072 | 4.0342 | +0.1730 | 3/0/6 |
| OASIS Dice | ConvexAdam | 19/19 | 0.8072 | 0.7633 | +0.0439 | 19/0/0 |
| OASIS HD95, voxels | ConvexAdam | 19/19 | 1.8523 | 2.3260 | -0.4737 | 19/0/0 |
| OASIS Dice | NIR | 19/19 | 0.8072 | 0.7089 | +0.0983 | 19/0/0 |
| OASIS HD95, voxels | NIR | 19/19 | 1.8523 | 3.1032 | -1.2509 | 19/0/0 |
| OASIS Dice | FireANTs | 19/19 | 0.8072 | 0.7437 | +0.0635 | 19/0/0 |
| OASIS HD95, voxels | FireANTs | 19/19 | 1.8523 | 2.4643 | -0.6120 | 19/0/0 |
| AbdomenCTCT Dice | ConvexAdam | 45/45 | 0.3922 | 0.4128 | -0.0206 | 12/0/33 |
| AbdomenCTCT HD95, voxels | ConvexAdam | 45/45 | 17.6492 | 16.4884 | +1.1608 | 15/0/30 |
| AbdomenCTCT Dice | NIR | 45/45 | 0.3922 | 0.1033 | +0.2889 | 43/0/2 |
| AbdomenCTCT HD95, voxels | NIR | 42 finite/45 | 17.6429 | 44.7550 | -27.1121 | 41/0/1 |
| AbdomenCTCT Dice | FireANTs | 45/45 | 0.3922 | 0.3543 | +0.0379 | 39/0/6 |
| AbdomenCTCT HD95, voxels | FireANTs | 45/45 | 17.6492 | 18.1551 | -0.5059 | 30/0/15 |

DIR-Lab case09 failed in the native suite, while all three external baselines
scored 10/10. The three nonfinite NIR Abdomen HD95 values are excluded only from
the finite paired HD95 row. Its official 45-pair HD95 mean is infinite. Case
counts and metrics are not pooled across datasets. The suite outputs record the
config and case inputs, but do not record a source commit. The run directory and
the later quick4 directory identify distinct code generations.

## Native DIR-Lab case severity

The next table uses the older native-suite run and its matched ConvexAdam case.
Initial TRE and registered TRE each average all 300 landmarks. Spacing and image
shape come from that case's saved `run.json`. The current crop-center-only
quick4 result is shown separately below.

| Case | Shape XYZ | Spacing XYZ, mm | Initial TRE | Suite TRE | ConvexAdam TRE |
| --- | --- | --- | ---: | ---: | ---: |
| 01 | 256×256×94 | 0.97, 0.97, 2.50 | 3.892 | 1.312 | 0.982 |
| 02 | 256×256×112 | 1.16, 1.16, 2.50 | 4.338 | 1.561 | 0.766 |
| 03 | 256×256×104 | 1.15, 1.15, 2.50 | 6.943 | 2.745 | 1.024 |
| 04 | 256×256×99 | 1.13, 1.13, 2.50 | 9.830 | 3.548 | 1.365 |
| 05 | 256×256×106 | 1.10, 1.10, 2.50 | 7.477 | 4.111 | 1.435 |
| 06 | 512×512×128 | 0.97, 0.97, 2.50 | 10.891 | 4.325 | 1.333 |
| 07 | 512×512×136 | 0.97, 0.97, 2.50 | 11.026 | 5.898 | 1.485 |
| 08 | 512×512×128 | 0.97, 0.97, 2.50 | 14.995 | 11.049 | 2.895 |
| 10 | 512×512×120 | 0.97, 0.97, 2.50 | 7.301 | 3.315 | 1.406 |

| Current quick4 case | Initial TRE | Current TRE | ConvexAdam TRE |
| --- | ---: | ---: | ---: |
| 01 | 3.892 | 1.209 | 0.982 |
| 03 | 6.943 | 2.661 | 1.024 |
| 04 | 9.830 | 3.350 | 1.365 |
| 05 | 7.477 | 3.205 | 1.435 |
| Mean | 7.036 | 2.606 | 1.202 |

All values in these two case tables are forward rounded TRE in mm. The apparent
cross-case increase with initial motion is descriptive. The current-policy
10-case results and frozen-checkpoint landmark analysis appear below.

## Existing mechanism evidence

- Training-decoder replay on the current quick4 checkpoints gave 3.319 mm
  continuous TRE over the covered landmarks. Global PCG on the same landmarks
  gave 2.638 mm. The crop-averaged local decoder gave 3.300 mm. The learned
  trajectory is already inaccurate before global recovery.
- ConvexAdam-oracle trajectories gave 1.199 mm direct rounded TRE. Known velocity
  with RK2 gave 1.209 mm. Brightness PCG using transport-consistent derivatives
  gave 1.215 mm, while separately differenced rendered frames gave 2.192 mm.
  All 64 PCG intervals converged in the paired derivative tests. The consistency
  of image derivative evidence matters even when the linear solver converges.
- Current quick4 ablations gave 2.606 mm with both PD-LK and box NCC, 2.989 mm
  with PD-LK only, and 6.512 mm with box NCC only. Each run completed 64/64 PCG
  intervals. Thus endpoint reconstruction and registration NCC alone did not
  recover the anatomy on these cases.

The next diagnostic was signed physical displacement along each landmark's
observed T00→T50 motion, perpendicular error, and signed z error, with counts
by initial motion bin. The user explicitly authorized frozen-checkpoint recovery
and training runs; the replay results appear below.

## Prepared frozen-checkpoint replay

`scripts/diagnose_dirlab_gap.py` prepares the next diagnostic for current native
quick4 checkpoints. It restores each frozen model, recovers both maps in
exception-safe temporary scratch, checks continuous and rounded TRE against the
saved metric within 0.01 mm, and reports physical along-motion, perpendicular,
and signed-z errors in the predefined motion bins. It also replays 256 ordinary
crop draws per case to count central particle-region exposure and samples 9 mm
endpoint windows for local correlation at observed and predicted matches. The
windows use a 10×10×10 grid at 1 mm spacing and flag low variance. Output is
scalar JSON only. A synthetic translation, anisotropic-spacing, image-correlation,
and crop-coverage check passed without loading checkpoints.

Use a fresh ignored output path, for example:

```bash
/venv/main/bin/python -m scripts.diagnose_dirlab_gap \
  --cases 1 3 4 5 --device cuda:0 \
  --output results/dirlab_gap_diagnostic_20261113
```

The replay completed on quick4. Its TRE parity gate reproduced all saved forward
and reverse continuous and rounded TRE within 0.01 mm, and recovery scratch was
cleaned. The current worktree and saved checkpoint config define this replay.

## Observed motion distribution before registration

The official 300 T00/T50 landmark correspondences yield the following initial
physical-displacement counts. These are read-only measurements from the prepared
landmark CSVs and image affines. They do not measure model error within a case.

| Case | <2 mm | 2–5 mm | 5–10 mm | ≥10 mm | 90th percentile, mm | Mean absolute z motion, mm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 01 | 53 | 148 | 76 | 23 | 7.81 | 3.56 |
| 02 | 99 | 93 | 60 | 48 | 10.07 | 3.78 |
| 03 | 14 | 93 | 101 | 92 | 12.92 | 6.33 |
| 04 | 5 | 52 | 80 | 163 | 15.71 | 9.39 |
| 05 | 34 | 86 | 77 | 103 | 15.36 | 6.68 |
| 06 | 12 | 54 | 87 | 147 | 21.51 | 10.02 |
| 07 | 5 | 62 | 97 | 136 | 22.59 | 10.24 |
| 08 | 4 | 36 | 76 | 184 | 27.59 | 13.52 |
| 09 | 19 | 67 | 98 | 116 | 12.87 | 6.53 |
| 10 | 40 | 102 | 67 | 91 | 17.18 | 6.64 |

Case08 has 184/300 landmarks moving at least 10 mm and the largest 90th
percentile motion, while case01 has only 23/300 in that bin. All cases use
2.5 mm through-plane spacing. On case08 the mean signed z motion is 12.80 mm,
close to its mean absolute z motion of 13.52 mm. Recovered landmark predictions
are required to determine whether errors grow with motion within each case.

## Current quick4 crop exposure, without recovery

A read-only sampler replay used the current native config, seed 42, 256 crop draws
per case, the prepared T00/T50 lung masks, and the normal 32768 reconstruction
indices per draw to preserve crop RNG consumption. A landmark counts as exposed
when its start lies in the crop's central 80% particle region. Each direction
replays the same crop sequence, with its own landmark starts and targets.

| Case | Forward landmarks exposed | Forward median crops | Forward observed-target exits / exposed crop observations | Reverse landmarks exposed | Reverse median crops | Reverse observed-target exits / exposed crop observations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 01 | 299/300 | 44.5 | 0/13579 | 299/300 | 46.0 | 0/13947 |
| 03 | 299/300 | 42.0 | 0/12878 | 299/300 | 43.0 | 0/13147 |
| 04 | 300/300 | 48.0 | 3/14539 | 300/300 | 48.0 | 144/14820 |
| 05 | 298/300 | 46.5 | 10/14470 | 298/300 | 48.0 | 52/14911 |

Every case's ≥10 mm landmarks have median forward exposure of 29–53 crops.
The four forward exposed counts exactly match the earlier decoder replay,
which independently used 256 crops. Even the largest reverse observed-target
exit rate is 0.972% of exposed crop observations (case04). This measures whether
the *observed target landmark* lies within a sampled crop, not whether the
learned trajectory exits it. It cannot establish how error varies with exposure
until per-landmark predictions are recovered.

## Observed-image local NCC control

A read-only endpoint-image diagnostic sampled identical 9 mm physical windows
(10 samples per axis, 1 mm apart) around the fixed landmark and either its true
moving correspondence or its unchanged voxel location. The images used the
adapter's independent min/max normalization and border interpolation. All 300
windows per case passed the low-variance threshold. This control tests whether
observed image patches contain correspondence signal. It does not use model
predictions or measure the training loss.

| Case | Forward true NCC | Forward identity NCC | True better, forward | Reverse true NCC | Reverse identity NCC | True better, reverse |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 01 | 0.830 | 0.366 | 272/300 | 0.830 | 0.370 | 271/300 |
| 03 | 0.827 | 0.180 | 291/300 | 0.827 | 0.195 | 293/300 |
| 04 | 0.766 | 0.060 | 296/300 | 0.766 | 0.091 | 298/300 |
| 05 | 0.759 | 0.207 | 277/300 | 0.759 | 0.191 | 279/300 |

The true correspondences usually score well above identity, especially in
cases 03–04. The pending checkpoint replay will compare true matches directly
with the model's predicted matches, which is the relevant test for choosing an
NCC-weight or similarity change.

## Frozen-checkpoint motion diagnostic

Source: `results/dirlab_gap_diagnostic_20261113/`, using the four current native
quick4 checkpoints, each with 300 landmarks. All four forward and reverse maps
passed the 0.01 mm saved-TRE parity gate. Results retain scalar values only, and
all recovery scratch was cleaned. Signed along-motion error is negative when the
model travels less far along the observed motion. The local NCC difference is
true correspondence minus model-predicted correspondence in 9 mm windows.

| Forward initial motion bin | Landmarks | Initial motion, mm | TRE, mm | Along error, mm | Perpendicular error, mm | True−predicted NCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| <2 mm | 106 | 0.851 | 0.976 | -0.516 | 0.820 | -0.034 |
| 2–5 mm | 379 | 2.984 | 1.340 | -0.776 | 0.705 | +0.023 |
| 5–10 mm | 334 | 6.493 | 1.912 | -1.343 | 0.895 | +0.073 |
| ≥10 mm | 381 | 13.263 | 5.020 | -4.556 | 1.400 | +0.285 |

Reverse trends agree. For its ≥10 mm group, mean along error is -4.574 mm,
perpendicular error 1.393 mm, and true−predicted NCC +0.258. The large-motion
shortfall is primarily along the correct motion direction. Local image evidence
at the observed match exceeds the model prediction. Crop replay and the
coverage-stratified TRE tables are in the result JSON. Coverage is not an
obvious universal explanation: case03 and case05 high-motion groups retain
large errors even in the higher-coverage half, whereas case04 benefits more
from exposure. This is an association, not a causal crop intervention.

The first controlled training change is `medical/configs/ablations/dirlab_native_registration_003.json`.
It copies the current native config and changes only registration NCC weight
from 0.01 to 0.03. The current ordinary-crop quick4 run is the matched control.

## Native registration NCC weight 0.03 quick4 trial

The first launch stopped before fitting because `/venv/main` lacked its declared
MONAI dependency. After installing MONAI 1.6.1 and verifying the case adapter
import, the identical config ran in the fresh
`results/dirlab_quick4_native_reg003_retry_20261113/` directory. It differs
from the current native quick4 control only in registration NCC weight 0.01→0.03.
All four completed seed 42 and 5000 steps. The original startup failure output
remains separate under `results/dirlab_quick4_native_reg003_20261113/`.

| Case | Control forward rounded TRE | NCC 0.03 forward rounded TRE | Control reverse rounded TRE | NCC 0.03 reverse rounded TRE |
| --- | ---: | ---: | ---: | ---: |
| 01 | 1.209 | 1.180 | 1.208 | 1.192 |
| 03 | 2.661 | 2.478 | 2.621 | 2.457 |
| 04 | 3.350 | 2.535 | 3.375 | 2.596 |
| 05 | 3.205 | 2.832 | 3.294 | 2.890 |
| Mean | 2.606 | 2.256 | 2.625 | 2.284 |

Mean forward continuous TRE fell 2.635→2.252 mm. All 64/64 PCG intervals
converged. Lung folding remained below 0.001% on each case. The trial passed the
predeclared quick4 promotion gate, improving mean forward rounded TRE by
0.350 mm and all four cases. A second frozen-checkpoint replay also passed the
saved-TRE parity gate. For the 381 forward landmarks with ≥10 mm initial motion,
mean along-motion error shrank from -4.556 to -3.370 mm, perpendicular error
from 1.400 to 1.294 mm, and true-minus-predicted local NCC from +0.285 to
+0.216. The reverse large-motion group improved similarly. This supports
stronger registration supervision as an effective current-code quick4 change,
while leaving a substantial displacement shortfall.

Full-cohort confirmation uses the existing current-policy quick4 control/trial
and runs the six other cases (02,06,07,08,09,10) under each frozen config. The
DIR-Lab launcher now accepts an explicit `--cases` list for this completion;
`--subset quick4` remains the frozen quick4 selection.

## Completed current-policy 10-case comparison

The four quick4 cases are the original matched runs above. Cases 02,06–10 were
trained and evaluated with seed 42 under each config in
`results/dirlab_native_remaining6_control_20261113/` and
`results/dirlab_native_remaining6_reg003_20261113/`. All checkpoints are frozen
during evaluation. ConvexAdam is the independent saved external baseline.
Scores below are forward **rounded** TRE in mm, 300 landmarks per case.

| Case | Native control | Registration NCC 0.03 | ConvexAdam |
| --- | ---: | ---: | ---: |
| 01 | 1.209 | 1.180 | 0.982 |
| 02 | 1.367 | 1.258 | 0.766 |
| 03 | 2.661 | 2.478 | 1.024 |
| 04 | 3.350 | 2.535 | 1.365 |
| 05 | 3.205 | 2.832 | 1.435 |
| 06 | 4.337 | 3.878 | 1.333 |
| 07 | 5.637 | 5.281 | 1.485 |
| 08 | 10.123 | 9.075 | 2.895 |
| 09 | 3.192 | 2.995 | 1.327 |
| 10 | 3.429 | 3.079 | 1.406 |
| Mean | 3.851 | 3.459 | 1.402 |

The stronger registration NCC term helps all 10 cases and both directions;
mean reverse rounded TRE falls 3.964→3.568 mm. Every PCG solve converged
(160/160 for each arm), and maximum lung folding among the 10 cases is
0.00168% for control and 0.00043% for the trial. This is a repeatable
single-seed, paired-case improvement, but it closes only 0.392 mm of the
2.449 mm control-to-ConvexAdam mean gap. Case08 remains the largest failure.

## Mechanism: learned temporal image change is too weak on large motion

`scripts/diagnose_dirlab_transport.py` evaluates the *frozen* SIREN at 33 times
on each observed moving→fixed landmark segment. At each time it measures the
analytic temporal derivative `q = ∂I/∂t` and the derivative required to carry
the observed displacement `d`, namely `−∇I·d`. It reports the uncentered
least-squares slope of `q` on `−∇I·d`, their Pearson correlation, and RMS
transport residual `q + ∇I·d`. The derivative is in native voxel coordinates;
the landmark grouping uses physical mm. The observed straight segment is an
approximation to an unknown path, so the slope is a mechanistic diagnostic,
not a measured ground-truth velocity or a training target.

| Case | ≥10 mm landmarks | Control slope | Trial slope | Control residual RMS | Trial residual RMS |
| --- | ---: | ---: | ---: | ---: | ---: |
| 01 | 23 | 0.830 | 0.899 | 0.227 | 0.177 |
| 02 | 48 | 1.028 | 1.024 | 0.202 | 0.180 |
| 03 | 92 | 0.865 | 0.878 | 0.276 | 0.245 |
| 04 | 163 | 0.838 | 0.941 | 0.433 | 0.312 |
| 05 | 103 | 0.764 | 0.870 | 0.388 | 0.287 |
| 06 | 147 | 0.938 | 0.961 | 0.282 | 0.259 |
| 07 | 136 | 0.609 | 0.630 | 0.339 | 0.312 |
| 08 | 184 | 0.366 | 0.614 | 0.531 | 0.414 |
| 09 | 116 | 0.819 | 0.848 | 0.196 | 0.193 |
| 10 | 91 | 0.829 | 0.881 | 0.415 | 0.313 |

The full-cohort table uses 33 temporal probes for every case; scalar results
are in `results/dirlab_transport_{control,reg003}_{quick4,remaining6}_33_20261113.json`.
Case08's severe-case slope was also 0.375→0.611 at 9 times, versus
0.366→0.614 at 33 times, so the result is stable to temporal sampling.
Case08 derivative correlation increases from 0.606 to
0.796 with stronger registration NCC; the derivative mismatch remains large.
The 33-time trapezoid integral of the residual matches the actual endpoint
model-intensity change to within 0.004 in the control and 0.002 in the trial.

The case08 frozen-map replay gives a direct anatomical consequence. For its 184
forward landmarks with at least 10 mm initial motion (mean 20.89 mm), mean TRE
is 14.63→12.94 mm and signed along-motion error is −14.00→−12.00 mm; mean
perpendicular error is only 2.69→2.78 mm. True-minus-predicted local NCC is
+0.627→+0.570. All 300 landmarks were exposed in 64 crop draws; observed
targets left the crop only 35 times among the exposed forward observations.
The reverse direction shows the same under-travel. These data implicate the
learned image trajectory and the registration signal for large displacement.
They do not yet prove whether endpoint SIREN fitting, parameter interpolation,
the local training decoder, or the global recovery field contributes most.

### Training-time decoder check on the severe case

The prior quick4 training-decoder replay is extended to case08 using each frozen
checkpoint, 64 ordinary seed-42 crops, the configured random local derivative
probes and RK2 trajectories, and the same matched 300 landmarks. The disposable
drivers and scalar outputs are in `results/dirlab_case08_decoder_{control,reg003}_20261113/`.
The PCG map was independently recovered from each same checkpoint; both runs
reproduced saved continuous TRE, converged all 16 PCG intervals, and cleaned
recovery scratch.

| Case08 forward continuous TRE | Control | NCC 0.03 |
| --- | ---: | ---: |
| Local decoder, mean single-crop prediction | 11.451 | 10.464 |
| Local decoder, averaged prediction | 11.410 | 10.407 |
| Global PCG on same 300 landmarks | 10.123 | 9.014 |

All 300 landmarks were covered, median 20 crop observations each. Only 0 and 1
active landmark trajectories respectively left crop support across 64 draws.
The local decoder is *worse* than global recovery on this severe case, as on
quick4. The case08 failure is therefore already present in the learned
trajectory under the decoder used for training. PCG is not the dominant source
of the large-motion error, though it may still limit final precision.

### Endpoint images lose local lung structure on larger cases

`scripts/diagnose_dirlab_endpoint_fidelity.py` compares each frozen endpoint
SIREN with the corresponding observed CT in 9 mm physical windows centered on
all 300 landmark points. It also compares moving/fixed patch correspondence at
the true landmark pair. This uses the same image normalization and border
extension as training and does not use the landmarks to fit a model. The mean
of moving and fixed same-position model-versus-observed patch NCC is:

| Case | Image shape XYZ | Control endpoint patch NCC | NCC 0.03 endpoint patch NCC | Observed true-pair NCC | Control model true-pair NCC |
| --- | --- | ---: | ---: | ---: | ---: |
| 01 | 256×256×94 | 0.789 | 0.787 | 0.830 | 0.840 |
| 02 | 256×256×112 | 0.712 | 0.710 | 0.840 | 0.831 |
| 03 | 256×256×104 | 0.746 | 0.750 | 0.827 | 0.813 |
| 04 | 256×256×99 | 0.793 | 0.792 | 0.766 | 0.798 |
| 05 | 256×256×106 | 0.750 | 0.748 | 0.759 | 0.750 |
| 06 | 512×512×128 | 0.584 | 0.578 | 0.754 | 0.704 |
| 07 | 512×512×136 | 0.551 | 0.543 | 0.762 | 0.649 |
| 08 | 512×512×128 | 0.564 | 0.560 | 0.767 | 0.667 |
| 09 | 512×512×128 | 0.690 | 0.683 | 0.769 | 0.763 |
| 10 | 512×512×120 | 0.618 | 0.611 | 0.785 | 0.766 |

All patches pass the low-variance check. For the control, the last 25 logged
training crops average global endpoint NCC 0.985–0.992 across cases, including
0.985/0.986 on case08. That high crop score hides the much weaker local
reconstruction at the points where registration is judged. Case08 true-match
image NCC is 0.767, but the modeled endpoints retain only 0.667. The same
registration-weight increase that improves TRE leaves local endpoint fitting
almost unchanged. Thus the local structure loss is a distinct, unresolved
bottleneck. The five 512×512 cases tend to fit worse than the five 256×256
cases under the same width-256 SIRENs, 5000 steps and globally pooled NCC;
case09 is a useful counterexample to a deterministic size-only rule.

The endpoint probe outputs are in
`results/dirlab_endpoint_fidelity_{control,reg003}_{quick4,remaining6}_20261113.json`.
The model outputs remain in memory; no reconstructed volume is saved.

### First endpoint-objective intervention: GW-NCC did not fix the gap

`medical/configs/ablations/dirlab_native_endpoint_gwncc.json` changes only endpoint
reconstruction from global NCC to the supported 50/50 global/window GW-NCC
with its default 8³ moment bins. Case01 and case08 completed seed42, 5000
steps and all 16 PCG intervals each. The same 9 mm endpoint patch diagnostic
and forward rounded TRE give:

| Case | Global NCC patch fit | GW-NCC patch fit | Global NCC TRE | GW-NCC TRE |
| --- | ---: | ---: | ---: | ---: |
| 01 | 0.789 | 0.743 | 1.209 | 1.357 |
| 08 | 0.564 | 0.522 | 10.123 | 10.369 |

The off-the-shelf GW-NCC windows are about 31 mm wide in X/Y on case08, much
larger than the 9 mm vessel-scale patches used here. This run establishes that
adding those windows did not recover the missing local details; it does not
exclude a more local objective or a larger endpoint representation. Results:
`results/dirlab_endpoint_gwncc_case01_case08_20261113/` and
`results/dirlab_endpoint_fidelity_gwncc_case{01,08}_20261113.json`.

### Second endpoint-objective intervention: MSE with auxiliary losses

`medical/configs/ablations/dirlab_native_endpoint_mse.json` changes only endpoint NCC
to MSE and removes the now-inapplicable NCC moment options. It retains PD-LK
weight 0.1 and registration NCC weight 0.01. Case08 completed
seed42 and 5000 steps. Its last crop endpoint MSEs are 0.00059/0.00054,
but the moving/fixed 9 mm patch NCCs are only 0.257/0.268, down from
0.562/0.566 with NCC. The model true-correspondence patch NCC falls
0.667→0.375. Thus a very low pooled intensity loss does not imply faithful
landmark-scale anatomy in this larger volume. The 198,657-parameter endpoint
SIREN and random crop samples can fit broad intensity structure while missing
smaller lung features; this is a plausible explanation, not a controlled
capacity test.

Forward rounded TRE nevertheless improves 10.123→9.160 mm, although it is
slightly worse than the NCC-0.03 trial's 9.075 mm. This means local endpoint
fidelity is **not the sole causal explanation** of TRE. The MSE checkpoint's
high-motion transport slope is 0.918, but its RMS temporal derivative is only
0.049 (versus 0.402 for NCC) and its RMS required spatial transport signal is
only 0.041 (versus 0.668). Both derivatives have largely collapsed, so that
slope cannot be read as faithful transport. All 16 PCG intervals hit the
500-iteration cap, versus 16/16 converged under NCC; the result is a diagnostic
and not a recommended preset. Results:
`results/dirlab_endpoint_mse_case08_20261113/`,
`results/dirlab_endpoint_fidelity_mse_case08_20261113.json`, and
`results/dirlab_transport_mse_case08_20261113.json`.

## Standard MSE reconstruction-only quick4 baseline

At the user's request, cases 01,03,04,05 were trained from initialization with
only `source_mse + target_mse`. Both auxiliary weights are zero. The native
width-256, four-layer, omega-30 SIRENs use seed42, 5000 Adam steps, 32768
reconstruction samples, learning rate 0.0002 decaying by factor 0.25 and the
usual equal-endpoint lung-mask crop-center sampler. Voxel sampling within each
crop is uniform. The masks select crop centers and define supplemental lung
evaluation. Intensities use the existing independent image min/max scaling.

The config is `medical/configs/ablations/dirlab_native_reconstruction_mse.json`.
`medical.reconstruction_method` uses the shared `fit_pair` training path and
streams exact whole-volume and lung-region reconstruction metrics. Every one
of the 1000 logged steps across the four runs contains only endpoint MSE terms,
and its total equals their sum. The registration-loss branch is skipped.
The results are in `results/dirlab_quick4_reconstruction_mse_20261113/`.

All table values average the moving and fixed endpoint scores within each case.
The last row averages the four case scores. PSNR uses normalized data range 1.
Patch NCC uses the existing 9 mm physical landmark-window diagnostic.

| Case | Whole-volume MSE | Whole-volume PSNR, dB | Lung PSNR, dB | Lung patch NCC |
| --- | ---: | ---: | ---: | ---: |
| 01 | 0.00006013 | 42.42 | 45.05 | 0.722 |
| 03 | 0.00004168 | 43.87 | 45.69 | 0.703 |
| 04 | 0.00018336 | 37.37 | 38.90 | 0.821 |
| 05 | 0.00075487 | 31.22 | 37.26 | 0.791 |
| Mean | 0.00026001 | 38.72 | 41.72 | 0.759 |

The existing normalization maps image maxima ranging from 4095 to 16414 in
quick4 to 1, so PSNR differences across cases also reflect their intensity
ranges. Local patch NCC measures spatial structure independently of affine
intensity scaling. `reconstruction_slices.png` shows observed T00, reconstructed
T00 and absolute error at each case's median lung-mask z coordinate. Both
observed and reconstructed views use the observed lung slice's 1st–99th
percentile display limits. Full per-endpoint metrics, checkpoint config,
training logs, patch scores and the aggregate `summary.json` are retained.

### Why the pure MSE baseline loses landmark structure

The frozen-model blur probe in `blur_diagnostic.json` compares reconstruction
patches with observed CT patches after Gaussian smoothing in physical units.
On case03, mean endpoint patch NCC rises from 0.703 against the original CT
to 0.839 against CT smoothed with sigma 2 mm. The median reconstructed patch
contrast is only 70–73% of the observed contrast. The fit loses fine spatial
structure even with all auxiliary objectives removed.

Input extrema also change the effective finite-budget reconstruction problem.
For case03 T50, only five of 6,815,744 voxels exceed 4095, but the maximum is
16414. For T00, eight voxels exceed 4095 and the maximum is 13374. The image's
99.99th percentiles are 2339 and 2337. These few extremes set the existing
min/max divisor, compressing lung feature contrast. Relative to a divisor of
4095, a given raw-intensity error contributes 16.1 times less squared error
at T50 and 10.7 times less at T00.

A controlled case03 reconstruction run changes only normalization to division
by 4095, without clipping any values. It uses the same pure MSE config, seed42,
architecture, crop samples and 5000-step optimizer schedule. Mean moving/fixed
landmark-patch NCC increases from **0.703 to 0.788** (moving 0.700→0.785,
fixed 0.707→0.791). NCC is invariant to affine intensity scaling, so this is
an improvement in fitted spatial structure. This experiment identifies
normalization scale as a causal contributor under the current training budget.
The residual smoothing remains to be explained by optimization budget and
representation capacity. Results and the frozen diagnostic adapter source are
in `results/dirlab_reconstruction_scale4095_case03_20261113/`.

### Six-layer reconstruction-only SIREN

The user requested six hidden layers under the same reconstruction-only recipe.
`medical/configs/ablations/dirlab_native_reconstruction_mse_6layers.json` differs from the
original four-layer pure-MSE baseline only in `model.layers: 6`. Width remains
256, omega 30, seed 42, steps 5000, batch size 32768 and both auxiliary weights
zero. This comparison uses the original independent min/max normalization.
Parameters increase from 198657 to 330241 per endpoint.

All four cases completed. All 1000 logged training records across the four runs
contain only the two endpoint MSE terms and their sum. The same 9 mm landmark
patch diagnostic gives the following moving/fixed endpoint averages.

| Case | Four-layer patch NCC | Six-layer patch NCC | Four-layer lung PSNR | Six-layer lung PSNR |
| --- | ---: | ---: | ---: | ---: |
| 01 | 0.7221 | 0.8064 | 45.05 | 46.23 |
| 03 | 0.7035 | 0.7930 | 45.69 | 46.83 |
| 04 | 0.8208 | 0.8821 | 38.90 | 40.25 |
| 05 | 0.7909 | 0.8626 | 37.26 | 38.90 |
| Mean | 0.7593 | 0.8360 | 41.72 | 43.05 |

Lung MSE decreases from 0.00009426 to 0.00006733, a 28.6% reduction. Exact
whole-volume MSE increases from 0.00026001 to 0.00031948, driven by higher
outside-lung error in cases 04 and 05. Mean whole-volume PSNR remains about
38.72 dB. The mean training time rises from 26.17 to 32.15 seconds per pair.
The improved lung structure under the same step budget demonstrates that
representation depth contributes to the reconstruction limitation. This is a
single-seed development comparison.

Checkpoints, logs and per-endpoint metrics are in
`results/dirlab_quick4_reconstruction_mse_6layers_20261113/`. Its
`comparison.json` contains the full paired comparison and `patch_fidelity.json`
contains the frozen landmark-patch scores.

### Six-layer reconstruction with 65536 points per minibatch

The user requested doubling the six-layer pure-MSE minibatch from 32768 to
65536 points. This is the only change in
`medical/configs/ablations/dirlab_native_reconstruction_mse_6layers_64k.json`. The four
frozen quick4 cases completed 5000 steps, seed 42, and both auxiliary weights
zero. The original min/max normalization and crop policy are retained. Changing
the number of sampled indices also changes subsequent crop RNG consumption.

| Case | Six-layer 32k patch NCC | Six-layer 64k patch NCC |
| --- | ---: | ---: |
| 01 | 0.8064 | 0.8132 |
| 03 | 0.7930 | 0.7969 |
| 04 | 0.8821 | 0.8913 |
| 05 | 0.8626 | 0.8692 |
| Mean | 0.8360 | 0.8427 |

Mean training time rises from 32.15 to 51.38 seconds per pair, about 60%.
Whole-volume MSE improves from 0.00031948 to 0.00030497, whereas lung MSE
increases from 0.00006733 to 0.00007028. Lung PSNR changes from 43.05 to
42.91 dB. Patch NCC improves slightly on all cases, with a smaller gain than
the earlier four-to-six-layer change. All 1000 training-log records contain
only the two endpoint MSEs and their sum.

Results, permitted checkpoints and the full paired `comparison.json` are in
`results/dirlab_quick4_reconstruction_mse_6layers_64k_20261113/`.

### Four hidden layers with width 512 and 32768 points

The user requested returning to 32768 points and four hidden layers while
increasing width to 512. The config
`medical/configs/ablations/dirlab_native_reconstruction_mse_width512.json` differs from
the original four-layer pure-MSE recipe only in width. The model has 790529
parameters per endpoint. All four cases completed seed 42, 5000 steps and
zero auxiliary loss weights with the original independent min/max scaling.
Every logged total equals the two endpoint MSE terms' sum.

| Case | Four layers, width 256 | Six layers, width 256 | Four layers, width 512 |
| --- | ---: | ---: | ---: |
| 01 | 0.7221 | 0.8064 | 0.8276 |
| 03 | 0.7035 | 0.7930 | 0.8281 |
| 04 | 0.8208 | 0.8821 | 0.9125 |
| 05 | 0.7909 | 0.8626 | 0.8899 |
| Mean patch NCC | 0.7593 | 0.8360 | 0.8645 |
| Mean training seconds | 26.17 | 32.15 | 51.73 |

The width-512 model's mean lung PSNR is 43.72 dB, versus 43.05 for the
six-layer width-256 model. Lung MSE falls from 0.00006733 to 0.00005566,
while whole-volume MSE rises slightly from 0.00031948 to 0.00032573.
At similar training time, width 512 with 32k points also exceeds the previous
width-256 six-layer 64k run's patch NCC of 0.8427. All these results use one
seed and the frozen quick4 development selection.

Results and the paired `comparison.json` are in
`results/dirlab_quick4_reconstruction_mse_width512_20261113/`.

### Four hidden layers with width 384

The user requested width 384 with four hidden layers. The config
`medical/configs/ablations/dirlab_native_reconstruction_mse_width384.json` changes only
width relative to the original pure-MSE baseline. All four quick4 cases
completed 5000 steps, seed 42, batch 32768 and zero auxiliary weights. There
are 445441 parameters per endpoint. Logged losses contain only the two endpoint
MSEs and their sum. The original min/max normalization is retained.

| Case | Width-384 patch NCC |
| --- | ---: |
| 01 | 0.7812 |
| 03 | 0.7669 |
| 04 | 0.8816 |
| 05 | 0.8584 |
| Mean | 0.8220 |

Mean training time is 38.31 seconds per pair. Mean lung PSNR is 42.84 dB,
lung MSE is 0.00007010 and whole-volume MSE is 0.00022868. Whole-volume MSE
is the lowest of the four tested 32k architectures. The six-layer width-256
model has higher lung patch NCC (0.8360) and lower training time (32.15 s),
while the four-layer width-512 model retains the highest lung patch NCC
(0.8645, 51.73 s). These are single-seed quick4 development comparisons.

All results and the four-architecture `comparison.json` are in
`results/dirlab_quick4_reconstruction_mse_width384_20261113/`.

## Mechanism-led improvement plan

1. **Repair endpoint structure and test the causal link.** The patch probe above
   establishes weak endpoint fidelity on the severe cases. Default 8³ GW-NCC
   and global MSE both worsen it, so prioritize a scale-aware endpoint
   representation or a genuinely local/structure-focused reconstruction signal,
   keeping evaluation labels out of training. For each proposal, measure patch
   fidelity and true-pair contrast before derivative residual, decoder TRE and
   recovered TRE. If endpoint structure improves without displacement
   improving, the temporal path is a separate bottleneck. The MSE result shows
   that improving TRE without improving endpoint structure is also possible.
2. **Test the capture-range hypothesis with a mechanism-changing training
   intervention.** Use a coarse-to-fine, spatially localized endpoint matching
   signal during training so large lung motion can be matched before fine-scale
   local flow refinement. Keep native units, the current masks-as-crop-centers
   policy, paired SIRENs, PD-LK, live gradients and PCG+RK2 recovery. Start
   with the frozen quick4; include case08 as a mandatory stress case. Compare
   signed along-motion error, true-minus-predicted patch NCC, transport
   residual, training-decoder TRE, and PCG TRE, not only final case mean.
3. **If local matching does not fix the image path, test a motion-aware temporal
   construction.** Preserve the fitted endpoints and change how intermediate
   images are produced or constrained so anatomical structure is carried through
   time. The diagnostic target is a smaller `q + ∇I·v` residual along plausible
   trajectories without fading anatomy. A midpoint transport consistency term
   or deformation-conditioned image interpolation is more directly aimed at
   this failure than a new PCG tolerance or smoothing coefficient. Gate it on
   endpoint fidelity and the decoder-replay result, then compare on all ten
   single-seed cases if it improves case08 and quick4.

The first intervention should be judged by whether it recovers **displacement
amplitude** on high-motion landmarks while maintaining low folding and the
already-good small-motion performance. A better TRE alone cannot distinguish
the proposed mechanism from another generic weight change.

## Physical scale of the native velocity grid

The native preset keeps an 8³ local velocity grid over half of each native image.
Using actual crop voxel counts and image spacing, its cell-center widths are:

| Case group | Typical crop cell width X/Y | Crop cell width Z | Initial 90th-percentile motion |
| --- | ---: | ---: | ---: |
| 01–05 | 15.4–18.3 mm | 14.4–16.2 mm | 7.8–15.7 mm |
| 06–08,10 | 30.9 mm | 18.4–20.9 mm | 17.2–27.6 mm |

A larger local cell may limit the spatial variation learned on cases 06–10.
The earlier 120 mm / 12³ physical-grid suite used roughly 10 mm cells but did
not consistently improve those large cases: case06 scored 5.376 versus native
4.325 mm, case07 7.349 versus 5.898 mm, and case08 12.063 versus 11.049 mm.
Those suites changed multiple geometry and sampling choices together. Cell
resolution is therefore an open mechanism, not an established fix. The current
full-cohort NCC-weight comparison will first test whether stronger data evidence
helps large native cases without changing their grid.

## Follow-up convergence experiments

The [SIREN convergence investigation](SIREN_CONVERGENCE.md) records the subsequent
5k-versus-40k screen, checkpoint probes and cross-dataset controls. Its passing
original-frequency stack combines width 512, Adam LR 1e-4 and registration NCC
0.05; the remaining method and default configuration are retained. The report
includes the matched frequency-by-capacity comparison and an initial-update
diagnostic explaining why widening requires a smaller Adam rate.
