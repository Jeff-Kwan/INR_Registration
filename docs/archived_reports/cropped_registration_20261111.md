# PD-LK-free half-box Euler2 registration-loss pilot

Completed one LPBA S02→S01 seed42, 5,000-step pilot. **PD-LK training loss was zero
throughout, including warm-up.** Crop sampling and masking were stable and faster,
but final registration essentially matched the existing unregularized MSE control.
No further coefficients, seeds, datasets or controls were trained.

## Fixed implementation and protocol

Preserve paired4×256 SIRENs, linear weight interpolation, MSE reconstruction, shared
initialization and original Adam/lr schedule. Set objective.weight=0; the base loss
never evaluates pointwise PD-LK energy. The local analytic 3×3 PD-LK-form velocity
solve still defines the registration surrogate, with c=4/eta=1e-6/rho=0; using that
formula is distinct from adding a PD-LK training loss. The CSV observer asserted
zero pdlk/weighted_pdlk at every logged step, including warm-up and activation.

Each minibatch samples a valid integer crop origin uniformly. Crop shape80×96×80
on the original160×192×160 grid. The32768 reconstruction sites are uniform voxel
centres in that crop. No images are resampled or rewritten. Original-image intensity
normalization and coordinate/axis conventions remain unchanged. A24³ velocity
lattice spans the crop, with original-volume derivative scales and coordinates.
Spacing3.435/4.130/3.435 mm; physical Gaussian sigma4 mm, compiled packed derivatives,
4096-point chunks and checkpoint recomputation. Euler2 queries times0,.5, totalling
27648 derivative queries per active pass. No global PCG or dense reconstruction is
used during training. Original final global PCG+RK2 recovery remains unchanged.

Registration activates after step2000, weight=.01, per-endpoint gradient budget=.1
relative to endpoint MSE, global clip1. This is an independently budgeted gradient,
not exact optimization of the nominal weighted loss sum. No NCC or PD-LK warm-up.

## Supported-particle mask

Avoid an external halo by sampling particles inside the supported crop interior.
Its margin is (ceil(4*sigma_lattice)+1) lattice cells: Gaussian support plus one
interpolation cell, about20.6 mm per face here. Thus registration starts are in an
inner region approximately37.8×53.7×37.8 mm, while reconstruction uses the full box.
This is stricter than merely rejecting particles outside the literal crop.

For initial a and both Euler endpoints x1,x2, use detached validity
m=1[a,x1,x2 all in the supported interior]. The retained trajectories remain live:
loss=sum(m*[M_data(a)-F_data(x2)]²)/sum(m). Rejected particles contribute no loss or
gradient; no partial trajectory is detached. Rejected intermediate queries use a
safe placeholder, whose output cannot contribute to the retained loss. Gaussian
reflection/border values therefore do not affect accepted trajectories.

If fewer than32/256 particles survive, skip the registration backward completely;
20 consecutive skipped batches abort. Nonfinite positions/gradients and original
reconstruction/runtime guards remain active. With valid particles entirely inside
the crop, no additional boundary penalty is applied. Masking has no escape penalty;
it creates survival-selection bias and is not a proof of valid/diffeomorphic motion.

## Results: standard fixed-grid PCG+RK2 evaluation

| Training | Dice | HD95 mm | Folded voxels / 4,915,200 |
|---|---:|---:|---:|
| Existing global MSE only | .634323 | 7.12078 | 5 |
| MSE crop + masked registration, PD-LK-free | .634556 | 7.05269 | 15 |
| Existing MSE + strict PD-LK | .698229 | 6.56515 | 0 |

MONAI1.6.1 macro56-label scores, no missing labels. Warped moving labels are evaluated
on the fixed grid; these are not the earlier moving-grid integrator-probe scores.
Dice improvement over the MSE-only control is only .000233; HD95 is .0681 mm lower,
with slightly more folding (.000305% versus .000102%). This does not establish a
useful registration gain. The PD-LK baseline remains clearly higher in Dice.

Final full-volume reconstruction PSNR31.383/31.856 dB, MSE .0007273/.0006522; both
pass25 dB. All16 PCG solves converged, max relative residual1.9953e-5 (tolerance2e-5).
Training129.21 s (~2m9s), standard recovery10.25 s; peak CUDA allocation684.1 MiB.
Active median37.54 ms; registration portion33.01 ms. First activation/compilation
is included in total time. Synthetic preflight warmed passes took36–38 ms.

The historical full-volume Euler2 augmentation cost248.1 s with Dice .698994, but
it included PD-LK and used global reconstruction. It is not a replacement control.
The MSE-only and PD-LK references are existing runs, not freshly matched-code fits.
Both objective and sampling differ from global MSE-only; no matched crop-only
control was trained, so we cannot isolate the registration term's causal effect.

## Stability, bias and limitations

Across150 logged active minibatches, median225/256 particles survived (87.9%),
range187–241 (73.0–94.1%). None of those logged batches skipped; unlogged individual
skip decisions were not persisted. No consecutive-skip guard fired. Whole-volume
sampled monitor minima28.62/29.34 dB; no reconstruction collapse was observed.

Both endpoint registration gradients were finite/nonzero at every active log.
Weighted raw norms reached3.84/.917, and budgets were active at all logged samples:
median gradient scales .00834/.0330. Hence this is a strongly budgeted loss; the
negative result does not prove that all crop-based registration objectives fail.
Logged unweighted registration loss median .0241, range .00266–.05896, with varying
crops and samples. It is not directly comparable to the earlier global-particle loss.

Uniform valid crop origins oversample central voxels; the supported interior further
restricts registration coverage. No importance correction or global reconstruction
mixture was added. Full-volume gates measure quality but do not remove this sampling
bias. Cropping, masking and parameter budgets all limit the interpretation.

## Checks and preservation

75 synthetic tests passed (74 medical, one Cameraman): global/crop coordinates,
original-volume indexing, exact zero contribution from a rejected particle, live
finite gradients to both endpoint SIRENs, skipped-backward behavior and PD-LK-free
configuration guards. Wheel audit48 entries; datasets/results/tests/archives excluded.
No additional optimizer smoke tests were run. No commit or push.

Input images, annotation files and all three protected earlier checkpoints have
unchanged hashes. All datasets remain available. Only final endpoint weights and
scalar evidence were saved; no deformation/velocity/warped volumes. Standard PCG
temporary scratch was cleaned and empty after recovery. Baseline presets unchanged.

Local run: results/cropped_gaussian_registration/20261111/; final checkpoint under
run/siren_gaussian_crop_registration/S02/endpoints.pt. Exact profile/train runners,
config, source hashes and scalar logs remain there; compact evidence accompanies
this report. The next experiment requires a new explicit user request.
