# Development and experiment protocol

## Current implementation

The maintained research question is whether relaxed pointwise PD-LK gives a
more useful weight-interpolated image trajectory than standard PD-LK, and under
what conditions. The two pipelines share the paired-SIREN and local differential
transport convention:

- `cameraman/` is the fast 2D synthetic prototype. It contains standard (`rho=0`)
  and relaxed (`rho=0.4`) configs and the bundled Cameraman image.
- `medical/` is the 3D registration implementation. `medical/configs/lpba_pdlk.json`
  is the relaxed default and `lpba_pdlk_standard.json` is its matched `rho=0`
  control. Both point to the existing local prepared data under `Data/`.
- Final deformation recovery is a separate global optical-flow and RK2 stage.
  Medical labels are used only for evaluation.

The medical training config preserves the completed run's architecture and
optimization settings: width 256, four sine layers, 5,000 steps, batch 32,768,
learning rate `2e-4` with final factor `0.25`, 4,096 PD-LK points, `c=4`,
`lambda=0.1`, `eta=1e-6`, and 16 analytical recovery intervals. Derivative
construction may compile on CUDA; the complete optimizer step is not compiled.
Cameraman uses eager analytical derivatives.

The 2D fitter has one objective: endpoint MSE plus weighted pointwise PD-LK.
Its public options cover relaxation, loss/sampling weights, initialization and
optimization schedules. Setting `lk_weight=0` gives an endpoint-only control.
Older objectives survive only as small mathematical references in
`cameraman/historical.py` and conceptual records in `historical_methods/`.
Oracle and cross-fade trajectories remain evaluation references.

Both runners reject nonempty output directories. The medical config loader
supports recognized saved pointwise run configs without modifying
saved files: it rebases old `Data/` paths to the current repository and removes
only unused pointwise patch-sampling fields. Actual retired patch/H1 objectives
remain unsupported by the maintained medical pipeline.

3D pointwise PD-LK is the only maintained medical objective. The full H1
training snapshot and old patch configuration were removed from the maintained
source tree after their formulation and evidence were recorded in
[`historical_methods/LPBA_H1_SWEEP.md`](historical_methods/LPBA_H1_SWEEP.md).

## Existing sweep evidence

See [curated results](BEST_RESULTS.md) for the retained four-hour LPBA sweep,
matched controls, convergence limitations and provenance. The evidence does not
establish a benefit from positive relaxation over standard PD-LK.

## Method questions to answer next

1. Derive how the local affine penalty and the `rho` attenuation transform under
   image-intensity scaling and spatial coordinate rescaling. Keep the derivation
   consistent with native-pixel/voxel units and the explicit ridge.
2. Explain when independent pointwise velocity elimination is a useful surrogate
   for a spatially coupled deformation objective, and identify its failure modes
   in low-texture regions.
3. Establish whether relaxation changes the learned trajectory or only changes
   optimization conditioning. Use endpoint-gated synthetic comparisons and oracle
   trajectory/recovery references before interpreting LPBA Dice.
4. Investigate LPBA global-flow convergence separately from the local PD-LK loss;
   preserve convergence diagnostics even when label metrics are favorable.
5. Choose seed counts and comparison panels before selecting another value of
   `rho`. Do not promote a single best seed to a default.

## Experiment protocol

For a new method claim, first check equations, derivative axes, units, limits,
and gradients on tiny synthetic inputs. Then compare matched `rho=0` and
positive-`rho` configurations on Cameraman, keeping architecture, sampling,
initialization and recovery fixed. Require endpoint PSNR at least 30 dB and
report any folding; separate representation, trajectory, flow and map errors.

Only transfer a controlled configuration to LPBA after those checks. Report
all nine moving subjects (S02–S10), mean Dice and HD95 under the recorded MONAI
definition, endpoint gate status, folding, and flow convergence. Include seeds,
config hashes, elapsed time, and failures. Use fresh output folders. For the
four-hour sweep, the current code prioritizes the `rho=0` and `rho=0.4` controls,
uses `rho` as the public coordinate, disables volumes throughout, and disables
checkpoints for screening/confirmation. Full/repeat candidates retain
checkpoints. Plan-only is safe and does not launch the trainer.
Later-stage deduplication ignores subject panels and artifact-retention flags,
while keeping repeat seeds distinct. A trainer launch error is recorded as a
failed attempt instead of leaving its status at running.

All run/results directories are ignored by Git. The LPBA dataset and archives
are local-only. Never create a run by reusing or overwriting an existing output
folder. Do not include data, volumes, checkpoints, credentials, or local result
folders in any publication archive.

## Non-training validation

Run the unit suites from the repository root:

```bash
python -m unittest discover -s cameraman/tests -t . -v
python -m unittest discover -s medical/tests -t . -v
```

The medical pipeline's synthetic recovery tests run only tiny in-memory cases;
they do not call `fit_pair`, optimizer steps, or read LPBA. To inspect command
interfaces, use `python -m cameraman.experiments.run_experiment --help`,
`python -m medical.registration.train --help`, and
`python -m medical.experiments.hyperparameter_sweep --help`.

The medical suite also drives the sweep controller with a fake trainer and
clock to check control priority, retention, deduplication, deadlines, launch
failure recording and duplicate-lock handling. No real trainer process starts.
Packaging uses regular packages rather than implicit namespace discovery so
ignored local output directories cannot become distribution packages; both
pipelines' default JSON configs and the Cameraman image are included.

An agent must not start training, sweeps, benchmarks, or saved-model recovery
unless the user explicitly asks in the active session. A plan-only sweep is
non-training; write its temporary files outside the repository and remove them
after inspection.

## Selected pipeline checks (2026-10-28)

User-authorized full 5,000-step runs exercised the current Cameraman relaxed
config and LPBA subject S02. Local provenance is under
`medical/results/selected_checks_20261028T082950Z/`; the 2D run is
`cameraman/runs/selected_check_20261028T082950Z/`.

LPBA `rho=0`, seed 43 reproduced every saved S02 metric except elapsed time
from sweep trial `033_full` exactly: Dice 0.698521, HD95 6.6295 mm, endpoint
PSNR 32.05/32.55 dB, zero folding. The relaxed default (`rho=0.4`, seed 42)
scored Dice 0.699745, HD95 6.5803 mm and endpoint PSNR 32.14/32.39 dB,
with zero folding. Both completed registration and retained the known flow
convergence warning. These different-seed, single-subject checks do not rank
methods. No LPBA volumes or checkpoints were saved; dataset hashes remained
unchanged.

Cameraman training and recovery completed with source/target PSNR
28.37/30.44 dB, INR map EPE 4.289 px and zero learned-map folding. The source
missed the 30 dB gate, consistent with earlier runs; this default is not a
qualified performance result. Plotting initially failed because the environment
lacked the declared Matplotlib dependency. Installing it and rendering the
saved tensors completed both the comparison PNG and animation without refitting.
