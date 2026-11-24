# Run the three baseline methods on all datasets

Run from `/workspace/INR_Registration`, using `/venv/main/bin/python`.
This launcher includes FireANTs SyN, ConvexAdam–MIND and NIR only. ccIDIR is
excluded from its method choices. Current audited configurations are preserved.

All three together have 222 fits. The configured seeds remain unchanged. Our method is not part of this launcher.

## Preview only

```bash
/venv/main/bin/python -m baselines.run_all --methods fireants convexadam nir --gpus 0 1 2 3 --output baselines/results/three_baselines
```

Without `--run`, this prints the job commands, counts, GPU IDs and missing inputs.
It does not create result directories or start registration processes. The plan
has been verified locally: 222 jobs, no missing inputs. All GPUs were idle after
preparation; no experiments were launched by preparing these scripts.

## Execute later

Combined command, sharing all four GPUs across the three methods:

```bash
cd /workspace/INR_Registration
/venv/main/bin/python -m baselines.run_all --methods fireants convexadam nir --gpus 0 1 2 3 --output baselines/results/three_baselines --run
```

Or run one method at a time, each using all four GPUs:

```bash
bash scripts/run_fireants_all.sh --gpus 0 1 2 3 --output baselines/results/fireants_all --run
bash scripts/run_convexadam_all.sh --gpus 0 1 2 3 --output baselines/results/convexadam_all --run
bash scripts/run_nir_all.sh --gpus 0 1 2 3 --output baselines/results/nir_all --run
```

The shell scripts resolve the repository location and use `/venv/main` explicitly. Omit `--run` from any command for a preview. Do not start multiple all-GPU launchers at the
same time: the combined command already shares a single four-worker pool.

## Scheduling and results

One worker owns each requested GPU and pulls the next independent case from a
shared queue. A case uses one GPU; four cases can run concurrently, including
cases of the same method. Each child sees its selected GPU as logical `cuda:0`.
OMP/MKL CPU threads are limited to four per process, matching the profile setup.
No source optimizer, inference or evaluation settings are changed by scheduling.

Everything persists under the selected `baselines/results/...` root:

- `configs/`: configuration snapshots frozen before launching any cases.
- `plan.json`: exact executed commands, GPU pool, seeds and upstream pins.
- `DATASET/METHOD/CASE/`: compact `metrics.json`, `run.json` and permitted checkpoint.
- `DATASET/METHOD/CASE.log`: native stdout/stderr, including failed runs.
- `progress.json`: completed-job records and intended count, updated atomically.
- `summary.json`: all return codes/times plus per-dataset/per-method metrics and
  scored/intended denominators. Failed cases do not prevent remaining jobs.

No deformation maps or warped images are retained. Native temporary artifacts
are cleaned by the audited adapters. Results are ignored by Git and excluded
from source packages. The full launcher has not been executed; its four-GPU
scheduling and failure accounting were tested with mocked subprocesses.
