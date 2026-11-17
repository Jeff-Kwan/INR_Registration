# Frozen 32-pair development cohorts

Prepared 17 November 2026 for hyperparameter selection. **128 fits total**, with
32 pairs per dataset, frozen seed 42. No fitting or recovery was launched.
Use the same pairs for every candidate configuration; do not redraw based on scores.
These are development results, separate from final comparison tables.

| Dataset | Selection | Distinct subjects |
|---|---|---:|
| OASIS | 32 directed pairs from L2R IDs0001–0364, no endpoint reused | 64 |
| ACDC | 32 training patients001–100; one phase direction each,16 ED→ES and16 ES→ED | 32 |
| LPBA40 | 32 distinct unordered pairs covering all25 author training subjects | 25 |
| IXI | Atlas moving→32 distinct author Train subjects fixed | 33 including shared atlas |

All test subjects are excluded; the IXI atlas remains shared as required by the
atlas-to-subject task. All58 IXI validation subjects and5 LPBA validation subjects
are reserved. OASIS excludes0365–0414, which match the Neurite evaluation subjects;
see [the exact subject-overlap audit](TUNING_DATA.md#confirmed-oasis-overlap).
OASIS development inputs use the L2R release, so their preprocessing differs from
the preserved Neurite release. LPBA subjects appear in either two or three pairs
(11 subjects twice,14 three times), avoiding concentrated reuse.

Selection uses independent NumPy PCG64 streams seeded with SeedSequence([42,index])
for OASIS/ACDC/LPBA/IXI indices0/1/2/3. LPBA uses a random25-edge cycle and7 extra
edges with distinct endpoints, then randomizes orientation and order. ACDC patients
are selected randomly without disease stratification; only phase direction balances.
No labels or registration scores influenced pair selection.

## Prepared inputs and provenance

The [frozen manifest](../Data/Tuning32/manifest.json) contains the ordered pairs,
all372 input-file SHA-256 hashes, shapes/affines, source-manifest hashes, protocol
identity and numerical dependency versions. [Pairs CSV](../Data/Tuning32/pairs.csv)
provides the same ordered128 pairs with repository-relative image/label paths.

OASIS, IXI and LPBA use existing input files without changes. OASIS requires the
existing L2R adapter's independent minmax at fitting; preserve IXI/LPBA intensities.
ACDC has64 prepared phase MRIs and64 GT volumes in
`Data/ACDC/prepared_training_tuning32`, with a separate provenance manifest.
It reuses `scripts.prepare_acdc.preprocess` directly: ES header spacing for both
phases,1.5×1.5×3.15mm array rescale, cubic/edge MRI, nearest/edge labels, independent
full-image minmax before128×128×min(Z,32) center crop, no padding.
Labels remain evaluation-only. Future tuning scores should use the common
Dice, HD95, NPJ and SDlogJ definitions in [DATASETS.md](DATASETS.md).

The official evaluation runners still enforce their frozen test membership.
The dedicated `medical.tuning_method` adapter validates only the frozen development
pairs; it calls the same canonical fitting/recovery functions and common array
evaluator. No official validator or test cohort is relaxed.

## Preparation and verification

Preparation script: [prepare_tuning_cohort.py](../scripts/prepare_tuning_cohort.py).
`--plan-only` prints the deterministic selection without preparing data.
Preparation refuses to replace an existing frozen cohort. Verify before reuse:

```bash
CUDA_VISIBLE_DEVICES='' /venv/main/bin/python -B scripts/prepare_tuning_cohort.py --verify-only --workers 4
```

All372 referenced volumes passed finite/nonconstant-image and image/label/pair
geometry checks. Selection tests reject test-subject substitutions and repeated
ACDC patients. Source hashes and selected ACDC source hashes were checked.

At the canonical5,000-update recipe, prior measured case times imply approximately
19 minutes of training or21 minutes including recovery/scoring with8 RTX5090 GPUs.
Allow **25–30 minutes per configuration** for scheduling/setup overhead. This is
an estimate from previous runs, not a new timing benchmark.

Selected ACDC disease-group counts: DCM=6, HCM=7, MINF=6, NOR=7, RV=6.

## Tuning launcher

[launch_tuning.sh](../scripts/launch_tuning.sh) accepts one complete canonical-format
JSON configuration per invocation. Planning is the default and creates no outputs:

```bash
bash scripts/launch_tuning.sh \
  --config medical/configs/default.json \
  --output results/tuning/canonical --plan-only
```

After explicit authorization, replace `--plan-only` with `--run`. Use `--gpus`
(default0–7) and optionally `--datasets oasis acdc lpba40 ixi`. All comparisons
should use the same selected datasets/pairs and seed42. The OASIS name here denotes
the L2R development inputs, not a new Neurite evaluation cohort. Each invocation
requires a fresh output directory outside Data and preserves a configuration
snapshot, plan, per-case logs/checkpoints/metrics, progress.json and summary.json.
All frozen inputs and protocol dependencies are checksum-verified before dispatch.
Each pair receives a fresh model/optimizer; one persistent process serves each GPU.

Summaries report per-dataset means/population SDs and valid denominators for Dice,
HD95, NPJ and SDlogJ; training/recovery/evaluation and wall times are recorded.
Failed cases remain visible and return a nonzero exit status. The equal-dataset
macro Dice is emitted only after every intended pair has a valid Dice result.
The automated search ranks categories by this equal-dataset Dice summary;
[the search protocol](TUNING_SEARCH.md) specifies final promotion criteria.

Validation: six CPU selection/synthetic/mocked tests passed, including plan-only
side effects, dispatch/config snapshots, normalization, map direction and scratch
cleanup after scoring failure. No GPU training or saved-model recovery was run
to validate this launcher. Search execution status is tracked in [TUNING_SEARCH.md](TUNING_SEARCH.md).

## Agreed candidate list — 20 sweeps plus one combined candidate

All candidates use the complete128-pair cohort. Constants and time-varying weights
are grouped together as schedules. Each candidate changes only its listed schedule
or scalar from candidate1. The user authorized all20 sweeps plus one combined-candidate run, with128 fits
each (2688 fits total). All20 configurations are generated in the search output,
and captured training supports the requested PD-LK cosine weight schedule.

| ID | Group | Setting |
|---:|---|---|
| 1 | Reference | Current default |
| 2 | PD-LK weight schedule | Cosine0.3→0.1 |
| 18 | PD-LK weight schedule | Constant0.05 |
| 19 | PD-LK weight schedule | Constant0.2 |
| 3 | Registration weight schedule | Cosine increase0.01→0.03 |
| 4 | Registration weight schedule | Cosine0.05→0.01 |
| 16 | Registration weight schedule | Constant0.01 |
| 17 | Registration weight schedule | Constant0.03 |
| 20 | Registration weight schedule | Constant0.02 |
| 5 | Inter-SIREN parameter L2 | 0 |
| 6 | Inter-SIREN parameter L2 | 0.1 |
| 7 | Inter-SIREN parameter L2 | 1.0 |
| 8 | PD-LK c | 2 |
| 9 | PD-LK c | 6 |
| 10 | PD-LK c | 8 |
| 11 | Moment weight | 0.001 |
| 12 | Moment weight | 0.002 |
| 13 | Moment weight | 0.01 |
| 14 | LR schedule | Cosine1e-4→1e-5 |
| 15 | LR schedule | Historical LR-only OneCycle2.5e-5→1e-4→2.5e-5 |

PD-LK c=4 is already candidate1. Default PD-LK schedule is constant0.1;
registration is cosine0.03→0.01; parameterL2 is0.01; moment is0.005;
LR is cosine1e-4→2e-5. Constants are schedules, not separate factor groups.

### Restored OneCycle candidate

[15_onecycle.json](../medical/configs/tuning/15_onecycle.json) changes only the
learning-rate schedule and final factor. It reproduces the historical source
`HEAD:medical/registration/optimization.py` and original
`HEAD:medical/configs/native_siren_onecycle_5k.json` LR settings, without restoring
any historical model, objective, sampling, optimizer or integrator branches.

For r=(step−1)/4999, rise from0.25 to1 with a half-cosine for r<0.1,
then decay from1 to0.25 with a half-cosine over the remaining90%. Multiply by
peak LR1e-4. The continuous peak is at10%; discrete update501 is closest to it.
AdamW betas stay(0.9,0.999); there is no momentum cycling or objective warmup.
The scalar schedule updates the same live CUDA LR tensor used by captured training.

The5000-update summed LR is0.3124625, versus0.3 for the default cosine.
Both schedule shape and endpoints/integrated LR differ, so this is a historical
recipe comparison, not an isolated causal test of warmup. Default.json remains
unchanged. Inspect candidate15 without fitting:

```bash
bash scripts/launch_tuning.sh --config medical/configs/tuning/15_onecycle.json --output results/tuning/15_onecycle --plan-only
```

[Automated rankings, combination and final promotion protocol](TUNING_SEARCH.md).
