# Per-case Dice advantage analysis

Script: [rank_case_advantage.py](../scripts/rank_case_advantage.py).
It reads completed per-case metric JSON and run metadata only; no checkpoint
loading, recovery, image processing, CUDA or training occurs.

For each directed case, compute the arithmetic mean Dice across four external
methods, then `advantage = (our_Dice - external_mean_Dice) / external_mean_Dice`.
Positive and negative values are retained. Percentage advantage is100×this ratio,
not a Dice percentage-point difference. The denominator belongs to the same case,
not the dataset-wide average. A zero denominator is undefined and raises an error.

Groups:

- Classical: ConvexAdam, FireANTs, Demons, NiftyReg, equal method weights.
- Neural: ccIDIR, NIR, NODEO, DNVF, equal method weights. The default neural root
  is `baselines/results/official_neural_20261116`, which contains the corrected
  NIR MRI/FP32 and DNVF TF32-off reruns, including the four repaired NIR ACDC cases.
  Superseded NIR/DNVF outputs are not used.

The script reports the highest and lowest signed advantage separately against
each group in each dataset. Every exact tie is retained. It exports all309 case
rows too, including baseline-specific Dice, group means, pair indices and endpoint
paths for later inspection. Both extremes may be positive or both may be negative.

Matching validates directed endpoint paths, label sets, standardized_v1 scoring,
preparation hashes and complete intended cohorts. Failed/undefined Dice values
raise an error; no silent method dropping, zero imputation or partial-run ranking.
The output includes hashes of all input metric files and plan/summary metadata.

The SIRENPath result root is mandatory: choose the original or manual-promotion
run when ready. These examples have **not** been executed:

```bash
# Original default, registration cosine0.03→0.01:
/venv/main/bin/python -B scripts/rank_case_advantage.py \
  --ours results/official_ours_cosine_20261116 \
  --output results/case_advantage_original

# Manually promoted constant registration0.02 (wait for complete results):
/venv/main/bin/python -B scripts/rank_case_advantage.py \
  --ours results/official_ours_reg002_20261117 \
  --output results/case_advantage_reg002
```

Optional `--datasets` selects among neurite_oasis,acdc,lpba40,ixi.
`--classical` and `--neural` can override external result directories explicitly.
Output must be a fresh directory outside Data. Outputs: `REPORT.md` with extrema,
`cases.csv` with all cases, and `advantage.json` with full results/provenance.

Validation: three synthetic tests passed for signed/zero-denominator handling,
separate group means and tied extrema, and failed/mismatched case rejection.
No real-result advantage analysis has been run.

## Completed original-default analysis —17 November2026

The user restored registration cosine0.03→0.01 and authorized analysis of the
original default official results. All309 cases matched successfully against
both four-method groups, including current corrected NIR/DNVF results.

[Extrema report](../results/case_advantage_original_20261117/REPORT.md) ·
[All-case CSV](../results/case_advantage_original_20261117/cases.csv) ·
[Full results/provenance](../results/case_advantage_original_20261117/advantage.json).
