# Choosing LPBA H1 hyperparameters

The reference sweep is
`Numerical_Results/hyperparameter_sweep/20261026T173311_657405Z/`.
It completed 30 three-subject screening runs (S02, S06, S10), followed by 10
full-panel confirmations on S02–S10 against fixed S01. Its 180 subject-pair fits
equal the budget of 20 nine-subject runs. This panel is not all 40 LPBA subjects.
Recorded total runtime was about 6 h 13 min.

## Recommended starting configuration

Use `configs/lpba_siren_h1_best.json`:

| Setting | Selected value |
|---|---|
| SIREN width / hidden layers / omega | 256 / 4 / 30 |
| Joint training steps / coordinate batch | 5,000 / 32,768 per endpoint |
| Initial LR / final factor | 2e-4 / 0.25 (exponential decay to about 5e-5) |
| Gradient clipping | 1 |
| H1 weight / gradient weight | 1 / 1 |
| Patch / shift radius / sampled center cube | 11³ / 1 / 32³ |
| Time separation | 0.2 |
| Frames / RK2 substeps per interval | 33 / 4 |
| Flow smoothness / ridge | 0.01 / 1e-6 |
| PCG maximum iterations / relative tolerance | 500 / 1e-5 |

## What the results support

Full-panel means (higher Dice, lower HD95 are better):

| Configuration | Dice | HD95, mm | Total runtime |
|---|---:|---:|---:|
| Baseline: 3 layers, LR 1e-4 constant, H1 weight 2 | 0.69040 | 5.46964 | 18.4 min |
| Baseline with H1 weight 1 | 0.69211 | 5.48237 | 18.0 min |
| Weight 1, initial LR 2e-4, final factor 0.25 | 0.69463 | 5.48029 | 18.2 min |
| Above with 4 layers (selected) | **0.69731** | **5.42629** | 22.2 min |

The selected model improved both metrics on every subject versus baseline,
passed endpoint PSNR checks on all nine, and had zero measured folding. It is
the only full-panel Pareto-front candidate: it has both the highest mean Dice
and lowest mean HD95 among confirmed candidates.

Screening-only findings are directional evidence, not full-panel confirmation:

- Removing H1 reduced Dice from 0.68913 to 0.61032 and increased HD95 from
  5.50866 to 6.50226 mm. Keep trajectory regularization.
- Radius zero gave Dice 0.62477. Keep radius one; larger radii were not tested.
- Patch sizes 3³ and 5³ gave Dice 0.67551 and 0.68327 versus 0.68913 for 11³.
  Smaller windows saved time but weakened performance on this panel.
- Time gaps 0.05, 0.1, and 0.3 gave Dice 0.64443, 0.68089, and 0.67240;
  retain 0.2 as the starting value.
- A 16³ center cube gave Dice 0.66567 versus 0.68913 for 32³. Reducing spatial
  support changes regularization coverage as well as cost.
- Gradient weights 0.25, 1, and 4 produced similar scores. Spend initial search
  budget on penalty strength, time gap, spatial support, and endpoint fitting.

H1 loss is not normalized by time gap. Changing the gap also changes effective
penalty magnitude; tune gap and weight together when exploring further.
Stronger penalty was not universally better, and combinations of independently
good settings did not automatically improve performance.

## Selection and interpretation

The sweep ranks equal-subject means using `Dice - 0.01 * HD95_mm`. One extra mm
of HD95 therefore costs 0.01 Dice; `--hd95-weight` changes that tradeoff.
Consult `pareto_frontier.json` when the metrics disagree. Undefined HD95
invalidates a candidate; do not substitute its finite-label-only mean.

Read `per_label_metrics.json` for anatomical detail and `flow_diagnostics.json`
for numerical convergence. No winning subject had every interval meet tolerance;
for S06 the maximum relative residual was 1.38e-5 against 1e-5, so its warning
was a near miss rather than evidence of gross solver failure. Check other
subjects individually rather than interpreting the boolean alone.

These are single-seed selection results on the same available nine subjects.
Mean improvements are modest: +0.00691 Dice and -0.04335 mm HD95 versus baseline.
Use repeated seeds and held-out subjects before claiming generalization.
