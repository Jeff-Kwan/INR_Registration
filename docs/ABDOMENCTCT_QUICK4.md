# ABDOMENCTCT frozen quick4 subset

Canonical cohort and scoring: [DATASETS.md](DATASETS.md). Current method and
recipe: [METHOD.md](METHOD.md). This subset does not select a different recipe.

Frozen pair/case indices: **2,8,18,41**. Selection: official45-pair Learn2Reg public-validation list; selection seed42.
The matching checked-in JSON under medical/protocols records the definition.
Do not redraw or choose cases based on scores. The full-cohort definition remains
unchanged. This is a development subset, not an independent test benchmark.

Preview only, without training or workers:

```bash
/venv/main/bin/python -m medical.abdomen_experiments --methods ours --subset quick4
```

Preparation and this subset do not authorize experiments. Use a single seed;
training requires an explicit user request. Older quick4 experiment reports are
archived and their configurations are unsupported.
