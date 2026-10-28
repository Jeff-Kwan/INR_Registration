# Paired-INR registration with PD-LK

This repository studies registration from a continuous image trajectory defined
by two endpoint SIRENs. The current methodological focus is standard and relaxed
pointwise differential Lucas–Kanade (PD-LK), with global optical-flow recovery
kept as a separate stage.

The workflow uses the bundled Cameraman image for quick 2D method development,
then evaluates the 3D pipeline on LPBA40. The prepared LPBA40 tree lives in the
repository's ignored `Data/` directory. It is never committed or uploaded.

## Layout

| Path | Purpose |
|---|---|
| `cameraman/` | 2D synthetic registration pipeline, standard/relaxed PD-LK configs, analysis and synthetic checks |
| `medical/` | 3D paired-SIREN registration, LPBA setup, sweep orchestration, analysis and checks |
| `docs/METHOD.md` | Equations, assumptions and coordinate conventions |
| `docs/DEVELOPMENT.md` | Current implementation, open questions and experiment protocol |
| `docs/RESEARCH_EVOLUTION.md` | Conceptual history and qualified evidence |
| `docs/BEST_RESULTS.md` | Curated result summaries and provenance |
| `docs/historical_methods/` | Short records for superseded methods |
| `Data/` | Existing prepared LPBA40 files, local-only and ignored by Git |
| `cameraman/runs/`, `medical/results/` | Local generated outputs; ignored by Git |

Start with [the method](docs/METHOD.md), then read [development notes](docs/DEVELOPMENT.md).
Agent navigation and safety instructions are in [AGENTS.md](AGENTS.md).

## Environment and use

Use an environment with PyTorch, NumPy, SciPy, Pillow, Matplotlib, tqdm,
NiBabel, and MONAI installed. On the supplied ML instance, `/venv/main` has the
PyTorch stack. Install this package without changing those dependencies:

```bash
/venv/main/bin/python -m pip install -e . --no-deps
```

Run the 2D relaxed PD-LK prototype from the repository root:

```bash
/venv/main/bin/python -m cameraman.experiments.run_experiment \
  --config cameraman/configs/pdlk_relaxed.json
```

For the exact standard-PD-LK control, use `cameraman/configs/pdlk_standard.json`.
Each run needs a fresh output directory; results are written under the ignored
`cameraman/runs/` tree.

The 3D config defaults to the existing prepared LPBA40 files under `Data/`:

```bash
/venv/main/bin/python -m medical.registration.train --help
/venv/main/bin/python -m medical.experiments.hyperparameter_sweep --help
```

The second command only shows sweep options. A sweep or registration run performs
training and must be started deliberately. Sweep screening and confirmation do
not save model checkpoints or volume files. Read [the experiment protocol](docs/DEVELOPMENT.md)
before launching either pipeline.

The LPBA setup tool accepts an already prepared dataset or authorized prepared
archive and refuses to overwrite existing destinations. The original raw-release
preprocessing recipe is unresolved; the tool does not guess it. See
[`medical/datasets/README.md`](medical/datasets/README.md).

## Non-training checks

```bash
/venv/main/bin/python -m unittest discover -s cameraman/tests -t . -v
/venv/main/bin/python -m unittest discover -s medical/tests -t . -v
```

These checks use synthetic inputs, analytical derivatives, local flow recovery,
and temporary dataset fixtures. They do not fit models or read LPBA images.
No experiment, benchmark, checkpoint, or volume output is part of the Git tree.
