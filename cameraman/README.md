# 2D Cameraman prototype

This package is the fast synthetic development path for paired-SIREN PD-LK. It
uses the bundled `data/cameraman.jpg` and a known elastic deformation so
trajectory fitting and global flow-recovery errors can be inspected separately.

From the repository root, run `python -m cameraman.experiments.run_experiment`
with `--config cameraman/configs/pdlk_relaxed.json`; use
`cameraman/configs/pdlk_standard.json` for the exact
`rho=0` control. Use fresh paths under the ignored `runs/` directory.

See the root [README](../README.md), [method](../docs/METHOD.md), and
[development protocol](../docs/DEVELOPMENT.md). Historical H1, patch and
metamorphosis work is summarized in `docs/historical_methods/`, with small
mathematical references in `cameraman/historical.py`. The maintained fitter uses
paired SIRENs, endpoint MSE and pointwise PD-LK only. `--lk-weight 0` disables
the regularizer for an endpoint-only control; `--lk-affine-rho 0` gives standard
PD-LK. Alternate architectures and historical training options are rejected.
