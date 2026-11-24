# Baseline integration

This directory contains comparison adapters, configs, setup tooling and ignored
upstream source checkouts. Keep it separate from development of SIRENPath.
Read ../AGENTS.md, ../README.md and README.md before editing here. On Vast, read
the complete /workspace/AGENTS.md and use /venv/main/bin/python.

Use the pinned authors' implementations in upstream/. Never substitute our own
losses, networks, solvers or recovery routines for theirs. Record required source
patches explicitly; sources.json pins exact revisions. Retained-method restrictions
in the root guide describe SIRENPath, not these authors' baseline algorithms.

Do not execute registration, training, optimizer steps or saved-model recovery
without an explicit request. The comparison launchers default to planning and
setup --check only verifies imports. Tests must mock all native optimizers.
Never persist deformation fields or warped images, including inside checkpoint
dictionaries. Remove native temporary logs and checkpoints on success and
exceptions. Never commit upstream checkouts, data or generated artifacts.
Read AUDIT.md for source and paper discrepancies and keep recipe limitations in
metrics.
