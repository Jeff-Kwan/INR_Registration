# Baseline integration

This directory contains comparison adapters, configs, setup tooling and ignored
upstream source checkouts. Keep it separate from development of our method.
Read ../AGENTS.md, ../README.md and README.md before editing here. On Vast, read
the complete /workspace/AGENTS.md and use /venv/main/bin/python.

Use the pinned authors' implementations in upstream/. Never substitute our own
losses, networks, solvers or recovery routines for theirs. Record required source
patches explicitly; sources.json pins exact revisions. Retained-method restrictions
in the root guide describe our method, not these authors' baseline algorithms.

Tests must mock all native optimizers. Never persist
deformation fields or warped images, including inside checkpoint dictionaries. Native temporary logs/checkpoints must be removed on success and exceptions. Never commit upstream checkouts, data, masks or generated artifacts.

Our method stays single-seed. Do not launch training or benchmarks without an explicit execution request. Read AUDIT.md for source/paper discrepancies; keep recipe limitations in metrics.
