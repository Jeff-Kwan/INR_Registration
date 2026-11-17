# GS-Reg hard phantom

The copied fixture is the rich 128³ analytic scene, scene/flow seeds 17/0, including fine texture, two textured polyhedral organs, target-frame contrast variation (0.12) and independent noise (SD 0.003). `far` reuses the exact saved GS deformation coefficients, whose foreground identity-map EPE is about 17.32 voxels. Identity/small/mid controls are also available.

The pure scene and invertible warp code is copied from GS's frozen experiment recipe. `medical/phantoms/provenance.json` records hashes and original paths. The compact coefficients and scene metadata are copied independently under ignored `Data/GSRegPhantom`; the neighboring checkout is read-only. The copied GS license is retained alongside the source.

Preview the case without fitting or recovery.

```bash
/venv/main/bin/python -m medical.phantoms.gs_hard --case far
```

When a registration experiment is authorized, use the explicit runner.

```bash
/venv/main/bin/python -m medical.phantoms.gs_hard --case far --run \
  --device cuda:0 --output results/phantom_far
```

Generation is in memory. Images use the GS shared source-range normalization and original noise streams, then transpose XYZ arrays into native ZYX tensors. Labels and continuous deformation truth are evaluation-only. The runner uses paired SIREN fitting and standard frozen PCG/RK2 recovery, reports bidirectional foreground/per-region EPE and min–max reconstruction PSNR, and retains only scalar metrics and a checkpoint. Recovery scratch is temporary. This port has source/input hash checks and a sparse positive-Jacobian/inverse-consistency check; no optimizer run was launched.
