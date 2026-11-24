# Single-case registration

Our current training samples a crop center from one equally selected endpoint mask, then uses that shared crop for both endpoint reconstructions and independent forward/reverse registration particles. Masks do not gate training losses or particle trajectories. PD-LK uses all crop grid cells; no registration loss is skipped for crop exits. This sampling policy is our method's choice, not the external ccIDIR recipe. See [METHOD.md](METHOD.md) for border extension and full-image boundary handling.

Both maps are evaluated in memory, without persistent image artifacts. Independent min/max intensity scaling is recorded as our preparation choice. Our method uses one configured seed.

Each update of the default and accuracy presets still samples a half-size crop within that domain. The checkpoint's resolved configuration records the crop and sampling recipe.

OASIS uses `medical.oasis_method` and `NAME_oasis.json` baseline configs; see [OASIS](OASIS.md). Other datasets require an evaluation adapter. An alternate method module may be supplied with `--method`; it exports `run_case(case, config, device)` and returns `metrics` and `checkpoint`.

Native comparison adapters are isolated in [baselines/](../baselines/README.md). Use `--method baselines.fireants`, `baselines.convexadam`, `baselines.ccidir` or
`baselines.nir` with the corresponding JSON in `baselines/configs/`. Dense-method
checkpoints contain provenance only, since their parameters are deformation maps
which must not persist. Network baselines retain their native weights. The
baseline multi-GPU launcher prints planned jobs unless `--run` is supplied.

To run all ten cases with one process per GPU when experiments are authorized:

The launcher uses a shared queue: each configured GPU takes the next case as soon
as its current case finishes. It writes a frozen config, per-case logs and a
return-code/timing summary. Only GPUs listed in `--gpus` participate; it does not
discover or claim other GPUs that become idle. Failed cases do not prevent other
workers from finishing. Existing output paths are refused; a failed run may leave
an incomplete directory. No experiment is started by import or help.

Our-method jobs now use one persistent spawned Python process per configured GPU.
The shared queue still assigns the next case to whichever worker finishes first.
Each case reloads its frozen config, resets Python/NumPy/PyTorch seeds, and creates
fresh models and optimizer state. Case tensors and unused CUDA allocations are
released afterward while compiled code and library caches stay in the process.
Worker PIDs and post-case allocated bytes are recorded in job summaries for
reuse/memory auditing. Ordinary case exceptions are logged and the queue continues;
an exited or unusable CUDA worker is replaced for later cases. External baselines
continue using isolated fresh subprocesses. Plan-only execution starts no workers.

The generic case runner initializes CUDA before resetting per-case peak counters.
CUDA metrics include peak allocated and reserved bytes. `run.json` records the
actual imported method module, source path and SHA-256 alongside the resolved
configuration and input geometry. These fields make cold-worker failures and
snapshot/import mistakes auditable. Adapter stage timings exclude the runner's
input loading and output serialization; queue dispatch-to-completion timings
include them. Optional [captured native training](CAPTURED_TRAINING.md) preserves
the same case interface and ordinary paired-SIREN checkpoint layout.

Add `--plan-only` to inspect it without launching workers.
