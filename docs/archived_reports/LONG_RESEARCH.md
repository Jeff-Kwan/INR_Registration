# Ten-hour registration research session

The session is complete. See the [final report](LONG_RESEARCH_REPORT.md) for six complete fitting cohorts, the one-minute speed frontier, mechanism tests and unresolved stretch targets. The instructions below record the adaptive research process used during the session.

Start with the current shared 5k recipe and one question: which part of the pipeline prevents accurate registration in one minute? Use the first full-cohort measurements to locate that cost. Prioritize verified speed improvements early because every later fit or recovery can benefit. Keep the paired SIRENs, endpoint reconstruction and auxiliary training objective, analytical velocity estimation and integrated recovery as the scientific scope. Width, depth, initialization, step budget and implementation are open research variables. [Shared reference preset](../medical/configs/native_siren_onecycle_5k.json)

## Clock and deliverable

Actual launch was **2026-11-13 16:41:37 UTC**. Following the user's one-hour extension and updated minimum, continue active research through at least **T0+9h, 2026-11-14 01:41:37 UTC**, and deliver by **T0+10h, 02:41:37 UTC**. These are 09:41:37 and 10:41:37 Hong Kong time. Use elapsed wall time across all four GPUs, including queue changes, retries and review. If validation finishes early, investigate the largest remaining error, test a mechanism on frozen checkpoints, or verify the speedup on a different shape. Deliver a completed, comparable full-cohort run of one frozen recipe, per-case results, and updated documentation. Report the actual outcome of each stretch target.

| Full-cohort objective | Stretch target | Companion evidence |
| --- | --- | --- |
| OASIS, 19 directed pairs | Mean official Dice ≥0.85 | HD95, SDlogJ, folding and label-level changes |
| Full registration pipeline | Mean ≤60 s per pair on one GPU | Report each dataset mean, pooled 74-pair mean, median and slowest cases. Time a dispatched case through completed recovery, scoring and output, including setup and synchronization. Record cold-start costs, warm-worker timings, peak memory and four-GPU throughput separately. |

Record successes over intended counts, nonfinite metrics, PCG convergence and retries. Report folding fractions with explicit units and the worst case as well as the mean. Review per-pair changes against the same reference. An improvement confined to one pair or accompanied by worse HD95, a folding spike, failed solves, or a slower dataset needs investigation before promotion. Keep the last well-rounded full-cohort recipe available as a fallback. [Protocols](DATASETS.md) · [External comparison conventions](EXTERNAL_BASELINES.md)

## How to allocate the next GPU

An opening allocation could put one GPU on the 74-pair reference, one on isolated profiling and two on independent quick4 cases. Start collecting reference and profile data immediately. While writing code or inspecting a trace, keep other available workers consuming already specified jobs with frozen code and configs. Keep one pair on one GPU. Use distinct device lists and output roots. Snapshot the code revision **and working-tree changes**, resolved config and software environment for each queue. Launch workers from the snapshot so edits for the next hypothesis cannot alter an in-flight comparison.

Reassign at case or wave boundaries. If the profile exposes a large, repeated cost, spend capacity on reproducing and validating the fix, then roll it into subsequent research jobs once parity and end-to-end savings are established. If a kernel saves little wall time, release that GPU and look at the next measured bottleneck. If method work needs evidence, run the cheapest probe that could change the decision. While a candidate trains, the other GPUs can finish the reference, replay difficult checkpoints or validate an independent change. For speed measurements, record concurrent host load and repeat under matched contention if necessary. Give final validation the workers it needs as its latest safe start approaches.

At each significant result record **observation → proposed mechanism → distinguishing prediction → evidence → decision**. Include the run paths, config/code identity, occupied devices, remaining GPU-hours and the next decision that depends on the result. Keep this log near the session outputs. Revisit the allocation when a job fails, throughput changes, a hypothesis is contradicted, or a candidate becomes eligible. Retries use the same seed and explicit failure accounting. If an audit reveals a scoring or training bug, mark affected comparisons, fix and verify it against an independent reference, then rerun affected cases on a consistent version.

| When the decision concerns… | Working notes |
| --- | --- |
| Time per case, algorithm cost or kernel fusion | [Performance investigations](PERFORMANCE_RESEARCH.md) |
| Whether a measurement, gradient or recovered displacement is correct | [Method audit](METHOD_AUDIT_PLAN.md) |
| Why a change could improve registration, and what would challenge it | [Research ideas](RESEARCH_IDEAS.md) |

## Long-horizon convergence comparison

Start it early enough to inform later decisions and budget its four fits against the final full-cohort reservation. Preserve the recipe's exact first 5k updates, including the 5k OneCycle schedule. For this diagnostic, continue the optimizer and sampling streams in the same process at the terminal LR through 40k. Record the continuation policy in the resolved run config. A OneCycle schedule stretched over 40k would change the early trajectory and answer a different question.

Log training terms, effective LR, gradient clipping and cumulative training seconds. Save checkpoints at 1k, 2k, 3k, 4k, 5k, 10k, 20k, 30k and 40k. Track actual recovery and evaluation cost separately so time-to-quality includes one deployment recovery and scoring pass. Keep diagnostic replay overhead visible in the session clock.

Overlay the existing original 40k and current 5k trajectories on matched cases. A recipe can reach the old target sooner and still have considerable headroom at 5k. A continuing gap at 40k motivates a matched-width optimizer or matched-optimizer width control before attributing it to capacity. Use early checkpoint gradient and phase probes to choose which comparison deserves the next GPU. [Mechanistic probes](RESEARCH_IDEAS.md)

## Protect the final full run

Run a full reference early. Quick4 is the fast development screen. Confirm promising changes on the other frozen quick4 sets before choosing a shared recipe. A speed candidate must demonstrate whole-pipeline savings and preserve numerical behavior or explicitly demonstrate acceptable method-level tradeoffs. Track paired changes against the reference on all three datasets and examine outliers before selecting the final recipe. A candidate with unresolved regressions can remain a research finding while the full run uses the strongest validated recipe.

Use observed full-reference throughput to budget the final run.

At each allocation decision estimate remaining work by dataset and shape from actual case times. Simulate dispatch over the GPUs that will really be free, including the tail of the longest case. Add time for compilation, retrying a slow or failed case and writing/checking the final report. Subtract that reservation from T0+10h to obtain the **latest safe freeze time**. Allowing 90 minutes for a four-GPU pass and 30 minutes for retry/review means freezing by **T0+8h**. Move that boundary earlier if the measured cost grows. Full validation may run across the nine-hour minimum. Freeze one shared configuration and implementation before the final pass. If a late experiment succeeds after the freeze, document its evidence for the next round.

Use fresh output roots and reconcile the final manifest with the cohort protocol. Preserve checkpoints, timed stages, diagnostics and failures under ignored results. Keep recovery scratch transient. At close, publish a repo-local report linking the reference, final config, comparable external baselines, per-case scorecard, timings, folding tails, audit fixes and remaining hypotheses. Update the README recommendation only if the complete evidence supports it.

## Launch preflight

From the repository root these commands preview the shared reference on the complete cohorts. At execution, use the actual session root, frozen config and assigned GPUs. Both Learn2Reg launchers execute with the explicit run flag. Launch separate, non-overlapping shards if changing GPU ownership between waves. Combine their per-case records by protocol identity.

Repeat the manifest and available-device check at launch.

## User deadline extension during execution

The user added one hour during the session. Preserve the original and revised deadlines in the session record. Use the extra hour for mechanism-led follow-ups and transfer checks, while retaining at least 90 minutes for final full-cohort validation plus 30 minutes for retry/review. The latest safe recipe freeze is T+8h unless measured throughput requires an earlier start.

The user explicitly updated the goal to require at least nine hours of work. This supersedes the opening eight-hour minimum. Do not finish before T+9h.
