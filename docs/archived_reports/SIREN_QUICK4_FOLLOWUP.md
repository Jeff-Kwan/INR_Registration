# SIREN follow-up with a 2.5-hour review deadline

Completed 2026-11-13: 132 fits, 660 verified milestone checkpoints and all final probes; execution 108.02 minutes from the original launch. See the [final review](SIREN_CONVERGENCE.md#final-25-hour-follow-up-review--2026-11-13). The revisions below record the authorized additions and the skipped numerically zero decay arm.

Use four RTX 5090s for one session of at most 150 minutes from launch to review. The purpose is to decide which optimizer or frequency effect deserves the next round. Width is fixed at 512. The existing common preset supplies the shared reference. See the [previous screen](SIREN_CONVERGENCE.md).

## Comparators and protocol

Use the [common 5k preset](../medical/configs/native_siren_width512_output30_5k.json) as the parent for each change below. It uses width 512, four sine layers, ω=30, output initialization ×30, Adam LR 1e-4 with exponential final factor 0.25, endpoint NCC with moment weight 0.01, PD-LK 0.1 and registration NCC 0.01.

| Existing comparator | DIR-Lab forward / reverse TRE, mm | OASIS Dice | Abdomen Dice |
| --- | ---: | ---: | ---: |
| Common 5k | 1.80454 / 1.87067 | 0.82364 | 0.41690 |
| Stronger registration NCC 5k, standard output initialization | 1.71776 / 1.76619 | 0.81902 | 0.40182 |
| Original milestone targets, default 40k on DIR-Lab and default 5k elsewhere | 1.89463 / 1.93786 | 0.81833 | 0.39645 |

The eligible frozen sets are DIR-Lab cases 01, 03, 04, 05, OASIS official pairs 1, 4, 8, 9 and AbdomenCTCT official pairs 2, 8, 18, 41. Schedule complete sets of four as specified below. Use seed 42 and the established native-grid sampling, full gradients, 32k endpoint samples, 32k separate registration particles and PCG+RK2 recovery. DIR-Lab masks select crop centers. Learn2Reg labels enter evaluation only. Use the same sample streams and probe sites across matched arms. Save 1k, 2k, 3k, 4k and 5k checkpoints within the 5k schedule.

Reuse an existing case when its resolved configuration, protocol, sampling and implementation behavior match. Record the source directory for each reused result. Freeze the case/config manifest and the contrasts before launch.

## Round-one queue

All settings inherit the common preset unless explicitly varied. Run each numbered comparison on DIR-Lab quick4 first, reusing the existing Adam/exponential, uniform-30 result. Use one case per GPU. Each wave is four case fits including final recovery and probes.

| Wave | Single change or combination | Contrast |
| --- | --- | --- |
| 1 | Adam with OneCycle, peak 1e-4 | Schedule versus the parent |
| 2 | AdamW, decay 0.01, exponential schedule | Decay/optimizer versus the parent |
| 3 | AdamW with the same decay and OneCycle | Completes the optimizer × schedule factorial |
| 4 | Uniform first-layer frequency 25 | Control for the mean of the ladder |
| 5 | Linear first-layer ladder 10–40 | Spread at the same arithmetic mean |
| 6 | Same ladder with inverse-frequency outgoing gains at unit RMS | Effect of balancing at identical frequencies |
| 7 | Standard output initialization ×1 on the common preset | Isolates output scaling at the successful width, LR and default losses |

For both OneCycle arms use the existing LR-only implementation, warm-up fraction 0.1, start factor 0.25 and final factor 0.25. Keep Adam betas fixed. Compare AdamW with Adam under each schedule and compare schedules within each optimizer. Record actual LR and cumulative LR. For the frequency arms only the first-layer initialization varies. Hidden ω remains 30.

## Cross-dataset check and clock

Validate all four frequency arms on both OASIS and abdomen quick4: uniform25,
plain20–30, plain10–40 and balanced10–40. The existing uniform30 common preset
is the reference on all three datasets. Compare each ladder with uniform25,
then compare narrow versus broad spread and balanced versus plain10–40. Report
both registration and reconstruction effects per case and label-size group.
This full comparison avoids dropping a frequency recipe based on DIR-Lab alone.

Select one optimizer-family candidate by lowest completed mean bidirectional
DIR-Lab TRE across the retuned LR bracket, including the new low-rate Adam
control. Evaluate it on both other quick4 sets. Report fixed-rate comparisons
and each optimizer/schedule's best tested rate separately.

The upper queue is15DIR-Lab recipes and5cross-dataset candidates,100new fits.
The session ends with review by minute150 from the original launch time.
Execution stops at minute140. Reserve45minutes for external validation when
admitting further screen work, using observed wave times to update estimates.
Record incomplete or deferred work explicitly and use the final ten minutes
for review. Further runs are decided at that review.

## Hidden ω mechanism probe

Use a short CPU diagnostic alongside the GPU queue. On one fixed case-01 batch and fixed evaluation coordinates, start from the same effective function. Compare hidden ω=30, hidden ω=15 with doubled hidden weights and biases, and hidden ω=30 with half-rate Adam updates on those same hidden parameters. Keep the input and output layers identical. Verify the endpoint predictions and spatial/temporal derivatives agree at initialization. Measure preactivation phase change, endpoint prediction change and temporal derivative after one matched update. If the ω=15 and half-rate arms agree while the ordinary ω=30 arm differs, the smaller phase update explains the immediate effect. If they diverge, inspect the layer contributions and optimizer update before proposing a long run. This diagnostic addresses the first update. The saved 5k hidden-ω results supply the longer-run observation.

## Review deliverable

For each completed arm show paired per-case changes from its matched control. Report both DIR-Lab TRE directions, ≥10 mm TRE, endpoint patch NCC, solver convergence and lung folding. On OASIS and abdomen report official Dice, HD95, folding and anatomical patch NCC. Use existing final probes and retain milestone checkpoints for targeted replay after review. Record elapsed time, completed and deferred waves, which effects transfer, and which mechanism warrants a follow-up even if aggregate TRE missed the target. Defer smaller-model efficiency, isolated timing repeats, broader sweeps and new combinations across the optimizer and frequency families until a recipe has been selected.

## LR-range revision during execution

Optimizer and schedule comparisons now include a local LR bracket of 5e-5,
1e-4 and 2e-4 for each of Adam/exponential, Adam/OneCycle,
AdamW/exponential and AdamW/OneCycle. Reuse the existing Adam/exponential
2e-4 result and the common 1e-4 result. Add seven new DIR-Lab recipes,
28 fits. The planned upper fit count becomes 72, subject to the same clock.
The original seven-recipe screen finishes first. Then run the local bracket,
reserving at least 30 minutes for validation and review preparation. Select the
best completed optimizer/schedule/rate candidate for cross-dataset evaluation.
Show fixed-rate effects and best-tested-rate effects separately. Record the
best Adam rate as its control, including the reused upper-rate provenance.
If the best result lies at an endpoint of the bracket, flag the unresolved range
for the next review. Weight decay and OneCycle warm-up settings remain fixed
within this pilot, so their optimal ranges remain an open question.

## Narrow-ladder and transferability revision

The user extended the session to 150 minutes from its original start. Preserve
that original clock. Execution has a 140-minute cutoff and review uses the final
ten minutes. Add a plain 20–30 first-layer ladder at the common LR 1e-4. Its mean
is 25, matching uniform25 and the broad10–40 ladder.

Evaluate uniform25, plain20–30, plain10–40 and balanced10–40 on both OASIS and
abdomen quick4, each using the same recipe across all datasets. These four
frequency validations supersede the earlier single frequency-candidate selection.
The existing uniform30 common preset supplies the reference in every dataset.
Report paired effects of narrow versus broad spread, each ladder versus uniform25,
and balancing within10–40. Compare reconstruction and small-structure accuracy
alongside registration and report any dataset-specific preference. Choose the
optimizer candidate only after its local LR bracket is completed or the clock
forces a partial review. All five validation candidates share the original
session deadline. The planned upper count is now100new fits, comprising60DIR-Lab
fits and40Learn2Reg fits. Reserve45minutes of the execution window for validation
when admitting further screen cases. Actual elapsed times determine deferrals.

## Weight-decay interpretation

AdamW in this pilot uses explicit weight decay 0.01, verified in the saved run
configurations and optimizer construction. At peak LR 1e-4, the real-arithmetic cumulative
decay-only shrinkage over 5k steps is 0.270% for exponential and 0.312% for
OneCycle. These theoretical products do not include float32 rounding; they do not measure the
full trajectory effect or establish an optimal decay. The user proposed lower
values 1e-3 and 1e-4 because SIREN frequency and derivative behavior may respond
differently to regularization. A future matched bracket should include zero,
1e-4, 1e-3 and 1e-2. This pilot has not tested that bracket; report its AdamW
findings as conditional on decay 0.01. Preserve the scheduled frequency-transfer
coverage within the current deadline.

## Lower-decay follow-up within the same clock

Following the user's repeated request to test lower decay, append two matched
DIR-Lab quick4 arms: AdamW/OneCycle at peak1e-4 with decay1e-4 and1e-3. Retain
all other common settings, all five milestones and final probes. Existing
Adam/OneCycle supplies zero decay on DIR-Lab; AdamW/OneCycle at0.01 is already
complete. Audit zero-decay Adam/AdamW update parity before treating the former
as the zero-decay control. After the original frequency validations, run these
two new arms. Then validate both lower-decay recipes and the zero-decay
Adam/OneCycle control on both OASIS and abdomen quick4, without DIR-Lab filtering.
This completes a four-value decay comparison at a fixed rate and schedule across
the three datasets. The revised upper count is132new fits:68DIR-Lab and64transfer,
with660milestone checkpoints. Preserve the original140-minute execution and
150-minute review deadlines; use measured remaining time to record any deferral.
No other optimizer, frequency, model or loss setting changes.

### Skip numerically zero decay1e-4

The user requested skipping decay1e-4 after isolated CPU/CUDA float32 updates
confirmed its multiplicative factor rounds to1 across the current LR schedule.
Do not train or validate that duplicate arm. Retain the measured zero-decay
Adam/OneCycle control, decay1e-3 and the already completed0.01 arm on all three
datasets. This supersedes the four-value bracket above. The upper queue becomes
120new fits (64DIR-Lab,56transfer), with600milestone checkpoints. The unused
1e-4 recipe was removed before launch. Preserve the same original session clock.

## One predefined interaction test after selection

The user previously authorized stacking promising strategies. After the independent
controls, select AdamW/OneCycle peak1e-4 with decay1e-3 (the best completed
DIR-Lab optimizer recipe) and combine it once with the20–30 first-layer ladder
(the strongest abdomen frequency recipe, with smaller OASIS cost than10–40).
Run the combination after all decay transfers, on all three frozen quick4 sets.
Keep every other setting fixed. Compare it with the parent and each single change,
and report the factorial interaction rather than assuming additive gains.

This meets the earlier condition that a recipe be selected before new combinations.
It adds12fits, bringing the revised upper queue to132new fits (68DIR-Lab,64transfer)
and660milestones. The skipped1e-4 decay arm stays skipped. Preserve the original
140-minute execution cutoff and150-minute review deadline; no further combination
sweep is included.
