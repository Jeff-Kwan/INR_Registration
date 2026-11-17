# Eight-hour generalist sweep

The user authorized multi-GPU training, adaptive hyperparameter selection and a
final full-cohort evaluation on 2026-11-16 (Hong Kong). This supersedes the usual
no-launch restriction only for this research effort. No commit or push is authorized.

The shared scientific starting point is width 512, 5,000 steps, seed 42,
four-quarter stochastic Euler, shared crops, 32k independent particles per
direction, native voxel units and full-gradient LK/NCC. The initial control
uses existing complete captured execution to reduce dispatch overhead, with
independent temporal and query sampling. All five cohorts receive the same
configuration; masks and intensity preparation keep their documented dataset
protocols. Original data and the repository default remain unchanged.

The frozen, score-blind selection is in ignored
`results/generalist_sweep_20261116/selection.json`, seed 20261116. Screening:

| Cohort | Two screening indices | Further validation indices |
| --- | --- | --- |
| DIR-Lab | 02, 07 | 08, 09 |
| OASIS | 12, 15 | 3, 5, 7, 10, 11, 16 |
| AbdomenCTCT | 14, 16 | 11, 13, 21, 23, 24, 27 |
| IXI | validation 35, 46 | validation 3, 9, 23, 38, 42, 53 |
| LPBA40 | validation 9, 10 | validation 1, 3, 4, 5, 6, 8 |

Old quick4 pairs and any reverse equivalents are excluded from tuning. Screening
and further validation pair indices are disjoint. This is pair-level separation;
subjects may recur between pairs within the published protocols. Previously
published full-cohort evaluations have already covered DIR-Lab/OASIS/abdomen;
these cases are new to quick screening, not claimed never previously evaluated.
Brain tuning uses validation subjects separate from the final test subjects.

Priorities are c, endpoint moment matching, LK weighting/schedules, registration
weighting/warmup, smoothing/schedules, box versus GW-NCC, and inter-SIREN L2.
First inspect the control runtime and errors. The first prepared c alternatives
are 0, 0.25, 1, 16 and 64 against control c=4. Follow-ups should refine promising
regions and test interactions, rather than selecting each coordinate in isolation.
The lower-priority L2 term means an explicitly defined differentiable penalty
between endpoint weights, not an unreported replacement with paired AdamW decay.
It has not yet been implemented or run in this effort.

Rank candidates using each cohort equally, relative to the identical-case control:
DIR-Lab primary TRE ratio, and `(1-DSC)/(1-control_DSC)` on the four label cohorts.
Report both DIR-Lab directions. Favor low mean normalized error while checking the
worst cohort, per-case regressions, folding, HD95, failures and PCG convergence.
No failed cases may be dropped to make a candidate win. Promote contenders to
the additional frozen pairs before locking one shared recipe. No dataset-specific
hyperparameters, rerandomized case sets or additional training seeds are allowed.

Time records include the worker's full case invocation, with phase timings also
retained. Report cold compilation separately when estimating steady-state cost;
do not silently subtract it from actual measured wall time. The final observed
mean must be below 120 seconds per case, with 90 seconds preferred. Reserve
approximately three hours of the eight-hour wall budget for the final evaluation
and reporting. `budget.json` records the original goal deadline.

The final locked candidate must run all 234 official evaluation entries:
10 DIR-Lab, 19 OASIS, 45 abdomen, 115 IXI test and 45 LPBA40 test. The final run
includes the previously excluded quick4 cases as part of each complete cohort.
Results influenced by screening/validation on the older public medical cohorts
must be described as such; they are not untouched hidden-test estimates.

`medical.generalist_sweep` plans or dispatches one batch through four persistent
GPU workers. It writes frozen config paths, input checks, per-case logs/checkpoints,
phase timings, progress and failure denominators, with no persistent maps or
warped image outputs. It refuses redispatch of an existing batch. Read current
supervisor state and progress before launching any follow-up. Never restart a
live job merely because a monitoring tool times out.

Initial live service: `inr-generalist-sweep`. Its first batch is
`screen__control`; stdout is in `/var/log/portal/inr-generalist-sweep.log`.

## First control and execution adjustment

All ten control fits succeeded. Mean case invocation time was 162.71 s,
including cold compilation. Per-cohort primary scores/time: DIR-Lab TRE
2.81831 mm / 203.5 s; OASIS DSC .823111 / 168.9 s; abdomen DSC .380296 /
152.2 s; IXI validation DSC .730524 / 137.4 s; LPBA40 validation DSC .730202 /
151.5 s. These small samples are calibration evidence, not final cohort scores.

The next batch compares a shared 4k-step speed control and c values
1,16,.25,64,0 (speed control c4). Recovery uses the existing native PCG/RK2
implementation with eight times, eight substeps and cap256 on every cohort.
Brain adapters now accept a cap only when it is at least their full native
axis length; brain recovery grids and label evaluation therefore remain exact.
DIR-Lab uses the existing spacing-aware capped recovery, returning full original
maps in memory for the usual landmark/folding evaluation. Original image input
preparation is unchanged. This execution adjustment is explicit because the
original 5k control exceeded the requested time limit.

The optional `objective.parameter_l2` term is now implemented and defaults to
zero. Its exact definition is lambda times the mean squared difference over all
corresponding SIREN weight and bias scalars. It uses ordinary loss gradients,
including in captured training; it does not change AdamW decay. Analytical
scalar/gradient checks, configuration guards and ordinary/captured full-parameter
gradient parity pass. Thirteen relevant synthetic/mocked tests passed. No
positive-L2 training experiment has been dispatched yet. Source snapshots before
and after this addition are in the ignored source/ and source_v2/ directories.

`--rank` on the sweep runner prints equal-cohort normalized error, worst cohort
ratio, runtime and completion counts; incomplete or failed candidates have no
selection score. Use `--stage validation --rank-baseline speed_control --rank`
after collecting matched validation controls. Inspect per-case folding/HD95 and
both DIR-Lab directions before promotion; the rank is not the full decision.

The first four speed-control cases completed at 92.9 s mean on OASIS and
93.9 s mean on DIR-Lab, including case invocation overhead. DIR-Lab recovery
fell to approximately 9–11 s. Preliminary scores were DIR-Lab TRE2.86283 mm
and OASIS DSC.819991; the small changes relative to the longer control must be
included when reporting the runtime tradeoff. Remaining cohorts and subsequent
c alternatives are still running, so no overall speed or winner is claimed yet.

## Adaptive continuation

`inr-generalist-campaign` waits for the live c batch to exit; it does not launch
competing GPU work. At the batch boundary it runs a six-update synthetic CUDA
self-test for the new captured Gaussian schedule plus L2. It then selects parents
from complete screening arms and dispatches moments, LK weighting/schedules,
registration weighting/warmup, smoothing, similarity and L2 in priority order.
Parent selection uses mean normalized cohort error plus 0.2 times positive
worst-cohort excess error. This adds a modest generalization preference while
retaining the raw mean and all cohort results for review. The original slow
control is calibration evidence rather than a promotion candidate.

Small screening arms may include cold compilation for each geometry, so their
promotion ceiling is 180s, followed by a strict measured <120s requirement on
expanded validation. This does not claim cold screening satisfies the target.
The final full-cohort mean remains strict and includes all invocation overhead.
Captured setup timing now separately records wall time of the first three
updates and graph construction; it includes warmup/compute as well as compilation
and is not labeled pure compiler time. It remains included in training/case time.

The moment stage also tests two settings on the runner-up c parent. Expanded
validation compares the speed control with three screened contenders on 26 new
pairs each. One shared configuration is locked before the 234-case final run.
The budget guard preserves the final-run reserve. The campaign records decisions
and rankings; completion still requires an agent audit and scientific report,
including failures, folding, HD95, both DIR-Lab directions and runtime.

Full Gaussian smoothing can now run in captured execution using the original
precomputed reflected-convolution matrices. CPU objective/all-parameter gradient
parity passes at three schedule positions, with nonzero inter-SIREN L2. The
complete GPU self-test is deferred until the active c sweep has released GPUs.
No source changes affect the already imported training code in the live c batch.
Source_v4 records the extended backend/campaign source for subsequent batches.

The completed c1/c16/c0.25 arms had normalized errors 1.02719, 1.00976 and
1.07270, respectively, against 0.99676 for the c4 speed control. This motivates
an adaptive half/double refinement around the eventual best c arm (c2/c8 if c4
remains best), before moment tuning. The waiting controller was updated before
it dispatched further work; the live broad c sweep was not interrupted.

`medical.generalist_report` produces an ignored live Markdown report, case CSV,
and structured requirement audit. It verifies success counts, exact final pair
membership, a single locked final recipe/seed and real runtime. Missing final
work currently fails those audit entries; the report is not a completion claim.

All 60 broad-c fits completed successfully. The remaining results were
c64 normalized error1.03857 and c0 error1.73111. c64 improved IXI's normalized
error by about7.2% but worsened the other four cohorts; it is not a generalist
winner. The complete captured CUDA schedule/L2 synthetic test passed in30.292s,
then c2/c8 refinement launched on all four GPUs. The original batch service
exited normally; the campaign service now owns the GPU work.

The20-fit refinement also completed without failures: c2 normalized error1.01203,
c8 error.99940, compared with c4 speed-control error.99676. The controller retained
c4 and began70 moment-matching fits: five alternative weights at c4 and two
weights at runner-up c8. This is still two-case screening per cohort; no final
configuration or significance claim follows from these small differences.

The zero-moment arm completed successfully with normalized error1.01433, worse
than the speed control. IXI DSC rose to.75354, while OASIS fell to.81039 and
abdomen fell to.33762. Final sampled OASIS output variances were about.029–.035
versus.309–.370 with the control, despite similar endpoint NCC losses. These
last-minibatch values are diagnostics on matching seeded crops, not whole-image
variance estimates. The changes are consistent with a gauge-dependent balance
between derivative losses and fixed inference regularization; this sweep does
not isolate the training and inference contributions to that mechanism. The
positive moment weights continue to test a shared compromise across cohorts.

Moment weight.001 completed at normalized error1.00358. Weight.003 reached
.99627, almost equal to the speed control's.99676 raw mean, but its abdomen DSC
was.35604 versus.37727 for the control. The brain Dice gains therefore do not
clearly establish a generalist improvement. Its worst-cohort penalty yields
utility1.00410 versus1.00029 for the control. The fixed generalist criterion
retains the control while higher weights and c8 interactions are evaluated.

All70 moment fits completed successfully. The c8/moment.003 interaction has
normalized error.994788 and utility1.000181 versus.996761/1.000289 for the speed
control. This is a near tie on two cases per cohort, not persuasive evidence of
superiority. The controller provisionally uses c8/.003 for LK weighting trials;
the original speed control remains a mandatory expanded-validation comparator.
LK trials cover constant weights.01,.03,.3,1 and linear/cosine ramps with the
inherited nominal weight.1. One shared config is used for every cohort in each arm.

All60 LK-weight/schedule fits completed successfully. The constant.1 parent
remains preferred. Alternative normalized errors: .01weight1.05413,
.03weight1.01171, .3weight1.02152,1weight1.06053, linear ramp1.02324,
cosine ramp1.04150. These arms averaged88.8–90.4s including real invocation
cost. In representative completed cases, separately measured setup/warmup/capture
was about2–4s; the reported total was not adjusted downward. Registration-weight
and warmup trials now run from the retained c8/moment.003/LK.1 parent.

All70 registration-weight/warmup fits completed successfully. Constant.03 leads:
normalized error.981402, utility.988374, mean88.9s. The500-step registration ramp
was next at.989978/.995327;1000-step ramp.993559/1.002948, joint warmup
.997898/1.009929, constant.1weight1.001120/1.015870 and zero registration
1.005739/1.016141. The controller selects constant.03 with no warmup for
smoothing trials. This improves the screen but still needs independent validation.

Smoothing compares three-tap cell sigmas.4,1,1.5 against the inherited.75, plus
full-Gaussian crop-relative static.1 and linear.2→.1,.4→.1,.6→.1 schedules.
The full matrices retain their wider support; these are not three-tap
approximations to a3.2-cell Gaussian. All other parameters are shared and fixed
within each arm, including c8, moments.003, LK.1 and registration.03.

After the0.4-cell arm improved the balanced screen (error.980481,
utility.984649), four extra narrow-bandwidth trials were proposed: cell sigma.2,
full box sigma.05, and schedules.2→.05/.4→.05. They have NOT been dispatched.
Automatic approval review rejected creating the extra managed service and holding
/resuming the scheduler, first citing the earlier no-launch instruction, then
refusing goal-tool output as authorization. Explicit human approval was requested.
The proposed helper/config/service files were not created. The existing approved
campaign was untouched and continues all its originally planned experiments.
Do not infer that the proposed extra tests have results or bypass the rejection.

The full-Gaussian.2→.1 schedule completed with error.980253,
utility.986665 and89.0s mean. Its raw average error slightly improves on the
0.4-cell blur (.980481), but the latter's utility.984649 remains preferable:
abdomen DSC.37168 versus.36043 for the schedule. Fixed full-Gaussian box.1
scored.988451/.996635. Remaining broad schedules continue. Optional additional
narrow-bandwidth experiments remain unlaunched while approval is pending.

All70 smoothing fits succeeded; cell sigma.4 is preferred by utility.984649.
The full.6→.1 schedule finished at error.990870/utility.999913. The proposed
extra narrow trials remain unlaunched. All20 approved similarity fits then
succeeded. GW-NCC.03 produced error.966655/utility.970690, improving the matched
box-NCC parent's error.980481/utility.984649. Its screen scores were DIR-Lab
2.61021mm, OASIS.81954, abdomen.37922, IXI.75936 and LPBA40.73230. Lower
GW-NCC weight.003 scored.990369/utility.995716. The cold.03 arm averaged136.4s;
the cached.003 arm averaged89.9s, so expanded validation must verify the
actual time gate. L2 trials now use GW-NCC.03 as parent and test six positive
penalty values.001,.01,.1,1,10,100 against the inherited zero-L2 parent.

Four L2 arms are complete: lambda.001 error.970359/utility.975691,
lambda.01 error.963110/utility.964966, lambda.1 error.967783/utility.971009,
and lambda1 error.973261/utility.976520. Lambda.01 leads the current screen;
lambda10/100 remain running. Cold means around131s still need the independent
validation runtime gate. A matched IXI checkpoint has mean squared parameter
difference3.51e-6 without L2,3.53e-6 at.001 and3.55e-6 at.01. The tiny penalties
and nonmonotonic agreement do not establish a useful shrinkage mechanism from
these small score changes; positive-L2 gains require expanded validation.

The report audit now directly checks each final fit's locked recipe, nonempty
checkpoint, final logged optimizer step, seed and complete recovery diagnostics.
It checks brain test splits and IXI17-group/LPBA4056-label metric coverage, and
records both brain directions, LPBA40 bidirectional Dice/Jacobian/IC, phase time,
setup time, folding maxima and folding-case counts. The final-run checks still
fail appropriately because validation and full evaluation have not occurred yet.

All440 screen fits succeeded across44 configurations (including the two execution
controls). The remaining L2 results were lambda10 error.975027/utility.980371
and lambda100 error.966623/utility.968019. Lambda.01 remains the screen leader;
lambda100 is next under the balanced criterion, followed by zero-L2 GW-NCC.03.
The campaign dispatched104 expanded-validation fits: speed_control plus these
three contenders, each on the same26 additional frozen pairs. All input checks
passed. No final recipe is locked yet. The read-only L2 checkpoint probe is
recorded in ignored l2_checkpoint_probe.json; no optimizer or recovery ran for it.

The validation control completed26/26 at96.480s mean, with primary scores
DIR5.65500/OASIS.801447/abdomen.357353/IXI.750440/LPBA40.714315.
The first promoted contender, L2.01, completed26/26 at91.859s and error.955822/
utility.958681. Scores are DIR4.95930/OASIS.798609/abdomen.362360/IXI.772512/
LPBA40.718864. Four cohorts improve; OASIS declines. Both recipes converged
all208 PCG intervals. Abdomen maximum folding fell from1.0107% to.2044%; this
is improvement, not a claim of nonfolding. L2.01 has slightly worse OASIS HD95,
with better abdomen/LPBA40 HD95. L2.100 and zero-L2 GW-NCC continue validation.
The final234-case input preflight again found no missing files. Detailed concise
report: docs/GENERALIST_SWEEP_REPORT_20261116.md (explicitly in progress).

L2.100 completed26/26 validation fits at92.396s mean, error.960349,
utility.964915. ScoresDIR4.96835/OASIS.796915/abdomen.362114/IXI.769664/
LPBA40.718661: L2.01 is slightly better on every cohort and faster. The
zero-L2 GW-NCC arm is last; no final winner is locked until it completes.
The configuration audit verified one frozen recipe per candidate across all
completed cohorts/stages, trainingseed42 only, and exactly the two frozen
screening pairs per cohort. A new input/protocol artifact audit was exercised
on the26 completed contender records; only the intentionally held-out brain
validation-split checks fail under its final-test-only rule. Its LPBA40 seed
check now uses the manifest's base XOR zero-based pair index, matching the
unchanged actual evaluator. The concise report and standalone PNG/PDF summary
are available; the final timing panel remains explicitly pending.

All104 validation fits succeeded. Zero-L2 GW-NCC scored error.958499,
utility.962233 and92.028s. ScoresDIR5.01263/OASIS.797740/abdomen.359063/
IXI.772743/LPBA40.720186. L2.01 wins the declared criterion (utility.958681),
with only.00268 mean-normalized-error advantage over zero L2; its gains are
DIR/OASIS/abdomen, with small IXI/LPBA40 losses. Positive L2 itself remains a
weak single-seed finding. The controller locked L2.01 before dispatching the
234-case full plan on four GPUs, with13324seconds of the original budget left.
The exact opt-in preset is medical/configs/native_generalist_euler4_4k.json;
its loaded JSON equals the locked config. Default.json remains unchanged.
Full child PID99864 is live under controller65393. No missing final inputs.

Full DIR-Lab10/10 completed: primary roundedTRE2.75830±1.77278mm acrosscases,
reverse2.89000±1.88381mm,112.55s mean,80/80PCGconverged. LungROI folding0;
wholegrid nonzero5/10,maxfraction5.19e-6. Theprimaryfolding_fraction field is
lungROI forDIR; report now retains wholegrid andlung separately toavoidclaiming
zero wholegridfolding. OASIS19/19 completed:.808155±.021043Dice,87.12s,
HD951.89779±.44069vox,shiftedSDlogJ.06718±.00412,152/152PCGconverged.
OASISfoldmeanfraction9.63e-7,max3.56e-6,12/19nonzero. Earlysame-seed publicpair
repeatsrequiredbythefullrun differed.018-.079mmDIR and.00086-.00167DiceOASIS.
Theexactnumericalsourceisnotisolated; noadditionalexperiments/seedswereadded.
ThisqualifiesthesmallL2benefit. Fullremainingcohortscontinuerunningnormally.

All45abdomenfullfitscompleted:.418188±.087362Dice,87.801smean,
HD9516.65343±5.62628vox,shiftedSDlogJ.175419±.087956. All360PCGintervals
converged. Folding42/45cases,mean.04063%,max.37989%. Fullpublic74complete,
IXItesthasstarted. All20same-seedpublicpairrepeatsfromthefullrunareavailable:
absoluteprimary-scoremean/maxDIR.04633/.07859mm,OASIS.00195/.00587Dice,
abdomen.00500/.00903Dice. Thisisrequiredfullevaluation,notnewrepeatormultiseed
experiments. Thetinylambda.01vs0differenceisnotarobustlyestablishedL2effect.
Publicpairsnotusedtothiscampaigntuning(6/11/37)aresavedseparately; theDIR6
cold-geometriesactualmean126.0s,sooveralltimeclaimsarenotper-cold-caseguarantees.

IXI115testfitscompletedwithnofailures:primary17groupDice.760293,
percaseSD.029075,group/caseSD.110391over1955scores. Reverse.773209±.020037.
Actualmean89.300s;920/920PCGintervalsconverged. Wholegridfolding93/115cases,
meanfraction1.84496e-6,max2.64486e-5. Full189/234complete,overallmean89.95s;
LPBA40istheonlyremainingcohort. Allfinalartifactcheckssofarhavenoissues.

COMPLETE:778/778fits(440screen,104validation,234full)succeeded. LPBA45forward
Dice.718451±.016208,reverse.719836±.016995,bidirectionalpairmean.719143±.016404.
LPBA87.035smean;360/360PCG. All1872/1872PCGintervalsconverged. Overallactual
caseinvocationmean89.392789s,std6.637462,median88.594629,p9591.086874,max136.417618;
fivecoldcases>120s. Finalbatch87.898min;fittingcampaign5.764hwithin8hbudget.
All234savedstatesandprotocolartifactsverified;all20automaticauditflagsand14
requirementchecks pass. Dataunchanged,scratchclean,GitHEAD993567eunchanged,
indexempty. ThetwoexitedownedSupervisorserviceswerearchivedandremovedto
preventfutureunapprovedautostarts;managementservicesunchanged. Finalreports,
caseCSV,PNG/PDF,checkpointSHA256auditandcompletion_audit.jsonareavailable.
ThetinyL2advantageisuncertain,andthenewrecipeisnotuniversallybetterthanthe
older3-cohortRK2/Sobolaccuracypreset:DIRimprovesbutOASIS/abdDicearelower.
Noadditionalexperimentsareauthorizedbythiscompletion. Defaultremainseuler4/5k.
