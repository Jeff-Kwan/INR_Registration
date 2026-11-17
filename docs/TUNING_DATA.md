# Available development data and evaluation-subject overlap

Inventory verified17 November2026. The [frozen tuning cohort](TUNING_COHORT.md)
is now prepared:32 pairs per dataset,128 total. No fitting or tuning was performed. Public validation subjects used as our reported evaluation
cohort must still be excluded from hyperparameter selection.

| Dataset | Related local development data | Readiness |
|---|---|---|
| OASIS |414 labeled L2R volumes; IDs0001–0364 exclude all50 subjects in our Neurite evaluation cohort | 32 pairs frozen across64 distinct development subjects; released L2R geometry |
| IXI |403 author training subjects and58 author validation subjects, separate from115 test subjects | All prepared images present; fixed atlas remains shared by the protocol |
| LPBA40 |25 author training subjects and5 author validation subjects, separate from10 test subjects | All40 prepared; five validation subjects define10 unordered combinations |
| ACDC |100 official training patients001–100, separate from test patients101–150 | Full raw cohort verified;32 patients selected and64 phase inputs prepared with KAN-IDIR recipe |

ACDC official training-source information:
https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html

## Confirmed OASIS overlap

Different dataset releases/preprocessing do **not** establish subject independence.
A read-only scan of all414 L2R label volumes found the50 subjects in our frozen
Neurite-OASIS evaluation cohort exactly at **L2R IDs0365–0414**. Label arrays are
voxelwise identical after axis permutation(0,2,1) and flipping the first two
resulting axes. This is stronger than a name or histogram-only association.

Consequently, all20 subjects in the existing L2R public-validation ablation cohort
(0395–0414) are part of our Neurite evaluation subject set. The corresponding
19-pair L2R ablation dataset cannot serve as an independent tuning/validation
cohort for the Neurite evaluation. Existing ablation results remain observations
on overlapping subjects; do not describe them as evidence from an independent
patient cohort or use them for supposedly test-independent hyperparameter search.

Exclude0365–0414 from L2R development data;0001–0364 remain the available
subject-disjoint candidate pool. A tuning split must separate subjects, with
all pairs restricted to the chosen development subjects. Splitting pair indices
alone is insufficient if endpoints recur between development and evaluation.

[Exact50-subject mapping and audit metadata](../results/data_audit_20261117/oasis_subject_overlap.json).
Original Data files, manifests, cohort definitions and scores were not changed.

## Other local data

`Data/ACDC_CorrMLP_v3` is an alternate preparation of the same official test
patients101–150, not additional training subjects. Neurite's full414-subject
archive is retained, but only the50 evaluation subjects are currently prepared
in its dedicated folder. The remaining364 subjects are available in that archive
and through the corresponding L2R release, rather than an independent new cohort.

AbdomenCTCT has30 labeled CT volumes; DIR-Lab has10 CT cases. These are useful
for their documented tasks or numerical robustness, but they are not matched
cardiac/brain-MRI development sets. Their prior uses and evaluation protocols
must be respected before defining any additional tuning study.

ACDC acquisition update: all100 official training patients are downloaded and verified.
[Acquisition details](ACDC.md#official-training-cohort-acquired--17-november2026) ·
[Download manifest](../Data/ACDC/training_download_manifest.json). Selected development inputs are prepared; no tuning was performed.
