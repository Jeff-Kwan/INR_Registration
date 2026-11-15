# Benchmark protocols

Use the 10 native DIR-Lab T00/T50 cases, the 19 directed Learn2Reg OASIS public-validation pairs and the 45 directed AbdomenCTCT public-validation pairs. Our method fits each pair independently at seed 42. Report directional TRE for lungs and official Dice, voxel HD95 and shifted SDlogJ for the label datasets. Include unshifted folding, PCG convergence, full pipeline time, failures and denominators. [Dataset protocols](DATASETS.md) · [Method and configurations](METHOD.md)

The external adapters cover FireANTs SyN, ConvexAdam–MIND, ccIDIR and NIR. The completed external comparison evaluates FireANTs, ConvexAdam and NIR. Each report identifies the evaluated recipe and cohort. [Adapter audit](../baselines/AUDIT.md) · [External results](EXTERNAL_BASELINES.md) · [Paired-SIREN results](RESULTS.md)

Prior cohort definitions and comparison settings are archived. [Historical benchmark specification](archived_reports/pre_cleanup_20261112/BENCHMARKS.md)
