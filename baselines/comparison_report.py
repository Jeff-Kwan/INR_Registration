"""Refresh the concise comparison document from current machine-readable results."""
from pathlib import Path
import json
import statistics

ROOT=Path(__file__).resolve().parents[1]
NAMES=dict(ours='SIRENPath',convexadam='ConvexAdam',fireants='FireANTs',demons='Demons',niftyreg='NiftyReg',ccidir='ccIDIR',nir='NIR',nodeo='NODEO',dnvf='DNVF')
DATASETS=dict(neurite_oasis='Neurite-OASIS',acdc='ACDC',lpba40='LPBA40',ixi='IXI')


def refresh(root=ROOT):
    root=Path(root)
    folders=dict(classical=root/'baselines/results/official_four_20261116',neural=root/'baselines/results/official_neural_20261116',ours=root/'results/official_ours_cosine_20261116')
    reports={k:json.loads((p/'summary.json').read_text()) for k,p in folders.items()}
    document=root/'docs/BASELINES_REPORT.md'
    old=document.read_text();tail=old[old.index('## Methods'):]
    tail=tail[:tail.index('\nAll results use')] if '\nAll results use' in tail else tail
    total=sum(len(v['jobs']) for v in reports.values());failures=sum(j['returncode']!=0 for v in reports.values() for j in v['jobs'])
    header=['| Dataset | Method | Fits succeeded | Dice ↑ | HD95 ↓ | NPJ (%) ↓ | SDlogJ ↓ | HD95 pairs | Runtime (s/case) | Backend |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|---|']
    lines=['# Baseline comparison aggregate report','',
           f'Finished evaluations: **{total}**; successful: **{total-failures}**; process failures: **{failures}**. '
           'Invalid original NIR/DNVF results were removed; only their corrected reruns appear below. '
           'Single seed42; mean ± population SD. Unscored values are —.']
    primary,corrected=[],[]
    for ds,ds_name in DATASETS.items():
        for method,label in NAMES.items():
            group='ours' if method=='ours' else 'classical' if method in ('convexadam','fireants','demons','niftyreg') else 'neural'
            source=reports[group];aggregate=source['datasets'][ds][method];m=aggregate['metrics']
            jobs=[j for j in source['jobs'] if j['dataset']==ds and j['method']==method and not j['returncode']]
            times=[j['seconds'] for j in jobs]
            cells=['—' if m[k]['mean'] is None else f"{m[k]['mean']:.{p}f} ± {m[k]['std']:.{p}f}" for k,p in [('Dice',3),('HD95',3),('NPJ',6),('SDlogJ',3)]]
            elapsed=f'{statistics.mean(times):.2f} ± {statistics.pstdev(times):.2f}' if times else '—'
            backend='CPU/OpenMP, 4 threads' if method=='niftyreg' else 'CPU, 4 threads' if method=='demons' else 'GPU (RTX5090)'
            row='| '+ds_name+' | '+label+f" | {aggregate['successful']}/{aggregate['intended']} | "+' | '.join(cells)+f" | {m['HD95']['scored_pairs']}/{aggregate['intended']} | {elapsed} | {backend} |"
            (corrected if method in ('nir','dnvf') else primary).append(row)
    lines+=['','## SIRENPath and other baselines','',*header,*primary,
            '','## Corrected NIR and DNVF reruns','',
            'NIR uses the corrected MRI LNCC/spectral recipe in IEEE FP32; DNVF uses its original recipe with TF32 disabled. '
            'All preparations and scoring remain frozen. Each metric mean uses successful cases with a defined score. '
            'ACDC/NIR uses the largest available odd NCC extent up to9 per sampled axis:9×9×7 for four thin cases,9×9×9 otherwise.',
            '',*header,*corrected]
    failed=[j for j in reports['neural']['jobs'] if j['method'] in ('nir','dnvf') and j['returncode']]
    if failed:
        pairs=', '.join(f"{j['dataset']}/{j['method']} pair{j['pair_index']:03d}" for j in failed)
        lines+=['',f'**Failed rerun cases:** {pairs}.']
        if all(j['method']=='nir' and j['dataset']=='acdc' and "Kernel size can't be greater" in Path(j['output']).with_suffix('.log').read_text() for j in failed):
            lines+=['NIR downsamples these thin ACDC volumes to depth7 or8, smaller than its native9-voxel NCC kernel. '
                    'These four cases fail before the first optimizer update. ACDC/NIR aggregates therefore cover96/100 fits; '
                    'the failures are not assigned zero scores.']
    lines+=['','Runtime is averaged over successful cases and includes case loading, fitting, recovery/transformation evaluation, scoring and output serialization. Neural externals use fresh subprocesses; SIRENPath uses persistent workers. CPU classical baselines ran in parallel with GPU baselines. FireANTs used its PyTorch fallback.',
            '', 'All preparations/cohorts and standardized_v1 scoring are frozen. HD95 is prepared-grid mm (IXI uses its unit-voxel benchmark frame). Its valid-pair coverage can differ between methods; direct ranking needs a common valid subset. Dice uses nearest-neighbour zero-outside label warps. NPJ is100×mean(detJ≤0); SDlogJ isstd(log(max(detJ,1e-9))), with central differences and two-voxel border exclusion.', '',tail,
            'All results use standardized_v1 scoring. [Classical aggregates](../baselines/results/official_four_20261116/summary.json) · [Current neural aggregates](../baselines/results/official_neural_20261116/summary.json) · [SIRENPath aggregates](../results/official_ours_cosine_20261116/summary.json) · [Correction/precision audit](NEURAL_BASELINE_DIAGNOSTICS.md) · [Rerun plan](NEURAL_RERUNS.md)','']
    document.write_text('\n'.join(lines))
