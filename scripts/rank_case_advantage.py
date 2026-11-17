"""Rank signed per-case Dice advantage over classical and corrected neural means.

Reads completed metric JSON only. Never loads checkpoints, images or CUDA.
The caller must explicitly choose the SIRENPath results directory.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
DATASETS={'neurite_oasis':49,'acdc':100,'lpba40':45,'ixi':115}
GROUPS={'classical':('convexadam','fireants','demons','niftyreg'),
        'neural':('ccidir','nir','nodeo','dnvf')}


def read_json(path):return json.loads(Path(path).read_text())


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def index_run(root,methods,datasets):
    root=Path(root).resolve();plan=read_json(root/'plan.json');summary=read_json(root/'summary.json')
    intended={};completed={}
    for row in plan['jobs']:
        if row['method'] not in methods or row['dataset'] not in datasets:continue
        key=(row['dataset'],row['method'],row['pair_index'])
        if key in intended:raise ValueError(f'Duplicate planned case: {key}')
        intended[key]=row
    for row in summary['jobs']:
        if row['method'] not in methods or row['dataset'] not in datasets:continue
        key=(row['dataset'],row['method'],row['pair_index'])
        if key in completed:raise ValueError(f'Duplicate completed case: {key}')
        completed[key]=row
    expected={(d,m,i) for d in datasets for m in methods for i in range(1,DATASETS[d]+1)}
    if set(intended)!=expected or set(completed)!=expected:raise ValueError(f'Incomplete cohort: {root}')
    records={}
    for key,job in intended.items():
        done=completed[key];dataset,method,index=key
        target=root/dataset/method/f'pair{index:03d}'
        if done['returncode'] or Path(done['output']).resolve()!=target:raise ValueError(f'Failed/mismatched case: {key}')
        path=target/'metrics.json';score=read_json(path)
        if score['dataset']!=dataset or score['metric_protocol']!='standardized_v1':raise ValueError(f'Metric protocol mismatch: {path}')
        value=score['Dice']['mean']
        if type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=1:raise ValueError(f'Undefined Dice: {path}')
        command=job['command']
        inputs={k:str(Path(command[command.index('--'+k)+1]).resolve()) for k in ('moving','fixed')}
        records[key]=dict(dice=value,inputs=inputs,labels=score['labels'],preparation=job['preparation_sha256'],
                          metric_path=str(path),metric_sha256=digest(path))
    return records


def advantage(ours,external):
    mean=statistics.mean(external)
    if mean<=0:raise ValueError('External mean Dice is zero; relative advantage is undefined')
    return mean,(ours-mean)/mean


def analyze(ours_root,classical_root,neural_root,datasets):
    roots={'ours':Path(ours_root).resolve(),'classical':Path(classical_root).resolve(),'neural':Path(neural_root).resolve()}
    indexed={'ours':index_run(roots['ours'],('ours',),datasets)}
    for group,methods in GROUPS.items():indexed[group]=index_run(roots[group],methods,datasets)
    rows=[];extrema={};evidence={}
    for dataset in datasets:
        extrema[dataset]={}
        for index in range(1,DATASETS[dataset]+1):
            own=indexed['ours'][(dataset,'ours',index)]
            row=dict(dataset=dataset,pair_index=index,moving=own['inputs']['moving'],fixed=own['inputs']['fixed'],ours_dice=own['dice'])
            evidence[own['metric_path']]=own['metric_sha256']
            for group,methods in GROUPS.items():
                values=[]
                for method in methods:
                    other=indexed[group][(dataset,method,index)]
                    if any(own[k]!=other[k] for k in ('inputs','labels','preparation')):
                        raise ValueError(f'Pair direction, labels or preparation mismatch: {dataset}/{method}/{index}')
                    row[method+'_dice']=other['dice'];values.append(other['dice'])
                    evidence[other['metric_path']]=other['metric_sha256']
                mean,relative=advantage(own['dice'],values)
                row[group+'_mean_dice']=mean;row[group+'_advantage']=relative;row[group+'_advantage_percent']=100*relative
            rows.append(row)
        cohort=[r for r in rows if r['dataset']==dataset]
        for group in GROUPS:
            key=group+'_advantage';low=min(r[key] for r in cohort);high=max(r[key] for r in cohort)
            extrema[dataset][group]=dict(lowest=[r for r in cohort if r[key]==low],highest=[r for r in cohort if r[key]==high])
    return dict(formula='(ours_case_Dice - mean_external_case_Dice) / mean_external_case_Dice',
                averaging='equal weight across the four methods within each group on the same directed case; no dataset-average denominator',
                groups=GROUPS,roots={k:str(v) for k,v in roots.items()},datasets=list(datasets),
                neural_results='Corrected NIR MRI/FP32 and DNVF TF32-off results, including repaired NIR ACDC cases; current official_neural_20261116 tree',
                rows=rows,extrema=extrema,metric_files_sha256=evidence,
                run_metadata_sha256={str(root/name):digest(root/name) for root in roots.values() for name in ('plan.json','summary.json')})


def write_report(report,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    (output/'advantage.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    with (output/'cases.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(report['rows'][0]));writer.writeheader();writer.writerows(report['rows'])
    lines=['# Per-case SIRENPath Dice advantage','',report['formula']+'. Positive means SIRENPath wins; negative means it loses.',
           '',f"SIRENPath results: `{report['roots']['ours']}`.",
           '', 'Classical mean: ConvexAdam, FireANTs, Demons, NiftyReg. Neural mean: ccIDIR, corrected NIR, NODEO, corrected DNVF.',
           '', 'Percentages below are relative changes, not Dice percentage-point differences. Exact ties retain every case.',
           '', '| Dataset | Group | Extreme | Pair | Our Dice | External mean Dice | Advantage |',
           '|---|---|---|---:|---:|---:|---:|']
    for dataset,groups in report['extrema'].items():
        for group,ends in groups.items():
            for extreme,rows in ends.items():
                for row in rows:
                    lines.append(f"| {dataset} | {group} | {extreme} | {row['pair_index']} | {row['ours_dice']:.6f} | {row[group+'_mean_dice']:.6f} | {row[group+'_advantage_percent']:+.3f}% |")
    lines+=['','cases.csv contains every directed pair, individual baseline Dice, group means, signed ratios, percentages and endpoint paths. advantage.json also records checksums and all extrema.','']
    (output/'REPORT.md').write_text('\n'.join(lines))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ours',type=Path,required=True,help='Explicit choice: original default or manual-promotion official run')
    parser.add_argument('--classical',type=Path,default=ROOT/'baselines/results/official_four_20261116')
    parser.add_argument('--neural',type=Path,default=ROOT/'baselines/results/official_neural_20261116',help='Current corrected NIR/DNVF result tree')
    parser.add_argument('--datasets',nargs='+',choices=DATASETS,default=list(DATASETS))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if len(set(args.datasets))!=len(args.datasets):parser.error('Datasets must be distinct')
    output=args.output.resolve()
    if output.exists():parser.error('Output must be a fresh directory')
    if output.is_relative_to((ROOT/'Data').resolve()):parser.error('Output cannot be under Data')
    report=analyze(args.ours,args.classical,args.neural,args.datasets)
    write_report(report,output)
    print('Wrote',len(report['rows']),'matched cases to',output)


if __name__=='__main__':main()
