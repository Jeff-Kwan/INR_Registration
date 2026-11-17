"""Frozen one-dimensional sweeps, independent rankings and conditional promotion."""
import argparse
import copy
import fcntl
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
from datetime import datetime, timezone

from medical.configuration import validate_config
from medical.brain_data import sha256
from medical.tuning_experiments import DATASETS
from scripts.prepare_tuning_cohort import ROOT, OUTPUT

CATEGORIES={
    'PDLK weight schedule':('objective.weight','objective.weight_schedule'),
    'Registration weight schedule':('objective.registration.weight','training.registration_weight_schedule'),
    'SIREN L2':('objective.parameter_l2',),
    'PDLK c':('objective.c',),
    'moment weight':('objective.ncc.moment_weight',),
    'LR schedule':('training.lr','training.lr_final_factor','training.lr_schedule'),
}


def get(config,path):
    for key in path.split('.'):config=config[key]
    return config


def put(config,path,value):
    keys=path.split('.')
    for key in keys[:-1]:config=config[key]
    config[keys[-1]]=copy.deepcopy(value)


def candidates(default):
    rows=[dict(id='01_default',category=None,description='Current default',config=copy.deepcopy(default))]
    def add(index,category,description,changes):
        config=copy.deepcopy(default)
        for path,value in changes.items():put(config,path,value)
        validate_config(config)
        rows.append(dict(id=f'{index:02d}',category=category,description=description,config=config))
    steps=default['training']['steps']
    add(2,'PDLK weight schedule','Cosine 0.3 to 0.1',{'objective.weight':.1,'objective.weight_schedule':dict(name='cosine',start_weight=.3,decay_steps=steps)})
    for index,start,final in ((3,.01,.03),(4,.05,.01),(16,.01,.01),(17,.03,.03),(20,.02,.02)):
        add(index,'Registration weight schedule',f'{"Constant" if start==final else "Cosine"} {start} to {final}',{'objective.registration.weight':final,'training.registration_weight_schedule':dict(name='cosine',start_weight=start,decay_steps=steps)})
    for index,value in ((5,0.),(6,.1),(7,1.)):
        add(index,'SIREN L2',str(value),{'objective.parameter_l2':value})
    for index,value in ((8,2.),(9,6.),(10,8.)):
        add(index,'PDLK c',str(value),{'objective.c':value})
    for index,value in ((11,.001),(12,.002),(13,.01)):
        add(index,'moment weight',str(value),{'objective.ncc.moment_weight':value})
    add(14,'LR schedule','Cosine 1e-4 to 1e-5',{'training.lr':1e-4,'training.lr_final_factor':.1,'training.lr_schedule':dict(name='cosine')})
    add(15,'LR schedule','Historical LR-only OneCycle',{'training.lr':1e-4,'training.lr_final_factor':.25,'training.lr_schedule':dict(name='onecycle',warmup_fraction=.1,start_factor=.25)})
    for index,value in ((18,.05),(19,.2)):
        add(index,'PDLK weight schedule',f'Constant {value}',{'objective.weight':value,'objective.weight_schedule':'constant'})
    return sorted(rows,key=lambda r:int(r['id'].split('_')[0]))


def validate_catalog(catalog):
    rows=catalog['candidates'];base=rows[0]['config']
    if rows[0]['id']!='01_default' or rows[0]['category'] is not None:raise ValueError('Default must be first')
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('Repeated candidate ID')
    fingerprints=set()
    for row in rows:
        config=row['config'];validate_config(config)
        if config['runtime']['seed']!=42:raise ValueError('Single seed42 required')
        fingerprint=json.dumps(config,sort_keys=True)
        if fingerprint in fingerprints:raise ValueError('Duplicate candidate')
        fingerprints.add(fingerprint)
        if row['category'] is not None:
            if row['category'] not in CATEGORIES:raise ValueError('Unknown category')
            restored=copy.deepcopy(config)
            for path in CATEGORIES[row['category']]:put(restored,path,get(base,path))
            if restored!=base:raise ValueError('Candidate changes more than one category')
    return catalog


def read_scores(directory,expected_config,manifest_hash):
    """Recompute rankings from all128 case results; never trust partial aggregates."""
    directory=Path(directory);plan=json.loads((directory/'plan.json').read_text())
    config_path=directory/'config.json'
    if json.loads(config_path.read_text())!=expected_config or sha256(config_path)!=plan['config_sha256']:
        raise ValueError(f'Configuration provenance mismatch: {directory}')
    if plan['tuning_manifest_sha256']!=manifest_hash or plan['split']!='development_tuning32' or plan['total_jobs']!=128:
        raise ValueError('Cohort provenance mismatch')
    summary=json.loads((directory/'summary.json').read_text())
    if not summary['complete'] or len(summary['jobs'])!=128:raise ValueError('Incomplete run')
    intended={(d,i) for d in DATASETS for i in range(1,33)}
    planned={(j['dataset'],j['pair_index']) for j in plan['jobs']}
    observed={(j['dataset'],j['pair_index']) for j in summary['jobs']}
    if len(plan['jobs'])!=128 or planned!=intended or observed!=intended:raise ValueError('Incomplete or duplicated pairs')
    metrics={d:{k:[] for k in ('Dice','HD95','NPJ','SDlogJ','seconds')} for d in DATASETS}
    for row in summary['jobs']:
        dataset,index=row['dataset'],row['pair_index'];target=directory/dataset/f'pair{index:03d}'
        if row['returncode'] or Path(row['output']).resolve()!=target.resolve():raise ValueError('Failed/misrouted case')
        score=json.loads((target/'metrics.json').read_text())
        if (score['split']!='development_tuning32' or score['dataset']!=dataset or score['pair_index']!=index
                or score['seed']!=42 or score['metric_protocol']!='standardized_v1'):raise ValueError('Case identity mismatch')
        for key in ('Dice','HD95','NPJ','SDlogJ'):
            value=score[key]['mean']
            if value is not None and (not isinstance(value,(int,float)) or not math.isfinite(value)):raise ValueError('Nonfinite metric')
            if key=='Dice' and (value is None or not 0<=value<=1):raise ValueError('Invalid Dice')
            if value is not None:metrics[dataset][key].append(value)
        metrics[dataset]['seconds'].append(row['seconds'])
    values={d:{k:dict(mean=statistics.mean(v) if v else None,scored_pairs=len(v)) for k,v in m.items()} for d,m in metrics.items()}
    return dict(macro_Dice=statistics.mean(values[d]['Dice']['mean'] for d in DATASETS),datasets=values)


def rank(catalog,root):
    validate_catalog(catalog);root=Path(root);base=catalog['candidates'][0]
    scores={};excluded={}
    for row in catalog['candidates']:
        try:scores[row['id']]=read_scores(root/'runs'/row['id'],row['config'],catalog['tuning_manifest_sha256'])
        except (ValueError,KeyError,OSError) as exc:excluded[row['id']]=str(exc)
    if base['id'] not in scores:raise ValueError('Complete default results required')
    # Require the entire requested sweep, preventing selection from a partial run.
    if excluded:raise ValueError(f'Sweep incomplete; excluded candidates: {excluded}')
    combined=copy.deepcopy(base['config']);rankings={};winners={}
    for category,paths in CATEGORIES.items():
        eligible=[row for row in catalog['candidates'] if row['category'] in (None,category)]
        eligible.sort(key=lambda row:(-scores[row['id']]['macro_Dice'],row['id']!=base['id'],row['id']))
        rankings[category]=[dict(id=r['id'],description=r['description'],**scores[r['id']]) for r in eligible]
        winner=eligible[0];winners[category]=winner['id']
        for path in paths:put(combined,path,get(winner['config'],path))
    validate_config(combined)
    return dict(objective='maximize equal-dataset mean Dice; default wins exact ties',winners=winners,rankings=rankings,scores=scores,combined_config=combined)


def compare_to_default(candidate,baseline):
    dice_wins=[];hd95_wins=[];deltas={}
    for dataset in DATASETS:
        a,b=candidate['datasets'][dataset],baseline['datasets'][dataset]
        dice=a['Dice']['mean']-b['Dice']['mean']
        valid_hd=(a['HD95']['scored_pairs']==32 and b['HD95']['scored_pairs']==32
                  and a['HD95']['mean'] is not None and b['HD95']['mean'] is not None)
        hd=a['HD95']['mean']-b['HD95']['mean'] if valid_hd else None
        if dice>0:dice_wins.append(dataset)
        if hd is not None and hd<0:hd95_wins.append(dataset)
        deltas[dataset]=dict(Dice=dice,HD95=hd)
    return dict(qualifies=len(dice_wins)>=3 and len(hd95_wins)>=3,
                Dice_improved_datasets=dice_wins,HD95_improved_datasets=hd95_wins,deltas=deltas)


def select_final(report,combined):
    baseline=report['scores']['01_default']
    pool={identifier:report['scores'][identifier] for identifier in set(report['winners'].values())}
    pool['combined']=combined
    comparisons={identifier:dict(**compare_to_default(scores,baseline),macro_Dice=scores['macro_Dice']) for identifier,scores in pool.items()}
    qualifying=[identifier for identifier,comparison in comparisons.items() if comparison['qualifies']]
    qualifying.sort(key=lambda identifier:(-pool[identifier]['macro_Dice'],identifier))
    selected=qualifying[0] if qualifying else None
    return dict(rule='strict mean Dice improvement on >=3/4 datasets AND strict mean HD95 improvement on >=3/4 datasets; complete HD95 required per counted dataset',
                comparisons=comparisons,selected=selected,promote=selected is not None,
                selected_scores=pool[selected] if selected else None,default_scores=baseline)


def write_ranking(root,report):
    root=Path(root);(root/'ranking.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (root/'combined.json').write_text(json.dumps(report['combined_config'],indent=2)+'\n')
    lines=['# Tuning category rankings','',report['objective']+'.','', 'Every candidate uses all128 frozen pairs. Other metrics are diagnostics, not hidden promotion gates.','']
    for category,rows in report['rankings'].items():
        lines += ['## '+category,'','| Rank | Candidate | Macro Dice | OASIS | ACDC | LPBA | IXI |','|---:|---|---:|---:|---:|---:|---:|']
        for i,row in enumerate(rows,1):
            lines.append('| '+str(i)+' | '+row['id']+' '+row['description']+' | '+f'{row["macro_Dice"]:.6f}'+' | '+' | '.join(f'{row["datasets"][d]["Dice"]["mean"]:.6f}' for d in DATASETS)+' |')
        lines.append('')
    lines+=['Full HD95, NPJ, SDlogJ and runtime summaries are in ranking.json.','']
    (root/'RANKING.md').write_text('\n'.join(lines))


def launch(config,output,gpus):
    subprocess.run([sys.executable,'-B','-m','medical.tuning_experiments','--config',str(config),'--output',str(output),'--gpus',*map(str,gpus),'--run'],cwd=ROOT,check=True)


def pipeline(root,gpus):
    root=Path(root);catalog=json.loads((root/'catalog.json').read_text());validate_catalog(catalog)
    if sha256(ROOT/'medical/configs/default.json')!=catalog['default_sha256']:raise ValueError('Default changed since search preparation')
    def status(phase,**fields):
        state=dict(phase=phase,updated_utc=datetime.now(timezone.utc).isoformat(),**fields)
        temporary=root/'status.partial.json';temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(root/'status.json')
        print(json.dumps(state),flush=True)
    for index,row in enumerate(catalog['candidates'],1):
        status('sweep',candidate=row['id'],candidate_number=index,total_sweeps=len(catalog['candidates']))
        output=root/'runs'/row['id']
        if output.exists():read_scores(output,row['config'],catalog['tuning_manifest_sha256'])
        else:launch(root/'configs'/(row['id']+'.json'),output,gpus)
    status('ranking')
    report=rank(catalog,root);write_ranking(root,report)
    combined_output=root/'combined'
    # Run the assembled combination once, even if it equals an existing candidate.
    status('combined_tuning',winners=report['winners'])
    if combined_output.exists():combined=read_scores(combined_output,report['combined_config'],catalog['tuning_manifest_sha256'])
    else:
        launch(root/'combined.json',combined_output,gpus)
        combined=read_scores(combined_output,report['combined_config'],catalog['tuning_manifest_sha256'])
    decision=select_final(report,combined)
    (root/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    if not decision['promote']:
        status('complete_without_promotion',decision=decision);return
    selected=decision['selected']
    selected_config=(report['combined_config'] if selected=='combined' else
                     next(row['config'] for row in catalog['candidates'] if row['id']==selected))
    (root/'selected.json').write_text(json.dumps(selected_config,indent=2)+'\n')
    status('promotion',decision=decision)
    promote(root,catalog,selected_config,decision)
    official=root/'official_promoted'
    if official.exists():raise FileExistsError('Official results already exist; inspect rather than overwrite')
    status('official_run',output=str(official))
    subprocess.run([sys.executable,'-B','-m','medical.comparison_experiments','--methods','ours','--config',str(root/'selected.json'),
                    '--output',str(official),'--gpus',*map(str,gpus),'--run'],cwd=ROOT,check=True)
    status('complete_with_promotion',decision=decision)


def promote(root,catalog,config,decision):
    default=ROOT/'medical/configs/default.json'
    validate_config(config)
    method_text=render_method_document(config)
    if sha256(default)!=catalog['default_sha256']:raise ValueError('Default changed before promotion')
    (root/'previous_default.json').write_text(default.read_text())
    temporary=default.with_suffix('.partial.json');temporary.write_text(json.dumps(config,indent=2)+'\n');temporary.replace(default)
    record=dict(search=str(root),decision=decision,config=config)
    (ROOT/'docs/TUNING_PROMOTION.json').write_text(json.dumps(record,indent=2)+'\n')
    (ROOT/'docs/METHOD.md').write_text(method_text)
    update_recipe_references(config)
    (ROOT/'docs/TUNING_PROMOTION.md').write_text('# Promoted tuning recipe\n\nSelected using only the128-pair development cohort; official test scores did not select it.\n\n'
        +f'Selected candidate: {decision["selected"]}; default macro Dice: {decision["default_scores"]["macro_Dice"]:.6f}; selected: {decision["selected_scores"]["macro_Dice"]:.6f}.\n\n'
        +f'Full provenance and previous default: `{root}`.\n\nResolved settings: [TUNING_PROMOTION.json](TUNING_PROMOTION.json).\n')


def render_method_document(config):
    """Replace scalar recipe prose at promotion; retain unchanged method equations."""
    p=ROOT/'docs/METHOD.md';text=p.read_text()
    # Replace schedule/weight prose with configuration-key references, preventing
    # a successful promotion from leaving the historical numbers as instructions.
    start=text.index('A Gaussian-moment penalty, weighted')
    end=text.index('This objective term is separate',start)
    text=text[:start]+'''A Gaussian-moment penalty with weight `objective.ncc.moment_weight` anchors
predicted means/variances toward0/1:

`moment_loss = 0.5 * sum_endpoints[mean² + (variance+epsilon) - log(variance+epsilon) - 1]`.

Variances are population variances. Parameter agreement is the mean squared
parameter difference with weight `objective.parameter_l2` in the active configuration.
'''+text[end:]
    text=text.replace('**c=4**','**c=`objective.c`**').replace('weighted **0.1**','weighted by `objective.weight` and its `objective.weight_schedule`')
    start=text.index('The registration coefficient follows');end=text.index('## Joint objective',start)
    text=text[:start]+'''The registration weight uses `training.registration_weight_schedule`, with
its terminal value in `objective.registration.weight`. Equal endpoints give a
constant schedule; either rising or falling cosine endpoints are supported.

'''+text[end:]
    start=text.index('The learning rate is');end=text.index('Model and optimizer state',start)
    text=text[:start]+'''The active learning rate uses `training.lr`, `training.lr_final_factor` and
`training.lr_schedule`. Cosine has no warmup; OneCycle uses the explicit warmup
fraction and start factor with fixed AdamW betas.

'''+text[end:]
    for line in text.splitlines():
        if line.startswith('| PD-LK transport loss |'):text=text.replace(line,'| PD-LK transport loss | `objective.weight` | See active recipe below |')
        elif line.startswith('| Differential residual'):text=text.replace(line,'| Differential residual in the local solve/PD-LK | `objective.c` | See active recipe below |')
        elif line.startswith('| Gaussian moments |'):text=text.replace(line,'| Gaussian moments | `objective.ncc.moment_weight` | See active recipe below |')
        elif line.startswith('| Registration loss |'):text=text.replace(line,'| Registration loss | `objective.registration.weight` | See active recipe below |')
        elif line.startswith('| Inter-SIREN parameter'):text=text.replace(line,'| Inter-SIREN parameter L2 | `objective.parameter_l2` | See active recipe below |')
    text+='\n## Promoted active scalar recipe\n\nSelected on the development cohort; [promotion evidence](TUNING_PROMOTION.md).\n\n```json\n'+json.dumps({category:{path:get(config,path) for path in paths} for category,paths in CATEGORIES.items()},indent=2)+'\n```\n'
    begin=text.index('## Explicit tuning option: historical OneCycle LR')
    end=text.index('## Frozen stage-3 recovery',begin)
    text=text[:begin]+text[end:]
    return text


def update_recipe_references(config):
    # Preserve the original protocol/method rules; point numerical summaries at
    # the promoted source instead of leaving the pre-search scalar values live.
    p=ROOT/'README.md';text=p.read_text()
    begin=text.index('It uses width 512,');end=text.index('The dataset reference',begin)
    text=text[:begin]+'''The active recipe uses width512,5000 updates and the loss/LR schedules selected
on the frozen development cohort. Exact active values and promotion evidence are
in [METHOD.md](docs/METHOD.md) and [TUNING_PROMOTION.md](docs/TUNING_PROMOTION.md).
Training and recovery equations, sampling and geometry remain unchanged.

'''+text[end:];p.write_text(text)
    p=ROOT/'docs/TUNING_COHORT.md';text=p.read_text().replace('Default.json remains\nunchanged.','Default was preserved during screening; see TUNING_PROMOTION.md for the final selection.');p.write_text(text)
    p=ROOT/'AGENTS.md';text=p.read_text()
    begin=text.index('paired SIRENs,');end=text.index('Keep all gradients',begin)
    text=text[:begin]+'''paired SIRENs, width512/four sine layers/omega30/output initialization×30;
5000 updates and AdamW decay0.001. Active PD-LK/registration/LR schedules,
parameterL2, differential c and moment weight are the promoted values in
medical/configs/default.json and docs/METHOD.md, with selection evidence in
docs/TUNING_PROMOTION.md. Reconstruction32768 samples,65536 separate particles
per direction, half-size crop, two orthogonal query pairs,8³ grid,sigma0.5,
two midpoint RK2 half-steps and scrambled2D Sobol draws remain unchanged.
Native voxel units and captured CUDA FP32 training remain the maintained path.
Frozen recovery uses brightness-flow PCG,8 intervals×8 RK2 substeps,axis cap256.

'''+text[end:];p.write_text(text)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','rank','run']);p.add_argument('--output',type=Path,required=True);p.add_argument('--gpus',type=int,nargs='+',default=list(range(8)))
    a=p.parse_args(argv);root=a.output.resolve()
    if root.is_relative_to((ROOT/'Data').resolve()) or (ROOT/'Data').resolve().is_relative_to(root):p.error('Output must be outside Data')
    if len(set(a.gpus))!=len(a.gpus) or any(g<0 for g in a.gpus):p.error('Distinct nonnegative GPU IDs required')
    if a.action=='prepare':
        base=ROOT/'medical/configs/default.json';default=json.loads(base.read_text())
        catalog=validate_catalog(dict(candidates=candidates(default),default_sha256=sha256(base),tuning_manifest_sha256=sha256(OUTPUT/'manifest.json')))
        root.mkdir(parents=True,exist_ok=False);(root/'configs').mkdir()
        for row in catalog['candidates']:(root/'configs'/(row['id']+'.json')).write_text(json.dumps(row['config'],indent=2)+'\n')
        (root/'catalog.json').write_text(json.dumps(catalog,indent=2)+'\n');print('Prepared20 sweep configurations; no fitting launched.')
    elif a.action=='rank':
        report=rank(json.loads((root/'catalog.json').read_text()),root);write_ranking(root,report)
    else:
        with (root/'.pipeline.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            pipeline(root,a.gpus)


if __name__=='__main__':main()
