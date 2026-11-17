"""Plan by default; explicitly launch one configuration on frozen tuning pairs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import queue
import statistics
import sys
import threading
import time

from medical.brain_data import sha256
from medical.configuration import validate_config
from medical.gpu_worker import GPUWorker
from scripts.prepare_tuning_cohort import ROOT, OUTPUT, select, paths, verify_frozen

DATASETS=('oasis','acdc','lpba40','ixi')


def make_jobs(config,output,datasets):
    settings=json.loads(Path(config).read_text());validate_config(settings)
    if settings['runtime']['seed']!=42:raise ValueError('Tuning remains single-seed42')
    manifest=json.loads((OUTPUT/'manifest.json').read_text())
    expected=[dict(r,**paths(r)) for group in select().values() for r in group]
    if manifest['pairs']!=expected:raise ValueError('Frozen selection changed')
    jobs=[]
    for pair in manifest['pairs']:
        if pair['dataset'] not in datasets:continue
        target=Path(output)/pair['dataset']/f'pair{pair["pair_index"]:03d}'
        command=[sys.executable,'-m','medical','--method','medical.tuning_method','--moving',str(ROOT/pair['moving_image']),
                 '--fixed',str(ROOT/pair['fixed_image']),'--config',str(config),'--output',str(target),'--device','cuda:0']
        jobs.append(dict(dataset=pair['dataset'],pair_index=pair['pair_index'],output=str(target),command=command))
    return jobs


def summarize(records,jobs):
    datasets={}
    for dataset in dict.fromkeys(j['dataset'] for j in jobs):
        intended=[j for j in jobs if j['dataset']==dataset];rows=[r for r in records if r['dataset']==dataset]
        metrics={key:[] for key in ('Dice','HD95','NPJ','SDlogJ')};times={key:[] for key in ('wall_seconds','training_seconds','recovery_seconds','evaluation_seconds')}
        successful=0
        for row in rows:
            path=Path(row['output'])/'metrics.json'
            if row['returncode'] or not path.is_file():continue
            scores=json.loads(path.read_text());successful+=1
            for key in metrics:
                value=scores[key]['mean']
                if value is not None:metrics[key].append(value)
            times['wall_seconds'].append(row['seconds'])
            for key in times:
                if key!='wall_seconds' and key in scores['timings']:times[key].append(scores['timings'][key])
        def aggregate(values):
            return dict(mean=statistics.mean(values) if values else None,std=statistics.pstdev(values) if values else None,scored_pairs=len(values))
        datasets[dataset]=dict(intended=len(intended),completed=len(rows),successful=successful,failed=sum(r['returncode']!=0 for r in rows),
                               metrics={k:dict(**aggregate(v),unscored_pairs=len(intended)-len(v)) for k,v in metrics.items()},timings={k:aggregate(v) for k,v in times.items()})
    # Equal dataset weight; never let cohort size select the winning recipe.
    means=[v['metrics']['Dice']['mean'] for v in datasets.values()]
    complete=all(v['successful']==v['intended'] and v['metrics']['Dice']['scored_pairs']==v['intended'] for v in datasets.values())
    return dict(split='development_tuning32',datasets=datasets,complete=complete,
                macro_dataset_Dice=statistics.mean(means) if complete and all(v is not None for v in means) else None)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'medical/configs/default.json')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--datasets',choices=DATASETS,nargs='+',default=list(DATASETS))
    p.add_argument('--gpus',type=int,nargs='+',default=list(range(8)))
    mode=p.add_mutually_exclusive_group();mode.add_argument('--run',action='store_true');mode.add_argument('--plan-only',action='store_true')
    args=p.parse_args(argv);output=args.output.resolve();config=args.config.resolve()
    if len(set(args.datasets))!=len(args.datasets) or len(set(args.gpus))!=len(args.gpus) or any(g<0 for g in args.gpus):p.error('Distinct datasets and nonnegative GPU IDs required')
    if output.is_relative_to((ROOT/'Data').resolve()) or (ROOT/'Data').resolve().is_relative_to(output):p.error('Use a fresh output directory outside Data')
    if output.exists():p.error('Output already exists; use a fresh directory')
    jobs=make_jobs(config,output,args.datasets)
    plan=dict(split='development_tuning32',execution=args.run,total_jobs=len(jobs),gpus=args.gpus,metric_protocol='standardized_v1',
              config_sha256=sha256(config),tuning_manifest_sha256=sha256(OUTPUT/'manifest.json'),jobs=jobs)
    if not args.run:print(json.dumps(plan,indent=2));return plan
    verify_frozen()
    output.mkdir(parents=True,exist_ok=False)
    snapshot=output/'config.json';snapshot.write_text(config.read_text())
    for job in jobs:job['command'][job['command'].index('--config')+1]=str(snapshot)
    (output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    pending=queue.Queue();records=[];lock=threading.Lock()
    for job in jobs:pending.put(job)
    def worker(gpu):
        with GPUWorker(gpu) as persistent:
            while True:
                try:job=pending.get_nowait()
                except queue.Empty:return
                target=Path(job['output']);target.parent.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
                status=persistent.run(job['command'],target.with_suffix('.log'))
                row={k:job[k] for k in ('dataset','pair_index','output')};row.update(status,gpu=gpu,seconds=time.perf_counter()-started)
                with lock:
                    records.append(row)
                    progress=dict(finished=len(records),intended=len(jobs),jobs=records,**summarize(records,jobs))
                    temporary=output/'progress.partial.json';temporary.write_text(json.dumps(progress,indent=2,allow_nan=False)+'\n');temporary.replace(output/'progress.json')
                pending.task_done()
    started=time.perf_counter()
    with ThreadPoolExecutor(len(args.gpus)) as pool:
        tasks=[pool.submit(worker,gpu) for gpu in args.gpus]
        for task in tasks:task.result()
    summary=dict(jobs=records,wall_seconds=time.perf_counter()-started,**summarize(records,jobs))
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    if not summary['complete']:raise SystemExit(1)


if __name__=='__main__':main()
