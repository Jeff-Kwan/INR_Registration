"""Plan FireANTs, ConvexAdam–MIND and NIR on all prepared datasets; --run executes."""
import argparse
from collections import Counter
import concurrent.futures
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time

import numpy as np

from .common import validate_config
from .sources import ROOT, provenance, verify_source
from .run_dirlab import make_jobs as dirlab_jobs, summarize as dirlab_summary
from medical.abdomen_experiments import make_jobs as label_jobs
from medical.abdomen_evaluation import PROTOCOLS

METHODS=('fireants','convexadam','nir')
DATASETS=('dirlab','abdomenctct','oasis')
REPO=ROOT.parent


def make_jobs(methods,datasets,data,output):
    if not methods or len(set(methods))!=len(methods) or set(methods)-set(METHODS):
        raise ValueError('Select distinct methods from fireants, convexadam, nir')
    if not datasets or len(set(datasets))!=len(datasets) or set(datasets)-set(DATASETS):
        raise ValueError('Select distinct supported datasets')
    for method in methods:
        verify_source(method)
    jobs=[]
    for dataset in datasets:
        if dataset=='dirlab':
            selected=dirlab_jobs(methods,list(range(1,11)),data/'DIRLab4DCT_IDIR',output/dataset,ROOT/'configs')
        else:
            folder='AbdomenCTCT' if dataset=='abdomenctct' else 'OASIS'
            selected=label_jobs(methods,list(range(1,len(PROTOCOLS[dataset]['eval_pairs'])+1)),data/folder,output/dataset,dataset)
        for job in selected:
            config_path=Path(job['command'][job['command'].index('--config')+1])
            validate_config(json.loads(config_path.read_text()),job['method'])
            job.update(dataset=dataset,job_id=f"{dataset}/{job['method']}/{Path(job['output']).name}")
            jobs.append(job)
    if len({j['output'] for j in jobs})!=len(jobs):
        raise ValueError('Duplicate job outputs')
    return jobs


def write_json(path,value):
    temporary=path.with_suffix('.partial.json')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def summarize(results):
    groups={}
    for dataset in DATASETS:
        rows=[r for r in results if r['dataset']==dataset]
        if not rows: continue
        if dataset=='dirlab':
            groups[dataset]=dirlab_summary(rows)
            continue
        groups[dataset]={}
        for method in sorted({r['method'] for r in rows}):
            selected=[r for r in rows if r['method']==method]
            metrics={}
            for name in ('DSC','HD95','LogJacDetStd'):
                values=[]
                for row in selected:
                    path=Path(row['output'])/'metrics.json'
                    if row['returncode']==0 and path.is_file():
                        value=json.loads(path.read_text())[name]['mean']
                        if value is not None: values.append(value)
                metrics[name]={'mean':float(np.mean(values)) if values else None,
                               'std':float(np.std(values)) if values else None,
                               '30':float(np.quantile(values,.7 if name=='LogJacDetStd' else .3)) if values else None,
                               'scored_pairs':len(values),'intended_pairs':len(selected)}
            groups[dataset][method]=metrics
    return groups


def execute(jobs,gpus,output):
    """One persistent worker per GPU; each pulls the next independent case."""
    if any(j['missing_inputs'] for j in jobs):
        raise ValueError('Missing prepared inputs; inspect the plan')
    for method in {j['method'] for j in jobs}: verify_source(method)
    output.mkdir(parents=True,exist_ok=False)
    configs=output/'configs';configs.mkdir()
    # Freeze configs before launching, so edits during a long run cannot change
    # settings for later cases. Per-case run.json also saves the resolved config.
    snapshots={}
    for job in jobs:
        command=job['command'][:]
        index=command.index('--config')+1
        source=Path(command[index])
        if source not in snapshots:
            target=configs/f"{job['method']}_{job['dataset']}.json"
            target.write_text(source.read_text());snapshots[source]=target
        job['command']=[*command[:index],str(snapshots[source]),*command[index+1:]]
    write_json(output/'plan.json',{'jobs':jobs,'gpus':gpus,'sources':{m:provenance(m) for m in {j['method'] for j in jobs}},
                                 'cpu_threads_per_process':4,'maps_saved':False})
    pending=queue.Queue()
    for job in jobs: pending.put(job)
    lock=threading.Lock();results=[]
    def worker(gpu):
        while True:
            try: job=pending.get_nowait()
            except queue.Empty: return
            started=time.perf_counter()
            target=Path(job['output']);target.parent.mkdir(parents=True,exist_ok=True)
            record={k:job[k] for k in ('method','dataset','job_id','seed','output')}
            for key in ('case','pair_index'):
                if key in job: record[key]=job[key]
            record['gpu']=gpu
            try:
                with target.with_suffix('.log').open('x') as log:
                    completed=subprocess.run(job['command'],cwd=REPO,
                        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1'),
                        stdout=log,stderr=subprocess.STDOUT,check=False)
                record['returncode']=completed.returncode
            except Exception as exc:
                record.update(returncode=-1,error=f'{type(exc).__name__}: {exc}')
            record['seconds']=time.perf_counter()-started
            with lock:
                results.append(record)
                write_json(output/'progress.json',{'finished':len(results),'intended':len(jobs),'jobs':results})
                print(f"[{len(results)}/{len(jobs)}] GPU {gpu}: {job['job_id']} exit={record['returncode']}",flush=True)
            pending.task_done()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures=[pool.submit(worker,gpu) for gpu in gpus]
        for future in futures: future.result()
    results.sort(key=lambda row:row['job_id'])
    summary={'jobs':results,'intended_jobs':len(jobs),'failed_jobs':sum(r['returncode']!=0 for r in results),
             'datasets':summarize(results)}
    write_json(output/'summary.json',summary)
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS))
    parser.add_argument('--datasets',nargs='+',choices=DATASETS,default=list(DATASETS))
    parser.add_argument('--gpus',nargs='+',type=int,default=[0,1,2,3])
    parser.add_argument('--data',type=Path,default=REPO/'Data')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--run',action='store_true',help='Execute; without this flag only print the plan')
    args=parser.parse_args(argv)
    if len(set(args.gpus))!=len(args.gpus) or any(g<0 for g in args.gpus): parser.error('GPUs must be distinct nonnegative indices')
    output=(args.output or ROOT/'results'/('_'.join(args.methods)+'_all')).resolve()
    data=args.data.resolve()
    if output.is_relative_to(data) or output.is_relative_to((REPO/'Data').resolve()): parser.error('Output cannot be inside Data')
    jobs=make_jobs(args.methods,args.datasets,data,output)
    plan={'execution':False,'gpus':args.gpus,'job_count':len(jobs),'output':str(output),
          'counts':dict(Counter(f"{j['method']}/{j['dataset']}" for j in jobs)),'jobs':jobs}
    if not args.run:
        print(json.dumps(plan,indent=2));return
    if any(j['missing_inputs'] for j in jobs):
        print(json.dumps(plan,indent=2));parser.error('Missing inputs; no jobs launched')
    if execute(jobs,args.gpus,output)['failed_jobs']: raise SystemExit(1)


if __name__=='__main__': main()
