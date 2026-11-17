"""Plan official AbdomenCTCT pairs; execute only when --run is explicitly supplied."""
import argparse
import hashlib
import concurrent.futures
import json
import os
import queue
import threading
import time
import numpy as np
from pathlib import Path
import subprocess
import sys
from medical.abdomen_evaluation import PROTOCOL,PROTOCOLS
from medical.gpu_worker import GPUWorker
from baselines.sources import SOURCES

METHODS={'ours':'medical.abdomen_method',**{name:f'baselines.{name}' for name in ('fireants','convexadam','ccidir','nir','demons','niftyreg')}}
ROOT=Path(__file__).resolve().parent.parent

def make_jobs(methods,pairs,data,output,dataset="abdomenctct",own_config=None):
    protocol=PROTOCOLS[dataset]
    jobs=[]
    for method in methods:
        config=(own_config or ROOT/'medical/configs/default.json') if method=='ours' else ROOT/f'baselines/configs/{method}_{dataset}.json'
        for index in pairs:
            pair=protocol['eval_pairs'][index-1]
            fixed,moving=(data/pair[key] for key in ('fixed','moving'))
            name=f'{fixed.name.split("_")[1]}_to_{moving.name.split("_")[1]}'
            inputs=[fixed,moving,*(data/pair[key].replace('imagesTr','labelsTr') for key in ('fixed','moving'))]
            target=output/method/name
            settings=json.loads(config.read_text())
            # Only ccIDIR's published DIR-Lab protocol mandates 50 fits. These
            # Learn2Reg adaptations and our method retain their configured seed.
            seed=settings['runtime']['seed'] if method=='ours' else settings['seed']
            module='medical' if method=='ours' else 'baselines.case_runner'
            adapter=('medical.oasis_method' if dataset=='oasis' else METHODS[method]) if method=='ours' else method
            device = SOURCES[method].get('device', 'cuda:0') if method != 'ours' else 'cuda:0'
            command=[sys.executable,'-m',module,'--method',adapter,'--moving',str(moving),'--fixed',str(fixed),
                     '--config',str(config),'--device',device,'--output',str(target)]
            if method!='ours': command+=['--seed',str(seed)]
            jobs.append({'method':method,'pair_index':index,'seed':seed,'output':str(target),'command':command,
                         'missing_inputs':[str(p) for p in inputs if not p.is_file()]})
    return jobs

def main(argv=None,*,dataset="abdomenctct"):
    count=len(PROTOCOLS[dataset]["eval_pairs"])
    name="AbdomenCTCT" if dataset=="abdomenctct" else "OASIS"
    p=argparse.ArgumentParser(description=f"Plan official {name} validation pairs; execution requires --run.")
    p.add_argument('--methods',choices=METHODS,nargs='+',default=list(METHODS))
    selection=p.add_mutually_exclusive_group()
    selection.add_argument('--pairs',type=int,nargs='+')
    selection.add_argument('--subset',choices=['quick4'],help='Frozen dataset-specific prototype subset')
    p.add_argument('--data',type=Path,default=Path(f'Data/{name}'))
    p.add_argument('--output',type=Path,default=Path(f'results/{dataset}'))
    p.add_argument('--gpus',type=int,nargs='+',default=[0,1,2,3])
    p.add_argument('--config',type=Path,help='Our method configuration; baseline presets remain separate')
    p.add_argument('--run',action='store_true')
    args=p.parse_args(argv)
    subset=None
    if args.subset:
        subset=json.loads((ROOT/f'medical/protocols/{dataset}_quick4.json').read_text())
        protocol_path=ROOT/f'medical/protocols/{dataset}_val.json'
        if hashlib.sha256(protocol_path.read_bytes()).hexdigest()!=subset['protocol_sha256']:
            p.error('Frozen quick4 protocol hash mismatch')
        args.pairs=subset['pair_indices']
        if [PROTOCOLS[dataset]['eval_pairs'][i-1] for i in args.pairs]!=subset['pairs']:
            p.error('Frozen quick4 pairs do not match official protocol')
    elif args.pairs is None:
        args.pairs=list(range(1,count+1))
    if len(set(args.pairs))!=len(args.pairs) or any(i not in range(1,count+1) for i in args.pairs): p.error(f'Distinct pair indices 1..{count} required')
    if len(set(args.gpus))!=len(args.gpus) or any(g<0 for g in args.gpus): p.error('Distinct nonnegative GPUs required')
    if len(set(args.methods))!=len(args.methods): p.error('Methods must be distinct')
    if args.output.resolve().is_relative_to((ROOT/'Data').resolve()): p.error('Output cannot be inside Data')
    jobs=make_jobs(args.methods,args.pairs,args.data,args.output,dataset,args.config)
    if not args.run:
        print(json.dumps({'execution':False,'single_seed':True,'subset':subset,'jobs':jobs},indent=2));return
    if any(j['missing_inputs'] for j in jobs): p.error('Missing prepared inputs')
    from baselines.sources import verify_source
    for name in args.methods:
        if name!='ours': verify_source(name)
    args.output.mkdir(parents=True,exist_ok=False)
    configs=args.output/'configs';configs.mkdir()
    snapshots={}
    for job in jobs:
        command=job['command']; index=command.index('--config')+1
        source=Path(command[index])
        if source not in snapshots:
            target=configs/source.name;target.write_text(source.read_text());snapshots[source]=target
        command[index]=str(snapshots[source])
    (args.output/'plan.json').write_text(json.dumps({'jobs':jobs,'gpus':args.gpus,'dispatch':'shared_queue','ours_workers':'persistent_processes','subset':subset},indent=2)+'\n')
    pending=queue.Queue()
    for job in jobs: pending.put(job)
    finished=[];lock=threading.Lock()
    def worker(gpu):
        with GPUWorker(gpu) as persistent:
            return work(gpu,persistent)
    def work(gpu,persistent):
        results=[]
        while True:
            try: job=pending.get_nowait()
            except queue.Empty: break
            target=Path(job['output']);target.parent.mkdir(parents=True,exist_ok=True)
            started=time.perf_counter()
            print(f'Starting pair {job["pair_index"]} on GPU {gpu}',flush=True)
            if job['method']=='ours':
                status=persistent.run(job['command'],target.with_suffix('.log'))
            else:
                with target.with_suffix('.log').open('x') as log:
                    completed=subprocess.run(job['command'],env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT,check=False)
                status={'returncode':completed.returncode}
            record={'method':job['method'],'pair_index':job['pair_index'],'gpu':gpu,'seed':job['seed'],**status,'output':job['output'],'seconds':time.perf_counter()-started}
            results.append(record)
            with lock:
                finished.append(record)
                temp=args.output/'progress.partial.json'
                temp.write_text(json.dumps({'finished':len(finished),'intended':len(jobs),'jobs':finished},indent=2)+'\n')
                temp.replace(args.output/'progress.json')
                print(json.dumps(record),flush=True)
            pending.task_done()
        return results
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures=[pool.submit(worker,gpu) for gpu in args.gpus]
        results=[row for future in futures for row in future.result()]
    summary={'jobs':results,'methods':{}}
    for method in args.methods:
        rows=[r for r in results if r['method']==method]
        aggregates={}
        for metric in ('DSC','HD95','LogJacDetStd','moving_psnr_db','fixed_psnr_db','seconds'):
            values=[]
            for row in rows:
                path=Path(row['output'])/'metrics.json'
                if row['returncode']==0 and path.is_file():
                    scores=json.loads(path.read_text())
                    reconstruction=scores.get('reconstruction',{})
                    value=(row['seconds'] if metric=='seconds' else
                           reconstruction.get(metric.split('_')[0],{}).get('psnr_db')
                       if metric.endswith('_psnr_db') else scores[metric]['mean'])
                    if value is not None: values.append(value)
            aggregates[metric]={'mean':float(np.mean(values)) if values else None,
                                'std':float(np.std(values)) if values else None,
                                '30':float(np.quantile(values,.7 if metric=='LogJacDetStd' else .3)) if values else None,
                                'scored_pairs':len(values),'intended_pairs':len(rows)}
        summary['methods'][method]=aggregates
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if any(row['returncode'] for row in results): raise SystemExit(1)

if __name__=='__main__': main()
