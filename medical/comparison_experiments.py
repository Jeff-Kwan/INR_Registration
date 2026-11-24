"""Plan or launch standardized SIRENPath and baseline comparisons on four cohorts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import queue
import statistics
import subprocess
import sys
import threading
import time

from medical.brain_data import pair_list, prepared_paths, sha256
from medical.cohort_data import definition
from medical.comparison_metrics import DATASETS
from medical.configuration import validate_config
from medical.gpu_worker import GPUWorker
from baselines.sources import verify_source, SOURCES
from baselines.common import validate_config as validate_baseline

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('ours', 'convexadam', 'fireants', 'nir', 'demons', 'niftyreg', 'ccidir', 'nodeo', 'dnvf')
FOLDERS = dict(neurite_oasis='NeuriteOASIS', acdc='ACDC', lpba40='LPBA40', ixi='IXI_TransMorph')


def make_jobs(methods, datasets, data, output, sirenpath_config=None):
    jobs = []
    for dataset in datasets:
        root = Path(data) / FOLDERS[dataset] / 'prepared'
        if dataset in ('ixi', 'lpba40'):
            pairs = []
            for i, pair in enumerate(pair_list(dataset, 'test'), 1):
                images = prepared_paths(root, pair)
                pairs.append(dict(pair_index=i, **{k+'_image': str(p.relative_to(root)) for k,p in images.items()}))
            metadata = [root/'protocol.json', root/'preparation.json']
        else:
            pairs = definition(root, dataset)['pairs']
            metadata = [root/'manifest.json']
        for method in methods:
            config = (Path(sirenpath_config) if sirenpath_config else ROOT/'medical/configs/default.json') if method=='ours' else ROOT/f'baselines/configs/{method}_{dataset}.json'
            settings = json.loads(config.read_text())
            if method=='ours':
                validate_config(settings)
            else:
                verify_source(method)
                validate_baseline(settings, method)
            for pair in pairs:
                images = {k: root/pair[k+'_image'] for k in ('moving','fixed')}
                from medical.comparison_metrics import label_path
                labels = [label_path(dict(path=str(p)),dataset) for p in images.values()]
                target = Path(output)/dataset/method/f'pair{pair["pair_index"]:03d}'
                device = 'cuda:0' if method=='ours' else SOURCES[method].get('device','cuda:0')
                command = [sys.executable,'-m','medical' if method=='ours' else 'baselines.case_runner',
                           '--method','medical.comparison_method' if method=='ours' else method,
                           '--moving',str(images['moving']),'--fixed',str(images['fixed']),
                           '--config',str(config),'--output',str(target),'--device',device]
                jobs.append(dict(dataset=dataset,method=method,pair_index=pair['pair_index'],output=str(target),device=device,
                                 seed=settings['runtime']['seed'] if method=='ours' else settings['seed'],command=command,
                                 config_sha256=sha256(config),preparation_sha256={p.name:sha256(p) for p in metadata},
                                 missing_inputs=[str(p) for p in [*images.values(),*labels,*metadata] if not p.is_file()]))
    return jobs


def summarize(records, jobs):
    result = {}
    for dataset in dict.fromkeys(j['dataset'] for j in jobs):
        result[dataset] = {}
        for method in dict.fromkeys(j['method'] for j in jobs if j['dataset']==dataset):
            intended = [j for j in jobs if j['dataset']==dataset and j['method']==method]
            rows = [r for r in records if r['dataset']==dataset and r['method']==method]
            metrics = {k: [] for k in ('Dice','HD95','NPJ','SDlogJ')}
            successful = 0
            for row in rows:
                path = Path(row['output'])/'metrics.json'
                if row['returncode'] or not path.is_file():
                    continue
                successful += 1
                scores = json.loads(path.read_text())
                for key in metrics:
                    value = scores[key]['mean']
                    if value is not None:
                        metrics[key].append(value)
            result[dataset][method] = dict(intended=len(intended),completed=len(rows),successful=successful,
                metrics={k:dict(mean=statistics.mean(v) if v else None,std=statistics.pstdev(v) if v else None,
                               scored_pairs=len(v),unscored_pairs=len(intended)-len(v)) for k,v in metrics.items()})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    # Stored plans and result directories use "ours" as the SIRENPath method ID.
    method_ids = {'sirenpath' if m == 'ours' else m: m for m in METHODS}
    parser.add_argument(
        '--methods', nargs='+', choices=method_ids, default=list(method_ids),
        type=lambda name: 'sirenpath' if name == 'ours' or name.lower() == 'sirenpath' else name,
        help='Select SIRENPath or one or more external baselines.',
    )
    parser.add_argument('--datasets',nargs='+',choices=DATASETS,default=list(DATASETS))
    parser.add_argument('--config',type=Path,help='SIRENPath configuration file')
    parser.add_argument('--data',type=Path,default=ROOT/'Data')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpus',type=int,nargs='+',default=[0,1,2,3,4,5,6,7])
    parser.add_argument('--cpu-workers',type=int,default=4)
    execution=parser.add_mutually_exclusive_group()
    execution.add_argument('--run',action='store_true')
    execution.add_argument('--plan-only',action='store_true')
    args=parser.parse_args(argv)
    args.methods = [method_ids[name] for name in args.methods]
    if args.cpu_workers < 1:parser.error('CPU workers must be positive')
    if any(len(set(v))!=len(v) for v in (args.methods,args.datasets,args.gpus)) or any(g<0 for g in args.gpus):
        parser.error('Distinct methods, datasets and nonnegative GPU IDs required')
    output=args.output.resolve()
    if output.is_relative_to(args.data.resolve()) or output.is_relative_to(ROOT/'Data'):
        parser.error('Output cannot be under Data')
    jobs=make_jobs(args.methods,args.datasets,args.data.resolve(),output,args.config)
    plan=dict(execution=args.run,metric_protocol='standardized_v1',jobs=jobs,total_jobs=len(jobs),gpus=args.gpus,cpu_workers=args.cpu_workers,metric_source_sha256=sha256(ROOT/'medical/comparison_metrics.py'))
    if not args.run:
        print(json.dumps(plan,indent=2));return plan
    if any(j['missing_inputs'] for j in jobs):
        parser.error('Missing inputs')
    output.mkdir(parents=True,exist_ok=False)
    configs=output/'configs';configs.mkdir()
    for job in jobs:
        command=job['command'];index=command.index('--config')+1
        destination=configs/f'{job["method"]}_{job["dataset"]}.json'
        if not destination.exists():destination.write_text(Path(command[index]).read_text())
        command[index]=str(destination)
    (output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    gpu_queue,cpu_queue,records,lock=queue.Queue(),queue.Queue(),[],threading.Lock()
    for job in jobs:
        (cpu_queue if job['device']=='cpu' else gpu_queue).put(job)
    def worker(gpu,pending):
        with GPUWorker(gpu) as persistent:
            while True:
                try:job=pending.get_nowait()
                except queue.Empty:return
                target=Path(job['output']);target.parent.mkdir(parents=True,exist_ok=True)
                started=time.perf_counter()
                if job['method']=='ours':
                    status=persistent.run(job['command'],target.with_suffix('.log'))
                else:
                    with target.with_suffix('.log').open('x') as log:
                        run=subprocess.run(job['command'],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT)
                    status=dict(returncode=run.returncode)
                row={k:job[k] for k in ('dataset','method','pair_index','output','device','seed')}
                row.update(status,seconds=time.perf_counter()-started)
                with lock:
                    records.append(row)
                    progress=dict(finished=len(records),intended=len(jobs),jobs=records,datasets=summarize(records,jobs))
                    temporary=output/'progress.partial.json';temporary.write_text(json.dumps(progress,indent=2,allow_nan=False)+'\n');temporary.replace(output/'progress.json')
                pending.task_done()
    with ThreadPoolExecutor(len(args.gpus)+args.cpu_workers) as pool:
        tasks=[pool.submit(worker,g,gpu_queue) for g in args.gpus]
        tasks += [pool.submit(worker,None,cpu_queue) for _ in range(args.cpu_workers)]
        for future in tasks:future.result()
    (output/'summary.json').write_text(json.dumps(dict(jobs=records,datasets=summarize(records,jobs)),indent=2,allow_nan=False)+'\n')
    if any(r['returncode'] for r in records):raise SystemExit(1)


if __name__=='__main__':main()
