"""Prepare complete replacement of NIR/DNVF results; execute only with --run."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import threading
import time
from medical.comparison_experiments import ROOT, make_jobs, summarize
from medical.brain_data import sha256
from medical.sirenpath_ablations import require_idle_gpus
from baselines.comparison_report import refresh

METHODS=('nir','dnvf')
DATASETS=('neurite_oasis','acdc','lpba40','ixi')
SOURCE_FILES=('baselines/nir.py','baselines/dnvf.py','baselines/precision.py','baselines/sources.json','baselines/patches/nir-mri.patch','medical/comparison_metrics.py')


def write_json(path,data):
    temporary=path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def replacement_plan(root):
    jobs=make_jobs(list(METHODS),list(DATASETS),ROOT/'Data',root)
    for job in jobs:
        command=job['command'];i=command.index('--config')+1
        command[i]=str(root/'configs'/f"{job['method']}_{job['dataset']}.json")
    return dict(methods=list(METHODS),jobs=jobs,total_jobs=len(jobs),
                execution=False,overwrite='replace; no archive or backup of invalid outputs',
                source_sha256={name:sha256(ROOT/name) for name in SOURCE_FILES})


def purge_method_outputs(root,methods=METHODS,datasets=DATASETS):
    # Fixed allowlist, no Data paths and no symlink traversal.
    for ds in datasets:
        for method in methods:
            if ds not in DATASETS or method not in METHODS:raise ValueError('Unsupported replacement scope')
            target=root/ds/method
            if target.is_symlink() or target.parent.is_symlink():raise ValueError('Refusing a symlinked result boundary')
            if not target.resolve().is_relative_to(root.resolve()):raise ValueError('Invalid replacement path')
    for ds in datasets:
        for method in methods:
            target=root/ds/method
            if target.exists():shutil.rmtree(target)


def snapshot(root,records,jobs,pending):
    data=dict(jobs=records,datasets=summarize(records,jobs),finished=len(records),intended=len(jobs),pending_replacement_methods=pending)
    write_json(root/'summary.json',data);write_json(root/'progress.json',data)
    refresh()


def prepare(root):
    if (root/'replacement.running').exists():raise ValueError('Replacement run is active; cannot prepare over it')
    plan=replacement_plan(root)
    if any(j['missing_inputs'] for j in plan['jobs']):raise ValueError('Missing frozen inputs')
    old=json.loads((root/'summary.json').read_text());original=json.loads((root/'plan.json').read_text())
    prior=root/'replacement_plan.json'
    fallback=json.loads(prior.read_text()).get('runtime_estimate',{}) if prior.exists() else {}
    estimates={}
    for method in METHODS:
        times=[j['seconds'] for j in old['jobs'] if j['method']==method]
        estimates[method]=dict(cases=309,old_gpu_seconds=sum(times),eta_minutes_8gpu=sum(times)*1.1/8/60) if times else fallback[method]
    plan['runtime_estimate']=estimates
    retained=[j for j in old['jobs'] if j['method'] not in METHODS]
    replacements={(j['dataset'],j['method'],j['pair_index']):j for j in plan['jobs']}
    jobs=[replacements.get((j['dataset'],j['method'],j['pair_index']),j) for j in original['jobs']]
    purge_method_outputs(root)
    for method in METHODS:
        for ds in DATASETS:
            source=ROOT/f'baselines/configs/{method}_{ds}.json';destination=root/'configs'/source.name
            destination.write_text(source.read_text())
    original.update(jobs=jobs,pending_replacement_methods=list(METHODS))
    write_json(root/'plan.json',original);write_json(root/'replacement_plan.json',plan)
    snapshot(root,retained,jobs,list(METHODS))
    return plan


def run(root,gpus):
    plan=json.loads((root/'replacement_plan.json').read_text())
    for name,expected in plan['source_sha256'].items():
        if sha256(ROOT/name)!=expected:raise ValueError(f'Prepared source changed: {name}; prepare again')
    for job in plan['jobs']:
        cfg=Path(job['command'][job['command'].index('--config')+1])
        if sha256(cfg)!=job['config_sha256']:raise ValueError('Prepared configuration changed')
        if Path(job['output']).exists():raise ValueError('A replacement case already exists; refusing duplicate fitting')
    require_idle_gpus(gpus)
    lockfile=root/'replacement.running'
    with lockfile.open('x') as stream:stream.write(str(os.getpid()))
    try:
        parent=json.loads((root/'plan.json').read_text());records=json.loads((root/'summary.json').read_text())['jobs']
        pending=queue.Queue()
        for job in plan['jobs']:pending.put(job)
        lock=threading.Lock()
        def worker(gpu):
            while True:
                try:job=pending.get_nowait()
                except queue.Empty:return
                target=Path(job['output']);target.parent.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
                with target.with_suffix('.log').open('x') as log:
                    process=subprocess.run(job['command'],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1'),stdout=log,stderr=subprocess.STDOUT)
                record={k:job[k] for k in ('dataset','method','pair_index','output','seed','device')}
                record.update(returncode=process.returncode,seconds=time.perf_counter()-started,gpu=gpu)
                with lock:
                    records.append(record)
                    still_pending=[m for m in METHODS if sum(r['method']==m for r in records)<309]
                    snapshot(root,records,parent['jobs'],still_pending)
                    print(json.dumps(record),flush=True)
                pending.task_done()
        with ThreadPoolExecutor(len(gpus)) as pool:
            tasks=[pool.submit(worker,gpu) for gpu in gpus]
            for task in tasks:task.result()
        parent['pending_replacement_methods']=[];write_json(root/'plan.json',parent)
        if any(r['returncode'] for r in records if r['method'] in METHODS):raise SystemExit(1)
    finally:lockfile.unlink()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'baselines/results/official_neural_20261116')
    mode=p.add_mutually_exclusive_group();mode.add_argument('--prepare',action='store_true');mode.add_argument('--run',action='store_true');mode.add_argument('--plan-only',action='store_true')
    p.add_argument('--gpus',nargs='+',type=int,default=list(range(8)))
    a=p.parse_args(argv);root=a.output.resolve()
    if root!=ROOT/'baselines/results/official_neural_20261116':p.error('Replacement is restricted to the existing official neural output root')
    if not a.gpus or len(set(a.gpus))!=len(a.gpus) or any(g<0 for g in a.gpus):p.error('Distinct nonnegative GPU IDs required')
    if a.run:return run(root,a.gpus)
    data=prepare(root) if a.prepare else replacement_plan(root)
    print(json.dumps(data,indent=2))


if __name__=='__main__':main()
