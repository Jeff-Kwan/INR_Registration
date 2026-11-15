"""Explicit one-case-per-dataset baseline profiling; does not launch full cohorts."""
import argparse
import concurrent.futures
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

METHODS=['fireants','convexadam','ccidir','nir']
DATASETS=['dirlab','abdomenctct','oasis']

def inputs(dataset):
    if dataset=='dirlab':
        root=Path('Data/DIRLab4DCT_IDIR/case06')
        return root/'T50.nii.gz',root/'T00.nii.gz'
    from medical.abdomen_evaluation import PROTOCOLS
    root=Path('Data')/('AbdomenCTCT' if dataset=='abdomenctct' else 'OASIS')
    pair=PROTOCOLS[dataset]['eval_pairs'][0]
    return root/pair['moving'],root/pair['fixed']

def child(method,dataset,output):
    import torch
    from medical.case_runner import run_case
    moving,fixed=inputs(dataset)
    config_name=method if dataset=='dirlab' else method+'_'+dataset
    config=json.loads((Path('baselines/configs')/(config_name+'.json')).read_text())
    config['seed']=0 if method=='ccidir' and dataset=='dirlab' else config['seed']
    torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    result={'method':method,'dataset':dataset,'seed':config['seed'],'moving':str(moving),'fixed':str(fixed)}
    try:
        module=importlib.import_module('baselines.'+method)
        run_case(moving,fixed,config,output/'case',module.run_case,'cuda:0')
        result['status']='ok'
    except Exception as exc:
        import traceback
        traceback.print_exc()
        result.update(status='failed',error=f'{type(exc).__name__}: {exc}')
    finally:
        torch.cuda.synchronize()
        result.update(seconds=time.perf_counter()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                      peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        (output/'profile.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
    return 0 if result['status']=='ok' else 1

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--method',choices=METHODS)
    p.add_argument('--dataset',choices=DATASETS)
    args=p.parse_args()
    if args.method:
        raise SystemExit(child(args.method,args.dataset,args.output))
    args.output.mkdir(parents=True,exist_ok=False)
    def worker(gpu,method):
        results=[]
        for dataset in DATASETS:
            output=args.output/method/dataset;output.mkdir(parents=True)
            command=[sys.executable,'-m','baselines.profile','--method',method,'--dataset',dataset,'--output',str(output)]
            started=time.perf_counter()
            with (output/'run.log').open('x') as log:
                completed=subprocess.run(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1'),stdout=log,stderr=subprocess.STDOUT)
            record=json.loads((output/'profile.json').read_text()) if (output/'profile.json').exists() else {'method':method,'dataset':dataset,'status':'process_failed'}
            record.update(gpu=gpu,process_seconds=time.perf_counter()-started,returncode=completed.returncode)
            results.append(record)
            (output/'profile.json').write_text(json.dumps(record,indent=2)+'\n')
        return results
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(worker,gpu,method) for gpu,method in enumerate(METHODS)]
        results=[r for future in futures for r in future.result()]
    (args.output/'summary.json').write_text(json.dumps(results,indent=2)+'\n')

if __name__=='__main__':main()
