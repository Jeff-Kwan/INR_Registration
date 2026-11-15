"""Run the 10 prepared DIR-Lab cases with one worker per GPU."""
import argparse
import concurrent.futures
import json
import queue
import sys
import time
from pathlib import Path

# Allow direct script execution from the repository.
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from medical.gpu_worker import GPUWorker


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('medical/configs/default.json'))
    p.add_argument('--data',type=Path,default=Path('Data/DIRLab4DCT_IDIR'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpus',type=int,nargs='+',default=[0,1,2,3])
    selection=p.add_mutually_exclusive_group()
    selection.add_argument('--subset',choices=['quick4'],help='Frozen four smallest prepared cases')
    selection.add_argument('--cases',type=int,nargs='+',help='Explicit prepared case numbers, 1 through 10')
    p.add_argument('--plan-only',action='store_true',help='Print jobs without creating outputs or starting workers')
    args=p.parse_args()
    if len(set(args.gpus))!=len(args.gpus) or any(i<0 for i in args.gpus): p.error('GPU IDs must be distinct nonnegative integers')
    if args.output.resolve().is_relative_to((Path.cwd()/'Data').resolve()): p.error('Output cannot be under Data')
    if args.cases is not None and (len(set(args.cases))!=len(args.cases) or any(n<1 or n>10 for n in args.cases)):
        p.error('Cases must be distinct numbers from 1 through 10')
    subset=None
    cases=list(range(1,11))
    if args.subset:
        subset=json.loads((Path(__file__).resolve().parents[1]/'medical/protocols/dirlab_quick4.json').read_text())
        cases=subset['case_indices']
    elif args.cases is not None:
        cases=args.cases
    if args.plan_only:
        jobs=[{'case':n,'moving':str(args.data/f'case{n:02d}/T50.nii.gz'),
               'fixed':str(args.data/f'case{n:02d}/T00.nii.gz'),'output':str(args.output/f'case{n:02d}')} for n in cases]
        print(json.dumps({'execution':False,'subset':subset,'config':str(args.config),'gpus':args.gpus,'dispatch':'shared_queue','jobs':jobs},indent=2))
        return
    args.output.mkdir(parents=True,exist_ok=False)
    config=args.output/'config.json'
    config.write_text(args.config.read_text())
    (args.output/'plan.json').write_text(json.dumps({'cases':cases,'subset':subset,'gpus':args.gpus,'dispatch':'shared_queue'},indent=2)+'\n')
    pending=queue.Queue()
    for n in cases: pending.put(n)
    def worker(gpu):
        with GPUWorker(gpu) as persistent:
            return work(gpu,persistent)
    def work(gpu,persistent):
        results=[]
        while True:
            try: n=pending.get_nowait()
            except queue.Empty: break
            case=args.data/f'case{n:02d}'; out=args.output/f'case{n:02d}'
            cmd=[sys.executable,'-m','medical','--moving',str(case/'T50.nii.gz'),'--fixed',str(case/'T00.nii.gz'),'--config',str(config),'--device',f'cuda:{gpu}','--output',str(out)]
            print(f'Starting case{n:02d} on GPU {gpu}',flush=True)
            started=time.perf_counter()
            completed=persistent.run(cmd,out.with_suffix('.log'))
            record={'case':n,'gpu':gpu,**completed,'seconds':time.perf_counter()-started}
            results.append(record)
            print(json.dumps(record),flush=True)
            pending.task_done()
        return results
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        tasks=[pool.submit(worker,gpu) for gpu in args.gpus]
        results=[r for task in tasks for r in task.result()]
    (args.output/'summary.json').write_text(json.dumps(sorted(results,key=lambda r:r['case']),indent=2)+'\n')
    if any(r['returncode'] for r in results): raise SystemExit(1)

if __name__=='__main__': main()
