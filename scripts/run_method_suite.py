"""Plan six single-seed full-dataset trials; --run explicitly starts GPU work."""
import argparse,concurrent.futures,hashlib,json,queue,sys,threading,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from medical.abdomen_experiments import make_jobs
from medical.abdomen_evaluation import PROTOCOLS
from medical.configuration import validate_config
from medical.gpu_worker import GPUWorker
ROOT=Path(__file__).resolve().parents[1]
TRIALS=('01_reference','02_target_moments','03_registration_003','04_pdlk_003','05_context120','06_native_grid')
DATASETS=('oasis','dirlab','abdomenctct')

def make_plan(trials,datasets,output,config_override=None):
    jobs=[];configs={}
    for trial in trials:
        config=Path(config_override).resolve() if config_override else ROOT/'medical/configs/full_suite'/f'{trial}.json'
        settings=json.loads(config.read_text());validate_config(settings);configs[trial]=settings
        for dataset in datasets:
            destination=output/trial/dataset
            if dataset!='dirlab':
                data=ROOT/'Data'/('OASIS' if dataset=='oasis' else 'AbdomenCTCT')
                rows=make_jobs(['ours'],list(range(1,len(PROTOCOLS[dataset]['eval_pairs'])+1)),data,destination,dataset,config)
            else:
                rows=[]
                for i in range(1,11):
                    data=ROOT/'Data/DIRLab4DCT_IDIR'/f'case{i:02d}';target=destination/f'case{i:02d}'
                    inputs=[data/f'{phase}{suffix}' for phase in ('T00','T50') for suffix in ('.nii.gz','_lung_mask.nii.gz','_landmarks_voxel.csv')]
                    command=[sys.executable,'-m','medical','--method','medical.paired_method','--moving',str(data/'T50.nii.gz'),'--fixed',str(data/'T00.nii.gz'),'--config',str(config),'--device','cuda:0','--output',str(target)]
                    rows.append({'method':'ours','pair_index':i,'seed':settings['runtime']['seed'],'command':command,'output':str(target),'missing_inputs':[str(p) for p in inputs if not p.is_file()]})
            jobs.extend({**row,'trial':trial,'dataset':dataset} for row in rows)
    return jobs,configs

def summarize(records,jobs):
    import statistics
    groups={}
    for trial,dataset in dict.fromkeys((j['trial'],j['dataset']) for j in jobs):
        rows=[r for r in records if (r['trial'],r['dataset'])==(trial,dataset)]
        values={};success=0;converged=0;intervals=0
        for row in rows:
            p=Path(row['output'])/'metrics.json'
            if row['returncode'] or not p.is_file():continue
            success+=1;m=json.loads(p.read_text())
            scalars={'seconds':row['seconds'],'folding_fraction':m.get('folding_fraction')}
            for key in ('DSC','HD95','LogJacDetStd'):
                if key in m:scalars[key]=m[key]['mean']
            if 'continuous_tre_mm' in m:
                scalars['forward_continuous_tre_mm']=m['continuous_tre_mm']['mean_mm']
                scalars['forward_rounded_tre_mm']=m['idir_rounded_tre_mm']['mean_mm']
                scalars['reverse_continuous_tre_mm']=m['reverse_T50_to_T00']['continuous_tre_mm']['mean_mm']
            for scope,scores in m.get('reconstruction',{}).items():
                if isinstance(scores,dict):
                    for endpoint in ('moving','fixed'):
                        if endpoint in scores:scalars[f'{scope}_{endpoint}_psnr_db']=scores[endpoint]['psnr_db']
            for key,value in scalars.items():
                if value is not None:values.setdefault(key,[]).append(value)
            diagnostics=m.get('recovery',[]);intervals+=len(diagnostics);converged+=sum(d['converged'] for d in diagnostics)
        groups.setdefault(trial,{})[dataset]={'completed':len(rows),'successful':success,'intended':sum(j['trial']==trial and j['dataset']==dataset for j in jobs),'pcg_converged':converged,'pcg_intervals':intervals,'metrics':{k:{'mean':statistics.mean(v),'scored':len(v)} for k,v in values.items()}}
    return groups

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trials',choices=TRIALS,nargs='+',default=list(TRIALS))
    p.add_argument('--datasets',choices=DATASETS,nargs='+',default=list(DATASETS))
    p.add_argument('--gpus',type=int,nargs='+',default=[0,1,2,3])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,help='Override the configuration for one explicitly selected trial')
    p.add_argument('--run',action='store_true')
    args=p.parse_args(argv);output=args.output.resolve()
    for name in ('trials','datasets','gpus'):
        value=getattr(args,name)
        if len(set(value))!=len(value):p.error(f'Duplicate {name}')
    if any(g<0 for g in args.gpus):p.error('GPU IDs must be nonnegative')
    if output.is_relative_to((ROOT/'Data').resolve()):p.error('Output cannot be under Data')
    if args.config and len(args.trials)!=1:p.error('--config requires exactly one --trials selection')
    jobs,configs=make_plan(args.trials,args.datasets,output,config_override=args.config)
    counts={t:{d:sum(j['trial']==t and j['dataset']==d for j in jobs) for d in args.datasets} for t in args.trials}
    missing=sorted({x for j in jobs for x in j['missing_inputs']})
    plan={'execution':args.run,'counts':counts,'total_jobs':len(jobs),'gpus':args.gpus,'dispatch':'one_shared_queue_all_trials_and_datasets','single_seed':42,'missing_inputs':missing}
    if not args.run:print(json.dumps(plan,indent=2));return
    if missing:p.error('Missing prepared inputs; inspect the plan')
    output.mkdir(parents=True,exist_ok=False);snapshots=output/'configs';snapshots.mkdir()
    for trial,config in configs.items():
        (snapshots/f'{trial}.json').write_text(json.dumps(config,indent=2)+'\n')
    for job in jobs:job['command'][job['command'].index('--config')+1]=str(snapshots/f"{job['trial']}.json")
    plan.update(jobs=jobs,config_sha256={t:hashlib.sha256((snapshots/f'{t}.json').read_bytes()).hexdigest() for t in configs})
    (output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    pending=queue.Queue()
    for job in jobs:pending.put(job)
    finished=[];lock=threading.Lock()
    def worker(gpu):
        with GPUWorker(gpu) as persistent:
            while True:
                try:job=pending.get_nowait()
                except queue.Empty:return
                target=Path(job['output']);target.parent.mkdir(parents=True,exist_ok=True)
                print(f"Starting {job['trial']} {job['dataset']} pair {job['pair_index']} on GPU {gpu}",flush=True)
                started=time.perf_counter();status=persistent.run(job['command'],target.with_suffix('.log'))
                row={k:job[k] for k in ('trial','dataset','pair_index','seed','output')}
                row.update(status,gpu=gpu,seconds=time.perf_counter()-started)
                with lock:
                    finished.append(row)
                    report={'finished':len(finished),'intended':len(jobs),'jobs':finished,'trials':summarize(finished,jobs)}
                    temp=output/'progress.partial.json';temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(output/'progress.json')
                    print(json.dumps(row),flush=True)
                pending.task_done()
    with concurrent.futures.ThreadPoolExecutor(len(args.gpus)) as pool:
        futures=[pool.submit(worker,g) for g in args.gpus]
        for future in futures:future.result()
    (output/'summary.json').write_text(json.dumps({'jobs':finished,'trials':summarize(finished,jobs)},indent=2)+'\n')
    if any(r['returncode'] for r in finished):raise SystemExit(1)

if __name__=='__main__':main()
