"""Plan DIR-Lab baseline jobs, or explicitly execute with one worker per GPU."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys

from .common import validate_config
from .protocols import seeds_for
from .sources import ROOT, SOURCES, verify_source


def make_jobs(methods, cases, data, output, config_dir):
    jobs = []
    for name in methods:
        verify_source(name)
        config = config_dir / f'{name}.json'
        settings=validate_config(json.loads(config.read_text()), name)
        for number in cases:
            case = data / f'case{number:02d}'
            inputs = [case / f'{phase}{suffix}' for phase in ('T00', 'T50')
                      for suffix in ('.nii.gz', '_landmarks_voxel.csv')]
            if name == 'ccidir':
                inputs += [case / f'{phase}_lung_mask.nii.gz' for phase in ('T00', 'T50')]
            seeds=seeds_for(name,'dirlab',settings['seed'])
            for seed in seeds:
                target = output / name / case.name
                if len(seeds)>1:
                    target=target/f'seed_{seed:04d}'
                command = [sys.executable, '-m', 'baselines.case_runner', '--method', name,
                           '--moving', str(case / 'T50.nii.gz'), '--fixed', str(case / 'T00.nii.gz'),
                           '--config', str(config), '--device', 'cuda:0', '--output', str(target),'--seed',str(seed)]
                jobs.append({'method': name, 'case': number, 'seed':seed, 'output': str(target), 'command': command,
                             'missing_inputs': [str(p) for p in inputs if not p.is_file()]})
    return jobs


def summarize(results):
    """Keep missing/crashed fits visible alongside measured TRE failure rates."""
    import numpy as np
    summary={}
    for name in sorted({r['method'] for r in results}):
        rows=[r for r in results if r['method']==name]
        measured=[]
        per_case={}
        for row in rows:
            path=Path(row['output'])/'metrics.json'
            if row['returncode'] or not path.is_file():
                continue
            scores=json.loads(path.read_text())
            values=[scores['idir_rounded_tre_mm']['mean_mm']]
            if name=='ccidir':
                values.append(scores['reverse_T50_to_T00']['idir_rounded_tre_mm']['mean_mm'])
            measured.extend(values)
            per_case.setdefault(str(row['case']),[]).append(values)
        expected=len(rows)*(2 if name=='ccidir' else 1)
        summary[name]={'intended_fits':len(rows),'intended_direction_results':expected,
            'scored_direction_results':len(measured),'unscored_direction_results':expected-len(measured),
            'measured_TRE_failures':sum(v>2 for v in measured),
            'measured_failure_fraction':sum(v>2 for v in measured)/len(measured) if measured else None,
            'failure_definition':'mean rounded landmark TRE > 2 mm',
            'per_case':{case:{'scored_fits':len(values),'direction_mean_mm':np.mean(values,axis=0).tolist(),
                              'direction_median_mm':np.median(values,axis=0).tolist()}
                        for case,values in per_case.items()}}
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--methods', choices=SOURCES, nargs='+', default=list(SOURCES))
    parser.add_argument('--cases', type=int, nargs='+', default=list(range(1, 11)))
    parser.add_argument('--data', type=Path, default=Path('Data/DIRLab4DCT_IDIR'))
    parser.add_argument('--configs', type=Path, default=ROOT / 'configs')
    parser.add_argument('--output', type=Path, default=Path('results/baselines'))
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3])
    parser.add_argument('--run', action='store_true', help='Execute the planned registrations; default prints a plan only')
    args = parser.parse_args(argv)
    if any(c not in range(1, 11) for c in args.cases) or len(set(args.cases)) != len(args.cases):
        parser.error('Cases must be distinct values in 1..10')
    if len(set(args.methods)) != len(args.methods):
        parser.error('Methods must be distinct')
    if any(g < 0 for g in args.gpus) or len(set(args.gpus)) != len(args.gpus):
        parser.error('GPUs must be distinct nonnegative indices')
    if args.output.resolve().is_relative_to((Path.cwd() / 'Data').resolve()):
        parser.error('Output cannot be under Data')
    jobs = make_jobs(args.methods, args.cases, args.data, args.output, args.configs)
    if not args.run:
        print(json.dumps({'execution': False, 'jobs': jobs}, indent=2))
        return
    if any(job['missing_inputs'] for job in jobs):
        print(json.dumps({'execution': False, 'jobs': jobs}, indent=2))
        parser.error('Missing inputs; prepare them before launching')
    args.output.mkdir(parents=True, exist_ok=False)
    def worker(gpu, assigned):
        results = []
        for job in assigned:
            target = Path(job['output'])
            target.parent.mkdir(parents=True, exist_ok=True)
            # Native ccIDIR/NIR contain bare .cuda() calls. Isolate the selected
            # physical GPU and use logical cuda:0 consistently in each child.
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            with target.with_suffix('.log').open('x') as log:
                result = subprocess.run(job['command'], env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
            results.append({'method': job['method'], 'case': job['case'], 'gpu': gpu,'seed':job['seed'],
                            'output':job['output'],'returncode': result.returncode})
        return results
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = [pool.submit(worker, gpu, jobs[i::len(args.gpus)]) for i, gpu in enumerate(args.gpus)]
        results = [entry for future in futures for entry in future.result()]
    (args.output / 'summary.json').write_text(json.dumps({'jobs':results,'methods':summarize(results)}, indent=2) + '\n')
    if any(result['returncode'] for result in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
