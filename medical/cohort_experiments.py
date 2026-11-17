"""Plan ACDC and Neurite-OASIS fits; --run explicitly starts the shared GPU queue."""
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
from medical.cohort_data import DATASETS, definition, validate_case
from medical.case_runner import load_volume
from medical.configuration import validate_config
from medical.gpu_worker import GPUWorker

ROOT = Path(__file__).resolve().parents[1]


def make_jobs(datasets, data_root, output, config):
    settings = json.loads(Path(config).read_text())
    validate_config(settings)
    jobs = []
    for dataset in datasets:
        root = Path(data_root) / DATASETS[dataset] / 'prepared'
        manifest = definition(root, dataset)
        for pair in manifest['pairs']:
            target = Path(output) / dataset / f'{pair["moving"]}_to_{pair["fixed"]}'
            inputs = [root / pair[f'{key}_{kind}'] for key in ('moving', 'fixed') for kind in ('image', 'label')]
            command = [sys.executable, '-m', 'medical', '--method', 'medical.cohort_method',
                       '--moving', str(root / pair['moving_image']), '--fixed', str(root / pair['fixed_image']),
                       '--config', str(Path(config).resolve()), '--device', 'cuda:0', '--output', str(target)]
            jobs.append(dict(dataset=dataset, pair_index=pair['pair_index'], seed=settings['runtime']['seed'],
                             output=str(target), command=command, manifest_sha256=sha256(root / 'manifest.json'),
                             missing_inputs=[str(p) for p in inputs if not p.is_file()]))
    return jobs, settings


def summarize(records, jobs):
    result = {}
    for dataset in dict.fromkeys(j['dataset'] for j in jobs):
        completed = [r for r in records if r['dataset'] == dataset]
        successful, values, label_dice = 0, {}, []
        converged, intervals = 0, 0
        for row in completed:
            path = Path(row['output']) / 'metrics.json'
            if row['returncode'] or not path.is_file():
                continue
            successful += 1
            scores = json.loads(path.read_text())
            for name, value in [('DSC', scores['DSC']['mean']), ('HD95', scores['HD95']['mean']),
                                ('folding_fraction', scores['folding_fraction']),
                                ('folding_percent', scores.get('folding_percent')),
                                ('LogJacDetStd', scores.get('LogJacDetStd', {}).get('mean')),
                                ('seconds', row['seconds'])]:
                if value is not None:
                    values.setdefault(name, []).append(value)
            intervals += len(scores['recovery'])
            converged += sum(r['converged'] for r in scores['recovery'])
            label_dice.extend(v for v in scores['DSC']['detailed'] if v is not None)
        metrics = {k: dict(mean=statistics.mean(v), std_across_cases=statistics.pstdev(v), scored_pairs=len(v))
                   for k, v in values.items()}
        if label_dice:
            metrics['pooled_label_DSC'] = dict(mean=statistics.mean(label_dice), std=statistics.pstdev(label_dice), observations=len(label_dice))
        result[dataset] = dict(intended=sum(j['dataset'] == dataset for j in jobs), completed=len(completed), successful=successful, pcg_converged=converged, pcg_intervals=intervals, metrics=metrics)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--data-root', type=Path, default=ROOT / 'Data')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/prepared_cohorts')
    parser.add_argument('--config', type=Path, default=ROOT / 'medical/configs/default.json')
    parser.add_argument('--gpus', nargs='+', type=int, default=[0, 1, 2, 3])
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument('--run', action='store_true')
    execution.add_argument('--plan-only', action='store_true')
    args = parser.parse_args(argv)
    if len(set(args.datasets)) != len(args.datasets) or len(set(args.gpus)) != len(args.gpus) or any(g < 0 for g in args.gpus):
        parser.error('Distinct datasets and nonnegative GPU IDs required')
    output = args.output.resolve()
    if output.is_relative_to(args.data_root.resolve()) or output.is_relative_to((ROOT / 'Data').resolve()):
        parser.error('Output cannot be under Data')
    try:
        jobs, settings = make_jobs(args.datasets, args.data_root.resolve(), output, args.config)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    missing = sorted({p for j in jobs for p in j['missing_inputs']})
    plan = dict(execution=args.run, total_jobs=len(jobs), single_seed=settings['runtime']['seed'],
                counts={d: sum(j['dataset'] == d for j in jobs) for d in args.datasets}, gpus=args.gpus,
                dispatch='shared_queue_persistent_GPU_workers', config=settings, config_sha256=sha256(args.config),
                missing_inputs=missing, jobs=jobs)
    if not args.run:
        print(json.dumps(plan, indent=2, allow_nan=False))
        return plan
    if missing:
        parser.error('Missing prepared inputs')
    # Validate every selected pair before creating outputs or starting workers.
    for job in jobs:
        command = job['command']
        case = {k: load_volume(command[command.index('--' + k) + 1]) for k in ('moving', 'fixed')}
        validate_case(case, job['dataset'])
    output.mkdir(parents=True, exist_ok=False)
    config = output / 'config.json'
    config.write_text(json.dumps(settings, indent=2) + '\n')
    for job in jobs:
        job['command'][job['command'].index('--config') + 1] = str(config)
    (output / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    pending, records, lock = queue.Queue(), [], threading.Lock()
    for job in jobs:
        pending.put(job)

    def worker(gpu):
        with GPUWorker(gpu) as persistent:
            while True:
                try:
                    job = pending.get_nowait()
                except queue.Empty:
                    return
                target = Path(job['output'])
                target.parent.mkdir(parents=True, exist_ok=True)
                started = time.perf_counter()
                status = persistent.run(job['command'], target.with_suffix('.log'))
                row = {k: job[k] for k in ('dataset', 'pair_index', 'seed', 'output')}
                row.update(status, gpu=gpu, seconds=time.perf_counter() - started)
                with lock:
                    records.append(row)
                    progress = dict(finished=len(records), intended=len(jobs), jobs=records, datasets=summarize(records, jobs))
                    temporary = output / 'progress.partial.json'
                    temporary.write_text(json.dumps(progress, indent=2, allow_nan=False) + '\n')
                    temporary.replace(output / 'progress.json')
                pending.task_done()
    with ThreadPoolExecutor(len(args.gpus)) as pool:
        for future in [pool.submit(worker, gpu) for gpu in args.gpus]:
            future.result()
    (output / 'summary.json').write_text(json.dumps(dict(jobs=records, datasets=summarize(records, jobs)), indent=2, allow_nan=False) + '\n')
    if any(r['returncode'] for r in records):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
