"""Plan our method on frozen IXI and LPBA40 benchmarks; --run opts into fitting."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import queue
import statistics
import sys
import threading
import time

from medical.brain_data import (DATASET_NAMES, PROTOCOL_PATH, pair_list, prepared_paths,
                                protocols, sha256, preparation_fingerprint,
                                require_sitreg_dependencies, EXPECTED_SHAPES)
from medical.configuration import validate_config
from medical.gpu_worker import GPUWorker

ROOT = Path(__file__).resolve().parents[1]


def make_jobs(datasets, split, data_root, output, config, pairs=None):
    settings = json.loads(Path(config).read_text())
    validate_config(settings)
    cap = settings['runtime'].get('recovery_max_axis')
    if any(cap < max(EXPECTED_SHAPES[d]) for d in datasets):
        raise ValueError('These benchmark adapters require full native-grid voxel recovery')
    if pairs is not None and len(datasets) != 1:
        raise ValueError('--pairs requires exactly one dataset')
    jobs = []
    for dataset in datasets:
        root = Path(data_root) / DATASET_NAMES[dataset] / 'prepared'
        definition = pair_list(dataset, split)
        selected = list(range(1, len(definition) + 1)) if pairs is None else list(pairs)
        if len(set(selected)) != len(selected) or any(i not in range(1, len(definition) + 1) for i in selected):
            raise ValueError('Distinct pair indices within the frozen list are required')
        for index in selected:
            pair = definition[index - 1]
            images = prepared_paths(root, pair)
            target = Path(output) / dataset / f'{pair["moving"]}_to_{pair["fixed"]}'
            required = [*images.values(), root / 'protocol.json', root / 'preparation.json',
                        *(root / 'labels' / p.name for p in images.values())]
            command = [sys.executable, '-m', 'medical', '--method', 'medical.brain_method',
                       '--moving', str(images['moving']), '--fixed', str(images['fixed']),
                       '--config', str(Path(config).resolve()), '--device', 'cuda:0', '--output', str(target)]
            jobs.append({'dataset': dataset, 'split': split, 'pair_index': index,
                         'moving_id': pair['moving'], 'fixed_id': pair['fixed'],
                         'seed': settings['runtime']['seed'], 'output': str(target),
                         'command': command, 'missing_inputs': [str(p) for p in required if not p.is_file()]})
    return jobs, settings


def summarize(records, jobs):
    result = {}
    for dataset in dict.fromkeys(j['dataset'] for j in jobs):
        selected = [row for row in records if row['dataset'] == dataset]
        values, groups, successful, converged, intervals = {}, [], 0, 0, 0
        for row in selected:
            path = Path(row['output']) / 'metrics.json'
            if row['returncode'] or not path.is_file():
                continue
            successful += 1
            scores = json.loads(path.read_text())
            scalars = {'DSC': scores['DSC']['mean'], 'folding_fraction': scores['folding_fraction'],
                       'seconds': row['seconds'], 'HD95': scores.get('HD95', {}).get('mean'),
                       'reverse_DSC': scores.get('reverse', {}).get('DSC', {}).get('mean'),
                       'bidirectional_DSC': scores.get('bidirectional_DSC')}
            for key, value in scalars.items():
                if value is not None:
                    values.setdefault(key, []).append(value)
            groups += list(scores['DSC'].get('groups', {}).values())
            intervals += len(scores['recovery'])
            converged += sum(row['converged'] for row in scores['recovery'])
        metrics = {key: {'mean': statistics.mean(v), 'std_across_cases': statistics.pstdev(v),
                         'scored_pairs': len(v)} for key, v in values.items()}
        if groups:
            metrics['DSC']['std_across_groups_and_cases'] = statistics.pstdev(groups)
        result[dataset] = {'intended': sum(j['dataset'] == dataset for j in jobs),
                           'completed': len(selected), 'successful': successful,
                           'pcg_converged': converged, 'pcg_intervals': intervals, 'metrics': metrics}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=DATASET_NAMES, default=list(DATASET_NAMES))
    parser.add_argument('--split', choices=['test', 'validation'], default='test')
    parser.add_argument('--pairs', nargs='+', type=int)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'Data')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/brain_benchmarks')
    parser.add_argument('--config', type=Path, default=ROOT / 'medical/configs/default.json')
    parser.add_argument('--gpus', nargs='+', type=int, default=[0, 1, 2, 3])
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument('--run', action='store_true')
    execution.add_argument('--plan-only', action='store_true', help='Explicit read-only planning (also the default)')
    args = parser.parse_args(argv)
    if len(set(args.datasets)) != len(args.datasets) or len(set(args.gpus)) != len(args.gpus) or any(g < 0 for g in args.gpus):
        parser.error('Distinct datasets and distinct nonnegative GPU IDs required')
    output = args.output.resolve()
    if output.is_relative_to((ROOT / 'Data').resolve()) or output.is_relative_to(args.data_root.resolve()):
        parser.error('Output cannot be under the data directory')
    try:
        jobs, settings = make_jobs(args.datasets, args.split, args.data_root.resolve(), output,
                                   args.config, args.pairs)
    except ValueError as exc:
        parser.error(str(exc))
    counts = {name: sum(j['dataset'] == name for j in jobs) for name in args.datasets}
    missing = sorted({p for j in jobs for p in j['missing_inputs']})
    plan = {'execution': args.run, 'single_seed': settings['runtime']['seed'], 'counts': counts,
            'total_jobs': len(jobs), 'gpus': args.gpus, 'dispatch': 'shared_queue_persistent_GPU_workers',
            'config': settings, 'config_sha256': sha256(args.config),
            'protocol_sha256': sha256(PROTOCOL_PATH), 'source_pins': protocols()['source_pins'],
            'missing_inputs': missing, 'jobs': jobs}
    if not args.run:
        print(json.dumps(plan, indent=2, allow_nan=False))
        return plan
    if missing:
        parser.error('Missing prepared inputs; inspect the plan before execution')
    if 'lpba40' in args.datasets:
        require_sitreg_dependencies()
    for dataset in args.datasets:
        meta = json.loads((args.data_root / DATASET_NAMES[dataset] / 'prepared/protocol.json').read_text())
        if meta['dataset'] != dataset or meta.get('preparation_protocol_sha256') != preparation_fingerprint(dataset):
            parser.error('Prepared inputs do not match the frozen protocol')
    output.mkdir(parents=True, exist_ok=False)
    config = output / 'config.json'
    config.write_text(json.dumps(settings, indent=2) + '\n')
    for job in jobs:
        job['command'][job['command'].index('--config') + 1] = str(config)
    (output / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    pending, finished, lock = queue.Queue(), [], threading.Lock()
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
                row = {key: job[key] for key in ('dataset', 'split', 'pair_index', 'seed', 'output')}
                row.update(status, gpu=gpu, seconds=time.perf_counter() - started)
                with lock:
                    finished.append(row)
                    progress = {'finished': len(finished), 'intended': len(jobs), 'jobs': finished,
                                'datasets': summarize(finished, jobs)}
                    temporary = output / 'progress.partial.json'
                    temporary.write_text(json.dumps(progress, indent=2, allow_nan=False) + '\n')
                    temporary.replace(output / 'progress.json')
                pending.task_done()
    with ThreadPoolExecutor(len(args.gpus)) as pool:
        for future in [pool.submit(worker, gpu) for gpu in args.gpus]:
            future.result()
    (output / 'summary.json').write_text(json.dumps({'jobs': finished, 'datasets': summarize(finished, jobs)}, indent=2) + '\n')
    if any(row['returncode'] for row in finished):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
