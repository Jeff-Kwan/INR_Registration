"""Plan or launch the seven user-defined SIRENPath ablations on frozen L2R OASIS."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
from pathlib import Path
import queue
import statistics
import subprocess
import sys
import threading
import time

from medical.abdomen_evaluation import PROTOCOLS
from medical.configuration import load_config, validate_config
from medical.gpu_worker import GPUWorker

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    1: ('reconstruction_only', 'Reconstruction only', 'mse', False, False),
    2: ('transport_only', 'Transport only', 'mse', True, False),
    3: ('registration_only', 'Registration only', 'mse', False, True),
    4: ('both_losses', 'Both losses', 'mse', True, True),
    5: ('sirenpath', 'SIRENPath', 'ncc', True, True),
    6: ('raw_crossfade', 'Raw intensity crossfade', None, False, False),
    7: ('reconstructed_crossfade', 'Canonical reconstructions crossfade', 'ncc (reused)', False, False),
}
METRICS = ('PSNR', 'PatchNCC', 'DSC', 'HD95', 'NPJ', 'SDlogJ', 'Runtime')


def settings_for(index):
    config = copy.deepcopy(load_config())
    _, _, endpoint, transport, registration = RUNS[index]
    if index in (6,7):
        # No training is invoked: the config supplies normalization and recovery.
        return config
    objective = config['objective']
    objective['endpoint'] = endpoint
    if endpoint == 'mse':
        objective['ncc']['moment_weight'] = 0.
    if not transport:
        objective['weight'] = 0.
    if not registration:
        objective['registration']['weight'] = 0.
        config['training']['registration_weight_schedule']['start_weight'] = 0.
    return validate_config(config, allow_ablation_mse=True)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def config_text(config):
    return json.dumps(config, indent=2)+'\n'


def make_plan(data, output, runs, pairs, gpus, canonical_root=None):
    data, output = Path(data).resolve(), Path(output).resolve()
    if output.is_relative_to((ROOT/'Data').resolve()) or output.is_relative_to(data):
        raise ValueError('Output cannot be under Data')
    preparation = data/'preparation.json'
    manifest = json.loads(preparation.read_text())
    if manifest['evaluation_pairs'] != PROTOCOLS['oasis']['eval_pairs']:
        raise ValueError('Frozen preparation pairs differ from the official protocol')
    if 7 in runs and 5 not in runs and canonical_root is None:
        raise ValueError('Run 7 requires run 5 in the selection or --canonical-root pointing to its case directories')
    checkpoint_root = Path(canonical_root).resolve() if canonical_root is not None else output/'05_sirenpath'
    configurations = {str(i):settings_for(i) for i in runs}
    if 7 in runs:
        configurations['7']['canonical_checkpoint_root'] = str(checkpoint_root)
    jobs = []
    for index in runs:
        slug, name, endpoint, transport, registration = RUNS[index]
        config = configurations[str(index)]
        snapshot = output/'configs'/f'{index:02d}_{slug}.json'
        for pair_index in pairs:
            pair = PROTOCOLS['oasis']['eval_pairs'][pair_index-1]
            images = {k:data/pair[k] for k in ('moving','fixed')}
            inputs = [*images.values(), *(data/pair[k].replace('imagesTr','labelsTr') for k in ('moving','fixed'))]
            target = output/f'{index:02d}_{slug}'/f'pair{pair_index:03d}'
            module = ('medical.reconstructed_crossfade_method' if index == 7 else
                      'medical.raw_crossfade_method' if index == 6 else 'medical.sirenpath_ablation_method')
            required_checkpoint = checkpoint_root/f'pair{pair_index:03d}'/'checkpoint.pt' if index==7 else None
            if index==7 and canonical_root is not None:
                inputs += [required_checkpoint, required_checkpoint.parent/'run.json', required_checkpoint.parent/'metrics.json']
            command = [sys.executable, '-m', 'medical', '--method', module,
                       '--moving', str(images['moving']), '--fixed', str(images['fixed']),
                       '--config', str(snapshot), '--device', 'cuda:0', '--output', str(target)]
            jobs.append(dict(run=index, name=name, pair_index=pair_index, seed=config['runtime']['seed'],
                             training_steps=0 if index in (6,7) else config['training']['steps'],
                             endpoint=endpoint, transport_loss=transport, registration_loss=registration,
                             output=str(target), command=command,
                             requires_checkpoint=str(required_checkpoint) if required_checkpoint else None,
                             config_sha256=hashlib.sha256(config_text(config).encode()).hexdigest(),
                             missing_inputs=[str(p) for p in inputs if not p.is_file()]))
    sources = ['configuration.py', 'sirenpath_ablations.py', 'sirenpath_ablation_method.py',
               'sirenpath_reconstruction.py', 'raw_crossfade_method.py', 'reconstructed_crossfade_method.py', 'abdomen_evaluation.py',
               'registration/objective.py', 'registration/captured_objective.py',
               'registration/captured_training.py', 'registration/optimization.py',
               'registration/recovery.py', 'registration/recovery_sampling.py',
               'registration/flow.py', 'registration/intensity_crossfade.py']
    return dict(dataset='L2R-OASIS', total_jobs=len(jobs), training_jobs=sum(j['training_steps']>0 for j in jobs),
                gpus=gpus, runs=runs, pair_indices=pairs, jobs=jobs,
                configurations=configurations, canonical_checkpoint_root=str(checkpoint_root),
                reuse_external_canonical=canonical_root is not None,
                preparation_sha256=sha256(preparation),
                protocol_sha256=sha256(ROOT/'medical/protocols/oasis_val.json'),
                source_sha256={name:sha256(ROOT/'medical'/name) for name in sources},
                metric_definitions=dict(DSC='pinned official Learn2Reg foreground Dice',
                    HD95='pinned official Learn2Reg HD95, voxel units',
                    NPJ='100*mean(detJ<=0), two-voxel interior',
                    SDlogJ='std(log(max(detJ,1e-9))), two-voxel interior; official shifted variant retained separately',
                    PSNR='mean of fixed/moving whole-grid PSNR dB, peak=1, no affine calibration',
                    PatchNCC='mean of fixed/moving signed Pearson NCC; all valid 9^3 windows, stride1, truth variance>1e-8',
                    Runtime='case worker wall seconds, including load, training/setup, reconstruction, recovery, scoring and checkpoint serialization'))


def aggregate(records, jobs):
    aggregates = {}
    for index in sorted({j['run'] for j in jobs}):
        intended = sum(j['run']==index for j in jobs)
        rows = [r for r in records if r['run']==index]
        values = {k:[] for k in METRICS}
        infinities = 0
        successful = 0
        for row in rows:
            path = Path(row['output'])/'metrics.json'
            if row['returncode'] or not path.is_file():
                continue
            successful += 1
            scores = json.loads(path.read_text())
            infinities += bool(scores['PSNR'].get('positive_infinity',False))
            for key in METRICS:
                value = row['seconds'] if key == 'Runtime' else scores[key]['mean']
                if value is not None:
                    values[key].append(value)
        metrics = {}
        for key, samples in values.items():
            infinite = infinities if key == 'PSNR' else 0
            metrics[key] = dict(mean=statistics.mean(samples) if samples and not infinite else None,
                                std=statistics.pstdev(samples) if samples and not infinite else None,
                                scored_pairs=len(samples)+infinite, infinite_pairs=infinite,
                                intended_pairs=intended)
        aggregates[str(index)] = dict(name=RUNS[index][1], intended=intended, completed=len(rows),
                                      successful=successful, failed=sum(r['returncode']!=0 for r in rows), metrics=metrics)
    return aggregates


def write_report(output, summary):
    lines = ['# SIRENPath L2R-OASIS ablations', '',
             'Case means ± population SD; PSNR/Patch NCC first average the two endpoints. '
             'Runtime includes loading through checkpoint serialization. NPJ is percent. '
             'HD95 uses official voxel units; SDlogJ is unshifted (official shifted scores remain in case files).', '',
             '| Run | Success/intended | PSNR (dB) | Patch NCC | DSC | HD95 | NPJ (%) | SDlogJ | Runtime (s) |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for index, result in summary['runs'].items():
        cells = []
        for key in METRICS:
            m = result['metrics'][key]
            if m['infinite_pairs']:
                value = '∞ (exact endpoints)'
            elif m['mean'] is None:
                value = '—'
            else:
                precision = 6 if key=='NPJ' else 1 if key in ('PSNR','Runtime') else 4
                value = f"{m['mean']:.{precision}f} ± {m['std']:.{precision}f}"
            cells.append(f"{value} [{m['scored_pairs']}/{m['intended_pairs']}]")
        lines.append(f"| {index}. {result['name']} | {result['successful']}/{result['intended']} | "+' | '.join(cells)+' |')
    lines += ['', 'Brackets show scored/intended pairs for each metric. Infinite raw-control PSNR '
              'is exact endpoint identity, not a fitted-model score. Null distances and failed jobs '
              'remain explicit; no score-based exclusions. See plan.json, progress.json, summary.json '
              'and the per-case metrics/logs for provenance and failures. Run 7 reuses run-5 training; '
              'its runtime is incremental decode/finite-difference/recovery cost. The reused training '
              'seconds and a method runtime including shared training are retained in its case metrics.', '']
    (Path(output)/'ABLATIONS_REPORT.md').write_text('\n'.join(lines))


def require_idle_gpus(gpus):
    """Honor the instruction not to disturb existing neural baseline jobs."""
    for gpu in gpus:
        completed = subprocess.run(['nvidia-smi', f'--id={gpu}', '--query-compute-apps=pid',
                                    '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True)
        if any(line.strip().isdigit() for line in completed.stdout.splitlines()):
            raise ValueError(f'GPU {gpu} has active compute jobs; leave them untouched and launch later')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'Data/OASIS')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--canonical-root', type=Path, help='Existing run-5 directory containing pairNNN/checkpoint.pt; otherwise depend on this study run 5')
    parser.add_argument('--runs', nargs='+', type=int, choices=RUNS, default=list(RUNS))
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--pairs', nargs='+', type=int)
    selection.add_argument('--subset', choices=['quick4'])
    parser.add_argument('--gpus', nargs='+', type=int, default=list(range(8)))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--plan-only', action='store_true')
    args = parser.parse_args(argv)
    if args.subset:
        frozen = json.loads((ROOT/'medical/protocols/oasis_quick4.json').read_text())
        if frozen['protocol_sha256'] != sha256(ROOT/'medical/protocols/oasis_val.json'):
            parser.error('Frozen quick4 protocol hash mismatch')
        args.pairs = frozen['pair_indices']
    elif args.pairs is None:
        args.pairs = list(range(1,20))
    for values, name in ((args.runs,'runs'),(args.pairs,'pairs'),(args.gpus,'GPUs')):
        if len(values)!=len(set(values)):
            parser.error(f'Distinct {name} required')
    if any(i not in range(1,20) for i in args.pairs) or any(i<0 for i in args.gpus):
        parser.error('Pairs must be 1..19 and GPUs nonnegative')
    output = args.output.resolve()
    try:
        plan = make_plan(args.data, output, args.runs, args.pairs, args.gpus, args.canonical_root)
    except ValueError as error:
        parser.error(str(error))
    plan['execution'] = args.run
    if not args.run:
        print(json.dumps(plan, indent=2)); return plan
    if any(j['missing_inputs'] for j in plan['jobs']):
        parser.error('Missing frozen inputs')
    try:
        require_idle_gpus(args.gpus)
    except ValueError as error:
        parser.error(str(error))
    output.mkdir(parents=True, exist_ok=False)
    (output/'configs').mkdir()
    for index in args.runs:
        slug = RUNS[index][0]
        (output/'configs'/f'{index:02d}_{slug}.json').write_text(config_text(plan['configurations'][str(index)]))
    (output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    pending = queue.Queue()
    records, lock = [], threading.Lock()
    def snapshot(final=False):
        summary = dict(finished=len(records), intended=len(plan['jobs']), jobs=records,
                       runs=aggregate(records, plan['jobs']))
        name = 'summary.json' if final else 'progress.json'
        partial = output/(name+'.partial')
        partial.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
        partial.replace(output/name)
        write_report(output, summary)
    snapshot()
    def worker(gpu):
        with GPUWorker(gpu) as persistent:
            while True:
                try:
                    job = pending.get_nowait()
                except queue.Empty:
                    return
                target = Path(job['output']); target.parent.mkdir(parents=True,exist_ok=True)
                started = time.perf_counter()
                print(f"Starting run {job['run']} pair {job['pair_index']} on GPU {gpu}",flush=True)
                dependency = job.get('requires_checkpoint')
                failed_fit = job['run']==7 and not plan['reuse_external_canonical'] and any(
                    r['run']==5 and r['pair_index']==job['pair_index'] and r['returncode'] for r in records)
                if dependency and (failed_fit or not all((Path(dependency).parent/name).is_file()
                                                        for name in ('checkpoint.pt','run.json','metrics.json'))):
                    status = dict(returncode=1, error='Canonical checkpoint dependency failed or is incomplete; no retraining')
                    target.with_suffix('.log').write_text(status['error']+'\n')
                else:
                    status = persistent.run(job['command'],target.with_suffix('.log'))
                row = dict(run=job['run'],pair_index=job['pair_index'],seed=job['seed'],gpu=gpu,
                           output=job['output'],seconds=time.perf_counter()-started,**status)
                with lock:
                    records.append(row); snapshot()
                pending.task_done()
    with ThreadPoolExecutor(len(args.gpus)) as pool:
        # Barrier guarantees run 7 only reads completed canonical checkpoints.
        for stage in ([j for j in plan['jobs'] if j['run']!=7],
                      [j for j in plan['jobs'] if j['run']==7]):
            if not stage:
                continue
            for job in stage:
                pending.put(job)
            tasks = [pool.submit(worker,gpu) for gpu in args.gpus]
            for task in tasks:
                task.result()
    snapshot(final=True)
    if any(r['returncode'] for r in records):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
