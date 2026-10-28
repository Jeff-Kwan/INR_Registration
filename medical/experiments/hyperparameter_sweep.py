"""Four-hour staged LPBA search. Standard-library orchestration; no work on import.

Run with ``python -m medical.experiments.hyperparameter_sweep`` from the repository root.
Use --plan-only to materialize the design without starting the trainer.
"""
import argparse
import copy
import csv
from datetime import datetime, timezone
import fcntl
import itertools
import json
import math
from pathlib import Path
import random
import signal
import statistics
import subprocess
import sys
import time

from medical.registration.configuration import load_config

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
SPACE = {
    'lk.affine_rho': [0., 0.0004, 0.004, 0.04, 0.4, 4., 40., 400., 4000.],
    'lk.weight': [0.003, 0.01, 0.03, 0.1, 0.3, 1.],
    'lk.gradient_weight': [0.25, 1., 4., 16., 64.],
    'lk.ridge': [1e-7, 1e-6, 1e-5, 1e-4],
    'lk.weight_schedule': ['constant', 'linear', 'cosine'],
    'lk.points': [4096, 8192],
    'training.lr': [0.0001, 0.0002, 0.0004],
    'training.lr_final_factor': [0.1, 0.25, 0.5],
    'training.batch_size': [16384, 32768, 65536],
    'model.width': [192, 256, 384],
    'model.layers': [3, 4, 5],
    'model.omega': [10., 20., 30., 40.],
}


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def changed(base, values):
    config = copy.deepcopy(base)
    for key, value in values.items():
        section, name = key.split('.')
        config[section][name] = value
    return config


def scientific_config(config, *, seed=None, include_seed=True):
    """Identity independent of subject panel and artifact-retention policy."""
    identity = copy.deepcopy(config)
    identity['dataset']['subjects'] = []
    identity.pop('output', None)
    if seed is not None:
        identity['runtime']['seed'] = seed
    if not include_seed:
        identity['runtime'].pop('seed', None)
    return identity


def config_key(config, *, seed=None):
    return json.dumps(scientific_config(config, seed=seed), sort_keys=True)


def design(base, rng):
    """Interleave anchored regularization and balanced joint candidates."""
    base = copy.deepcopy(base)
    base['training'].update(steps=5000, grad_clip=1.0)
    control = changed(base, {'lk.affine_rho': 0.})
    configs = [base, control]
    # Separate penalty scale c from attenuation rho=c/beta.
    configs += [changed(base, {'lk.affine_rho': r, 'lk.gradient_weight': c})
                for r, c in itertools.product(SPACE['lk.affine_rho'], [1., 4., 16.])]
    configs += [changed(base, {'lk.weight': w, 'lk.ridge': r})
                for w, r in itertools.product([0.01, 0.03, 0.3], [1e-6, 1e-5])]
    # Stratify every marginal, avoiding an enormous Cartesian product.
    columns = {}
    for key, values in SPACE.items():
        column = (values * (72 // len(values) + 1))[:72]
        rng.shuffle(column)
        columns[key] = column
    joint = [changed(base, {k: v[i] for k, v in columns.items()}) for i in range(72)]
    anchors = configs[1:]
    configs = [base, control]
    for anchor, candidate in itertools.zip_longest(anchors, joint):
        if anchor is not None:
            configs.append(anchor)
        if candidate is not None:
            configs.append(candidate)
    unique = {}
    for config in configs:
        unique.setdefault(config_key(config), config)
    return list(unique.values())


def estimate(config, subjects):
    # User-provided 600s/full cohort. No benchmark or historical timing scan.
    training = config['training']
    scale = ((config['model']['width'] / 256) ** 2 * config['model']['layers'] / 4
             * training['steps'] / 5000
             * (0.5 * training['batch_size'] / 32768 + 0.5 * config['lk']['points'] / 4096))
    return 600 * len(subjects) / 9 * (0.5 + 0.5 * scale)


def collect_metrics(run, subjects):
    from medical.registration.io import LABEL_METRIC_DEFINITION
    rows = []
    for subject in subjects:
        folder = run / subject
        if not json.loads((folder / 'status.json').read_text()).get('complete'):
            raise ValueError(f'{subject}: incomplete')
        m = json.loads((folder / 'metrics.json').read_text())
        if not m.get('endpoint_quality_pass'):
            raise ValueError(f'{subject}: endpoint reconstruction gate failed')
        labels = m.get('labels') or {}
        if labels.get('metric_definition') != {**LABEL_METRIC_DEFINITION, 'hd95_enabled': True}:
            raise ValueError(f'{subject}: metric definition mismatch')
        dice, hd = labels.get('dice'), labels.get('hd95_mm')
        if (type(dice) not in (int, float) or type(hd) not in (int, float)
                or not math.isfinite(dice) or not math.isfinite(hd)
                or not 0 <= dice <= 1 or hd < 0):
            raise ValueError(f'{subject}: invalid label metrics')
        rows.append({'subject': subject, 'dice': dice, 'hd95_mm': hd,
                     'endpoint_quality_pass': m.get('endpoint_quality_pass'),
                     'folding_fraction': m.get('folding_fraction'),
                     'flow_converged': m.get('flow_all_intervals_converged', False)})
    return rows


def measure(run, subjects):
    rows = collect_metrics(run, subjects)
    if any(not r['endpoint_quality_pass'] or r['folding_fraction'] != 0 for r in rows):
        raise ValueError('Endpoint/folding gate failed')
    return {'dice': statistics.mean(r['dice'] for r in rows),
            'hd95_mm': statistics.mean(r['hd95_mm'] for r in rows),
            'subject_metrics': rows}


def ranked(records, stage):
    return sorted([r for r in records if r['stage'] == stage and r['status'] == 'complete'],
                  key=lambda r: (-r['dice'], r['hd95_mm']))


def config_runtimes(records):
    """Group wall times across panels/seeds; never used by ranked()."""
    groups = {}
    for record in records:
        config = scientific_config(record['config'], include_seed=False)
        key = json.dumps(config, sort_keys=True)
        group = groups.setdefault(key, {'config': config, 'attempts': [], 'total_run_seconds': 0.})
        duration = record.get('total_run_seconds', record.get('elapsed_seconds', 0.))
        group['total_run_seconds'] += duration
        group['attempts'].append({'index': record['index'], 'stage': record['stage'],
            'subjects': record['subjects'], 'seed': record['config']['runtime']['seed'],
            'status': record['status'], 'total_run_seconds': duration})
    return list(groups.values())


def export_summary(output, records, status, elapsed=0.):
    """Refresh a small, centralized index without reading bulky trial artifacts."""
    summary = output / 'summary'
    summary.mkdir(exist_ok=True)
    stages = {stage: ranked(records, stage) for stage in ('screen', 'confirm', 'full', 'reserve_repeat')}
    save(summary / 'leaderboards.json', stages)
    save(summary / 'trials.json', records)
    save(summary / 'config_runtimes.json', config_runtimes(records))
    failures = [r for r in records if r['status'] not in ('complete', 'running')]
    save(summary / 'failures.json', failures)
    full = sorted(stages['full'] + stages['reserve_repeat'], key=lambda r: (-r['dice'], r['hd95_mm']))
    best = full[0] if full else None
    save(summary / 'overview.json', {'status': status, 'elapsed_seconds': elapsed,
        'attempts': len(records), 'completed': sum(r['status'] == 'complete' for r in records),
        'failed_or_interrupted': len(failures), 'best_full_trial': best['index'] if best else None,
        'warning': 'Best observed individual run; compare seeds before adoption. Full cohort participates in selection.'})
    if best:
        config = copy.deepcopy(best['config'])
        config['dataset']['subjects'] = []
        save(summary / 'best_observed_config.json', config)
        save(summary / 'best_observed_metrics.json', best)
    fields = ['index', 'stage', 'status', 'seed', 'subjects', 'dice', 'hd95_mm',
              'total_run_seconds', 'flow_warning_subjects', 'output', 'error']
    temporary = summary / 'trials.csv.tmp'
    with temporary.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {k: record.get(k) for k in fields}
            row.update(seed=record['config']['runtime']['seed'], subjects=','.join(record['subjects']),
                flow_warning_subjects=','.join(r['subject'] for r in record.get('subject_metrics', []) if not r['flow_converged']))
            writer.writerow(row)
    temporary.replace(summary / 'trials.csv')
    lines = ['# Sweep results', '', f'Status: **{status}**. Wall time: {elapsed/3600:.2f} hours.', '',
        '[Plan](../plan.json) · [Overview](overview.json) · [All attempts](trials.csv) · '
        '[Leaderboards](leaderboards.json) · [Config runtimes](config_runtimes.json) · [Failures](failures.json)', '',
        'Quality ranking uses Dice, then HD95; runtime is informational. '
        'Best observed runs require endpoint, folding and finite-label gates. '
        'Inspect seed consistency and flow warnings before adoption. The full cohort is used for selection.', '']
    if best:
        lines += [f"Best observed full-cohort trial: **{best['index']}**, Dice {best['dice']:.4f}, HD95 {best['hd95_mm']:.3f} mm.", '',
            '[Configuration](best_observed_config.json) · [Metrics and subjects](best_observed_metrics.json)', '']
    else:
        lines += ['No eligible full-cohort result yet.', '']
    for stage, ranking in stages.items():
        lines += [f'## {stage}', '', '| Trial | Seed | Dice | HD95 mm | Seconds | Flow warnings | Details |',
                  '|---|---:|---:|---:|---:|---:|---|']
        for r in ranking:
            relative = Path(r['output']).relative_to(output).as_posix()
            warnings = sum(not item['flow_converged'] for item in r.get('subject_metrics', []))
            lines.append(f"| {r['index']} | {r['config']['runtime']['seed']} | {r['dice']:.4f} | {r['hd95_mm']:.3f} | {r.get('total_run_seconds', 0):.1f} | {warnings} | [Files](../{relative}/) |")
        lines.append('')
    temporary = summary / 'REPORT.md.tmp'
    temporary.write_text('\n'.join(lines))
    temporary.replace(summary / 'REPORT.md')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/lpba_pdlk.json')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--deadline-utc', help='Optional original UTC cutoff when replacing an interrupted run')
    args = parser.parse_args()
    lock = None
    if not args.plan_only:
        lock = (REPO_ROOT / '.relaxed_pdlk_sweep.lock').open('a')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            parser.exit(1, 'A sweep is already running and holds the GPU lock. '
                        'Check the running sweep process and its live results report before retrying.\n')
    base = load_config(args.config)
    base['lk'].update(mode='pointwise', affine_beta=None, affine_rho=0.4, differentiate_solve=False)
    base['runtime']['seed'] = args.seed
    base['training'].update(steps=5000, grad_clip=1.0)
    base['evaluation'].update(min_endpoint_psnr_db=30., hd95=True)
    base['dataset']['require_labels'] = True
    base['output'].update(save_frames=False, save_velocities=False, save_volumes=False)
    rng = random.Random(args.seed)
    candidates = design(base, rng)
    output = args.output or ROOT / 'results/relaxed_pdlk_sweeps' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output.mkdir(parents=True, exist_ok=False)
    plan = {'budget_seconds': 14400, 'stages_seconds': {'screen': 3600, 'confirm': 2700, 'full': 6300},
            'reserve_seconds': 1800, 'fixed': {'training.steps': 5000, 'training.grad_clip': 1.0}, 'space': SPACE, 'candidates': candidates,
            'screen_subjects': ['S02', 'S06'], 'confirm_subjects': ['S02', 'S04', 'S06', 'S09'],
            'full_subjects': [f'S{i:02}' for i in range(2, 11)],
            'ranking': 'Require every endpoint >=30dB, zero folding, finite label metrics; maximize macro Dice, HD95 tie-break.',
            'parameterization': 'rho=c/beta; c_eff=c/(1+rho*|grad I|^2). rho=0 is the exact standard PD-LK control; positive rho implies beta=c/rho.',
            'recovery': 'Held fixed to isolate training/architecture/regularization effects'}
    save(output / 'plan.json', plan)
    (output / 'trials').mkdir()
    (output / 'README.md').write_text('# Relaxed PD-LK sweep\n\nStart with [summary/REPORT.md](summary/REPORT.md).\n\n'
        'The summary directory centralizes metrics, rankings, runtimes and failures. '
        'Per-attempt configs, logs and subject artifacts are under trials/.\n')
    export_summary(output, [], 'planned')
    if args.plan_only:
        print(output)
        return
    # Advisory lock prevents two copies of this sweep sharing the GPU.
    with lock:
        started = time.monotonic()
        deadline = started + plan['budget_seconds']
        if args.deadline_utc:
            cutoff = datetime.fromisoformat(args.deadline_utc).timestamp()
            deadline = min(deadline, started + max(0., cutoff - time.time()))
        records = []
        ratios = []

        def trial(config, stage, subjects, stage_end, seed=None):
            prediction = estimate(config, subjects) * max(1., statistics.median(ratios) if ratios else 1.)
            remaining = min(stage_end, deadline) - time.monotonic()
            if remaining < 1.3 * prediction:
                return False
            index = len(records)
            folder = output / 'trials' / f'{index:03}_{stage}'
            folder.mkdir()
            config = copy.deepcopy(config)
            config['output']['save_checkpoints'] = stage not in ('screen', 'confirm')
            config['dataset']['subjects'] = subjects
            if seed is not None:
                config['runtime']['seed'] = seed
            save(folder / 'config.json', config)
            record = {'index': index, 'stage': stage, 'config': config, 'subjects': subjects,
                      'status': 'running', 'output': str(folder)}
            records.append(record)
            export_summary(output, records, 'running', time.monotonic() - started)
            before = time.monotonic()
            with (folder / 'trainer.log').open('w') as log:
                process = None
                try:
                    process = subprocess.Popen([sys.executable, '-m', 'medical.registration.train',
                        '--config', str((folder / 'config.json').resolve()), '--output-dir', str((folder / 'run').resolve())],
                        cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    code = process.wait(timeout=max(1, min(remaining, 2 * prediction + 120)))
                    if code:
                        raise ValueError(f'trainer exit {code}; see trainer.log')
                    record.update(measure(folder / 'run', subjects), status='complete')
                except (subprocess.TimeoutExpired, KeyboardInterrupt, SystemExit):
                    import os
                    if process is not None:
                        try:
                            os.killpg(process.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            try:
                                os.killpg(process.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                            process.wait()
                    record.update(status='timeout_or_interrupted')
                    record['total_run_seconds'] = time.monotonic() - before
                    export_summary(output, records, 'interrupted' if sys.exc_info()[0] is not subprocess.TimeoutExpired else 'running', time.monotonic() - started)
                    if sys.exc_info()[0] is not subprocess.TimeoutExpired:
                        raise
                except (OSError, ValueError, KeyError, TypeError) as error:
                    record.update(status='failed', error=str(error))
            record['elapsed_seconds'] = time.monotonic() - before
            record['total_run_seconds'] = record['elapsed_seconds']
            if record['status'] == 'complete':
                ratios.append(record['elapsed_seconds'] / estimate(config, subjects))
            export_summary(output, records, 'running', time.monotonic() - started)
            return True

        screen_end = started + plan['stages_seconds']['screen']
        for config in candidates:
            trial(config, 'screen', plan['screen_subjects'], screen_end)
        # Spend spare screening time refining one or two regularization coordinates.
        for _ in range(100):
            leaders = ranked(records, 'screen')[:5]
            if not leaders or time.monotonic() >= screen_end:
                break
            config = copy.deepcopy(rng.choice(leaders)['config'])
            for key in rng.sample(list(SPACE)[:5], 2):
                config = changed(config, {key: rng.choice(SPACE[key])})
            trial(config, 'screen', plan['screen_subjects'], screen_end)
        confirm_end = screen_end + plan['stages_seconds']['confirm']
        leaders = ranked(records, 'screen')[:16]
        # Carry the matched standard control through every panel regardless of rank.
        control = changed(base, {'lk.affine_rho': 0.})
        confirmation = [control, base] + [r['config'] for r in leaders]
        seen = set()
        for config in confirmation:
            key = config_key(config)
            if key not in seen:
                seen.add(key)
                trial(config, 'confirm', plan['confirm_subjects'], confirm_end)
        # Full-cohort comparisons first at matched seed, then repeat at a new seed.
        finalists = [r['config'] for r in ranked(records, 'confirm')[:4]]
        full = [control, base] + finalists
        seen = set()
        for seed in [args.seed, args.seed + 1]:
            for config in full:
                key = config_key(config, seed=seed)
                if key not in seen:
                    seen.add(key)
                    trial(config, 'full', plan['full_subjects'], deadline - plan['reserve_seconds'], seed)
        # Reserve can finish a matched comparison; never launch past the hard deadline.
        seen = set()
        for config in [control, base] + finalists:
            key = config_key(config, seed=args.seed + 2)
            if key not in seen:
                seen.add(key)
                trial(config, 'reserve_repeat', plan['full_subjects'], deadline - 15, args.seed + 2)
        export_summary(output, records, 'finished', time.monotonic() - started)
    print(output)


if __name__ == '__main__':
    main()
