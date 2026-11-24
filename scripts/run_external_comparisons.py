"""Run the authorized external comparison and maintain its central Markdown report."""
import argparse
from datetime import datetime
import json
import os
import statistics
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'docs/EXTERNAL_BASELINES.md'
METHODS = ('convexadam', 'fireants', 'demons', 'niftyreg')
NAMES = dict(convexadam='ConvexAdam', fireants='FireANTs-SyN', demons='Demons', niftyreg='NiftyReg')


def value(number):
    return '—' if number is None else f'{number:.5f}'


def update_report(output, exit_code):
    plan_path = output / 'plan.json'
    if not plan_path.exists():
        return
    plan = json.loads(plan_path.read_text())
    progress_path = output / 'progress.json'
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else dict(jobs=[], datasets={})
    records = progress['jobs']
    completed = {(r['dataset'], r['method'], r['pair_index']): r for r in records}
    failures = sum(r['returncode'] != 0 for r in records)
    status = 'Running' if exit_code is None else ('Completed' if exit_code == 0 else 'Finished with failures')
    now = datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y-%m-%d %H:%M:%S HKT')
    relative = os.path.relpath(output, REPORT.parent)
    lines = [f'## {output.name}', '', f'{status}. Updated {now}.', '',
             f'{len(records)} / {len(plan["jobs"])} fits finished, with {failures} process failures.', '',
             'One seed (42) per pair. Eight GPU workers and eight independent CPU workers, four native CPU threads per process.', '',
             f'[Frozen plan]({relative}/plan.json) · [Progress]({relative}/progress.json) · [Final summary]({relative}/summary.json)', '',
             'All runs use standardized_v1. Dice is the equal foreground-label mean. HD95 uses prepared-grid mm. '
             'NPJ is a percentage. SDlogJ uses log(max(detJ,1e-9)). Jacobians use central differences with two boundary voxels excluded. '
             'IXI HD95 is reported in mm using nominal 1 × 1 × 1 mm benchmark spacing. Missing surfaces produce null HD95 and explicit scored counts.', '',
             '| Dataset | Method | Finished | Successful | Dice | HD95 | NPJ % | SDlogJ | Backend | Runtime s/case | Scored pairs (Dice / HD95 / NPJ / SDlogJ) |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |']
    for dataset in dict.fromkeys(j['dataset'] for j in plan['jobs']):
        for method in METHODS:
            intended = sum(j['dataset'] == dataset and j['method'] == method for j in plan['jobs'])
            data = progress['datasets'].get(dataset, {}).get(method, {})
            metrics = data.get('metrics', {})
            numbers = [value(metrics.get(k, {}).get('mean')) for k in ('Dice', 'HD95', 'NPJ', 'SDlogJ')]
            counts = ' / '.join(str(metrics.get(k, {}).get('scored_pairs', 0)) for k in ('Dice', 'HD95', 'NPJ', 'SDlogJ'))
            durations = [r['seconds'] for r in records if r['dataset'] == dataset and r['method'] == method]
            runtime = f'{statistics.mean(durations):.2f} ± {statistics.pstdev(durations):.2f}' if durations else '—'
            backend = 'CPU, 4 threads' if method == 'demons' else ('CPU/OpenMP, 4 threads' if method == 'niftyreg' else 'GPU (RTX 5090)')
            lines.append(f'| {dataset} | {NAMES[method]} | {data.get("completed", 0)}/{intended} | {data.get("successful", 0)} | ' + ' | '.join(numbers) + f' | {backend} | {runtime} | {counts} |')
    lines += ['', 'Runtime is mean ± population SD of per-case subprocess wall time, including startup, input loading, registration, transformation evaluation and scoring. FireANTs used its PyTorch fallback with fused operations unavailable.', '', '### Per-case results', '',
              '| Dataset | Method | Pair | Status | Dice | HD95 | NPJ % | SDlogJ | Seconds | Artifacts |',
              '| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for job in plan['jobs']:
        row = completed.get((job['dataset'], job['method'], job['pair_index']))
        if row is None:
            continue
        target = Path(job['output'])
        path = target / 'metrics.json'
        scores = json.loads(path.read_text()) if row['returncode'] == 0 and path.exists() else {}
        numbers = [value(scores.get(k, {}).get('mean')) for k in ('Dice', 'HD95', 'NPJ', 'SDlogJ')]
        state = 'success' if row['returncode'] == 0 else f'failed ({row["returncode"]})'
        link = os.path.relpath(target, REPORT.parent)
        lines.append(f'| {job["dataset"]} | {NAMES.get(job["method"], job["method"])} | {job["pair_index"]} | {state} | ' + ' | '.join(numbers) +
                     f' | {row["seconds"]:.2f} | [metrics]({link}/metrics.json) · [log]({link}.log) |')
    start, end = f'<!-- BEGIN {output.name} -->', f'<!-- END {output.name} -->'
    block = start + '\n' + '\n'.join(lines) + '\n' + end
    existing = REPORT.read_text() if REPORT.exists() else '# External baseline results\n\nCentral record of standardized comparisons on Neurite-OASIS, ACDC, LPBA40 and IXI.\n'
    if start in existing:
        before, remainder = existing.split(start, 1)
        _, after = remainder.split(end, 1)
        content = before + block + after
    else:
        content = existing.rstrip() + '\n\n' + block + '\n'
    temporary = REPORT.with_suffix('.partial.md')
    temporary.write_text(content)
    temporary.replace(REPORT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    command = [sys.executable, '-u', '-m', 'medical.comparison_experiments', '--methods', *METHODS,
               '--output', str(output), '--cpu-workers', '8', '--gpus', *map(str, range(8)), '--run']
    process = subprocess.Popen(command, cwd=ROOT)
    try:
        while process.poll() is None:
            update_report(output, None)
            time.sleep(30)
        update_report(output, process.returncode)
    except BaseException:
        process.terminate()
        process.wait()
        update_report(output, process.returncode)
        raise
    raise SystemExit(process.returncode)


if __name__ == '__main__':
    main()
