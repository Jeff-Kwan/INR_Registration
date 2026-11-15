"""Aggregate the authorized convergence trials from their saved evidence."""
import csv
import io
import json
from pathlib import Path
import statistics


ROOTS = [Path('results') / name for name in (
    'siren_convergence_20261113', 'siren_stacked_20261113',
    'siren_strong_registration_20261113', 'siren_frequency_balance_20261113',
    'siren_capacity_20261113', 'siren_output_gauge_20261113',
    'siren_loss_curriculum_20261113', 'siren_generalization_refinement_20261113',
    'siren_combination_20261113')]
CASES = [1, 3, 4, 5]
REFERENCE = Path('results/dirlab_quick4_default_40k_20261113')


def means(directory):
    files = [directory / f'case{case:02d}' / 'metrics.json' for case in CASES]
    rows = [json.loads(path.read_text()) for path in files if path.exists()]
    if len(rows) != 4:
        return None
    forward = statistics.mean(row['idir_rounded_tre_mm']['mean_mm'] for row in rows)
    reverse = statistics.mean(row['reverse_T50_to_T00']['idir_rounded_tre_mm']['mean_mm'] for row in rows)
    return {'forward_tre_mm': forward, 'reverse_tre_mm': reverse,
            'bidirectional_tre_mm': (forward + reverse) / 2,
            'mean_lung_folding': statistics.mean(row['folding_fraction_lung'] for row in rows),
            'converged_intervals': sum(interval['converged'] for row in rows for interval in row['recovery']),
            'total_intervals': sum(len(row['recovery']) for row in rows),
            'per_case': {str(case): {
                'forward_tre_mm': row['idir_rounded_tre_mm']['mean_mm'],
                'reverse_tre_mm': row['reverse_T50_to_T00']['idir_rounded_tre_mm']['mean_mm'],
                'lung_folding': row['folding_fraction_lung'],
                'training_seconds': row.get('timings', {}).get('training_seconds')}
                for case, row in zip(CASES, rows)}}


def probes(directory):
    gap_paths = [directory / f'case{case:02d}' / 'diagnostics.json' for case in CASES]
    fit_paths = [directory / f'case{case:02d}_fidelity.json' for case in CASES]
    transport_paths = [directory / f'case{case:02d}_transport.json' for case in CASES]
    count = sum(gap.exists() and fit.exists() and transport.exists()
                for gap, fit, transport in zip(gap_paths, fit_paths, transport_paths))
    result = {'probed_cases': count, 'large_motion_tre_mm': None, 'endpoint_ncc': None}
    if count == 4:
        gaps = [json.loads(path.read_text()) for path in gap_paths]
        fits = [json.loads(path.read_text()) for path in fit_paths]
        bins = [row['forward']['motion']['motion_bins']['10-inf'] for row in gaps]
        result['large_motion_tre_mm'] = sum(row['count'] * row['continuous_tre_mm'] for row in bins) / sum(row['count'] for row in bins)
        result['endpoint_ncc'] = statistics.mean(
            (row['moving_fit']['mean_ncc'] + row['fixed_fit']['mean_ncc']) / 2 for row in fits)
    return result


def main():
    reference = means(REFERENCE)
    assert reference is not None
    trials = []
    pending = []
    for root in ROOTS:
        plan = json.loads((root / 'plan.json').read_text())
        for variant in plan['variants']:
            directory = root / variant
            metrics = means(directory)
            if metrics is None:
                pending.append(str(directory))
                continue
            config = json.loads((directory / 'config.json').read_text())
            model = config['model']
            width, layers = model['width'], model['layers']
            parameters = width * 4 + (layers - 1) * (width * width + width) + width + 1
            row = {'recipe': variant, 'directory': str(directory),
                   'steps': config['training']['steps'], 'parameters_per_endpoint': parameters,
                   'model': model, 'training': config['training'], 'objective': config['objective'],
                   **metrics, **probes(root / 'probes' / variant / 'step05000')}
            row['dirlab_target_met'] = (
                row['steps'] == 5000 and row['probed_cases'] == 4
                and row['forward_tre_mm'] <= reference['forward_tre_mm']
                and row['reverse_tre_mm'] <= reference['reverse_tre_mm'])
            trials.append(row)
    trials.sort(key=lambda row: row['bidirectional_tre_mm'])
    destination = ROOTS[0]
    result = {'reference_directory': str(REFERENCE), 'reference': reference,
              'completed_recipes': len(trials), 'pending_directories': pending, 'trials': trials}
    temporary = destination / 'all_trials.tmp'
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    temporary.replace(destination / 'all_trials.json')
    fields = ['recipe', 'directory', 'steps', 'parameters_per_endpoint', 'forward_tre_mm',
              'reverse_tre_mm', 'bidirectional_tre_mm', 'large_motion_tre_mm',
              'endpoint_ncc', 'mean_lung_folding', 'converged_intervals',
              'total_intervals', 'probed_cases', 'dirlab_target_met']
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fields, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(trials)
    (destination / 'all_trials.csv').write_text(buffer.getvalue())
    lines = [f'{len(trials)} completed recipes, {len(pending)} pending recipes.', '',
             '| Recipe | Forward mm | Reverse mm | Large-motion mm | Endpoint NCC | DIR-Lab target |',
             '| --- | ---: | ---: | ---: | ---: | --- |']
    for row in trials[:12]:
        large = f"{row['large_motion_tre_mm']:.3f}" if row['large_motion_tre_mm'] is not None else 'pending'
        ncc = f"{row['endpoint_ncc']:.3f}" if row['endpoint_ncc'] is not None else 'pending'
        status = 'met' if row['dirlab_target_met'] else (
            'probes pending' if row['probed_cases'] < 4 else 'unmet')
        lines.append(f"| {row['recipe']} | {row['forward_tre_mm']:.3f} | {row['reverse_tre_mm']:.3f} | {large} | {ncc} | {status} |")
    report = Path('docs/SIREN_CONVERGENCE.md')
    text = report.read_text()
    start, end = '<!-- ranking-start -->', '<!-- ranking-end -->'
    before, remainder = text.split(start)
    _, after = remainder.split(end)
    report.write_text(before + start + '\n' + '\n'.join(lines) + '\n' + end + after)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
