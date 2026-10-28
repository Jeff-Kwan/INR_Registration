"""Evaluate saved paired LPBA SIRENs without training or overwriting the run."""
import argparse
import gc
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np
import torch

from medical.models.siren3d import Siren3D
from medical.registration.configuration import load_config
from medical.registration.io import (LABEL_METRIC_DEFINITION, label_metrics, load_image,
                                     load_label, require_same_grid, write_json)
from medical.registration.train import (RunETA, discover_subjects, evaluate, label_path,
                                        prepare_recovery, recover, resolve_dataset_path,
                                        validate_config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path, help='Existing run with config.json and subject/models.pt')
    parser.add_argument('--output', type=Path, required=True, help='New, empty evaluation directory')
    parser.add_argument('--intervals', type=int, default=8)
    parser.add_argument('--method', choices=('analytical', 'finite'), default='analytical')
    parser.add_argument('--inference-chunk', type=int, default=262144)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    config = load_config(args.run/'config.json')
    baseline_recovery = dict(config['recovery'])
    baseline_key = 'frame' if baseline_recovery.get('method', 'finite') == 'finite' else 'saved_analytical'
    config['recovery'].pop('frames', None)
    config['recovery'].update(method=args.method, intervals=args.intervals)
    config['runtime'].update(device=args.device, inference_chunk=args.inference_chunk)
    config['output'].update(root=str(args.output.resolve()), save_frames=False, save_velocities=False)
    validate_config(config)
    subjects = discover_subjects(config)
    for subject, _ in subjects:
        if not (args.run/subject/'models.pt').is_file():
            raise FileNotFoundError(args.run/subject/'models.pt')
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f'Output must be empty: {args.output}')
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output/'config.json', config)
    write_json(args.output/'manifest.json', {'source_run': str(args.run.resolve()),
               'training': False, 'label_metric_definition': LABEL_METRIC_DEFINITION, 'baseline_recovery': baseline_recovery, 'recovery': config['recovery'],
               'gpu': torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else 'CPU',
               'timing': 'recovery_seconds includes endpoint decoding, flow, integration and map/reconstruction writes; excludes metrics'})
    device = torch.device(args.device)
    reference, fixed_np, _ = load_image(resolve_dataset_path(config, config['dataset']['fixed']))
    fixed_label = load_label(resolve_dataset_path(config, config['dataset']['fixed_label']), reference)
    fixed = torch.from_numpy(fixed_np).to(device)
    eta, rows = RunETA(len(subjects), config), []
    for index, (subject, path) in enumerate(subjects):
        eta.begin_pair(index)
        output = args.output/subject
        output.mkdir()
        require_same_grid(reference, nib.load(str(path)), str(path))
        _, moving_np, _ = load_image(path)
        moving = torch.from_numpy(moving_np).to(device)
        labels = load_label(label_path(config, subject), reference)
        checkpoint = torch.load(args.run/subject/'models.pt', map_location='cpu', weights_only=True)
        source, target = [Siren3D(**config['model']).to(device).eval() for _ in range(2)]
        source.load_state_dict(checkpoint['source'])
        target.load_state_dict(checkpoint['target'])
        del checkpoint
        prepare_recovery(source, target)
        print(f'{subject}: recovering {args.method}, {args.intervals} intervals', flush=True)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        forward, inverse, src, tgt, diagnostics = recover(source, target, reference.shape,
            reference, config, output, device, eta)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        recovery_seconds = time.perf_counter()-started
        metrics = evaluate(moving, fixed, labels, fixed_label, forward, inverse,
                           src, tgt, reference, config, output)
        metrics.update(subject=subject, recovery_seconds=recovery_seconds,
            flow_all_intervals_converged=all(d['converged'] for d in diagnostics),
            flow_max_relative_residual=max(d['relative_linear_residual'] for d in diagnostics))
        write_json(output/'metrics.json', metrics)
        old = json.loads((args.run/subject/'metrics.json').read_text())
        # Compare both recovery methods using the same metric definition.
        recomputed = old['labels'].get('metric_definition') != metrics['labels']['metric_definition']
        if recomputed:
            baseline_labels = load_label(args.run/subject/'warped_label.nii.gz', reference)
            old['labels'], baseline_per_label = label_metrics(fixed_label, baseline_labels,
                reference.header.get_zooms()[:3], compute_hd95=config['evaluation']['hd95'])
            write_json(output/'baseline_labels_recomputed.json', old['labels'])
            write_json(output/'baseline_per_label_metrics.json', baseline_per_label)
        row = {'subject': subject, 'recovery_seconds': recovery_seconds,
               'baseline_labels_recomputed': recomputed}
        for name, result in [(baseline_key, old), ('analytical' if args.method == 'analytical' else 'new_finite', metrics)]:
            row[name] = {'dice': result['labels']['dice'], 'hd95_mm': result['labels']['hd95_mm'],
                         'warped_mse': result['after']['mse'], 'folding_fraction': result['folding_fraction']}
        rows.append(row)
        methods = [baseline_key, 'analytical' if args.method == 'analytical' else 'new_finite']
        means = {method: {key: float(np.mean([r[method][key] for r in rows]))
                          for key in rows[0][method]} for method in methods}
        write_json(args.output/'comparison.json', {'source_run': str(args.run.resolve()),
                   'completed': len(rows), 'total': len(subjects),
                   'label_metric_definition': LABEL_METRIC_DEFINITION, 'means': means, 'subjects': rows})
        print(json.dumps(row), flush=True)
        del moving, source, target, forward, inverse, metrics, src, tgt
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    print(json.dumps(means, indent=2), flush=True)


if __name__ == '__main__':
    main()
