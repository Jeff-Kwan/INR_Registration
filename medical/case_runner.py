"""Dataset-independent single-case runner. Methods are supplied as Python plugins.

A plugin exports run_case(case, config, device) and returns a mapping with
'metrics' (JSON-compatible) and 'checkpoint' (torch-serializable). The case
contains moving/fixed float32 arrays and their NIfTI affines. Plugins must not
write image maps. This module does not choose a training objective or method.
"""
from __future__ import annotations

import argparse
import importlib
import hashlib
import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import torch


def load_volume(path):
    image = nib.load(str(path))
    data = np.asarray(image.dataobj, dtype=np.float32)
    if data.ndim != 3 or not np.isfinite(data).all():
        raise ValueError(f'{path}: expected a finite three-dimensional volume')
    return {'data': data, 'affine': image.affine.copy(), 'path': str(Path(path).resolve())}


def run_case(moving, fixed, config, output, method, device='cpu'):
    """Run one case and return its metrics; output must be a new directory."""
    moving_volume, fixed_volume = load_volume(moving), load_volume(fixed)
    if moving_volume['data'].shape != fixed_volume['data'].shape or not np.allclose(
            moving_volume['affine'], fixed_volume['affine'], rtol=1e-5, atol=1e-5):
        raise ValueError('Prepared volumes must share a grid and affine')
    output = Path(output)
    if output.resolve().is_relative_to((Path.cwd() / 'Data').resolve()):
        raise ValueError('Output cannot be under Data')
    output.mkdir(parents=True, exist_ok=False)
    cuda = torch.device(device).type == 'cuda'
    if cuda:
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)
    result = method({'moving': moving_volume, 'fixed': fixed_volume}, config, device)
    if not isinstance(result, dict) or 'metrics' not in result or 'checkpoint' not in result:
        raise ValueError('Method must return a mapping with metrics and checkpoint')
    if cuda:
        result['metrics']['runtime_memory'] = {
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(device)}
    metrics_json = json.dumps(result['metrics'], indent=2, allow_nan=False)
    evidence = {'moving': moving_volume['path'], 'fixed': fixed_volume['path'],
                'shape': list(moving_volume['data'].shape),
                'affine': moving_volume['affine'].tolist(), 'device': device, 'config': config}
    module = getattr(method, '__module__', None)
    source = getattr(sys.modules.get(module), '__file__', None)
    evidence['method_module'] = module
    if source and Path(source).is_file():
        evidence['method_source'] = str(Path(source).resolve())
        evidence['method_source_sha256'] = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    evidence_json = json.dumps(evidence, indent=2, allow_nan=False)
    torch.save(result['checkpoint'], output / 'checkpoint.pt')
    (output / 'metrics.json').write_text(metrics_json + '\n')
    (output / 'run.json').write_text(evidence_json + '\n')
    return result['metrics']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--moving', required=True)
    parser.add_argument('--fixed', required=True)
    parser.add_argument('--config', required=True, help='JSON configuration passed to the method')
    parser.add_argument('--output', required=True, help='New case output directory')
    parser.add_argument('--method', default='medical.paired_method', help='Importable module exposing run_case')
    parser.add_argument('--device', default='cuda:0', help='PyTorch device, e.g. cuda:0')
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text())
    method = importlib.import_module(args.method).run_case
    scores = run_case(args.moving, args.fixed, config, args.output, method, args.device)
    print(json.dumps(scores, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
