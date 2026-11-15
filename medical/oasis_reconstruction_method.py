"""Endpoint-only MSE fitting on native OASIS images."""
import json
import time

import torch

from medical.configuration import validate_config
from medical.models.siren import endpoint_reconstruction
from medical.registration.training import fit_pair


def run_case(case, config, device):
    validate_config(config)
    objective = config['objective']
    if (objective['endpoint'] != 'mse' or objective['weight'] != 0
            or objective['registration']['weight'] != 0):
        raise ValueError('Reconstruction requires endpoint MSE and zero auxiliary weights')
    if 'working_grid' in config or config.get('spatial_units', 'voxel') != 'voxel':
        raise ValueError('This reconstruction experiment uses native voxel grids')
    images, normalization = [], {}
    for name in ('moving', 'fixed'):
        data = case[name]['data']
        low, high = float(data.min()), float(data.max())
        if high <= low:
            raise ValueError('Cannot normalize a constant image')
        images.append(torch.from_numpy(((data-low)/(high-low)).transpose(2, 1, 0).copy()).to(device))
        normalization[name] = {'min': low, 'max': high}
    affine = case['moving']['affine'].copy()
    affine[:3, :3] = affine[:3, :3][:, ::-1]
    started = time.perf_counter()
    pair = fit_pair(*images, config, affine=affine,
                    report=lambda row: print(json.dumps({'training': row}), flush=True))
    training_seconds = time.perf_counter() - started
    started = time.perf_counter()
    reconstruction = endpoint_reconstruction(pair, images, config['runtime']['inference_chunk'])
    return {
        'metrics': {'reconstruction': reconstruction, 'seed': config['runtime']['seed'],
                    'intensity_normalization': {'method': 'independent_minmax', **normalization},
                    'timings': {'training_seconds': training_seconds,
                                'evaluation_seconds': time.perf_counter() - started}},
        'checkpoint': {'format_version': 1, 'model': pair.model_config,
                       'training': pair.training_metadata, 'config': config,
                       'state': {key: value.detach().cpu() for key, value in pair.state_dict().items()}},
    }
