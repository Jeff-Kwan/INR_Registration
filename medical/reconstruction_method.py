"""Native DIR-Lab endpoint-MSE baseline through the shared case runner."""
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from medical.case_runner import load_volume
from medical.configuration import validate_config
from medical.models.siren import coordinates_from_indices, endpoint_reconstruction
from medical.registration.training import fit_pair


@torch.no_grad()
def lung_reconstruction(pair, images, masks, chunk):
    scores = {'scope': 'lung_mask_evaluation_only', 'data_range': 1.0}
    for name, model, image, mask in zip(
            ('moving', 'fixed'), (pair.source, pair.target), images, masks):
        indices = mask.flatten().nonzero().flatten()
        error = image.new_zeros((), dtype=torch.float64)
        for selected in indices.split(chunk):
            coordinates = coordinates_from_indices(selected, image.shape).to(image.dtype)
            difference = model(coordinates).flatten() - image.flatten()[selected]
            error += difference.square().sum(dtype=torch.float64)
        mse = float(error / len(indices))
        scores[name] = {'voxel_count': len(indices), 'mse': mse,
                        'psnr_db': -10 * math.log10(mse) if mse else None}
    return scores


def run_case(case, config, device):
    validate_config(config)
    objective = config['objective']
    if (objective['endpoint'] != 'mse' or objective['weight'] != 0
            or objective['registration']['weight'] != 0):
        raise ValueError('Reconstruction baseline requires MSE and zero auxiliary weights')
    if 'working_grid' in config or config.get('spatial_units', 'voxel') != 'voxel':
        raise ValueError('Reconstruction baseline uses native voxel grids')
    images, masks, normalization = [], [], {}
    for name in ('moving', 'fixed'):
        volume = case[name]
        data = volume['data']
        low, high = float(data.min()), float(data.max())
        if high <= low:
            raise ValueError('Cannot normalize a constant image')
        images.append(torch.from_numpy(((data-low)/(high-low)).transpose(2, 1, 0).copy()).to(device))
        path = Path(volume['path'])
        stem = path.name.removesuffix('.gz').removesuffix('.nii')
        mask = load_volume(path.with_name(stem + '_lung_mask.nii.gz'))
        if (mask['data'].shape != data.shape or not np.allclose(mask['affine'], volume['affine'])
                or not np.isin(mask['data'], [0, 1]).all() or not mask['data'].any()):
            raise ValueError('Expected a nonempty binary lung mask on the image grid')
        masks.append(torch.from_numpy(mask['data'].transpose(2, 1, 0).copy()).to(device).bool())
        normalization[name] = {'min': low, 'max': high}
    affine = case['moving']['affine'].copy()
    affine[:3, :3] = affine[:3, :3][:, ::-1]
    started = time.perf_counter()
    pair = fit_pair(*images, config, affine=affine, moving_mask=masks[0], fixed_mask=masks[1],
                    report=lambda row: print(json.dumps({'training': row}), flush=True))
    training_seconds = time.perf_counter() - started
    started = time.perf_counter()
    chunk = config['runtime']['inference_chunk'] or 65536
    reconstruction = endpoint_reconstruction(pair, images, chunk)
    lung = lung_reconstruction(pair, images, masks, chunk)
    return {
        'metrics': {'reconstruction': reconstruction, 'lung_reconstruction': lung,
                    'seed': config['runtime']['seed'],
                    'intensity_normalization': {'method': 'independent_minmax', **normalization},
                    'timings': {'training_seconds': training_seconds,
                                'evaluation_seconds': time.perf_counter() - started}},
        'checkpoint': {'format_version': 1, 'model': pair.model_config,
                       'training': pair.training_metadata, 'config': config,
                       'state': {key: value.detach().cpu() for key, value in pair.state_dict().items()}},
    }
