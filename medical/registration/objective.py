"""Endpoint reconstruction: MSE or global/Gaussian-window NCC with Gaussian-moment matching."""
from dataclasses import dataclass
from functools import lru_cache
import math

import torch

from medical.models.siren_functional import siren_forward, validate_coordinates
from medical.registration.ncc import endpoint_statistics, ncc_options


@dataclass
class TrainingBatch:
    coordinates: torch.Tensor
    source: torch.Tensor
    target: torch.Tensor
    target_coordinates: torch.Tensor = None
    source_cells: torch.Tensor = None
    target_cells: torch.Tensor = None


def endpoint_loss(pair, batch, settings, *, compile_endpoints=False):
    target_coordinates = batch.coordinates if batch.target_coordinates is None else batch.target_coordinates
    options = ncc_options(settings.get('ncc')) if settings['endpoint'] in ('ncc','gw_ncc','lncc') else {}
    if settings['endpoint'] == 'gw_ncc' and (batch.source_cells is None or batch.target_cells is None):
        raise ValueError('Endpoint GW-NCC requires cell indices in the reconstruction domain')
    endpoints = pair.prepare(temporal=False)
    validate_coordinates(batch.coordinates, endpoints.source)
    validate_coordinates(target_coordinates, endpoints.target)
    function = _compiled_endpoints() if compile_endpoints and batch.coordinates.is_cuda else _endpoint_kernel
    return function(endpoints.source, endpoints.target, endpoints.omegas, batch.coordinates,
                    target_coordinates, batch.source, batch.target, settings['endpoint'], options, batch.source_cells, batch.target_cells, settings.get('endpoint_gw_grid_size', 8), settings.get('endpoint_lncc', {}).get('size', 9))


def _endpoint_kernel(source, target, omegas, source_coordinates, target_coordinates,
                     source_truth, target_truth, mode, options, source_cells=None, target_cells=None, gw_grid_size=8, window_size=9, *, fuse_scalar_affines=False):
    source_values = siren_forward(source_coordinates, source, omegas, fuse_scalar_affines=fuse_scalar_affines).flatten()
    target_values = siren_forward(target_coordinates, target, omegas, fuse_scalar_affines=fuse_scalar_affines).flatten()
    source = (source_values - source_truth).square().mean()
    target = (target_values - target_truth).square().mean()
    extra = {}
    endpoint = source+target
    if mode in ('ncc','gw_ncc','lncc'):
        extra = endpoint_statistics(source_values, target_values, source_truth, target_truth, **options)
        similarity = extra['endpoint_ncc_loss']
        if mode == 'gw_ncc':
            from medical.registration.symmetric_registration import gaussian_weighted_ncc
            source_gw, _, source_local = gaussian_weighted_ncc(source_values, source_truth, source_cells,
                grid_size=gw_grid_size, sigma_cells=.75, gaussian_kernel='three_tap', epsilon=options['epsilon'])
            target_gw, _, target_local = gaussian_weighted_ncc(target_values, target_truth, target_cells,
                grid_size=gw_grid_size, sigma_cells=.75, gaussian_kernel='three_tap', epsilon=options['epsilon'])
            similarity = source_gw + target_gw
            extra.update(endpoint_gw_ncc_loss=similarity, source_local_ncc_loss=source_local,
                         target_local_ncc_loss=target_local)
        if mode == 'lncc':
            from medical.registration.endpoint_lncc import window_loss
            source_local = window_loss(source_values, source_truth, window_size, options['epsilon'])
            target_local = window_loss(target_values, target_truth, window_size, options['epsilon'])
            similarity = .5 * (similarity + source_local + target_local)
            extra.update(endpoint_lncc_loss=similarity, source_local_ncc_loss=source_local,
                         target_local_ncc_loss=target_local)
        endpoint = similarity+extra['weighted_moments']
    return {'total': endpoint, 'source_mse': source, 'target_mse': target, **extra}


@lru_cache(maxsize=1)
def _compiled_endpoints():
    return torch.compile(_endpoint_kernel, fullgraph=True)


def scheduled_weight(settings, step, steps):
    fraction = (step - 1) / max(steps - 1, 1) if steps > 1 else 1.
    factor = {'constant': 1., 'linear': fraction,
              'cosine': .5 * (1 - math.cos(math.pi * fraction))}[settings['weight_schedule']]
    return settings['weight'] * factor
