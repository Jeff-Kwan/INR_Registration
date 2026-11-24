"""Endpoint reconstruction with moment matching and the endpoint parameter penalty."""
from dataclasses import dataclass
from functools import lru_cache
import torch

from medical.models.siren_functional import siren_forward, validate_coordinates
from medical.registration.ncc import endpoint_statistics, ncc_options


@dataclass
class TrainingBatch:
    coordinates: torch.Tensor
    source: torch.Tensor
    target: torch.Tensor
    target_coordinates: torch.Tensor = None


def endpoint_parameter_penalty(source, target):
    """Mean squared difference over all paired SIREN weights and biases."""
    pairs = [(a, b) for layer_a, layer_b in zip(source, target)
             for a, b in zip(layer_a, layer_b)]
    return sum((a - b).square().sum() for a, b in pairs) / sum(a.numel() for a, _ in pairs)


def endpoint_loss(pair, batch, settings, *, compile_endpoints=False):
    target_coordinates = batch.coordinates if batch.target_coordinates is None else batch.target_coordinates
    endpoints = pair.prepare(temporal=False)
    validate_coordinates(batch.coordinates, endpoints.source)
    validate_coordinates(target_coordinates, endpoints.target)
    function = _compiled_endpoints() if compile_endpoints and batch.coordinates.is_cuda else _endpoint_kernel
    return function(endpoints.source, endpoints.target, endpoints.omegas, batch.coordinates,
                    target_coordinates, batch.source, batch.target, settings['endpoint'],
                    ncc_options(settings.get('ncc')), parameter_l2=settings.get('parameter_l2', 0.),
                    endpoint_weight=settings.get('endpoint_weight', 1.))


def _endpoint_kernel(source, target, omegas, source_coordinates, target_coordinates,
                     source_truth, target_truth, mode, options, *, fuse_scalar_affines=False,
                     parameter_l2=0., endpoint_weight=1.):
    if mode not in ('ncc', 'mse'):
        raise ValueError('Expected canonical NCC or explicitly requested ablation MSE')
    parameter_penalty = endpoint_parameter_penalty(source, target) if parameter_l2 else None
    source_values = siren_forward(source_coordinates, source, omegas, fuse_scalar_affines=fuse_scalar_affines).flatten()
    target_values = siren_forward(target_coordinates, target, omegas, fuse_scalar_affines=fuse_scalar_affines).flatten()
    source_mse = (source_values - source_truth).square().mean()
    target_mse = (target_values - target_truth).square().mean()
    extra = endpoint_statistics(source_values, target_values, source_truth, target_truth, **options)
    fitting = extra['endpoint_ncc_loss'] if mode == 'ncc' else source_mse + target_mse
    endpoint = endpoint_weight * fitting + extra['weighted_moments']
    if parameter_penalty is not None:
        extra.update(parameter_l2=parameter_penalty, weighted_parameter_l2=parameter_l2 * parameter_penalty)
        endpoint = endpoint + extra['weighted_parameter_l2']
    return {'total': endpoint, 'source_mse': source_mse, 'target_mse': target_mse, **extra}


@lru_cache(maxsize=1)
def _compiled_endpoints():
    return torch.compile(_endpoint_kernel, fullgraph=True)
