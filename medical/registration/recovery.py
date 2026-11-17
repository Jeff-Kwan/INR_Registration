"""Frozen trajectory -> midpoint optical flow -> in-memory forward/inverse maps."""
from dataclasses import dataclass, field
import math

import numpy as np
import torch
from medical.registration.cuda_kernels import supported

from medical.artifacts import recovery_scratch
from medical.models.siren import coordinates_from_indices
from medical.models.siren_functional import coordinate_scale
from medical.registration.flow import estimate_velocity_from_derivatives
from medical.registration.geometry import identity_grid, sample


@dataclass
class RecoveryResult:
    forward: torch.Tensor
    inverse: torch.Tensor
    diagnostics: list = field(default_factory=list)


@torch.no_grad()
def derivative_grid(pair, shape, time, chunk=65536, compile_derivatives=True):
    parameter = next(pair.parameters())
    if type(chunk) is not int or chunk < 0:
        raise ValueError('Derivative chunk must be a nonnegative integer')
    scale = coordinate_scale(shape, parameter)
    prepared = pair.prepare(compile_parameters=compile_derivatives).at(time)
    total = math.prod(shape)
    gradient = parameter.new_empty((total, len(shape)))
    temporal = parameter.new_empty(total)
    for start in range(0, total, chunk or total):
        end = min(start + (chunk or total), total)
        coordinates = coordinates_from_indices(torch.arange(start, end, device=parameter.device), shape).to(parameter.dtype)
        g, q = prepared.derivatives(coordinates, scale, compile_derivatives=compile_derivatives)
        gradient[start:end], temporal[start:end] = g, q
    return gradient.reshape(*shape, len(shape)), temporal.reshape(shape)


def integrate_interval(mapping, velocity, dt, substeps=4):
    if type(substeps) is not int or substeps < 1:
        raise ValueError('RK2 substeps must be a positive integer')
    if (not torch.is_grad_enabled() and supported(mapping.device) and mapping.dtype == torch.float32
            and velocity.dtype == mapping.dtype and velocity.device == mapping.device
            and mapping.ndim == 4 and mapping.shape[-1] == 3 and mapping.shape == velocity.shape
            and min(mapping.shape[:3]) >= 2 and mapping.numel() < 2**31):
        from medical.registration.rk2_cuda import integrate
        return integrate(mapping, velocity, dt, substeps)
    step = dt / substeps
    for _ in range(substeps):
        midpoint = mapping + .5 * step * sample(velocity, mapping)
        mapping = mapping + step * sample(velocity, midpoint)
    return mapping


def _velocity_history(scratch, shape, intervals, device):
    """Keep modest CUDA histories on-device. Large histories use disposable scratch."""
    history_shape = (intervals, *shape, len(shape))
    size = 4 * math.prod(history_shape)
    if device.type == 'cuda':
        free, _ = torch.cuda.mem_get_info(device)
        # Leave the solver, dense SIREN derivatives and other GPU clients headroom.
        if size <= min(free // 12, 2 << 30):
            try:
                return torch.empty(history_shape, dtype=torch.float32, device=device)
            except torch.OutOfMemoryError:
                pass
    backing = np.lib.format.open_memmap(scratch / 'velocities.npy', mode='w+',
                                        dtype=np.float32, shape=history_shape)
    return torch.from_numpy(backing)


@torch.no_grad()
def recover(pair, shape, settings, scratch_root, *, chunk=65536, compile_derivatives=True, compile_solver=True,
            velocity_solver=estimate_velocity_from_derivatives, integrator=integrate_interval, max_axis=None):
    if (getattr(pair, 'requires_working_grid_metadata', False)
            or getattr(pair, 'requires_physical_units_metadata', False)
            or getattr(pair, 'working_grid_metadata', None) is not None
            or getattr(pair, 'physical_units_metadata', None) is not None):
        raise ValueError('Recovery supports native-voxel models only')
    if max_axis is not None:
        if velocity_solver is not estimate_velocity_from_derivatives or integrator is not integrate_interval:
            raise ValueError('Capped recovery supports the ordinary native brightness-flow solver')
        from medical.registration.recovery_sampling import recover_capped
        return recover_capped(pair, shape, settings, scratch_root, max_axis=max_axis,
                              chunk=chunk, compile_derivatives=compile_derivatives,
                              compile_solver=compile_solver)
    if set(settings) != {'intervals','smoothness','ridge','iterations','tolerance','substeps'}:
        raise ValueError('Expected only PCG+RK2 recovery settings')
    device = next(pair.parameters()).device
    pair.zero_grad(set_to_none=True)
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    grid = identity_grid(shape, device)
    forward = grid.clone()
    dt = 1 / settings['intervals']
    diagnostics = []
    previous_velocity = None
    with recovery_scratch(scratch_root) as scratch:
        velocities = _velocity_history(scratch, shape, settings['intervals'], device)
        try:
            for index in range(settings['intervals']):
                extra = ({"compile_solver": compile_solver}
                         if velocity_solver is estimate_velocity_from_derivatives else {})
                gradient, temporal = derivative_grid(pair, shape, (index + .5) * dt, chunk, compile_derivatives)
                if velocity_solver is estimate_velocity_from_derivatives and previous_velocity is not None:
                    extra['initial_velocity'] = previous_velocity
                velocity, diagnostic = velocity_solver(gradient, temporal, **extra, **{
                    k: settings[k] for k in ('smoothness', 'ridge', 'iterations', 'tolerance')})
                diagnostics.append({'interval': index, 'time_a': index * dt,
                                    'time_b': (index + 1) * dt,
                                    **diagnostic})
                if velocity_solver is estimate_velocity_from_derivatives:
                    previous_velocity = velocity
                velocities[index].copy_(velocity)
                forward = integrator(forward, velocity, dt, settings['substeps'])
                del gradient, temporal, velocity
            previous_velocity = None
            inverse = grid.clone()
            for index in range(settings['intervals'] - 1, -1, -1):
                velocity = velocities[index].to(device)
                inverse = integrator(inverse, velocity, -dt, settings['substeps'])
                del velocity
        finally:
            previous_velocity = None
            del velocities
    return RecoveryResult(forward, inverse, diagnostics)
