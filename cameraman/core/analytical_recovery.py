"""Continuous-SIREN image derivatives feeding existing global flow and RK2.

Only image derivative construction changes. Velocity is solved on the full
spatial grid at interval midpoints and held fixed within each RK2 interval.
"""
import math
import time

import torch

from cameraman.core.flow import estimate_velocity_from_derivatives, integrate
from cameraman.core.lk_loss import analytical_siren_derivatives


def synchronized_time(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    return time.perf_counter()


@torch.no_grad()
def image_derivatives(source, target, coordinates, time_point, shape, chunk=0):
    """Full-grid g,q; optional query chunks bound SIREN activation memory only."""
    if len(shape) not in (2, 3) or min(shape) < 2 or math.prod(shape) != len(coordinates):
        raise ValueError('Coordinate count must match a 2D/3D grid with axes >=2')
    if chunk < 0:
        raise ValueError('Chunk must be nonnegative; 0 means one pass')
    scale = [2/(n-1) for n in reversed(shape)]
    step = chunk or len(coordinates)
    gradient = coordinates.new_empty((len(coordinates), len(shape)))
    temporal = coordinates.new_empty(len(coordinates))
    for start in range(0, len(coordinates), step):
        end = min(start+step, len(coordinates))
        g, q = analytical_siren_derivatives(source, target, coordinates[start:end], time_point, scale)
        gradient[start:end], temporal[start:end] = g, q
    return gradient.reshape(*shape, len(shape)), temporal.reshape(shape)


@torch.no_grad()
def midpoint_derivatives(source, target, grid, intervals=8, chunk=0):
    """Build g,q for each temporal midpoint on a full 2D/3D spatial grid."""
    if type(intervals) is not int or intervals < 1:
        raise ValueError('intervals must be a positive integer')
    shape = grid.shape[:-1]
    coordinates = (2*grid/grid.new_tensor([n-1 for n in reversed(shape)])-1).reshape(-1, len(shape))
    gradient = grid.new_empty((intervals, *shape, len(shape)))
    temporal = grid.new_empty((intervals, *shape))
    for i in range(intervals):
        gradient[i], temporal[i] = image_derivatives(source, target, coordinates,
            grid.new_tensor((i+.5)/intervals), shape, chunk)
    return gradient, temporal


@torch.no_grad()
def recover_2d(source, target, grid, intervals=8, chunk=0, *, smoothness=.01,
               ridge=1e-6, iterations=500, tolerance=1e-5, substeps=4):
    """Analytical midpoint derivatives + existing batched 2D flow/RK2."""
    started = synchronized_time(grid.device)
    gradient, temporal = midpoint_derivatives(source, target, grid, intervals, chunk)
    differentiated = synchronized_time(grid.device)
    velocity, residual, diagnostics = estimate_velocity_from_derivatives(gradient, temporal,
        smoothness=smoothness, ridge=ridge, iterations=iterations, tolerance=tolerance)
    solved = synchronized_time(grid.device)
    forward = integrate(velocity, grid, substeps)
    inverse = integrate(velocity, grid, substeps, inverse=True)
    finished = synchronized_time(grid.device)
    return forward, inverse, residual, {'derivatives_seconds': differentiated-started,
        'flow_seconds': solved-differentiated, 'integration_seconds': finished-solved,
        'total_seconds': finished-started, **diagnostics}


@torch.no_grad()
def recover_3d(source, target, grid, backend, intervals=8, kind='analytical', chunk=0,
               *, smoothness=.01, ridge=1e-6, iterations=500, tolerance=1e-5, substeps=4):
    """Streaming volume recovery; backend is the existing INR flow3d module.

    Keep velocity fields for reverse integration. Query chunks, if requested,
    only bound model activations; the global flow solve always uses full volumes.
    """
    if intervals < 1 or chunk < 0 or kind not in ('analytical', 'finite'):
        raise ValueError('Positive interval count and kind analytical/finite required')
    shape = grid.shape[:-1]
    if len(shape) != 3:
        raise ValueError('Expected a volume grid [D,H,W,3]')
    coordinates = (2*grid/grid.new_tensor(list(reversed([n-1 for n in shape])))-1).reshape(-1, 3)
    timings = dict(derivatives_seconds=0., flow_seconds=0., integration_seconds=0.)
    started = synchronized_time(grid.device)

    def image(time_point):
        first, second = dict(source.named_parameters()), dict(target.named_parameters())
        state = {k: torch.lerp(v, second[k], time_point) for k, v in first.items()}
        result = coordinates.new_empty(len(coordinates))
        step = chunk or len(coordinates)
        for start in range(0, len(coordinates), step):
            result[start:start+step] = torch.func.functional_call(source, state,
                (coordinates[start:start+step],)).flatten()
        return result.reshape(shape)

    first = image(grid.new_tensor(0.)) if kind == 'finite' else None
    timings['derivatives_seconds'] += synchronized_time(grid.device)-started
    forward, velocities, diagnostics = grid.clone(), [], []
    for index in range(intervals):
        mark = synchronized_time(grid.device)
        if kind == 'analytical':
            g, q = image_derivatives(source, target, coordinates,
                grid.new_tensor((index+.5)/intervals), shape, chunk)
        else:
            second = image(grid.new_tensor((index+1)/intervals))
            gz, gy, gx = torch.gradient((first+second)*.5)
            g = torch.stack((gx, gy, gz), -1)
            q = (second-first)*intervals
            first = second
        end = synchronized_time(grid.device)
        timings['derivatives_seconds'] += end-mark
        velocity, info = backend.estimate_velocity_from_derivatives(g, q, smoothness,
            ridge, iterations, tolerance)
        mark = synchronized_time(grid.device)
        timings['flow_seconds'] += mark-end
        forward = backend.integrate_interval(forward, velocity, 1/intervals, substeps)
        end = synchronized_time(grid.device)
        timings['integration_seconds'] += end-mark
        velocities.append(velocity)
        diagnostics.append(info)
    mark = synchronized_time(grid.device)
    inverse = grid.clone()
    for field in reversed(velocities):
        inverse = backend.integrate_interval(inverse, field, -1/intervals, substeps)
    finished = synchronized_time(grid.device)
    timings['integration_seconds'] += finished-mark
    timings.update(total_seconds=finished-started,
        max_relative_linear_residual=max(d['relative_linear_residual'] for d in diagnostics),
        max_iterations=max(d['iterations'] for d in diagnostics),
        unconverged_intervals=sum(not d['converged'] for d in diagnostics))
    return forward, inverse, timings
