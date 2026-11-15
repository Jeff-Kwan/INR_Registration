"""Global brightness-flow PCG; separate from the local training penalty."""
from functools import lru_cache, partial
import math
import torch
from medical.registration.cuda_kernels import supported

def laplacian(values, axis_weights=None):
    result = torch.zeros_like(values)
    for axis in range(values.ndim - 1):
        high, low = [slice(None)] * values.ndim, [slice(None)] * values.ndim
        high[axis], low[axis] = slice(1, None), slice(None, -1)
        difference = values[tuple(high)] - values[tuple(low)]
        if axis_weights is not None: difference = difference * axis_weights[axis]
        result[tuple(high)] += difference
        result[tuple(low)] -= difference
    return result


def stencil_laplacian(value, axis_weights=None):
    result = torch.zeros_like(value)
    for axis in range(value.ndim - 1):
        first = value.narrow(axis, 0, 1)
        last = value.narrow(axis, value.shape[axis]-1, 1)
        before = torch.cat((first, value.narrow(axis, 0, value.shape[axis]-1)), dim=axis)
        after = torch.cat((value.narrow(axis, 1, value.shape[axis]-1), last), dim=axis)
        term = (value-before) + (value-after)
        result = result + (term if axis_weights is None else term*axis_weights[axis])
    return result


def pcg_step(velocity, residual, direction, rz, gradient, inverse_diagonal, smoothness, ridge, axis_weights=None, *, use_stencil=False):
    data_applied = gradient * (gradient*direction).sum(-1, keepdim=True)
    applied = (data_applied
               + smoothness*(stencil_laplacian(direction, axis_weights) if use_stencil else laplacian(direction, axis_weights)) + ridge*direction)
    alpha = rz / (direction*applied).sum().clamp_min(1e-30)
    velocity = velocity + alpha*direction
    residual = residual - alpha*applied
    z = residual*inverse_diagonal
    next_rz = (residual*z).sum()
    direction = z + (next_rz/rz.clamp_min(1e-30))*direction
    return velocity, residual, direction, next_rz


@lru_cache(maxsize=1)
def _compiled_step():
    return torch.compile(partial(pcg_step, use_stencil=True), fullgraph=True)


@torch.no_grad()
def estimate_velocity_from_derivatives(gradient, temporal, smoothness=.01, ridge=1e-6,
                                      iterations=500, tolerance=2e-5, *,
                                      initial_velocity=None, compile_solver=True, spacing=None):
    """Global PCG; spacing in array-axis order, with physical gradients/velocity."""
    dimensions = temporal.ndim
    if dimensions not in (2, 3) or gradient.shape != (*temporal.shape, dimensions):
        raise ValueError('Expected matching 2D/3D g[...,D] and q[...]')
    if not all(math.isfinite(x) and x > 0 for x in (smoothness, ridge, tolerance)):
        raise ValueError('Flow smoothness, ridge, tolerance must be finite and positive')
    if type(iterations) is not int or iterations < 1:
        raise ValueError('Flow iterations must be a positive integer')
    axis_weights = None
    if spacing is not None:
        if len(spacing)!=dimensions or not all(math.isfinite(h) and h>0 for h in spacing):
            raise ValueError('Expected positive finite spacing in array-axis order')
        axis_weights = gradient.new_tensor(spacing).square().reciprocal()
    rhs = -gradient * temporal[..., None]

    def original_operator(value):
        # Independent original expression for the reported true residual.
        data = gradient * (gradient * value).sum(-1, keepdim=True)
        result = (data
                + smoothness * laplacian(value, axis_weights) + ridge * value)
        return result

    cuda_step = None
    if (compile_solver and supported(gradient.device) and dimensions == 3
            and gradient.dtype == temporal.dtype == torch.float32
            and 0 < gradient.numel() <= 2**31 - 1):
        from medical.registration.flow_cuda import CudaPcgStep, planar
        weights = (1., 1., 1.) if axis_weights is None else axis_weights.tolist()
        cuda_step = CudaPcgStep(gradient, smoothness, ridge, weights)
        inverse_diagonal = cuda_step.inverse
        step = cuda_step
    else:
        degree = torch.zeros_like(temporal)
        for axis in range(dimensions):
            weight=1. if axis_weights is None else axis_weights[axis]
            axis_degree=torch.full_like(temporal,2.)
            for side in (0,-1):
                boundary=[slice(None)]*dimensions;boundary[axis]=side
                axis_degree[tuple(boundary)]-=1
            degree+=weight*axis_degree
        diagonal = gradient.square() + smoothness * degree[..., None] + ridge
        inverse_diagonal = diagonal.reciprocal()
        reference_step = _compiled_step() if compile_solver and gradient.is_cuda else pcg_step

        def step(velocity, residual, direction, rz):
            return reference_step(velocity, residual, direction, rz, gradient,
                                  inverse_diagonal, smoothness, ridge, axis_weights)

    def dot(a, b):
        return (a * b).sum()

    if initial_velocity is None:
        velocity = torch.zeros_like(rhs)
        residual = rhs.clone()
    else:
        if (initial_velocity.shape != rhs.shape or initial_velocity.device != rhs.device
                or initial_velocity.dtype != rhs.dtype or not torch.isfinite(initial_velocity).all()):
            raise ValueError('Initial velocity must be finite and match flow shape, device and dtype')
        velocity = initial_velocity.clone()
        residual = rhs - original_operator(velocity)
    if initial_velocity is not None and not bool(rhs.count_nonzero()):
        velocity = torch.zeros_like(rhs)
        residual = rhs.clone()
    if cuda_step is not None:
        velocity, residual = planar(velocity), planar(residual)
    direction = residual * inverse_diagonal
    rz = dot(residual, direction)
    rhs_norm = dot(rhs, rhs).sqrt().clamp_min(1e-20)
    completed, restarts = 0, 0
    true_norm = None
    for iteration in range(iterations):
        if iteration % 10 == 0 and residual.norm() <= tolerance * rhs_norm:
            true_residual = rhs - original_operator(velocity)
            checked_norm = true_residual.norm()
            if checked_norm <= tolerance * rhs_norm:
                true_norm = checked_norm
                break
            residual = planar(true_residual) if cuda_step is not None else true_residual
            direction = residual * inverse_diagonal
            rz = dot(residual, direction)
            restarts += 1
        velocity, residual, direction, rz = step(velocity, residual, direction, rz)
        completed = iteration + 1
    # Report the true residual, not only the recursively updated CG residual.
    if true_norm is None:
        true_norm = (rhs - original_operator(velocity)).norm()
    relative = true_norm / rhs_norm
    brightness = temporal + (gradient * velocity).sum(-1)
    if not torch.isfinite(velocity).all() or not torch.isfinite(relative):
        raise RuntimeError("Non-finite optical-flow solution")
    diagnostic = {"iterations": completed, "relative_linear_residual": float(relative),
                      "converged": bool(relative <= tolerance),
                      "recursive_relative_residual": float(residual.norm()/rhs_norm),
                      "residual_restarts": restarts,
                      "termination_reason": "tolerance" if relative <= tolerance else "iteration_limit",
                      "brightness_residual_rms": float(brightness.square().mean().sqrt())}
    return velocity.contiguous() if cuda_step is not None else velocity, diagnostic
