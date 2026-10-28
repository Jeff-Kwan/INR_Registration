"""Regularized differential optical flow solved by batched preconditioned CG.

Minimize sum (I_tau + grad(I_mid) dot v)^2 + lambda * sum_edges |dv|^2
+ ridge * sum |v|^2, with natural (zero flux) smoothness boundaries.
All derivatives use pixels and unit trajectory time; v is pixels/unit time.
"""
import torch
from cameraman.core.synthetic import sample


def laplacian(v):
    result = torch.zeros_like(v)
    dx = v[:, :, 1:] - v[:, :, :-1]
    dy = v[:, 1:] - v[:, :-1]
    result[:, :, 1:] += dx
    result[:, :, :-1] -= dx
    result[:, 1:] += dy
    result[:, :-1] -= dy
    return result


@torch.no_grad()
def estimate_velocity(sequence, horizontal=False, smoothness=0.01,
                      iterations=500, tolerance=1e-5, ridge=1e-6):
    dt = 1 / (sequence.shape[0] - 1)
    mid = (sequence[:-1] + sequence[1:]) / 2
    gy, gx = torch.gradient(mid, dim=(1, 2))
    temporal = (sequence[1:] - sequence[:-1]) / dt
    return estimate_velocity_from_derivatives(torch.stack((gx, gy), -1), temporal,
        horizontal, smoothness, iterations, tolerance, ridge)


@torch.no_grad()
def estimate_velocity_from_derivatives(gradient, temporal, horizontal=False,
                                      smoothness=.01, iterations=500, tolerance=1e-5, ridge=1e-6):
    """Same PCG objective, accepting g[K,H,W,2], q[K,H,W] in pixel/time units."""
    if gradient.shape != (*temporal.shape, 2) or temporal.ndim != 3:
        raise ValueError('Expected g[K,H,W,2] and q[K,H,W]')
    full_gradient = gradient
    gradient = gradient[..., :1] if horizontal else gradient
    rhs = -gradient * temporal[..., None]
    degree = torch.full_like(temporal, 4)
    degree[:, [0, -1], :] -= 1
    degree[:, :, [0, -1]] -= 1
    diagonal = gradient.square() + smoothness * degree[..., None] + ridge

    def operator(v):
        return (gradient * (gradient * v).sum(-1, keepdim=True)
                + smoothness * laplacian(v) + ridge * v)

    def dot(a, b):
        return (a * b).sum((1, 2, 3), keepdim=True)

    velocity = torch.zeros_like(rhs)
    residual = rhs.clone()
    z = residual / diagonal
    direction = z.clone()
    rz = dot(residual, z)
    rhs_norm = dot(rhs, rhs).sqrt().clamp_min(1e-20)
    completed = 0
    for iteration in range(iterations):
        active = dot(residual, residual).sqrt() > tolerance * rhs_norm
        # Keep convergence masking on device; avoid a host synchronization per CG step.
        if iteration % 10 == 0 and not active.any():
            break
        applied = operator(direction)
        alpha = torch.where(active, rz / dot(direction, applied).clamp_min(1e-30), 0)
        velocity += alpha * direction
        residual -= alpha * applied
        z = residual / diagonal
        next_rz = dot(residual, z)
        beta = torch.where(active, next_rz / rz.clamp_min(1e-30), 0)
        direction = z + beta * direction
        rz = next_rz
        completed = iteration + 1
    relative = (dot(residual, residual).sqrt() / rhs_norm).flatten()
    if horizontal:
        velocity = torch.cat((velocity, torch.zeros_like(velocity)), -1)
    brightness = temporal + full_gradient[..., 0] * velocity[..., 0] + full_gradient[..., 1] * velocity[..., 1]
    return velocity, brightness, {"iterations": completed,
                                 "max_relative_linear_residual": relative.max().item()}


@torch.no_grad()
def integrate(velocity, grid, substeps=4, inverse=False, method="rk2"):
    """Advect coordinates through Eulerian fields with RK2 (default) or RK4.

    Reversing field order and time gives a target-to-source sampling map.
    """
    if method not in ('rk2', 'rk4') or type(substeps) is not int or substeps < 1:
        raise ValueError('Expected method rk2/rk4 and positive integer substeps')
    mapping = grid.clone()
    dt = (-1 if inverse else 1) / (len(velocity) * substeps)
    fields = velocity.flip(0) if inverse else velocity
    for field in fields:
        for _ in range(substeps):
            k1 = sample(field, mapping)
            if method == 'rk2':
                midpoint = mapping + 0.5 * dt * k1
                mapping = mapping + dt * sample(field, midpoint)
            else:
                k2 = sample(field, mapping + 0.5 * dt * k1)
                k3 = sample(field, mapping + 0.5 * dt * k2)
                k4 = sample(field, mapping + dt * k3)
                mapping = mapping + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    return mapping
