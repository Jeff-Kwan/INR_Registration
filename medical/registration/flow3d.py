"""Regularized 3D optical flow and RK2/RK4 map integration.

Arrays are [D,H,W]; vector channels are pixel (x=W,y=H,z=D).
The solver is the 3D extension of ``cameraman.core.flow``.
"""
import torch
import torch.nn.functional as F


def identity_grid(shape, device):
    z, y, x = torch.meshgrid(*(torch.arange(n, device=device, dtype=torch.float32)
                              for n in shape), indexing="ij")
    return torch.stack((x, y, z), -1)


def normalized_grid(mapping, shape):
    return 2 * mapping / mapping.new_tensor([shape[2] - 1, shape[1] - 1, shape[0] - 1]) - 1


def sample(volume, mapping, mode="bilinear", padding_mode="border"):
    """Sample a scalar [D,H,W] or vector [D,H,W,C] volume."""
    vector = volume.ndim == 4
    image = volume.permute(3, 0, 1, 2)[None] if vector else volume[None, None]
    result = F.grid_sample(image, normalized_grid(mapping, volume.shape[:3])[None],
                           mode=mode, padding_mode=padding_mode, align_corners=True)[0]
    return result.permute(1, 2, 3, 0) if vector else result[0]


def laplacian(values):
    result = torch.zeros_like(values)
    for axis in range(3):
        high, low = [slice(None)] * 4, [slice(None)] * 4
        high[axis], low[axis] = slice(1, None), slice(None, -1)
        difference = values[tuple(high)] - values[tuple(low)]
        result[tuple(high)] += difference
        result[tuple(low)] -= difference
    return result


@torch.no_grad()
def estimate_velocity(first, second, dt, smoothness=.01, ridge=1e-6,
                      iterations=500, tolerance=1e-5):
    """PCG for (g g^T + smoothness*L + ridge*I)v = -g*I_t.

    Central derivatives use voxel units; natural zero-flux smoothness edges.
    One full-resolution time interval is solved at once to bound VRAM usage.
    """
    midpoint = (first + second) * .5
    gz, gy, gx = torch.gradient(midpoint)
    gradient = torch.stack((gx, gy, gz), -1)
    temporal = (second - first) / dt
    return estimate_velocity_from_derivatives(gradient, temporal, smoothness, ridge, iterations, tolerance)


@torch.no_grad()
def estimate_velocity_from_derivatives(gradient, temporal, smoothness=.01, ridge=1e-6,
                                      iterations=500, tolerance=1e-5):
    """Same global PCG solve with supplied voxel/time derivatives g[D,H,W,3], q."""
    if gradient.shape != (*temporal.shape, 3) or temporal.ndim != 3:
        raise ValueError('Expected g[D,H,W,3] and q[D,H,W]')
    rhs = -gradient * temporal[..., None]
    degree = torch.full_like(temporal, 6)
    for axis in range(3):
        boundary = [slice(None)] * 3
        boundary[axis] = [0, -1]
        degree[tuple(boundary)] -= 1
    diagonal = gradient.square() + smoothness * degree[..., None] + ridge

    def operator(value):
        return (gradient * (gradient * value).sum(-1, keepdim=True)
                + smoothness * laplacian(value) + ridge * value)

    def dot(a, b):
        return (a * b).sum()

    velocity = torch.zeros_like(rhs)
    residual = rhs.clone()
    z = residual / diagonal
    direction = z.clone()
    rz = dot(residual, z)
    rhs_norm = dot(rhs, rhs).sqrt().clamp_min(1e-20)
    completed = 0
    for iteration in range(iterations):
        if iteration % 10 == 0 and dot(residual, residual).sqrt() <= tolerance * rhs_norm:
            break
        applied = operator(direction)
        alpha = rz / dot(direction, applied).clamp_min(1e-30)
        velocity.add_(alpha * direction)
        residual.sub_(alpha * applied)
        z = residual / diagonal
        next_rz = dot(residual, z)
        direction = z + (next_rz / rz.clamp_min(1e-30)) * direction
        rz = next_rz
        completed = iteration + 1
    # Report the true residual, not only the recursively updated CG residual.
    relative = (rhs - operator(velocity)).norm() / rhs_norm
    brightness = temporal + (gradient * velocity).sum(-1)
    if not torch.isfinite(velocity).all() or not torch.isfinite(relative):
        raise RuntimeError("Non-finite optical-flow solution")
    return velocity, {"iterations": completed, "relative_linear_residual": float(relative),
                      "converged": bool(relative <= tolerance),
                      "brightness_residual_rms": float(brightness.square().mean().sqrt())}


@torch.no_grad()
def integrate_interval(mapping, velocity, dt, substeps=4, method="rk2"):
    """RK2 (default) or classical RK4 for one fixed Eulerian velocity interval."""
    if method not in ('rk2', 'rk4') or type(substeps) is not int or substeps < 1:
        raise ValueError('Expected method rk2/rk4 and positive integer substeps')
    step = dt / substeps
    for _ in range(substeps):
        k1 = sample(velocity, mapping)
        if method == 'rk2':
            midpoint = mapping + .5 * step * k1
            mapping = mapping + step * sample(velocity, midpoint)
        else:
            k2 = sample(velocity, mapping + .5 * step * k1)
            k3 = sample(velocity, mapping + .5 * step * k2)
            k4 = sample(velocity, mapping + step * k3)
            mapping = mapping + step/6 * (k1 + 2*k2 + 2*k3 + k4)
    return mapping
