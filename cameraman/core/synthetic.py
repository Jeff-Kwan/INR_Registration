"""Pixel coordinates are (x, y); maps are [H, W, 2] in pixels."""
from pathlib import Path
import math
import warnings

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def identity_grid(size, device):
    y, x = torch.meshgrid(torch.arange(size, device=device),
                          torch.arange(size, device=device), indexing="ij")
    return torch.stack((x, y), -1).float()


def sample(field, coordinates, mode="bilinear", padding_mode="border"):
    """Bilinear pull sampling, align_corners=True, border padding.

    field: [H,W] or [H,W,C]; coordinates: [H,W,2].
    """
    scalar = field.ndim == 2
    if scalar:
        field = field[..., None]
    h, w = field.shape[:2]
    scale = coordinates.new_tensor([w - 1, h - 1])
    grid = 2 * coordinates / scale - 1
    result = F.grid_sample(field.permute(2, 0, 1)[None], grid[None],
                           mode=mode, padding_mode=padding_mode,
                           align_corners=True)[0].permute(1, 2, 0)
    return result[..., 0] if scalar else result


def stationary_map(velocity, grid, time=1., steps=7):
    """exp(time*v) in pixels using displacement scaling and squaring."""
    displacement = time * velocity / (2**steps)
    for _ in range(steps):
        displacement = displacement + sample(displacement, grid + displacement)
    return grid + displacement


def map_jacobian(mapping):
    dy, dx = torch.gradient(mapping, dim=(0, 1))
    return dx[..., 0]*dy[..., 1] - dx[..., 1]*dy[..., 0]


def minimum_cell_jacobian(mapping):
    """Minimum determinant at all four corners of every bilinear mesh cell."""
    top = mapping[:-1, 1:] - mapping[:-1, :-1]
    bottom = mapping[1:, 1:] - mapping[1:, :-1]
    left = mapping[1:, :-1] - mapping[:-1, :-1]
    right = mapping[1:, 1:] - mapping[:-1, 1:]
    return torch.stack([dx[..., 0]*dy[..., 1]-dx[..., 1]*dy[..., 0]
                        for dx, dy in ((top, left), (top, right), (bottom, left), (bottom, right))]).min()


def elastic_geometry(grid, amplitude=10., sigma=12., seed=42, steps=7, min_jacobian=.2,
                     boundary="taper", min_identity_epe=0.):
    """Gaussian-smoothed random stationary velocity, tapered at image edges.

    Positive/negative integration gives forward/inverse maps. If sampled maps
    fail discrete Jacobian or bounds checks, reduce velocity amplitude and retry.
    """
    if amplitude < 0 or sigma <= 0 or steps < 1 or not 0 < min_jacobian < 1:
        raise ValueError("Elastic settings require amplitude>=0, sigma>0, steps>=1, 0<min_jacobian<1")
    h, w = grid.shape[:2]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(1, 2, h, w, generator=generator).to(grid.device)
    radius = min(math.ceil(3*sigma), min(h, w)-1)
    axis = torch.arange(-radius, radius+1, device=grid.device, dtype=grid.dtype)
    kernel = torch.exp(-.5*(axis/sigma).square())
    kernel = kernel / kernel.sum()
    noise = F.conv2d(F.pad(noise, (radius, radius, 0, 0), mode="reflect"),
                     kernel.view(1, 1, 1, -1).expand(2, 1, 1, -1), groups=2)
    noise = F.conv2d(F.pad(noise, (0, 0, radius, radius), mode="reflect"),
                     kernel.view(1, 1, -1, 1).expand(2, 1, -1, 1), groups=2)
    velocity = noise[0].permute(1, 2, 0)
    if boundary == "taper":
        taper = (torch.sin(math.pi*grid[..., 0]/(w-1))*torch.sin(math.pi*grid[..., 1]/(h-1))).square()
        velocity = velocity * taper[..., None]
        velocity[[0, -1], :, :] = 0
        velocity[:, [0, -1], :] = 0
    elif boundary == "normal":
        # Only normal velocity vanishes at each edge; tangential motion remains.
        # A narrow collar replaces the old broad 2D center-weighting window.
        width = max(2., 2*sigma)
        distance_x = torch.minimum(grid[..., 0], w-1-grid[..., 0])
        distance_y = torch.minimum(grid[..., 1], h-1-grid[..., 1])
        tx = torch.sin(.5*math.pi*(distance_x/width).clamp(0, 1)).square()
        ty = torch.sin(.5*math.pi*(distance_y/width).clamp(0, 1)).square()
        velocity = velocity * torch.stack((tx, ty), -1)
    else:
        raise ValueError("boundary must be taper or normal")
    velocity = velocity * (amplitude / velocity.norm(dim=-1).max().clamp_min(1e-12))
    for attempt in range(13):
        minimum = float("inf")
        minimum_cell = float("inf")
        valid = True
        for time in torch.linspace(-1, 1, 65, device=grid.device):
            mapping = stationary_map(velocity, grid, time, steps)
            determinant = map_jacobian(mapping)
            minimum = min(minimum, determinant.min().item())
            minimum_cell = min(minimum_cell, minimum_cell_jacobian(mapping).item())
            valid = valid and bool(torch.isfinite(mapping).all() and torch.isfinite(determinant).all())
            valid = valid and bool((mapping[..., 0] >= -1e-4).all() and (mapping[..., 0] <= w-1+1e-4).all()
                                   and (mapping[..., 1] >= -1e-4).all() and (mapping[..., 1] <= h-1+1e-4).all())
        if valid and minimum >= min_jacobian and minimum_cell > 1e-4:
            break
        velocity *= .5
    else:
        raise RuntimeError("Could not construct an elastic map passing numerical fold checks")
    forward = stationary_map(velocity, grid, 1., steps)
    inverse = stationary_map(velocity, grid, -1., steps)
    composition_error = (sample(inverse, forward)-grid).norm(dim=-1)
    lengths = (forward-grid).norm(dim=-1)
    identity_epe = lengths.mean().item()
    if min_identity_epe > 0 and identity_epe <= min_identity_epe:
        raise ValueError(f"Transformation rejected: full regular-grid EPE {identity_epe:.4f} <= {min_identity_epe:.4f}")
    diagnostics = {"requested_max_speed_pixels": amplitude, "actual_max_speed_pixels": velocity.norm(dim=-1).max().item(),
                   "amplitude_halvings": attempt, "sampled_min_jacobian": minimum,
                   "sampled_min_cell_jacobian": minimum_cell,
                   "forward_min_jacobian": map_jacobian(forward).min().item(),
                   "inverse_min_jacobian": map_jacobian(inverse).min().item(),
                   "inverse_composition_mean_pixels": composition_error.mean().item(),
                   "inverse_composition_max_pixels": composition_error.max().item(),
                   "max_displacement_pixels": lengths.max().item(),
                   "identity_grid_epe_pixels": identity_epe,
                   "identity_grid_fraction_above_5px": (lengths > 5).float().mean().item(),
                   "boundary_mode": boundary, "checked_times": 65, "minimum_required_identity_epe": min_identity_epe}
    return {"kind": "elastic", "velocity": velocity, "integration_steps": steps,
            "forward_map": forward, "inverse_map": inverse, "diagnostics": diagnostics}


def oracle_frame(source, grid, displacement, time, geometry=None):
    inverse = (stationary_map(geometry["velocity"], grid, -time, geometry["integration_steps"])
               if geometry is not None and geometry.get("kind") == "elastic" else grid-time*displacement)
    return sample(source, inverse)


def make_pair(image_path, size, amplitude, frames, device, generate_oracle=True,
              deformation="shear", elastic_sigma=12., deformation_seed=42,
              elastic_steps=7, elastic_min_jacobian=.2, return_geometry=False,
              elastic_boundary="taper", min_transform_epe=0.):
    path = Path(image_path)
    if not path.exists() and path.name == "cameraman.png":
        alternative = path.with_suffix(".jpg")
        if alternative.exists():
            warnings.warn(f"{path} missing; using {alternative}.")
            path = alternative
    with Image.open(path) as image:
        image = image.convert("L").resize((size, size), Image.Resampling.BICUBIC)
        source = torch.tensor(np.asarray(image).copy(), device=device).float() / 255
    grid = identity_grid(size, device)
    displacement = torch.zeros_like(grid)
    displacement[..., 0] = amplitude * torch.sin(2 * math.pi * grid[..., 1] / size)
    geometry = {"kind": "shear"}
    if deformation == "elastic":
        geometry = elastic_geometry(grid, amplitude, elastic_sigma, deformation_seed, elastic_steps,
                                    elastic_min_jacobian, elastic_boundary, min_transform_epe)
        displacement = geometry["forward_map"] - grid
    elif deformation != "shear":
        raise ValueError("deformation must be shear or elastic")
    oracle = None
    if generate_oracle:
        times = torch.linspace(0, 1, frames, device=device)
        oracle = torch.stack([oracle_frame(source, grid, displacement, t, geometry) for t in times])
    target = oracle[-1] if oracle is not None else oracle_frame(source, grid, displacement, 1., geometry)
    speed_bound = geometry["diagnostics"]["actual_max_speed_pixels"] if deformation == "elastic" else abs(amplitude)
    margin = math.ceil(speed_bound) + 3
    if 2 * margin >= size:
        raise ValueError("Image too small for amplitude and evaluation margin.")
    mask = torch.zeros_like(source, dtype=torch.bool)
    mask[margin:-margin, margin:-margin] = True
    result = (source, target, grid, displacement, oracle, mask, path.resolve(), margin)
    return (*result, geometry) if return_geometry else result
