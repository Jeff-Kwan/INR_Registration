"""Explicit sampled-window endpoint NCC; offsets are defined in physical mm."""
import math
import torch
from medical.registration.image_sampling import sample_points
from medical.registration.objective import TrainingBatch


def validate_windows(options):
    if not isinstance(options, dict) or set(options) != {'windows', 'size', 'spacing_mm'}:
        raise ValueError('endpoint_lncc requires windows, size and spacing_mm')
    if (type(options['windows']) is not int or options['windows'] < 1
            or type(options['size']) is not int or options['size'] < 3 or options['size'] % 2 != 1
            or type(options['spacing_mm']) not in (int, float)
            or not math.isfinite(options['spacing_mm']) or options['spacing_mm'] <= 0):
        raise ValueError('Invalid LNCC window geometry')


def sample_window_batch(branch, moving, fixed, generator, options, working_spacing):
    size = options['size']
    step = options['spacing_mm'] / working_spacing
    radius = (size // 2) * step
    if bool((branch.domain_extent < 2 * radius).any()):
        raise ValueError('LNCC window exceeds reconstruction crop')
    offsets = torch.arange(size, device=moving.device, dtype=moving.dtype) * step - radius
    offsets = torch.stack(torch.meshgrid(offsets, offsets, offsets, indexing='ij'), -1).reshape(-1, 3)
    centers = branch.origin + radius + torch.rand((options['windows'], 3), device=moving.device,
        dtype=moving.dtype, generator=generator) * (branch.domain_extent - 2 * radius)
    points = (centers[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    coordinates = 2 * points / branch.extent - 1
    return TrainingBatch(coordinates, sample_points(moving, points), sample_points(fixed, points))


def window_loss(prediction, truth, size, epsilon):
    a, b = prediction.reshape(-1, size ** 3), truth.reshape(-1, size ** 3)
    a, b = a - a.mean(-1, keepdim=True), b - b.mean(-1, keepdim=True)
    rho = (a * b).mean(-1) / ((a.square().mean(-1) + epsilon)
                                      * (b.square().mean(-1) + epsilon)).sqrt()
    return (1 - rho.clamp(-1, 1)).mean()
