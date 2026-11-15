"""Shared-grid geometry and differentiable Gaussian smoothing."""
import math
import torch
import torch.nn.functional as F


def smooth_velocity(field, sigma, *, kernel='full', compile_smoothing=False):
    """Separable Gaussian; sigma is in lattice array-axis units."""
    if kernel not in ('full', 'three_tap', 'five_tap'):
        raise ValueError('Unsupported Gaussian kernel')
    if compile_smoothing and field.is_cuda and kernel == 'three_tap':
        if any(not math.isfinite(w) or w <= 0 for w in sigma) or min(field.shape[:3]) < 2:
            raise ValueError('Invalid Gaussian smoothing geometry')
        from medical.registration.compiled_decoder import compiled_three_tap_smoothing
        return compiled_three_tap_smoothing()(field,tuple(sigma))
    channels = field.shape[-1]
    values = field.movedim(-1, 0)[None]
    for axis, width in enumerate(sigma):
        if width <= 0 or not math.isfinite(width):
            raise ValueError('Gaussian widths must be finite and positive')
        radius = (1 if kernel == 'three_tap' else 2 if kernel == 'five_tap'
                  else math.ceil(4 * width))
        if radius >= field.shape[axis]:
            raise ValueError('Gaussian support exceeds reflect-padding domain')
        x = torch.arange(-radius, radius + 1, device=field.device, dtype=field.dtype)
        taps = torch.exp(-.5 * (x / width).square())
        taps = taps / taps.sum()
        shape = [1, 1, 1]; shape[axis] = taps.numel()
        weights = taps.reshape(1, 1, *shape).expand(channels, 1, *shape)
        padding = [0] * 6
        padding[2 * (2-axis):2 * (2-axis)+2] = [radius, radius]
        values = F.conv3d(F.pad(values, padding, mode='reflect'), weights, groups=channels)
    return values[0].movedim(0, -1)


class SharedGridGeometry:
    """Shared geometry and configuration for paired registration."""
    def __init__(self, moving, fixed, affine, *, grid_size=16, particles=4096,
                 sigma_cells=.75, sigma_mm=None, chunk=4096, recompute=True, c=4., eta=1e-6,
                 compile_velocity=False, gaussian_kernel='full'):
        if moving.shape != fixed.shape or moving.ndim != 3 or min(moving.shape) < 2:
            raise ValueError('Require matching 3D endpoint images')
        if any(type(n) is not int or n < 1 for n in (grid_size, particles, chunk)) or grid_size < 2:
            raise ValueError('Invalid lattice, particle count or chunk size')
        if (c < 0 or eta <= 0 or not math.isfinite(c) or not math.isfinite(eta)
                or type(sigma_cells) not in (int, float) or not math.isfinite(sigma_cells) or sigma_cells <= 0
                or (sigma_mm is not None and (type(sigma_mm) not in (int, float)
                    or not math.isfinite(sigma_mm) or sigma_mm <= 0))):
            raise ValueError('Invalid local solve or Gaussian settings')
        matrix = torch.as_tensor(affine, dtype=torch.float64)[:3, :3]
        spacing = matrix.norm(dim=0)
        if (spacing <= 0).any() or not torch.allclose(matrix.T @ matrix, torch.diag(spacing.square()), atol=1e-6):
            raise ValueError('Separable physical Gaussian requires orthogonal image axes')
        self.moving, self.fixed = moving, fixed
        self.grid_size, self.particles = grid_size, particles
        self.chunk, self.recompute, self.c, self.eta = chunk, recompute, c, eta
        if gaussian_kernel not in ('full', 'three_tap'):
            raise ValueError('Unsupported Gaussian kernel')
        self.gaussian_kernel = gaussian_kernel
        self.compile_velocity = compile_velocity
        self.extent = moving.new_tensor(list(reversed([n-1 for n in moving.shape])))
        self.sigma_cells = sigma_cells
        self.sigma_mm = sigma_mm
