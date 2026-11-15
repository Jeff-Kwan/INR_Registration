"""Label-free endpoint sampling; see docs/STRUCTURE_SAMPLING.md for citations."""
import math
import numpy as np
import torch
from scipy.ndimage import gaussian_filter


def validate_sampling(options):
    if (not isinstance(options, dict) or set(options) != {'structure_fraction', 'sigma_mm'}
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in options.values())
            or not 0 <= options['structure_fraction'] <= 1 or options['sigma_mm'] <= 0):
        raise ValueError('endpoint_sampling requires structure_fraction in [0,1] and positive sigma_mm')


def structure_scores(image, affine, sigma_mm):
    """Gaussian derivatives in physical units; input/output arrays and affine columns are ZYX.

    Nearest-edge extension and four-sigma Gaussian truncation are explicit.
    Only observed inputs enter this preprocessing; training graphs stay live.
    """
    if image.ndim != 3 or min(image.shape) < 2:
        raise ValueError('Structure sampling requires a 3D image with axes of length >= 2')
    matrix = np.asarray(affine, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError('Structure sampling requires a finite 4x4 affine')
    basis = matrix[:3, :3]
    spacing = np.linalg.norm(basis, axis=0)
    if np.any(spacing <= 0) or not np.allclose((basis / spacing).T @ (basis / spacing), np.eye(3), atol=1e-6):
        raise ValueError('Structure sampling requires orthogonal nondegenerate image axes')
    values = image.detach().cpu().numpy().astype(np.float32, copy=False)
    if not np.isfinite(values).all():
        raise ValueError('Structure sampling requires finite input intensities')
    squared = np.zeros_like(values)
    for axis in range(3):
        order = [0, 0, 0]
        order[axis] = 1
        derivative = gaussian_filter(values, sigma_mm / spacing, order=order,
                                     mode='nearest', truncate=4.0) / spacing[axis]
        squared += derivative * derivative
    return np.sqrt(squared)


class StructureEndpointSampler:
    """Fixed uniform/gradient mixture, conditional on each endpoint's support.

    Inverse-CDF sampling avoids scanning a whole volume on every training step.
    FP64 CDFs avoid losing low-mass regions to cumulative FP32 rounding.
    """
    def __init__(self, image, active, affine, options):
        validate_sampling(options)
        self.active, self.device = active, image.device
        self.size = image.numel() if active is None else active.numel()
        self.fraction = options['structure_fraction']
        score = structure_scores(image, affine, options['sigma_mm']).reshape(-1)
        if active is not None:
            score = score[active.cpu().numpy()]
        cdf = np.cumsum(score, dtype=np.float64)
        self.cdf = None
        if cdf.size and cdf[-1] > 0 and self.fraction:
            cdf /= cdf[-1]
            self.cdf = torch.from_numpy(cdf).to(self.device)
        self.metadata = {**options, 'fallback_uniform': self.cdf is None,
                         'support': 'whole_volume' if active is None else 'endpoint_mask'}

    def sample(self, count, generator):
        indices = torch.randint(self.size, (count,), device=self.device, generator=generator)
        if self.cdf is not None:
            draws = torch.rand(count, device=self.device, dtype=torch.float64, generator=generator)
            structured = torch.searchsorted(self.cdf, draws, right=True).clamp_max(self.size - 1)
            choose = torch.rand(count, device=self.device, generator=generator) < self.fraction
            indices = torch.where(choose, structured, indices)
        return indices if self.active is None else self.active[indices]


class StructureCropSampler:
    """Choose usual crops or endpoint-gradient centers once per minibatch.

    The outer mixture is over crops; structured draws use pure gradient weights.
    Maps span observed images, independently of any lung crop-center masks.
    """
    def __init__(self, moving, fixed, affine, options, *, seed):
        validate_sampling(options)
        self.fraction = options['structure_fraction']
        self.generator = torch.Generator(device=moving.device).manual_seed(seed + 4000067)
        self.samplers = [StructureEndpointSampler(image, None, affine,
            {**options, 'structure_fraction': 1.}) for image in (moving, fixed)]
        self.metadata = {**options, 'method': 'direct_structure_crop_center',
            'endpoint_probability': [.5, .5], 'transport': False,
            'support': 'whole_volume', 'inverse_probability_weighting': False,
            'endpoints': [sampler.metadata for sampler in self.samplers]}

    def sample_center(self, shape):
        """Return an XYZ voxel center, or None for the ordinary crop sampler."""
        device = self.samplers[0].device
        if torch.rand((), device=device, generator=self.generator) >= self.fraction:
            return None
        endpoint = int(torch.randint(2, (), device=device, generator=self.generator))
        index = self.samplers[endpoint].sample(1, self.generator)[0]
        z = torch.div(index, shape[1] * shape[2], rounding_mode='floor')
        y = torch.div(index, shape[2], rounding_mode='floor') % shape[1]
        x = index % shape[2]
        return torch.stack((x, y, z)).to(torch.float32)
