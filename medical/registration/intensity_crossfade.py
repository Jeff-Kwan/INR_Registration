"""No-fit intensity path implementing the frozen stage-3 derivative interface."""
from dataclasses import dataclass
import torch
import torch.nn.functional as F


def _sample(volume, coordinates):
    # Channel-last volume, normalized XYZ points, border extension as in recovery.
    grid = coordinates.reshape(1, 1, 1, -1, 3)
    return F.grid_sample(volume.movedim(-1, 0)[None], grid, mode='bilinear',
                         padding_mode='border', align_corners=True)[0, :, 0, 0].T


class IntensityCrossfade(torch.nn.Module):
    """I(x,t)=(1-t)*moving(x)+t*fixed(x), in the recovery ZYX array layout.

    Spatial derivatives are central voxel differences (one-sided at the boundary).
    The temporal derivative is exactly fixed-moving. Only derivatives are supplied;
    the ordinary PCG solver, warm starts and RK2 integrator are reused unchanged.
    """
    def __init__(self, moving, fixed):
        super().__init__()
        if moving.shape != fixed.shape or moving.ndim != 3 or min(moving.shape) < 2:
            raise ValueError('Expected matching 3D endpoint arrays with axes >=2')
        if moving.dtype != fixed.dtype or moving.device != fixed.device:
            raise ValueError('Endpoint dtype and device must match')
        self.anchor = torch.nn.Parameter(moving.new_zeros(()), requires_grad=False)
        with torch.no_grad():
            gradients = [torch.stack(tuple(reversed(torch.gradient(image, edge_order=1))), -1)
                         for image in (moving, fixed)]
            self.register_buffer('moving_gradient', gradients[0], persistent=False)
            self.register_buffer('fixed_gradient', gradients[1], persistent=False)
            self.register_buffer('temporal', (fixed-moving)[..., None], persistent=False)
        self.register_buffer('native_half_extent', moving.new_tensor(
            [.5*(n-1) for n in reversed(moving.shape)]), persistent=False)

    def prepare(self, *, temporal=True, compile_parameters=False):
        return self

    def at(self, time):
        return CrossfadeDerivatives(self, time)


@dataclass
class CrossfadeDerivatives:
    path: IntensityCrossfade
    time: float

    def derivatives(self, coordinates, scale, *, compile_derivatives=True):
        g0 = _sample(self.path.moving_gradient, coordinates)
        g1 = _sample(self.path.fixed_gradient, coordinates)
        # Convert native differences to normalized-coordinate derivatives, then
        # to the requested recovery grid. Capped recovery handles spacing ratios.
        gradient = ((1-self.time)*g0+self.time*g1)*self.path.native_half_extent*scale
        temporal = _sample(self.path.temporal, coordinates).flatten()
        return gradient, temporal
