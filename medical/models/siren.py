"""Paired spatial SIRENs with linear parameter interpolation."""
import copy
import math

import torch
from torch import nn

from medical.models.siren_functional import coordinate_scale, prepare_endpoints


@torch.no_grad()
def endpoint_reconstruction(pair, images, chunk=65536):
    """Exact whole-volume endpoint MSE/PSNR, streamed without saving images."""
    scores={'scope':'whole_volume','data_range':1.0}
    for name, model, image in zip(('moving','fixed'),(pair.source,pair.target),images):
        error=image.new_zeros((),dtype=torch.float64)
        for start in range(0,image.numel(),chunk):
            indices=torch.arange(start,min(start+chunk,image.numel()),device=image.device)
            coordinates=coordinates_from_indices(indices,image.shape).to(image.dtype)
            error+=(model(coordinates).flatten()-image.flatten()[indices]).square().sum(dtype=torch.float64)
        mse=float(error/image.numel())
        if not math.isfinite(mse): raise RuntimeError('Nonfinite endpoint reconstruction')
        scores[name]={'mse':mse,'psnr_db':-10*math.log10(mse) if mse>0 else None,
                      'perfect_reconstruction':mse==0}
    return scores






class SineLayer(nn.Module):
    def __init__(self, inputs, outputs, omega, first=False):
        super().__init__()
        self.linear = nn.Linear(inputs, outputs)
        self.omega = omega
        bound = 1 / inputs if first else math.sqrt(6 / inputs) / omega
        with torch.no_grad():
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)

    def forward(self, coordinates):
        values = self.linear(coordinates)
        return torch.sin(self.omega * values) if torch.is_grad_enabled() else values.mul_(self.omega).sin_()


class Siren(nn.Module):
    def __init__(self, dimensions=3, width=512, layers=4, omega=30., output_init_scale=30.):
        super().__init__()
        if (type(dimensions) is not int or dimensions not in (2, 3)
                or type(width) is not int or width < 1 or type(layers) is not int or layers < 1
                or type(omega) not in (int, float) or not math.isfinite(omega) or omega <= 0
                or type(output_init_scale) not in (int, float)
                or not math.isfinite(output_init_scale) or output_init_scale <= 0):
            raise ValueError('Expected positive SIREN dimensions, width, layers, omega and output scale')
        blocks = [SineLayer(dimensions, width, omega, first=True)]
        blocks += [SineLayer(width, width, omega) for _ in range(layers - 1)]
        output = nn.Linear(width, 1)
        with torch.no_grad():
            bound = math.sqrt(6 / width) / omega
            output.weight.uniform_(-bound, bound)
            if dimensions == 3:
                output.bias.zero_()
            output.weight.mul_(output_init_scale)
            output.bias.mul_(output_init_scale)
        self.net = nn.Sequential(*blocks, output)

    def forward(self, coordinates):
        return self.net(coordinates)


def coordinates_from_indices(indices, shape):
    """Array-index order -> normalized sampler coordinates (reversed axes)."""
    if len(shape) not in (2, 3) or any(type(n) is not int or n < 2 for n in shape):
        raise ValueError('Image axes must have integer sizes of at least two')
    stride, components = 1, []
    for size in reversed(shape):
        components.append((torch.div(indices, stride, rounding_mode='floor') % size) / (size - 1))
        stride *= size
    return 2 * torch.stack(components, -1) - 1


class PairedSiren(nn.Module):
    def __init__(self, dimensions=3, **model):
        super().__init__()
        self.source = Siren(dimensions, **model)
        self.target = copy.deepcopy(self.source)
        self.dimensions, self.model_config = dimensions, model

    def prepare(self, *, temporal=True, compile_parameters=False):
        return prepare_endpoints(self.source, self.target, temporal=temporal,
                                 compile_parameters=compile_parameters)

    def image(self, coordinates, time):
        return self.prepare(temporal=False).at(time).image(coordinates)

    def derivatives(self, coordinates, time, shape, *, differential=False, compile_derivatives=True):
        return self.prepare(compile_parameters=compile_derivatives).at(time).derivatives(
            coordinates, coordinate_scale(shape, coordinates),
            differential=differential, compile_derivatives=compile_derivatives)

    @torch.no_grad()
    def decode(self, shape, time, chunk=65536):
        parameter = next(self.parameters())
        device = parameter.device
        if type(chunk) is not int or chunk < 0:
            raise ValueError('Decode chunk must be a nonnegative integer')
        coordinate_scale(shape, parameter)
        prepared = self.prepare(temporal=False).at(time)
        result = torch.empty(math.prod(shape), device='cpu')
        step = chunk or result.numel()
        for start in range(0, result.numel(), step):
            end = min(start + step, result.numel())
            coordinates = coordinates_from_indices(torch.arange(start, end, device=device), shape)
            result[start:end] = prepared.image(coordinates.to(parameter.dtype)).flatten().cpu()
        return result.reshape(shape)
