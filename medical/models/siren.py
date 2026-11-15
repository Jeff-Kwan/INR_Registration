"""Paired spatial SIRENs with linear parameter interpolation."""
import copy
import math

import torch
from torch import nn
from torch.nn.utils import parametrize

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


@torch.no_grad()
def endpoint_rescaled_reconstruction(pair, images, chunk=65536):
    """Supplemental PSNR after matching prediction extrema to target extrema.

    Two streaming passes, no images saved. Constant predictions are explicitly
    undefined rather than assigned an arbitrary intensity calibration.
    """
    scores = {'scope': 'whole_volume', 'normalization': 'prediction_minmax_to_target_minmax'}
    for name, model, image in zip(('moving', 'fixed'), (pair.source, pair.target), images):
        low, high = image.new_tensor(float('inf')), image.new_tensor(-float('inf'))
        def predictions(start):
            indices = torch.arange(start, min(start+chunk, image.numel()), device=image.device)
            coordinates = coordinates_from_indices(indices, image.shape).to(image.dtype)
            return indices, model(coordinates).flatten()
        for start in range(0, image.numel(), chunk):
            _, values = predictions(start)
            low, high = torch.minimum(low, values.min()), torch.maximum(high, values.max())
        target_low, target_high = image.min(), image.max()
        bounds = [float(v) for v in (low, high, target_low, target_high)]
        if not all(math.isfinite(v) for v in bounds):
            raise RuntimeError('Nonfinite reconstruction extrema')
        result = dict(prediction_min=bounds[0], prediction_max=bounds[1],
                      target_min=bounds[2], target_max=bounds[3])
        if high <= low or target_high <= target_low:
            result.update(mse=None, psnr_db=None, status='degenerate_intensity_range')
        else:
            error = image.new_zeros((), dtype=torch.float64)
            for start in range(0, image.numel(), chunk):
                indices, values = predictions(start)
                scaled = target_low+(values-low)/(high-low)*(target_high-target_low)
                error += (scaled-image.flatten()[indices]).square().sum(dtype=torch.float64)
            mse = float(error/image.numel())
            result.update(mse=mse, psnr_db=10*math.log10(float(target_high-target_low)**2/mse)
                          if mse else None, perfect_reconstruction=mse==0, status='evaluated')
        scores[name] = result
    return scores


class FixedRowScale(nn.Module):
    """Fixed frequency parameterization with live gradients to raw parameters."""
    def __init__(self, scales):
        super().__init__()
        self.register_buffer('scales', scales)

    def forward(self, value):
        return value * self.scales


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
    def __init__(self, dimensions=3, width=256, layers=4, omega=30., init_scale=1.,
                 first_omega_range=None, layer_init_scales=None, hidden_omega=None,
                 balance_first_frequencies=False, first_omega_mode='initialization',
                 output_init_scale=1.):
        super().__init__()
        if (type(dimensions) is not int or dimensions not in (2, 3)
                or type(width) is not int or width < 1 or type(layers) is not int or layers < 1
                or type(omega) not in (int, float) or not math.isfinite(omega) or omega <= 0):
            raise ValueError('Expected 2D/3D SIREN with positive width, layers and omega')
        hidden_omega = omega if hidden_omega is None else hidden_omega
        if (type(hidden_omega) not in (int, float) or not math.isfinite(hidden_omega)
                or hidden_omega <= 0):
            raise ValueError('Hidden omega must be finite and positive')
        blocks = [SineLayer(dimensions, width, omega, first=True)]
        blocks += [SineLayer(width, width, hidden_omega) for _ in range(layers - 1)]
        if (type(init_scale) not in (int, float) or not math.isfinite(init_scale)
                or init_scale <= 0):
            raise ValueError('SIREN initialization scale must be finite and positive')
        # Yeom et al. (ICLR 2025): scale initial non-output weight matrices.
        with torch.no_grad():
            for block in blocks:
                block.linear.weight.mul_(init_scale)
        if layer_init_scales is not None:
            if (not isinstance(layer_init_scales, (list, tuple)) or len(layer_init_scales) != layers
                    or any(type(value) not in (int, float) or not math.isfinite(value)
                           or value <= 0 for value in layer_init_scales)):
                raise ValueError('Layer initialization scales require one positive finite value per sine layer')
            with torch.no_grad():
                for block, scale in zip(blocks, layer_init_scales):
                    block.linear.weight.mul_(scale)
        if first_omega_range is not None:
            if (not isinstance(first_omega_range, (list, tuple)) or len(first_omega_range) != 2
                    or any(type(value) not in (int, float) or not math.isfinite(value)
                           or value <= 0 for value in first_omega_range)
                    or first_omega_range[0] > first_omega_range[1]):
                raise ValueError('First-layer omega range must contain two ordered positive finite values')
            # Encode the frequency ladder in the initial parameters. Subsequent
            # training and analytical derivatives use the existing scalar omega.
            first = blocks[0].linear
            with torch.no_grad():
                scales = torch.linspace(*first_omega_range, width,
                                        device=first.weight.device, dtype=first.weight.dtype) / omega
            if first_omega_mode == 'parameterized':
                # Fixed row factors give explicit per-neuron frequency update
                # scales. Effective weights still interpolate linearly in time.
                parametrize.register_parametrization(first, 'weight', FixedRowScale(scales[:, None]))
                parametrize.register_parametrization(first, 'bias', FixedRowScale(scales))
            else:
                with torch.no_grad():
                    first.weight.mul_(scales[:, None])
                    first.bias.mul_(scales)
        if (first_omega_mode not in ('initialization', 'parameterized')
                or (first_omega_mode == 'parameterized' and first_omega_range is None)):
            raise ValueError('First omega mode must be initialization or parameterized with a range')
        output = nn.Linear(width, 1)
        if (type(output_init_scale) not in (int, float) or not math.isfinite(output_init_scale)
                or output_init_scale <= 0):
            raise ValueError('Output initialization scale must be finite and positive')
        with torch.no_grad():
            output.weight.uniform_(-math.sqrt(6 / width) / omega, math.sqrt(6 / width) / omega)
            if dimensions == 3:
                output.bias.zero_()
            output.weight.mul_(output_init_scale)
            output.bias.mul_(output_init_scale)
        self.net = nn.Sequential(*blocks, output)
        if type(balance_first_frequencies) is not bool:
            raise ValueError('First-layer frequency balancing must be boolean')
        if balance_first_frequencies:
            if first_omega_range is None:
                raise ValueError('Frequency balancing requires a first-layer omega range')
            # Unit-RMS outgoing gains preserve expected affine variance while
            # inverse-frequency weighting balances initial derivative amplitudes.
            outgoing = blocks[1].linear if layers > 1 else output
            with torch.no_grad():
                frequencies = torch.linspace(*first_omega_range, width,
                                             device=outgoing.weight.device, dtype=outgoing.weight.dtype)
                gains = frequencies.reciprocal()
                gains = gains / gains.square().mean().sqrt()
                outgoing.weight.mul_(gains[None, :])

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
