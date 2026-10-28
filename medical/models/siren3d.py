"""Paired spatial SIRENs and differentiable weight interpolation in 3D."""
import math

import torch
from torch import nn


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
        if not torch.is_grad_enabled():
            # Full-volume activations can each be several GiB. In inference,
            # scaling and sine may reuse this newly allocated linear output.
            return values.mul_(self.omega).sin_()
        return torch.sin(self.omega * values)


class Siren3D(nn.Module):
    def __init__(self, width=128, layers=3, omega=30.):
        super().__init__()
        blocks = [SineLayer(3, width, omega, first=True)]
        blocks += [SineLayer(width, width, omega) for _ in range(layers - 1)]
        output = nn.Linear(width, 1)
        with torch.no_grad():
            bound = math.sqrt(6 / width) / omega
            output.weight.uniform_(-bound, bound)
            output.bias.zero_()
        self.net = nn.Sequential(*blocks, output)

    def forward(self, coordinates):
        return self.net(coordinates)


def interpolate(source, target, coordinates, time):
    """Gradients reach both endpoint networks; time selects model weights."""
    target_parameters = dict(target.named_parameters())
    parameters = {name: torch.lerp(value, target_parameters[name], time)
                  for name, value in source.named_parameters()}
    return torch.func.functional_call(source, parameters, (coordinates,))


def coordinates_from_indices(indices, shape):
    """NIfTI array axes (D,H,W) -> normalized sampler coordinates (x,y,z)."""
    d, h, w = shape
    z = torch.div(indices, h * w, rounding_mode="floor")
    y = torch.div(indices, w, rounding_mode="floor") % h
    x = indices % w
    return 2 * torch.stack((x / (w - 1), y / (h - 1), z / (d - 1)), -1) - 1


@torch.no_grad()
def decode(source, target, shape, time, chunk_size=0):
    """One spatial pass by default; an explicit chunk size bounds memory."""
    device = next(source.parameters()).device
    result = torch.empty(math.prod(shape), dtype=torch.float32, device="cpu")
    chunk_size = chunk_size or result.numel()
    for start in range(0, result.numel(), chunk_size):
        end = min(start + chunk_size, result.numel())
        indices = torch.arange(start, end, device=device)
        coordinates = coordinates_from_indices(indices, shape)
        result[start:end] = interpolate(source, target, coordinates, time).flatten().cpu()
    return result.reshape(shape).numpy()
