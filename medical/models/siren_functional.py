"""Tensor-only SIRENs and short-lived differentiable parameter bundles.

Never cache these bundles on a module or across optimizer updates.
"""
from dataclasses import dataclass
from functools import lru_cache
import math

import torch
import torch.nn.functional as F


def siren_parameters(model):
    """Validate a scalar sine network and read parameters without name dicts."""
    net = getattr(model, 'net', ())
    if len(net) < 2 or not isinstance(net[-1], torch.nn.Linear) or net[-1].out_features != 1:
        raise ValueError('Expected sine hidden layers and a scalar linear output')
    parameters, omegas = [], []
    inputs = None
    for layer in net[:-1]:
        linear, omega = getattr(layer, 'linear', None), getattr(layer, 'omega', None)
        if (not isinstance(linear, torch.nn.Linear)
                or getattr(layer, 'activation', torch.sin) is not torch.sin
                or type(omega) not in (int, float) or not math.isfinite(omega) or omega <= 0):
            raise ValueError('SIREN hidden layers require linear maps and finite positive omega')
        if inputs is not None and linear.in_features != inputs:
            raise ValueError('SIREN layer dimensions must agree')
        inputs = linear.out_features
        parameters.append((linear.weight, linear.bias))
        omegas.append(float(omega))
    parameters.append((net[-1].weight, net[-1].bias))
    if parameters[0][0].shape[1] not in (2, 3) or net[-1].in_features != inputs:
        raise ValueError('Expected a 2D/3D scalar SIREN')
    reference = parameters[0][0]
    if any(bias is None for _, bias in parameters):
        raise ValueError('SIREN linear layers require biases')
    if any(not value.is_floating_point() or value.device != reference.device
           or value.dtype != reference.dtype for layer in parameters for value in layer):
        raise ValueError('SIREN parameters must share a floating dtype and device')
    return tuple(parameters), tuple(omegas)


def coordinate_scale(shape, reference):
    if len(shape) not in (2, 3) or any(type(n) is not int or n < 2 for n in shape):
        raise ValueError('Image axes must have integer sizes of at least two')
    return reference.new_tensor([2 / (n - 1) for n in reversed(shape)])


def validate_coordinates(coordinates, parameters):
    reference = parameters[0][0]
    if (coordinates.ndim < 1 or coordinates.shape[-1] != reference.shape[1]
            or coordinates.device != reference.device or coordinates.dtype != reference.dtype):
        raise ValueError('Coordinates must match the SIREN dimension, dtype and device')


def siren_forward(coordinates, parameters, omegas, *, fuse_scalar_affines=False):
    """Functional forward with ordinary autograd and no module rebinding."""
    values = coordinates
    for index, ((weight, bias), omega) in enumerate(zip(parameters[:-1], omegas)):
        if index == 0 and fuse_scalar_affines:
            # Expose the three-coordinate affine to the compiler for fusion with sine.
            values = torch.addcmul(
                torch.addcmul(values[..., :1] * weight[:, 0], values[..., 1:2], weight[:, 1]),
                values[..., 2:3], weight[:, 2],
            ) + bias
        else:
            values = F.linear(values, weight, bias)
        values = (torch.sin(omega * values) if torch.is_grad_enabled()
                  else values.mul_(omega).sin_())
    if fuse_scalar_affines:
        weight, bias = parameters[-1]
        return (values * weight).sum(-1, keepdim=True) + bias
    return F.linear(values, *parameters[-1])


@dataclass(frozen=True)
class PreparedSiren:
    parameters: tuple
    direction: tuple
    omegas: tuple

    def image(self, coordinates):
        validate_coordinates(coordinates, self.parameters)
        return siren_forward(coordinates, self.parameters, self.omegas)

    def derivatives(self, coordinates, scale, *, differential=False, compile_derivatives=True):
        from medical.registration.derivatives import evaluate_derivatives
        return evaluate_derivatives(self.parameters, self.direction, self.omegas,
            coordinates, scale, differential=differential,
            compile_derivatives=compile_derivatives)


@dataclass(frozen=True)
class SirenEndpoints:
    source: tuple
    target: tuple
    direction: tuple
    omegas: tuple
    compile_parameters: bool = False

    def at(self, time):
        reference = self.source[0][0]
        if isinstance(time, torch.Tensor):
            if time.ndim != 0 or time.device != reference.device or time.dtype != reference.dtype:
                raise ValueError('Time must be a scalar tensor matching the SIREN dtype/device')
        else:
            if type(time) not in (int, float) or not math.isfinite(time):
                raise ValueError('Time must be finite and scalar')
            time = reference.new_tensor(time)
        function = _compiled_parameters(True) if self.compile_parameters else _interpolate
        parameters = function(self.source, self.target, time)
        return PreparedSiren(parameters, self.direction, self.omegas)


def _interpolate(source, target, time):
    return tuple(tuple(torch.lerp(a, b, time) for a, b in zip(first, second))
                 for first, second in zip(source, target))


def _parameter_direction(source, target):
    return tuple(tuple(b - a for a, b in zip(first, second))
                 for first, second in zip(source, target))


@lru_cache(maxsize=2)
def _compiled_parameters(interpolate=False):
    return torch.compile(_interpolate if interpolate else _parameter_direction, fullgraph=True)


def prepare_endpoints(source, target, *, temporal=True, compile_parameters=False):
    first, omegas = siren_parameters(source)
    second, target_omegas = siren_parameters(target)
    if (omegas != target_omegas or len(first) != len(second)
            or any(a.shape != b.shape or a.dtype != b.dtype or a.device != b.device
                   for left, right in zip(first, second) for a, b in zip(left, right))):
        raise ValueError('Endpoint SIRENs must match in architecture, omega, dtype and device')
    compile_parameters = compile_parameters and first[0][0].is_cuda
    function = _compiled_parameters() if compile_parameters else _parameter_direction
    direction = function(first, second) if temporal else ()
    return SirenEndpoints(first, second, direction, omegas, compile_parameters)
