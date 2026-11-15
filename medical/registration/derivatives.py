"""Exact sine-chain derivatives with live parameter, coordinate and time gradients."""
from functools import lru_cache
import math

import torch
import torch.nn.functional as F

from medical.models.siren_functional import validate_coordinates


def evaluate_derivatives(parameters, direction, omegas, coordinates, scale, *,
                         differential=False, compile_derivatives=True):
    """Validate metadata outside the tensor-only compiled forward/backward."""
    validate_coordinates(coordinates, parameters)
    if not direction:
        raise ValueError('Derivative evaluation requires a prepared parameter direction')
    if not isinstance(scale, torch.Tensor):
        if len(scale) != coordinates.shape[-1] or any(
                not math.isfinite(value) or value <= 0 for value in scale):
            raise ValueError('Spatial scale must be finite and positive for each coordinate axis')
        scale = coordinates.new_tensor(scale)
    if (scale.shape != (coordinates.shape[-1],) or scale.device != coordinates.device
            or scale.dtype != coordinates.dtype):
        raise ValueError('Spatial scale must match the coordinate dimension, dtype and device')
    function = _differential_chain if differential else _basic_chain
    if compile_derivatives and coordinates.is_cuda:
        function = _compiled_analytical(differential)
    return function(parameters, direction, omegas, coordinates, scale)


def _basic_chain(parameters, direction, omegas, coordinates, scale):
    """Scalar-output reverse chain for g and parameter-direction contraction q."""
    h = coordinates.reshape(-1, coordinates.shape[-1])
    states = []
    for (weight, bias), omega in zip(parameters[:-1], omegas):
        phase = omega * F.linear(h, weight, bias)
        factor = omega * phase.cos()
        states.append((h, factor, weight))
        h = phase.sin()
    adjoint = parameters[-1][0]
    q = F.linear(h, *direction[-1]).squeeze(-1)
    for index in range(len(states) - 1, -1, -1):
        h, factor, weight = states[index]
        adjoint = adjoint * factor
        q = q + (adjoint * F.linear(h, *direction[index])).sum(-1)
        adjoint = adjoint @ weight
    g = adjoint.reshape(coordinates.shape) * scale
    return g, q.reshape(coordinates.shape[:-1])


def _layer_tangents(index, h, scale, state, weight, bias, delta_weight, delta_bias, omega):
    dimensions, points = h.shape[-1], h.shape[0]
    if index == 0:
        phase = omega * F.linear(h, weight, bias)
        spatial = (weight * scale).T[:, None, :].expand(-1, points, -1)
        temporal = F.linear(h, delta_weight, delta_bias)
        tangents = torch.cat((spatial, temporal[None]))
    else:
        linear = F.linear(state, weight)
        phase = omega * (linear[0] + bias)
        temporal = F.linear(state[0], delta_weight, delta_bias)
        tangents = torch.cat((linear[1:1 + dimensions], (linear[-1] + temporal)[None]))
    sine, factor = phase.sin(), omega * phase.cos()
    return sine, factor, tangents


def _differential_chain(parameters, direction, omegas, coordinates, scale):
    """Packed spatial/time tangents and reverse adjoints for g, q, H and m.

    The first affine map has known coordinate tangents and zero time tangent;
    avoid multiplying a packed identity/zero input through it. The scalar output
    similarly has zero spatial adjoint tangents. All simplifications are exact
    and retain ordinary autograd through every endpoint parameter.
    """
    dimensions = coordinates.shape[-1]
    h = coordinates.reshape(-1, dimensions)
    state = None
    history = []
    for index, ((weight, bias), (delta_weight, delta_bias), omega) in enumerate(
            zip(parameters[:-1], direction[:-1], omegas)):
        sine, factor, tangents = _layer_tangents(
            index, h, scale, state, weight, bias, delta_weight, delta_bias, omega)
        factor_tangents = -(omega ** 2) * sine[None] * tangents
        history.append((factor, factor_tangents, weight, delta_weight))
        if index + 1 < len(omegas):
            state = torch.cat((sine[None], factor[None] * tangents))
        else:
            # Spatial state lanes after the final hidden layer are never read.
            final_time = factor * tangents[-1]
    output_weight = parameters[-1][0]
    output_delta_weight, output_delta_bias = direction[-1]
    q = (F.linear(final_time, output_weight).squeeze(-1)
         + F.linear(sine, output_delta_weight, output_delta_bias).squeeze(-1))
    adjoint = output_weight
    adjoint_tangents = None
    for factor, factor_tangents, weight, delta_weight in reversed(history):
        value_adjoint = adjoint * factor
        if adjoint_tangents is None:
            tangent_adjoint = torch.cat((adjoint[None] * factor_tangents[:dimensions],
                (output_delta_weight * factor + adjoint * factor_tangents[-1])[None]))
        else:
            tangent_adjoint = adjoint_tangents * factor[None] + adjoint[None] * factor_tangents
        propagated = F.linear(torch.cat((value_adjoint[None], tangent_adjoint)), weight.T)
        adjoint = propagated[0]
        adjoint_tangents = torch.cat((propagated[1:1 + dimensions],
            (propagated[-1] + F.linear(value_adjoint, delta_weight.T))[None]))
    g = adjoint.reshape(coordinates.shape) * scale
    hessian = adjoint_tangents[:dimensions].permute(1, 2, 0) * scale[:, None]
    mixed = adjoint_tangents[-1] * scale
    return (g, q.reshape(coordinates.shape[:-1]),
            hessian.reshape(*coordinates.shape, dimensions), mixed.reshape(coordinates.shape))


def _layer_curvature_chain(parameters, direction, omegas, coordinates, scale):
    """Exact scalar-output Hessian and mixed derivative with one reverse adjoint.

    For each sine layer, J carries preactivation derivatives in physical or voxel
    coordinates and time. Its curvature contribution is J_x.T diag(a f'') J,
    where a is the output adjoint of that layer. A time-varying affine weight
    also contributes its weight direction contracted with the input's spatial
    tangent. Both contributions retain autograd through all network parameters.
    """
    dimensions = coordinates.shape[-1]
    h = coordinates.reshape(-1, dimensions)
    state = None
    history = []
    for index, ((weight, bias), (delta_weight, delta_bias), omega) in enumerate(
            zip(parameters[:-1], direction[:-1], omegas)):
        sine, factor, tangents = _layer_tangents(
            index, h, scale, state, weight, bias, delta_weight, delta_bias, omega)
        state = torch.cat((sine[None], factor[None] * tangents))
        history.append((factor, -(omega ** 2) * sine, tangents, weight, delta_weight))

    output_weight = parameters[-1][0]
    output_delta_weight, output_delta_bias = direction[-1]
    q = (F.linear(state[-1], output_weight).squeeze(-1)
         + F.linear(sine, output_delta_weight, output_delta_bias).squeeze(-1))
    adjoint = output_weight
    affine_mixed = (state[1:1 + dimensions] * output_delta_weight).sum(-1).T
    curvature = 0
    for index in reversed(range(len(history))):
        factor, second, tangents, weight, delta_weight = history[index]
        weighted_spatial = tangents[:dimensions] * (adjoint * second)[None]
        # Spatial curvature is symmetric. Each row also carries one space-time term.
        dot = [[(weighted_spatial[i] * tangents[j]).sum(-1)
                for j in range(i, dimensions + 1)] for i in range(dimensions)]
        rows = [torch.stack([dot[min(i, j)][abs(i - j)]
                             for j in range(dimensions + 1)], dim=-1)
                for i in range(dimensions)]
        curvature = curvature + torch.stack(rows, dim=-2)

        value_adjoint = adjoint * factor
        weight_direction_adjoint = value_adjoint @ delta_weight
        if index:
            previous_factor, _, previous_tangents, _, _ = history[index - 1]
            affine_mixed = affine_mixed + (
                (previous_factor * weight_direction_adjoint)[None] *
                previous_tangents[:dimensions]).sum(-1).T
        else:
            affine_mixed = affine_mixed + weight_direction_adjoint * scale
        adjoint = value_adjoint @ weight

    g = (adjoint * scale).reshape(coordinates.shape)
    hessian = curvature[:, :, :dimensions].reshape(*coordinates.shape, dimensions)
    mixed = (curvature[:, :, -1] + affine_mixed).reshape(coordinates.shape)
    return g, q.reshape(coordinates.shape[:-1]), hessian, mixed


@lru_cache(maxsize=2)
def _compiled_analytical(differential=False):
    function = _differential_chain if differential else _basic_chain
    return torch.compile(function, fullgraph=True)
