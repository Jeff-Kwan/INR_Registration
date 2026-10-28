"""Exact-derivative local Lucas–Kanade temporal penalties.

Spatial dimensions are independent of time: Cameraman uses D=2; a volume
uses D=3. Coordinates passed to the INR are normalized to [-1, 1].
"""
import math
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.func import functional_call, jvp


def exact_image_derivatives(source, target, coordinates, time, spatial_scale):
    """Return g=dI/d(pixel), q=dI/dt without summing q across coordinates.

    JVP is through the interpolated parameters in direction theta_F-theta_M.
    Both its primal and tangent retain reverse-mode graphs to both endpoints.
    Summing intensities for spatial autograd is valid for pointwise INRs only.
    spatial_scale converts normalized input derivatives to desired spatial units.
    """
    coordinates = coordinates.detach().requires_grad_(True)
    first, second = dict(source.named_parameters()), dict(target.named_parameters())
    parameters = {key: torch.lerp(value, second[key], time) for key, value in first.items()}
    direction = {key: second[key] - value for key, value in first.items()}
    shape = coordinates.shape[:-1]

    def evaluate(state):
        return functional_call(source, state, (coordinates.reshape(-1, coordinates.shape[-1]),)).reshape(shape)

    intensity, q = jvp(evaluate, (parameters,), (direction,))
    g, = torch.autograd.grad(intensity.sum(), coordinates, create_graph=True)
    scale = torch.as_tensor(spatial_scale, device=coordinates.device, dtype=coordinates.dtype)
    return g * scale, q


def analytical_siren_derivatives(source, target, coordinates, time, spatial_scale,
                                 temporal=True, differential=False):
    """Explicit sine chain rule, with an ordinary parameter-autograd graph.

    Supports the 2D Siren and LPBA Siren3D (sine hidden layers, scalar linear
    output). No autograd.grad/JVP is used to construct image derivatives.
    The reverse chain computes g; the same adjoints contracted with each
    layer's parameter displacement give per-coordinate q. temporal=False
    skips q for spatial-only benchmarks. Endpoint parameters are never detached.
    differential=True also returns H and d(g)/dt via the analytical sine chain
    rule. Their parameter derivatives retain higher-order autograd support.
    """
    if differential and not temporal:
        raise ValueError('Differential LK requires temporal derivatives')
    for model in (source, target):
        if (not hasattr(model, 'net') or not isinstance(model.net[-1], torch.nn.Linear)
                or model.net[-1].out_features != 1
                or any(not hasattr(layer, 'linear') or not hasattr(layer, 'omega')
                       or getattr(layer, 'activation', torch.sin) is not torch.sin
                       for layer in model.net[:-1])):
            raise ValueError('Analytical derivatives require sine hidden layers and scalar linear output')
    first, second = dict(source.named_parameters()), dict(target.named_parameters())
    if first.keys() != second.keys() or any(first[k].shape != second[k].shape for k in first):
        raise ValueError('Endpoint SIREN architectures must match')
    parameters = {key: torch.lerp(value, second[key], time) for key, value in first.items()}
    direction = {key: second[key] - value for key, value in first.items()} if temporal else {}
    shape = coordinates.shape[:-1]
    h = coordinates.detach().reshape(-1, coordinates.shape[-1])
    scale = torch.as_tensor(spatial_scale, device=coordinates.device, dtype=coordinates.dtype)
    if differential:
        # Forward spatial Jacobians are in physical pixel/voxel units.
        hx = torch.diag(scale).expand(h.shape[0], -1, -1)
    states = []
    for index, layer in enumerate(source.net[:-1]):
        prefix = f'net.{index}.linear.'
        weight = parameters[prefix + 'weight']
        phase = layer.omega * F.linear(h, weight, parameters[prefix + 'bias'])
        sine, factor = phase.sin(), layer.omega * phase.cos()
        if differential:
            zx = F.linear(hx.transpose(-1, -2), weight).transpose(-1, -2)
            factor_x = -(layer.omega ** 2) * sine[..., None] * zx
            states.append((h, factor, weight, prefix, hx, factor_x))
            hx = factor[..., None] * zx
        else:
            states.append((h, factor, weight, prefix, None, None))
        h = sine
    prefix = f'net.{len(source.net) - 1}.'
    adjoint = parameters[prefix + 'weight'].expand(h.shape[0], -1)
    q = F.linear(h, direction[prefix + 'weight'], direction[prefix + 'bias']).squeeze(-1) if temporal else None
    if differential:
        adjoint_x = torch.zeros_like(hx)
        mixed = F.linear(hx.transpose(-1, -2), direction[prefix + 'weight']).squeeze(-1)
    for h, factor, weight, prefix, hx, factor_x in reversed(states):
        if differential:
            adjoint_x = adjoint_x * factor[..., None] + adjoint[..., None] * factor_x
        adjoint = adjoint * factor
        if temporal:
            local_tangent = F.linear(h, direction[prefix + 'weight'], direction[prefix + 'bias'])
            q = q + (adjoint * local_tangent).sum(-1)
            if differential:
                tangent_x = F.linear(hx.transpose(-1, -2), direction[prefix + 'weight']).transpose(-1, -2)
                mixed = mixed + (adjoint_x * local_tangent[..., None]
                                  + adjoint[..., None] * tangent_x).sum(-2)
        adjoint = adjoint @ weight
        if differential:
            adjoint_x = F.linear(adjoint_x.transpose(-1, -2), weight.T).transpose(-1, -2)
    g = adjoint.reshape(coordinates.shape) * scale
    if differential:
        hessian = adjoint_x * scale[:, None]
        return (g, q.reshape(shape), hessian.reshape(*shape, len(scale), len(scale)),
                mixed.reshape(coordinates.shape))
    return g, q.reshape(shape) if temporal else None


def minimized_transport_energy(g, q, ridge, differentiate_solve=True):
    """Batched normal equations and explicit residual + ridge, with full autograd.

    g: [P,N,D], q: [P,N]. Returns (mean minimized energy, [P,D] velocity).
    The ridge uses velocity in the spatial units of g, per unit normalized time.
    """
    if (g.ndim != 3 or g.shape[-1] not in (2, 3) or q.shape != g.shape[:-1]
            or min(g.shape[:2]) < 1):
        raise ValueError('Expected nonempty g[P,N,2 or 3] and q[P,N]')
    if not math.isfinite(ridge) or ridge <= 0:
        raise ValueError('LK ridge must be finite and strictly positive')
    # Solve in at least float32, preserving float64 for numerical checks.
    dtype = torch.float64 if g.dtype == torch.float64 else torch.float32
    g, q = g.to(dtype), q.to(dtype)
    with nullcontext() if differentiate_solve else torch.no_grad():
        matrix = g.transpose(-1, -2) @ g / g.shape[1]
        matrix = matrix + ridge * torch.eye(g.shape[-1], device=g.device, dtype=dtype)
        rhs = -(g * q[..., None]).mean(1)
        velocity = torch.linalg.solve(matrix, rhs[..., None]).squeeze(-1)
    residual = q + (g * velocity[:, None]).sum(-1)
    energy = residual.square().mean(1) + ridge * velocity.square().sum(-1)
    return energy.mean(), velocity


def minimized_differential_transport_energy(g, q, hessian, mixed, ridge, alpha, differentiate_solve=True):
    """Minimize brightness + alpha*gradient transport + ridge per patch.

    differentiate_solve=False holds the optimum fixed in backward (envelope
    theorem); derivative tensors still retain their endpoint parameter graphs.
    The default preserves the established through-solve baseline.

    H[...,k,j]=d(g_k)/dx_j. Alpha has squared-spatial-unit dimensions; there
    is no division by D, N*D, or a time gap in the gradient-constancy term.
    """
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError('LK gradient weight must be finite and nonnegative')
    if alpha == 0:
        return minimized_transport_energy(g, q, ridge, differentiate_solve=differentiate_solve)
    if (g.ndim != 3 or g.shape[-1] not in (2, 3) or q.shape != g.shape[:-1]
            or min(g.shape[:2]) < 1 or hessian.shape != (*g.shape, g.shape[-1])
            or mixed.shape != g.shape):
        raise ValueError('Expected g[P,N,D], q[P,N], H[P,N,D,D], mixed[P,N,D]')
    if not math.isfinite(ridge) or ridge <= 0:
        raise ValueError('LK ridge must be finite and positive')
    dtype = torch.float64 if g.dtype == torch.float64 else torch.float32
    g, q, hessian, mixed = [value.to(dtype) for value in (g, q, hessian, mixed)]
    n, d = g.shape[1:]
    with nullcontext() if differentiate_solve else torch.no_grad():
        matrix = (g.transpose(-1, -2) @ g
                  + alpha * torch.einsum('pnki,pnkj->pij', hessian, hessian)) / n
        matrix = matrix + ridge * torch.eye(d, device=g.device, dtype=dtype)
        rhs = -((g * q[..., None]).sum(1)
                + alpha * torch.einsum('pnki,pnk->pi', hessian, mixed)) / n
        velocity = torch.linalg.solve(matrix, rhs[..., None]).squeeze(-1)
    residual = q + (g * velocity[:, None]).sum(-1)
    gradient_residual = mixed + (hessian @ velocity[:, None, :, None]).squeeze(-1)
    energy = (residual.square().mean(1)
              + alpha * gradient_residual.square().sum(-1).mean(1)
              + ridge * velocity.square().sum(-1))
    return energy.mean(), velocity


def lk_temporal_loss(source, target, coordinates, time, shape, ridge=1e-6,
                     derivatives='analytical', gradient_weight=0., differentiate_solve=False):
    """Local constant velocity in pixels/voxels per t∈[0,1]; no finite differences."""
    if coordinates.shape[-1] != len(shape):
        raise ValueError('Spatial dimensions of coordinates and grid must match')
    if derivatives not in ('analytical', 'autograd'):
        raise ValueError('LK derivatives must be analytical or autograd')
    if not math.isfinite(gradient_weight) or gradient_weight < 0:
        raise ValueError('LK gradient weight must be finite and nonnegative')
    scale = [2 / (n - 1) for n in reversed(shape)]
    if gradient_weight:
        if derivatives != 'analytical':
            raise ValueError('Differential LK uses analytical SIREN derivatives')
        g, q, hessian, mixed = analytical_siren_derivatives(
            source, target, coordinates, time, scale, differential=True)
        return minimized_differential_transport_energy(
            g, q, hessian, mixed, ridge, gradient_weight, differentiate_solve=differentiate_solve)[0]
    function = analytical_siren_derivatives if derivatives == 'analytical' else exact_image_derivatives
    g, q = function(source, target, coordinates, time, scale)
    return minimized_transport_energy(g, q, ridge, differentiate_solve=differentiate_solve)[0]


def regularization_weight(settings,step,steps):
    """Optional rise from zero to configured weight across optimizer minibatches."""
    if settings.get('weight_schedule','constant') == 'constant':
        return settings['weight']
    if settings['weight_schedule'] not in ('cosine','linear'):
        raise ValueError('Unknown LK weight schedule')
    fraction=(step-1)/max(steps-1,1) if steps>1 else 1.
    factor=fraction if settings['weight_schedule']=='linear' else .5*(1-math.cos(math.pi*fraction))
    return settings['weight']*factor
