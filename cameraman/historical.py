"""Small reference objectives from earlier experiments; no training dispatch.

These records document ideas, not supported experiment configurations.
Patch LK uses core.lk_loss.minimized_transport_energy on grouped derivatives.
"""
import math
from contextlib import nullcontext

import torch
import torch.nn.functional as F


def shifted_patch_h1(frames, patch_size=11, radius=1, gradient_weight=1.):
    """Bidirectional central-patch H1 minimum over integer shifts, with full halo."""
    if type(patch_size) is not int or patch_size < 1 or patch_size % 2 != 1:
        raise ValueError('patch_size must be a positive odd integer')
    if type(radius) is not int or radius < 0 or not math.isfinite(gradient_weight) or gradient_weight < 0:
        raise ValueError('Invalid radius or gradient weight')
    halo = radius + int(gradient_weight > 0)
    size = patch_size + 2 * halo
    if frames.ndim != 4 or frames.shape[0] < 2 or tuple(frames.shape[1:]) != (1, size, size):
        raise ValueError('Expected [T,1,size,size] with full shift and gradient halo')
    features = frames
    if gradient_weight:
        gx = F.pad(.5 * (frames[..., 2:] - frames[..., :-2]), (1, 1, 0, 0))
        gy = F.pad(.5 * (frames[..., 2:, :] - frames[..., :-2, :]), (0, 0, 1, 1))
        features = torch.cat((frames, gx, gy), 1)
    first = torch.cat((features[:-1], features[1:]))
    second = torch.cat((features[1:], features[:-1]))
    central = first[..., halo:halo+patch_size, halo:halo+patch_size]
    costs = []
    for dy in range(-radius, radius+1):
        for dx in range(-radius, radius+1):
            shifted = second[..., halo+dy:halo+dy+patch_size, halo+dx:halo+dx+patch_size]
            residual = (central-shifted).square().mean((-1, -2))
            costs.append(residual[:, 0] + gradient_weight * residual[:, 1:].sum(1))
    return torch.stack(costs).min(0).values.mean()


def motion_explainability_loss(frames, window=11, ridge=1e-4):
    """Minimize local mean (Ix*u+Iy*v+It)^2 + ridge*(u^2+v^2).

    Uses frame differences (no division by dt), so u,v are pixels per pair.
    The closed-form minimum includes the ridge cost. No oracle or inferred
    ground-truth motion is supplied. Windows touching boundaries are excluded.
    """
    if frames.ndim != 4 or frames.shape[0] < 2 or frames.shape[1] != 1:
        raise ValueError("frames must have shape [T,1,H,W], T>=2")
    if window < 1 or window % 2 != 1 or min(frames.shape[-2:]) < window + 2:
        raise ValueError("window must be positive and odd; image dimensions must be >=window+2")
    if not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge must be finite and positive")
    frames = frames.float()
    a, b = frames[:-1], frames[1:]
    mid = .5 * (a + b)
    ix = .5 * (mid[..., 1:-1, 2:] - mid[..., 1:-1, :-2])
    iy = .5 * (mid[..., 2:, 1:-1] - mid[..., :-2, 1:-1])
    it = (b - a)[..., 1:-1, 1:-1]
    terms = torch.cat((ix*ix, ix*iy, iy*iy, ix*it, iy*it, it*it), dim=1)
    stats = F.avg_pool2d(terms, window, stride=1, padding=window//2)
    xx, xy, yy, xt, yt, tt = stats.chunk(6, dim=1)
    a, d = xx + ridge, yy + ridge
    det = (a*d - xy*xy).clamp_min(1e-12)
    explained = (d*xt.square() - 2*xy*xt*yt + a*yt.square()) / det
    residual = (tt - explained).clamp_min(0)
    r = window // 2
    return (residual[..., r:-r, r:-r] if r else residual).mean()


def stationary_energy(g,q,h,mixed,qtt,ridge,c,tau,differentiate_solve=False):
    """Minimize brightness + c spatial + tau² temporal residuals jointly per point."""
    with nullcontext() if differentiate_solve else torch.no_grad():
        matrix=g[..., :,None]*g[...,None,:]+c*h.transpose(-1,-2)@h
        matrix=matrix+tau**2*mixed[..., :,None]*mixed[...,None,:]
        matrix=matrix+ridge*torch.eye(g.shape[-1],device=g.device,dtype=g.dtype)
        rhs=-(g*q[...,None]+c*(h.transpose(-1,-2)@mixed[...,None]).squeeze(-1)+tau**2*mixed*qtt[...,None])
        v=torch.linalg.solve(matrix,rhs[...,None]).squeeze(-1)
    r=q+(g*v).sum(-1)
    rx=mixed+(h@v[...,None]).squeeze(-1)
    rt=qtt+(mixed*v).sum(-1)
    return (r.square()+c*rx.square().sum(-1)+tau**2*rt.square()+ridge*v.square().sum(-1)).mean()


def pointwise_energy(gradient, temporal, alpha):
    """Closed form beta=0; never constructs a velocity field or a solver graph."""
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError('alpha must be finite and positive')
    return (alpha*temporal.square()/(gradient.square().sum(-1)+alpha)).mean()
