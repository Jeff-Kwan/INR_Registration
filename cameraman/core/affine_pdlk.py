"""Locally affine velocity PD-LK with its Jacobian eliminated analytically."""
from contextlib import nullcontext
import math
import torch
from cameraman.core.lk_loss import (
    analytical_siren_derivatives,lk_temporal_loss,minimized_transport_energy)


def validate_affine_settings(beta,average_velocity):
    if not math.isfinite(beta) or beta<=0:
        raise ValueError('Affine PD-LK beta must be finite and strictly positive')
    if type(average_velocity) is not bool:
        raise ValueError('Affine average_velocity must be boolean')


def adaptive_transport_energy(g,q,hessian,mixed,ridge,c,beta=None,average_velocity=False,differentiate_solve=False,rho=None):
    """Independent a per coordinate; c_eff=c/(1+rho*|g|²), rho=c/beta.

    beta is the effective local Jacobian penalty by default. If
    average_velocity=True, it denotes the explicit Jacobian penalty instead,
    and beta_eff=beta+ridge*c includes the averaged velocity-magnitude ridge.
    Only the optimal a is detached; c_eff remains differentiable through g.
    Accepts flat queries or singleton patch dimension; never averages neighbors.
    """
    if rho is not None:
        if beta is not None or not math.isfinite(rho) or rho < 0:
            raise ValueError('Specify finite nonnegative rho or positive beta, not both')
        if type(average_velocity) is not bool:
            raise ValueError('average_velocity must be boolean')
    else:
        beta = 10. if beta is None else beta
        validate_affine_settings(beta,average_velocity)
    if not math.isfinite(c) or c<0 or not math.isfinite(ridge) or ridge<=0:
        raise ValueError('Require finite c>=0 and ridge>0')
    if (g.ndim<2 or g.shape[-1] not in (2,3) or q.shape!=g.shape[:-1]
            or hessian.shape!=(*g.shape,g.shape[-1]) or mixed.shape!=g.shape):
        raise ValueError('Expected g[...,D],q[...],H[...,D,D],mixed[...,D]')
    if c==0:
        loss,a=minimized_transport_energy(g.reshape(-1,1,g.shape[-1]),q.reshape(-1,1),ridge,
                                         differentiate_solve=differentiate_solve)
        return loss,a.reshape_as(g),torch.zeros_like(q)
    dtype=torch.float64 if g.dtype==torch.float64 else torch.float32
    g,q,hessian,mixed=[x.to(dtype) for x in (g,q,hessian,mixed)]
    ratio = rho if rho is not None else c/beta
    if average_velocity:
        ratio = ratio/(1+ridge*ratio)
    effective_c=c/(1+ratio*g.square().sum(-1))
    with nullcontext() if differentiate_solve else torch.no_grad():
        matrix=g[..., :,None]*g[...,None,:]+effective_c[...,None,None]*(hessian.transpose(-1,-2)@hessian)
        matrix=matrix+ridge*torch.eye(g.shape[-1],device=g.device,dtype=dtype)
        rhs=-(g*q[...,None]+effective_c[...,None]*(hessian.transpose(-1,-2)@mixed[...,None]).squeeze(-1))
        a=torch.linalg.solve(matrix,rhs[...,None]).squeeze(-1)
    brightness=q+(g*a).sum(-1)
    gradient=mixed+(hessian@a[...,None]).squeeze(-1)
    energy=brightness.square()+effective_c*gradient.square().sum(-1)+ridge*a.square().sum(-1)
    return energy.mean(),a,effective_c


def affine_temporal_loss(source,target,coordinates,time,shape,ridge=1e-6,c=4.,beta=None,
                         average_velocity=False,differentiate_solve=False,rho=None):
    """Exact SIREN derivatives; rho is canonical, beta accepted for compatibility."""
    if rho is not None:
        if beta is not None or not math.isfinite(rho) or rho < 0:
            raise ValueError('Specify finite nonnegative rho or positive beta, not both')
        if type(average_velocity) is not bool:
            raise ValueError('average_velocity must be boolean')
    else:
        beta = 10. if beta is None else beta
        validate_affine_settings(beta,average_velocity)
    if coordinates.ndim!=3 or coordinates.shape[1]!=1 or coordinates.shape[-1]!=len(shape):
        raise ValueError('Affine PD-LK requires independent point queries [N,1,D]')
    if c==0:
        return lk_temporal_loss(source,target,coordinates,time,shape,ridge,gradient_weight=0.,
                                differentiate_solve=differentiate_solve)
    scale=[2/(n-1) for n in reversed(shape)]
    values=analytical_siren_derivatives(source,target,coordinates,time,scale,differential=True)
    return adaptive_transport_energy(*values,ridge,c,beta,average_velocity,differentiate_solve,rho=rho)[0]
