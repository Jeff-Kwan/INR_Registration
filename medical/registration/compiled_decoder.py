"""Compiled smoothing, all-particle statistics and registration trajectories."""
from functools import lru_cache
import torch
from medical.registration.training_integrators import DEFAULT_INTEGRATOR


def three_tap_smoothing(field,sigma):
    for axis,width in enumerate(sigma):
        ids=torch.arange(field.shape[axis],device=field.device)
        before=torch.where(ids==0,1,ids-1)
        after=torch.where(ids==field.shape[axis]-1,field.shape[axis]-2,ids+1)
        x=field.new_tensor([-1.,0.,1.])
        weights=torch.exp(-.5*(x/width).square());weights=weights/weights.sum()
        field=weights[0]*field.index_select(axis,before)+weights[1]*field+weights[2]*field.index_select(axis,after)
    return field


@lru_cache(maxsize=1)
def compiled_three_tap_smoothing():
    return torch.compile(three_tap_smoothing,fullgraph=True)


def box_decoder(starts,fields,source,target,origin,domain,extent,grid_size,
                reverse, *, similarity='gw_ncc', particle_sampling_fraction=.8,
                gw_ncc_sigma_cells=.75, gw_ncc_kernel='three_tap', integrator=DEFAULT_INTEGRATOR):
    """Integrate one direction with two midpoint RK2 half-steps and score it."""
    from medical.registration.geometry import sample
    from medical.registration.image_sampling import sample_points
    from medical.registration.training_integrators import time_strata
    count = time_strata(integrator)
    if len(fields) != count:
        raise ValueError('One velocity field is required per training time stratum')
    def supported(p):
        return ((p>=origin)&(p<=origin+domain)).all(-1)
    def velocity(field,p):
        local=(p-origin)/domain*grid_size-.5
        return sample(field,local.reshape(-1,1,1,3)).reshape(-1,3)
    p=starts;valid=supported(p)
    finite=torch.ones((),dtype=torch.bool,device=p.device)
    dt=(-1 if reverse else 1)/count
    for field in reversed(fields) if reverse else fields:
        v=velocity(field,p)
        middle=p+.5*dt*v
        finite=finite&torch.isfinite(middle).all();valid=valid&supported(middle)
        v=velocity(field,middle)
        p=p+dt*v
        finite=finite&torch.isfinite(p).all();valid=valid&supported(p)
    a=sample_points(source,starts);b=sample_points(target,p)
    if similarity != 'gw_ncc':
        raise ValueError('Unsupported captured registration similarity')
    mse=(a-b).square().mean()
    from medical.registration.symmetric_registration import source_grid_indices,gaussian_weighted_ncc
    bins=source_grid_indices(starts,origin,domain,particle_sampling_fraction)
    ncc,global_ncc,local_ncc=gaussian_weighted_ncc(a,b,bins,
        sigma_cells=gw_ncc_sigma_cells,gaussian_kernel=gw_ncc_kernel)
    zero=mse.new_zeros(())
    unit=p/extent
    boundary=((-unit).clamp_min(0)+(unit-1).clamp_min(0)).square().sum(-1).mean()
    return dict(mse=mse,similarity=ncc,global_ncc_loss=global_ncc,local_ncc_loss=local_ncc,
                boundary=boundary,outside=(~valid).float().mean(),
                valid_particles=a.new_tensor(starts.shape[0]),skipped=zero),finite


@lru_cache(maxsize=1)
def compiled_box_decoder():
    return torch.compile(box_decoder,fullgraph=True)
