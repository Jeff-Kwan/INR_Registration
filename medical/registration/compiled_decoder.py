"""Compiled smoothing, all-particle statistics and spatial RK2 registration decoding."""
from functools import lru_cache
import torch


def box_statistics(original, warped):
    da = original-original.mean()
    db = warped-warped.mean()
    correlation = (da*db).mean()/((da.square().mean()+1e-8)*(db.square().mean()+1e-8)).sqrt()
    return (original-warped).square().mean(), 1-correlation


@lru_cache(maxsize=1)
def compiled_box_statistics():
    return torch.compile(box_statistics,fullgraph=True)


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
                reverse, *, similarity='ncc', particle_sampling_fraction=.8,
                gw_ncc_sigma_cells=.75, gw_ncc_kernel='three_tap'):
    """Integrate one direction with spatial midpoint RK2 and score all particles."""
    from medical.registration.geometry import sample
    from medical.registration.image_sampling import sample_points
    def supported(p):
        return ((p>=origin)&(p<=origin+domain)).all(-1)
    def velocity(field,p):
        local=(p-origin)/domain*grid_size-.5
        return sample(field,local.reshape(-1,1,1,3)).reshape(-1,3)
    p=starts;valid=supported(p)
    finite=torch.ones((),dtype=torch.bool,device=p.device)
    dt=-.5 if reverse else .5
    for field in reversed(fields) if reverse else fields:
        middle=p+.5*dt*velocity(field,p)
        finite=finite&torch.isfinite(middle).all();valid=valid&supported(middle)
        p=p+dt*velocity(field,middle)
        finite=finite&torch.isfinite(p).all();valid=valid&supported(p)
    a=sample_points(source,starts);b=sample_points(target,p)
    mse,ncc=box_statistics(a,b)
    global_ncc,local_ncc=ncc,mse.new_zeros(())
    if similarity == 'gw_ncc':
        from medical.registration.symmetric_registration import source_grid_indices,gaussian_weighted_ncc
        bins=source_grid_indices(starts,origin,domain,particle_sampling_fraction)
        ncc,global_ncc,local_ncc=gaussian_weighted_ncc(a,b,bins,
            sigma_cells=gw_ncc_sigma_cells,gaussian_kernel=gw_ncc_kernel)
    elif similarity != 'ncc':
        raise ValueError('Unsupported captured registration similarity')
    zero=mse.new_zeros(())
    unit=p/extent
    boundary=((-unit).clamp_min(0)+(unit-1).clamp_min(0)).square().sum(-1).mean()
    return dict(mse=mse,similarity=ncc,global_ncc_loss=global_ncc,local_ncc_loss=local_ncc,
                boundary=boundary,outside=(~valid).float().mean(),
                valid_particles=a.new_tensor(starts.shape[0]),skipped=zero),finite


@lru_cache(maxsize=1)
def compiled_box_decoder():
    return torch.compile(box_decoder,fullgraph=True)
