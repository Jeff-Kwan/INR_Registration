"""Joint endpoint reconstruction, local transport loss and trajectory-based registration."""
import torch
from medical.models.siren_functional import _interpolate,_parameter_direction
from medical.registration.objective import _endpoint_kernel
from medical.registration.symmetric_registration import _pair_velocity_chunk
from medical.registration.derivatives import _differential_chain, _layer_curvature_chain
from medical.registration.compiled_decoder import box_decoder,three_tap_smoothing
from medical.registration.training_integrators import DEFAULT_INTEGRATOR


def joint_loss(source,target,omegas,source_coordinates,target_coordinates,source_truth,target_truth,
               query_coordinates,spatial_scale,positions,moving,fixed,origin,domain,extent,
               times,sigma,grid_size,c,eta,local_transport_weight,registration_weight,ncc_options,
               registration_options=None,integrator=DEFAULT_INTEGRATOR,parameter_l2=0.,endpoint_weight=1.,brightness_weight=1.,
               endpoint_mode='ncc',transport_enabled=True,registration_enabled=True):
    input_weight=source[0][0]
    fuse_affines=(input_weight.is_cuda and input_weight.dtype==torch.float32
                  and input_weight.shape==(512,3))
    terms=_endpoint_kernel(source,target,omegas,source_coordinates,target_coordinates,
                           source_truth,target_truth,endpoint_mode,ncc_options,fuse_scalar_affines=fuse_affines,
                           parameter_l2=parameter_l2,endpoint_weight=endpoint_weight)
    if not transport_enabled and not registration_enabled:
        zero = terms['total'].new_zeros(())
        terms.update(local_transport_loss=zero, registration_loss=zero)
        return terms, torch.ones((), device=zero.device, dtype=torch.bool)
    direction=_parameter_direction(source,target)
    differential = _layer_curvature_chain if fuse_affines else _differential_chain
    fields=[]
    energies=[]
    for time in times:
        parameters=_interpolate(source,target,time)
        velocity,energy=_pair_velocity_chunk(parameters,direction,omegas,query_coordinates,
            spatial_scale,c,eta,True,derivative_function=differential,velocity_function=checked_velocity,
            brightness_weight=brightness_weight)
        field=velocity.reshape(grid_size,grid_size,grid_size,3)
        field=three_tap_smoothing(field,sigma)
        fields.append(field)
        energies.append(energy.mean())
    fields=tuple(fields)
    local_transport_loss=torch.stack(energies).mean()
    registration=local_transport_loss.new_zeros(())
    finite=torch.ones((),device=local_transport_loss.device,dtype=torch.bool)
    if registration_enabled:
        options=registration_options or {}
        decoder_options=dict(similarity=options.get('registration_similarity','gw_ncc'),
            particle_sampling_fraction=options.get('particle_sampling_fraction',.8),
            gw_ncc_sigma_cells=options.get('gw_ncc_sigma_cells',.75),
            gw_ncc_kernel=options.get('gw_ncc_kernel','three_tap'),integrator=integrator)
        forward,finite_forward=box_decoder(positions[0],fields,moving,fixed,origin,domain,extent,grid_size,False,**decoder_options)
        reverse,finite_reverse=box_decoder(positions[1],fields,fixed,moving,origin,domain,extent,grid_size,True,**decoder_options)
        registration=.5*(forward['similarity']+reverse['similarity']+forward['boundary']+reverse['boundary'])
        finite=finite_forward & finite_reverse
    terms.update(local_transport_loss=local_transport_loss,registration_loss=registration)
    terms['total']=terms['total']+local_transport_weight*local_transport_loss+registration_weight*registration
    return terms,finite



def checked_velocity(g,q,hessian,mixed,c,eta,*,brightness_weight=1.):
    matrix=(brightness_weight*(g[..., :,None]*g[...,None,:])+c*(hessian.transpose(-1,-2)@hessian)).mean(-3)
    rhs=-(brightness_weight*(g*q[...,None])+c*(hessian.transpose(-1,-2)@mixed[...,None]).squeeze(-1)).mean(-2)
    value,info=torch.linalg.solve_ex(matrix+eta*torch.eye(3,device=g.device,dtype=g.dtype),
                                    rhs[...,None],check_errors=False)
    torch._assert_async((info==0).all(),'Local shared-grid solve failed')
    return value.squeeze(-1)
