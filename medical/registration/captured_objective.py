"""Joint endpoint NCC, shared-grid PD-LK and box or Gaussian-window registration NCC."""
import torch
from medical.models.siren_functional import _interpolate,_parameter_direction
from medical.registration.objective import _endpoint_kernel
from medical.registration.symmetric_registration import _pair_velocity_chunk
from medical.registration.derivatives import _differential_chain, _layer_curvature_chain
from medical.registration.compiled_decoder import box_decoder,three_tap_smoothing


def joint_loss(source,target,omegas,source_coordinates,target_coordinates,source_truth,target_truth,
               query_coordinates,spatial_scale,positions,moving,fixed,origin,domain,extent,
               early,late,sigma,grid_size,c,eta,weight,registration_weight,ncc_options,
               registration_options=None):
    input_weight=source[0][0]
    fuse_affines=(input_weight.is_cuda and input_weight.dtype==torch.float32
                  and input_weight.shape==(512,3))
    terms=_endpoint_kernel(source,target,omegas,source_coordinates,target_coordinates,
                           source_truth,target_truth,'ncc',ncc_options,fuse_scalar_affines=fuse_affines)
    direction=_parameter_direction(source,target)
    differential = _layer_curvature_chain if fuse_affines else _differential_chain
    fields=[]
    energies=[]
    for time in [early,late]:
        parameters=_interpolate(source,target,time)
        velocity,energy=_pair_velocity_chunk(parameters,direction,omegas,query_coordinates,
            spatial_scale,c,eta,True,derivative_function=differential,regularizer='pdlk',velocity_function=checked_velocity)
        fields.append(three_tap_smoothing(velocity.reshape(grid_size,grid_size,grid_size,3),sigma))
        energies.append(energy.mean())
    fields=tuple(fields)
    options=registration_options or {}
    decoder_options=dict(similarity=options.get('registration_similarity','ncc'),
        particle_sampling_fraction=options.get('particle_sampling_fraction',.8),
        gw_ncc_sigma_cells=options.get('gw_ncc_sigma_cells',.75),
        gw_ncc_kernel=options.get('gw_ncc_kernel','three_tap'))
    forward,finite_forward=box_decoder(positions[0],fields,moving,fixed,origin,domain,extent,grid_size,False,**decoder_options)
    reverse,finite_reverse=box_decoder(positions[1],fields,fixed,moving,origin,domain,extent,grid_size,True,**decoder_options)
    penalty=.5*(energies[0]+energies[1])
    registration=.5*(forward['similarity']+reverse['similarity']+forward['boundary']+reverse['boundary'])
    terms.update(pdlk=penalty,registration=registration)
    terms['total']=terms['total']+weight*penalty+registration_weight*registration
    return terms,finite_forward & finite_reverse



def checked_velocity(g,q,hessian,mixed,c,eta):
    matrix=(g[..., :,None]*g[...,None,:]+c*(hessian.transpose(-1,-2)@hessian)).mean(-3)
    rhs=-(g*q[...,None]+c*(hessian.transpose(-1,-2)@mixed[...,None]).squeeze(-1)).mean(-2)
    value,info=torch.linalg.solve_ex(matrix+eta*torch.eye(3,device=g.device,dtype=g.dtype),
                                    rhs[...,None],check_errors=False)
    torch._assert_async((info==0).all(),'Local shared-grid solve failed')
    return value.squeeze(-1)
