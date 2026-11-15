"""Fixed coarse sampling of the same native-voxel PCG+RK2 equations.

No image preprocessing or estimator change. Derivatives and velocity retain
original voxel units; grid spacing enters the PCG stencil and map integration.
Returned maps are temporary full-grid tensors for unchanged evaluation.
"""
import torch
from medical.registration.flow import estimate_velocity_from_derivatives
from medical.registration.recovery import recover as original_recover,integrate_interval,RecoveryResult


def upsample_map(mapping, shape, spacing_xyz):
    result=torch.nn.functional.interpolate(mapping.movedim(-1,0)[None],size=shape,
        mode='trilinear',align_corners=True)[0].movedim(0,-1)
    return result*spacing_xyz


@torch.no_grad()
def recover_capped(pair,shape,settings,scratch_root,*,max_axis=256,chunk=65536,
            compile_derivatives=True,compile_solver=True,**unsupported):
    if unsupported or getattr(pair,'requires_physical_units_metadata',False) or getattr(pair,'requires_working_grid_metadata',False):
        raise ValueError('Capped recovery requires a frozen native-voxel model')
    if len(shape)!=3:raise ValueError('Capped recovery requires a 3D native grid')
    if type(max_axis) is not int or max_axis<2: raise ValueError('Recovery cap must be at least two')
    if max(shape)<=max_axis:
        return original_recover(pair,shape,settings,scratch_root,chunk=chunk,compile_derivatives=compile_derivatives,compile_solver=compile_solver)
    coarse=tuple(min(n,max_axis) for n in shape)
    reference=next(pair.parameters())
    ratios=tuple((n-1)/(c-1) for n,c in zip(shape,coarse))
    spacing_xyz=reference.new_tensor(ratios[::-1])
    previous=None
    def solver(gradient,temporal,**kwargs):
        nonlocal previous
        velocity,diagnostic=estimate_velocity_from_derivatives(gradient/spacing_xyz,temporal,
            spacing=ratios,initial_velocity=previous,compile_solver=compile_solver,**kwargs)
        previous=velocity
        diagnostic.update({'native_recovery_max_axis':max_axis,
                           'recovery_shape_zyx':list(coarse),
                           'recovery_spacing_in_original_voxels_zyx':list(ratios),
                           'velocity_units':'original_native_voxels'})
        return velocity,diagnostic
    def integrate(mapping,velocity,dt,substeps):
        return integrate_interval(mapping,velocity/spacing_xyz,dt,substeps)
    try:
        result=original_recover(pair,coarse,settings,scratch_root,chunk=chunk,
            compile_derivatives=compile_derivatives,compile_solver=compile_solver,
            velocity_solver=solver,integrator=integrate)
        return RecoveryResult(upsample_map(result.forward,shape,spacing_xyz),
                              upsample_map(result.inverse,shape,spacing_xyz),result.diagnostics)
    finally:
        previous=None
