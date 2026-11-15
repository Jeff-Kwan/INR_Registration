"""Optional native-grid millimetre contract; no image resampling."""
import numpy as np


def metadata(shape, affine):
    shape=tuple(shape)
    matrix=np.asarray(affine,dtype=np.float64)
    if (len(shape)!=3 or any(type(n) is not int or n<2 for n in shape)
            or matrix.shape!=(4,4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3],[0,0,0,1])):
        raise ValueError('Physical units require a valid native shape and affine')
    basis=matrix[:3,:3];spacing=np.linalg.norm(basis,axis=0)
    if (np.any(spacing<=0) or not np.allclose((basis/spacing).T@(basis/spacing),np.eye(3),atol=1e-6)):
        raise ValueError('Physical units require orthogonal nondegenerate axes')
    return {'version':1,'units':'mm','shape_zyx':list(shape),'affine_zyx':matrix.tolist(),
            'spacing_xyz_mm':spacing[::-1].tolist()}


def validate_metadata(value, shape):
    if not isinstance(value,dict) or set(value)!={'version','units','shape_zyx','affine_zyx','spacing_xyz_mm'}:
        raise ValueError('Missing or invalid physical-units metadata')
    expected=metadata(tuple(value['shape_zyx']),value['affine_zyx'])
    if (value['version']!=1 or value['units']!='mm' or tuple(shape)!=tuple(value['shape_zyx'])
            or not np.allclose(value['spacing_xyz_mm'],expected['spacing_xyz_mm'],atol=1e-8,rtol=1e-8)):
        raise ValueError('Physical-units metadata does not match native geometry')
    return value


def restore_checkpoint(payload, device='cpu'):
    from medical.configuration import validate_config
    from medical.models.siren import PairedSiren
    validate_config(payload['config'])
    if payload['config'].get('spatial_units','voxel')!='mm':
        raise ValueError('Expected a native physical-units checkpoint')
    value=payload.get('physical_units')
    if not isinstance(value,dict):raise ValueError('Missing physical-units checkpoint geometry')
    validate_metadata(value,value.get('shape_zyx',()))
    pair=PairedSiren(3,**payload['model']).to(device)
    pair.load_state_dict(payload['state'],strict=True)
    pair.requires_physical_units_metadata=True
    pair.physical_units_metadata=value
    return pair.eval()
