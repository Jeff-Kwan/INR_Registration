"""Convert native physical transforms to the shared fixed-to-moving voxel API."""
import numpy as np
from .common import sample_xyz_field

LPS = np.diag([-1., -1., 1.])


def image_geometry(volume):
    affine = np.asarray(volume['affine'])
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    direction = LPS @ affine[:3, :3] / spacing
    if not np.isfinite(affine).all() or np.any(spacing <= 0) or not np.allclose(direction.T @ direction, np.eye(3), atol=1e-5):
        raise ValueError('ITK baselines require an orthogonal physical image grid')
    return tuple(spacing), tuple(LPS @ affine[:3, 3]), direction


def sitk_image(module, volume):
    image = module.GetImageFromArray(volume['data'].transpose(2, 1, 0).copy())
    spacing, origin, direction = image_geometry(volume)
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    image.SetDirection(tuple(direction.ravel()))
    return image


def lps_displacement_predictor(field, case):
    field = np.asarray(field)
    expected = (*case['fixed']['data'].shape, 3)
    if field.shape != expected or not np.isfinite(field).all():
        raise ValueError('Invalid native physical displacement field')
    fixed = np.asarray(case['fixed']['affine'])
    inverse = np.linalg.inv(case['moving']['affine'])

    def predict(points):
        points = np.asarray(points)
        delta = sample_xyz_field(field, points)
        # ITK vector interpolation extends edge samples inside its half-voxel
        # buffer boundary. Displacement transforms are identity outside it.
        inside = np.all((points >= -.5) & (points < np.asarray(field.shape[:3]) - .5), axis=1)
        delta[~inside] = 0
        physical = points @ fixed[:3, :3].T + fixed[:3, 3] + delta @ LPS
        return physical @ inverse[:3, :3].T + inverse[:3, 3]
    return predict


def ras_mapping_predictor(field, case):
    field = np.asarray(field)
    if field.shape != (*case['fixed']['data'].shape, 3) or not np.isfinite(field).all():
        raise ValueError('Invalid NiftyReg absolute physical deformation')
    inverse = np.linalg.inv(case['moving']['affine'])

    def predict(points):
        physical = sample_xyz_field(field, points)
        return physical @ inverse[:3, :3].T + inverse[:3, 3]
    return predict


def provenance_checkpoint():
    return {'kind': 'provenance_only', 'parameter_checkpoint': False,
            'reason': 'Native dense field or control grid is confined to temporary evaluation state'}
