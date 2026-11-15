"""ANTs, elastix and Demons on a common canonical physical grid.

Inputs share one orthogonal NIfTI lattice. Toolkits receive a zero-origin,
axis-aligned grid with the original spacing. Returned displacements are native
voxel vectors (x=W,y=H,z=D); the evaluator uses the original affine.
"""
import tempfile
from pathlib import Path
import numpy as np
import torch

from medical.registration.geometry import identity_grid, sample
from medical.registration.recovery import RecoveryResult


def ants_syn(fixed, moving, spacing, iterations=(50, 50, 30), metric='cc',
             flow_sigma=3., total_sigma=0., gradient_step=.2, syn_sampling=2):
    """SyNOnly on prepared grids; compose the full image-warp transform chain."""
    import ants
    options = dict(spacing=tuple(float(s) for s in reversed(spacing)))
    fixed_image = ants.from_numpy(np.ascontiguousarray(fixed.transpose(2, 1, 0)), **options)
    moving_image = ants.from_numpy(np.ascontiguousarray(moving.transpose(2, 1, 0)), **options)
    with tempfile.TemporaryDirectory(prefix='ants_syn_') as directory:
        registration = ants.registration(fixed=fixed_image, moving=moving_image,
            type_of_transform='SyNOnly', initial_transform='Identity',
            syn_metric=metric, syn_sampling=int(syn_sampling),
            reg_iterations=tuple(int(v) for v in iterations), flow_sigma=float(flow_sigma),
            total_sigma=float(total_sigma), grad_step=float(gradient_step),
            outprefix=str(Path(directory) / 'syn'), verbose=False)
        inverse = ants_chain_displacement(ants, fixed_image, moving_image,
            registration['fwdtransforms'], spacing, directory)
    return {'inverse': inverse, 'method': 'ants_syn_' + metric}


def ants_chain_displacement(ants, fixed, moving, transforms, spacing, directory):
    composed = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=transforms,
        compose=str(Path(directory) / 'composite'), singleprecision=True)
    field = ants.image_read(composed).numpy()
    array = np.transpose(field, (2, 1, 0, 3))
    return np.ascontiguousarray(array / np.asarray(tuple(reversed(spacing))), dtype=np.float32)


@torch.no_grad()
def invert_displacement(displacement, iterations=30, tolerance=1e-4):
    field = torch.as_tensor(np.ascontiguousarray(displacement), dtype=torch.float32)
    grid = identity_grid(field.shape[:3], 'cpu')
    mapping = grid.clone()
    change = float('inf')
    for index in range(iterations):
        update = grid - sample(field, mapping)
        change = float((update - mapping).abs().max())
        mapping = update
        if change < tolerance:
            break
    return mapping, {'iterations': index + 1, 'max_update_voxels': change,
                     'converged': change < tolerance, 'approximate': True}


def register(method, fixed, moving, affine, settings):
    methods = {'ants_syn': ants_syn, 'elastix_bspline': elastix_bspline, 'sitk_demons': sitk_demons}
    if method not in methods:
        raise ValueError(f'Unknown baseline: {method}')
    spacing = np.linalg.norm(np.asarray(affine)[:3, :3], axis=0)
    payload = methods[method](fixed, moving, spacing, **settings)
    displacement = payload['inverse']
    if displacement.shape != (*fixed.shape, 3) or not np.isfinite(displacement).all():
        raise ValueError('Baseline returned an invalid displacement')
    forward, inversion = invert_displacement(displacement)
    inverse = identity_grid(fixed.shape, 'cpu') + torch.from_numpy(displacement)
    return RecoveryResult(forward, inverse), {'method': payload['method'],
        'options': settings, 'forward_inversion': inversion}

def _sitk():
    import SimpleITK as sitk
    return sitk

def _to_sitk(image, spacing):
    """[D,H,W] array -> SimpleITK image with x=W, y=H, z=D (spacing reversed)."""
    sitk = _sitk()
    volume = sitk.GetImageFromArray(np.ascontiguousarray(image, dtype=np.float32))
    volume.SetSpacing((float(spacing[2]), float(spacing[1]), float(spacing[0])))
    return volume

def _sitk_displacement_to_voxels(field, spacing):
    """SimpleITK displacement image (mm) -> [D,H,W,3] voxel units (x,y,z)."""
    sitk = _sitk()
    array = sitk.GetArrayFromImage(field)  # [D,H,W,3] mm, channels (x,y,z)
    scale = np.asarray([spacing[2], spacing[1], spacing[0]], dtype=np.float64)
    return (array / scale).astype(np.float32)

def sitk_demons(fixed, moving, spacing, kind='diffeomorphic', iterations=100,
                standard_deviations=1.0):
    """SimpleITK Demons; ``kind`` is ``diffeomorphic`` or ``fast_symmetric``."""
    sitk = _sitk()
    fixed_image = _to_sitk(fixed, spacing)
    moving_image = _to_sitk(moving, spacing)
    constructors = {
        'diffeomorphic': sitk.DiffeomorphicDemonsRegistrationFilter,
        'fast_symmetric': sitk.FastSymmetricForcesDemonsRegistrationFilter,
        'symmetric': sitk.SymmetricForcesDemonsRegistrationFilter,
    }
    if kind not in constructors:
        raise ValueError(f'Unknown demons kind: {kind}')
    registration = constructors[kind]()
    registration.SetNumberOfIterations(int(iterations))
    registration.SetStandardDeviations(float(standard_deviations))
    field = registration.Execute(fixed_image, moving_image)
    return {'inverse': _sitk_displacement_to_voxels(field, spacing),
            'method': f'sitk_demons_{kind}', 'iterations': int(iterations)}

def _itk_image(array, spacing):
    """[D,H,W] -> ITK image (x=W, y=H, z=D) in a zero-origin grid."""
    import itk
    image = itk.image_from_array(np.ascontiguousarray(array, dtype=np.float32))
    image.SetSpacing((float(spacing[2]), float(spacing[1]), float(spacing[0])))
    return image

def elastix_bspline(fixed, moving, spacing, metric='ncc', affine=True):
    """Elastix B-spline with optional affine initialization.

    Uses the tuned elastix default parameter maps, then extracts the transform's
    deformation field with transformix.  ``metric`` is ``ncc`` (correlation, for
    same-modality data) or ``mi``.
    """
    import itk
    fixed_image = _itk_image(fixed, spacing)
    moving_image = _itk_image(moving, spacing)
    parameters = itk.ParameterObject.New()
    metric_name = ('AdvancedNormalizedCorrelation' if metric == 'ncc'
                   else 'AdvancedMattesMutualInformation')
    if affine:
        affine_map = parameters.GetDefaultParameterMap('affine')
        affine_map['Metric'] = [metric_name]
        parameters.AddParameterMap(affine_map)
    # Preserve the library's per-resolution defaults; override only the metric.
    bspline_map = parameters.GetDefaultParameterMap('bspline')
    bspline_map['Metric'] = [metric_name]
    parameters.AddParameterMap(bspline_map)
    with tempfile.TemporaryDirectory(prefix="elastix_") as directory:
        _, transform_parameters = itk.elastix_registration_method(
            fixed_image, moving_image, parameter_object=parameters,
            output_directory=directory, log_to_console=False, log_to_file=False)
        field = itk.transformix_deformation_field(fixed_image,
            transform_parameter_object=transform_parameters, output_directory=directory,
            log_to_console=False, log_to_file=False)
        array = np.asarray(itk.array_from_image(field)).copy()  # [D,H,W,3] mm, channels (x,y,z)
    scale = np.asarray([spacing[2], spacing[1], spacing[0]], dtype=np.float64)
    return {'inverse': (array / scale).astype(np.float32),
            'method': f'elastix_bspline_{metric}'}
