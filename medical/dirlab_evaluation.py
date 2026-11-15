"""DIR-Lab 300-landmark evaluation for a T00-to-T50 coordinate map.

Predictions are zero-based T50 voxel coordinates in x,y,z order, one per
T00 landmark and in the same row order. The registration method should supply
these coordinates in memory; the CLI also accepts an explicit CSV for analysis.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import nibabel as nib


def score_landmarks(fixed_xyz, moving_xyz, predicted_xyz, affine, *, grid_shape=None):
    """Score 300 correspondences in zero-based voxel coordinates.

    IDIR rounds the original one-based predicted and reference voxel coordinates
    before multiplying axis differences by spacing. Continuous TRE uses the full
    affine basis. Standard deviations use NumPy's population convention.
    """
    import torch
    on_cuda = torch.is_tensor(predicted_xyz) and predicted_xyz.is_cuda
    if on_cuda:
        def array(value):
            return torch.as_tensor(value, device=predicted_xyz.device, dtype=torch.float64)
        xp = torch
    else:
        def array(value):
            return np.asarray(value, dtype=np.float64)
        xp = np
    fixed, moving, predicted = map(array, (fixed_xyz, moving_xyz, predicted_xyz))
    affine = np.asarray(affine, dtype=np.float64)
    if any(x.shape != (300, 3) or not xp.isfinite(x).all() for x in (fixed, moving, predicted)):
        raise ValueError('Expected three finite 300-by-3 landmark arrays')
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError('Expected a finite 4-by-4 affine')
    basis = affine[:3, :3]
    spacing = np.linalg.norm(basis, axis=0)
    if np.any(spacing <= 0) or not np.allclose(basis.T @ basis, np.diag(spacing**2), rtol=1e-5, atol=1e-8):
        raise ValueError('IDIR-compatible voxel-spacing TRE requires an orthogonal grid')
    basis, spacing = array(basis), array(spacing)
    def distances(a, b):
        return xp.linalg.norm((a-b) @ basis.T, axis=1)
    def std(value, axis=None):
        return value.std(dim=axis, correction=0) if on_cuda else value.std(axis=axis)
    # Rounding after the shift preserves the official one-based half-voxel ties.
    component = xp.abs(xp.round(predicted + 1) - xp.round(moving + 1)) * spacing
    rounded = xp.linalg.norm(component, axis=1)
    initial = distances(fixed, moving)
    continuous = distances(predicted, moving)
    outside = None
    if grid_shape is not None:
        outside = int(((predicted < 0) | (predicted > array(grid_shape)-1)).any(axis=1).sum())
    values = xp.stack([initial.mean(), std(initial), rounded.mean(), std(rounded),
                       continuous.mean(), std(continuous)]).tolist()
    axis_stats = xp.stack([component.mean(axis=0), std(component, 0)]).tolist()
    return {'count': 300,
            'initial_tre_mm': {'mean_mm': values[0], 'std_mm': values[1]},
            'idir_rounded_tre_mm': {'mean_mm': values[2], 'std_mm': values[3]},
            'idir_rounded_axis_mean_mm_xyz': axis_stats[0],
            'idir_rounded_axis_std_mm_xyz': axis_stats[1],
            'continuous_tre_mm': {'mean_mm': values[4], 'std_mm': values[5]},
            'prediction_outside_voxel_grid': outside}


def read_points(path):
    return np.loadtxt(path, delimiter=',', skiprows=1, ndmin=2)


def evaluate_case(case_dir, predicted_xyz):
    case_dir = Path(case_dir)
    fixed_image = nib.load(str(case_dir / 'T00.nii.gz'))
    moving_image = nib.load(str(case_dir / 'T50.nii.gz'))
    if fixed_image.shape != moving_image.shape or not np.allclose(fixed_image.affine, moving_image.affine):
        raise ValueError('T00 and T50 must have the same grid')
    fixed = read_points(case_dir / 'T00_landmarks_voxel.csv')
    moving = read_points(case_dir / 'T50_landmarks_voxel.csv')
    result = score_landmarks(fixed, moving, predicted_xyz, fixed_image.affine)
    pred = np.asarray(predicted_xyz)
    result['prediction_outside_voxel_grid'] = int(np.any((pred < 0) | (pred > np.array(moving_image.shape)-1), axis=1).sum())
    result['case'] = case_dir.name
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-dir', required=True, type=Path)
    parser.add_argument('--predicted', required=True, type=Path, help='CSV with x,y,z header and 300 zero-based predicted T50 voxel coordinates')
    args = parser.parse_args(argv)
    print(json.dumps(evaluate_case(args.case_dir, read_points(args.predicted)), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
