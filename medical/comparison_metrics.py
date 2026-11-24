"""Versioned common comparison metrics on original prepared evaluation grids."""
from pathlib import Path
import json
import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import map_coordinates, find_objects
import surface_distance
from medical.learn2reg_evaluation import dense_mapping, _interior_jacobian, _surface_distances
from medical.brain_data import protocols

DATASETS = ('neurite_oasis', 'acdc', 'lpba40', 'ixi')


def dataset_for(case):
    root = Path(case['fixed']['path']).resolve().parent.parent
    if (root / 'protocol.json').exists():
        return json.loads((root / 'protocol.json').read_text())['dataset']
    protocol = json.loads((root / 'manifest.json').read_text())['protocol']
    return {'acdc_kan_idir_v1': 'acdc', 'neurite_oasis_gaussiandir_last50_consecutive49_v1': 'neurite_oasis'}[protocol]


def validate_case(case, dataset):
    if dataset in ('ixi', 'lpba40'):
        from medical.brain_data import validate_case as validate
    else:
        from medical.cohort_data import validate_case as validate
    validate(case, dataset)


def label_path(volume, dataset):
    path = Path(volume['path'])
    if dataset == 'neurite_oasis':
        return path.with_name('aligned_seg35.nii.gz')
    return path.parent.parent / 'labels' / path.name


def labels_for(dataset):
    if dataset == 'ixi':
        return sorted({int(v['freesurfer_id']) for group in protocols()['ixi_transmorph']['score_groups'].values() for v in group})
    if dataset == 'lpba40':
        return sorted({int(v) for codes in protocols()['lpba40_sitreg']['labels'].values() for v in codes})
    return list(range(1, 4 if dataset == 'acdc' else 36))


def evaluate(case, prediction, dataset):
    fixed, moving = [np.asarray(nib.load(label_path(case[k], dataset)).dataobj) for k in ('fixed', 'moving')]
    if callable(prediction):
        mapping = dense_mapping(prediction, fixed.shape)
    elif isinstance(prediction, torch.Tensor):
        mapping = prediction.detach().cpu().numpy()
    else:
        mapping = np.asarray(prediction)
    return evaluate_arrays(fixed, moving, mapping, labels_for(dataset),
                           np.linalg.norm(case['fixed']['affine'][:3, :3], axis=0), dataset)


def evaluate_arrays(fixed, moving, mapping, labels, spacing, dataset):
    if mapping.shape != (*fixed.shape, 3) or not np.isfinite(mapping).all():
        raise ValueError('Expected a finite fixed-to-moving absolute XYZ voxel map')
    warped = map_coordinates(moving, mapping.transpose(3, 0, 1, 2), order=0, mode='constant', cval=0)
    dice, distances = [], []
    boxes = [find_objects(array.astype(np.int32, copy=False), max_label=max(labels)) for array in (fixed, warped)]
    for label in labels:
        truth_box, pred_box = (items[label-1] for items in boxes)
        available = [box for box in (truth_box, pred_box) if box is not None]
        if available:
            roi = tuple(slice(min(box[axis].start for box in available), max(box[axis].stop for box in available)) for axis in range(3))
            truth, predicted = fixed[roi] == label, warped[roi] == label
        else:
            truth = predicted = np.zeros((1,1,1), dtype=bool)
        total = np.count_nonzero(truth) + np.count_nonzero(predicted)
        dice.append(float(2 * np.count_nonzero(truth & predicted) / total) if total else None)
        distance = None
        if truth.any() and predicted.any():
            surfaces = _surface_distances()(truth, predicted, spacing)
            value = surface_distance.compute_robust_hausdorff(surfaces, 95.)
            distance = float(value) if np.isfinite(value) else None
        distances.append(distance)
    jac = _interior_jacobian(mapping)
    if not jac.size or not np.isfinite(jac).all():
        raise ValueError('A finite two-voxel-interior Jacobian grid is required')
    def metric(values):
        finite = [v for v in values if v is not None]
        return dict(mean=float(np.mean(finite)) if finite else None, detailed=values, scored_labels=len(finite), intended_labels=len(values))
    hd = metric(distances)
    # Missing surfaces are explicit failed distances, with a strict case mean.
    if hd['scored_labels'] != hd['intended_labels']:
        hd['mean'] = None
    hd['units'] = 'mm in prepared benchmark geometry'
    return dict(metric_protocol='standardized_v1', dataset=dataset, labels=labels, Dice=metric(dice), HD95=hd,
                NPJ=dict(mean=float(100*np.mean(jac <= 0)), units='percent', definition='100*mean(det(Dphi)<=0)'),
                SDlogJ=dict(mean=float(np.std(np.log(np.maximum(jac, 1e-9)))), epsilon=1e-9),
                jacobian=dict(stencil='central differences, two-voxel boundary excluded', region='whole evaluation grid interior', samples=int(jac.size)),
                missing_label_policy='Dice both-empty null; missed foreground Dice zero; missing surface HD95 null and strict case mean null',
                label_warp='nearest neighbor, zero outside image', distance_geometry='IXI uses its released unit-voxel benchmark frame' if dataset=='ixi' else 'prepared image affine spacing')
