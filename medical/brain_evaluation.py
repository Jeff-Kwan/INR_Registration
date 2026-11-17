"""Dataset-specific brain scores from in-memory absolute array-axis voxel maps.

IXI uses the TransMorph 46-channel label warp and 17-group score. LPBA40
uses the pinned SITReg reference methods; neither uses the OASIS evaluator.
"""
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F

from medical.brain_data import label_path, protocols


def finite(value):
    value = float(value)
    return value if np.isfinite(value) else None


def mapping_tensor(mapping, shape):
    tensor = mapping.detach() if isinstance(mapping, torch.Tensor) else torch.as_tensor(mapping)
    if tensor.shape != (*shape, 3) or not torch.isfinite(tensor).all():
        raise ValueError('Expected a finite absolute moving XYZ map on the fixed array grid')
    return tensor.float()


def identity(shape, device, dtype=torch.float32):
    return torch.stack(torch.meshgrid(*(torch.arange(n, device=device, dtype=dtype)
                                       for n in shape), indexing='ij'), -1)


def remap_ixi(labels):
    spec = protocols()['ixi_transmorph']['freesurfer_id_to_consecutive_id']
    remapped = np.zeros(labels.shape, dtype=np.uint8)
    for source, target in spec.items():
        remapped[labels == int(source)] = target
    return remapped


@torch.no_grad()
def warp_ixi(labels, mapping, class_chunk=8):
    """Stream classes, preserving the author's argmax tie order and zero padding."""
    source = torch.as_tensor(remap_ixi(labels).copy(), device=mapping.device)
    grid = (2 * mapping / mapping.new_tensor(np.asarray(labels.shape) - 1) - 1).flip(-1)[None]
    best = torch.full(mapping.shape[:-1], -torch.inf, device=mapping.device)
    winner = torch.zeros(mapping.shape[:-1], dtype=torch.uint8, device=mapping.device)
    for start in range(0, 46, class_chunk):
        indices = torch.arange(start, min(start + class_chunk, 46), device=mapping.device)
        channels = (source[None] == indices[:, None, None, None]).float()[None]
        warped = F.grid_sample(channels, grid, mode='bilinear', padding_mode='zeros', align_corners=True)[0]
        values, local_index = warped.max(dim=0)
        update = values > best
        winner[update] = (local_index[update] + start).to(torch.uint8)
        best = torch.maximum(best, values)
    return winner.cpu().numpy()


def ixi_folding(mapping):
    """TransMorph whole-grid NumPy gradients of voxel displacement + identity."""
    values = mapping.detach().cpu().numpy() if isinstance(mapping, torch.Tensor) else np.asarray(mapping)
    grid = np.stack(np.meshgrid(*(np.arange(n) for n in values.shape[:3]), indexing='ij'), -1)
    absolute = (values - grid.astype(np.float32)) + grid
    a, b, c = np.gradient(absolute, axis=(0, 1, 2))
    det = (a[..., 0] * (b[..., 1] * c[..., 2] - b[..., 2] * c[..., 1])
           - a[..., 1] * (b[..., 0] * c[..., 2] - b[..., 2] * c[..., 0])
           + a[..., 2] * (b[..., 0] * c[..., 1] - b[..., 1] * c[..., 0]))
    return float(np.mean(det <= 0))


def ixi_direction(moving, fixed, mapping):
    mapping = mapping_tensor(mapping, fixed.shape)
    prediction = warp_ixi(moving, mapping)
    target = remap_ixi(fixed)
    pred_count = np.bincount(prediction.ravel(), minlength=46)
    target_count = np.bincount(target.ravel(), minlength=46)
    equal = prediction == target
    intersection = np.bincount(target[equal], minlength=46)
    dices = 2 * intersection / (pred_count + target_count + 1e-5)
    groups = {name: float(np.mean([dices[item['remapped_id']] for item in members]))
              for name, members in protocols()['ixi_transmorph']['score_groups'].items()}
    return {'DSC': {'mean': float(np.mean(list(groups.values()))), 'groups': groups,
                    'all_46_label_dice': dices.tolist(), 'aggregation': 'equal mean of 17 groups from 30 labels'},
            'folding_fraction': ixi_folding(mapping), 'folding_definition': 'whole-grid NumPy detJ <= 0'}


@torch.no_grad()
def lpba_direction(moving, fixed, mapping, *, pair_index, jacobian_samples=1000000):
    from composable_mapping import CoordinateSystem, DataFormat, LinearInterpolator, samplable_volume
    from medical.protocols.sitreg_reference import SITRegMetrics
    mapping = mapping_tensor(mapping, fixed.shape)
    spec = protocols()['lpba40_sitreg']
    seed = spec['data_config']['jacobian_sampling_base_seed'] ^ (pair_index - 1)
    reference = SITRegMetrics(spec['labels'], samples=jacobian_samples, seed=seed)
    source = torch.as_tensor(moving.astype(np.float32, copy=True), device=mapping.device)[None]
    target = torch.as_tensor(fixed.astype(np.float32, copy=True), device=mapping.device)[None]
    displacement = (mapping - identity(mapping.shape[:3], mapping.device)).movedim(-1, 0)
    warped = reference._transform_mask(source, displacement)
    scores = reference._compute_segmentation_metrics(warped, target, '')
    coordinate_system = CoordinateSystem.centered_normalized(
        spatial_shape=mapping.shape[:3], dtype=mapping.dtype, device=mapping.device)
    field = samplable_volume(displacement[None], coordinate_system=coordinate_system,
                            sampler=LinearInterpolator(), data_format=DataFormat.voxel_displacements())
    sampled = reference._compute_sampled_determinant_metrics(field, '', mapping.device)
    dice = {name: finite(scores[name + '_dice']) for name in spec['labels']}
    hd95 = {name: finite(scores[name + '_hd95']) for name in spec['labels']}
    return {'DSC': {'mean': finite(scores['dice']), 'per_label': dice,
                    'excluded_nonfinite_labels': [k for k, v in dice.items() if v is None]},
            'HD95': {'mean': finite(scores['hd95']), 'per_label': hd95, 'units': 'voxels',
                     'excluded_nonfinite_labels': [k for k, v in hd95.items() if v is None]},
            'folding_fraction': sampled['proportion_negative_determinants'],
            'folding_definition': 'SITReg sampled float64 detJ < 0',
            'determinant_std': sampled['det_std'],
            'jacobian_samples': jacobian_samples, 'jacobian_seed': seed}


def evaluate(case, prediction, dataset, *, pair_index, inverse_prediction=None):
    moving = np.asarray(nib.load(label_path(case['moving'])).dataobj)
    fixed = np.asarray(nib.load(label_path(case['fixed'])).dataobj)
    if dataset == 'ixi':
        scores = ixi_direction(moving, fixed, prediction)
        if inverse_prediction is not None:
            scores['reverse'] = ixi_direction(fixed, moving, inverse_prediction)
    elif dataset == 'lpba40':
        if inverse_prediction is None:
            raise ValueError('LPBA40 paired evaluation requires both map directions')
        scores = lpba_direction(moving, fixed, prediction, pair_index=pair_index)
        scores['reverse'] = lpba_direction(fixed, moving, inverse_prediction, pair_index=pair_index)
        from medical.protocols.sitreg_reference import SITRegMetrics
        forward = mapping_tensor(prediction, fixed.shape)
        inverse = mapping_tensor(inverse_prediction, moving.shape)
        forward_ddf = (forward - identity(fixed.shape, forward.device)).movedim(-1, 0)
        inverse_ddf = (inverse - identity(moving.shape, inverse.device)).movedim(-1, 0)
        reference = SITRegMetrics(protocols()['lpba40_sitreg']['labels'])
        scores['inverse_consistency'] = {k: finite(v) for k, v in
            reference._compute_inverse_consistency_metrics(forward_ddf, inverse_ddf, '').items()}
        values = [scores['DSC']['mean'], scores['reverse']['DSC']['mean']]
        scores['bidirectional_DSC'] = float(np.mean(values)) if all(v is not None for v in values) else None
    else:
        raise ValueError('Unknown benchmark')
    scores.update(dataset=dataset, pair_index=pair_index,
                  primary_direction='warp moving labels onto fixed grid',
                  source_revision=protocols()['source_pins']['transmorph' if dataset == 'ixi' else 'sitreg']['commit'])
    return scores
