"""In-memory scoring for the selected ACDC and Neurite-OASIS protocols."""
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from medical.learn2reg_evaluation import evaluate_labels, json_metric


def acdc_metrics(fixed, warped):
    from medpy.metric.binary import hd95
    # Match the pinned author's set-intersection and first-element omission.
    labels = list(set(np.unique(warped)) & set(np.unique(fixed)))[1:]

    def score(codes):
        dice, distances = [], []
        for code in codes:
            truth, prediction = fixed == code, warped == code
            total = truth.sum() + prediction.sum()
            dice.append(2. * np.count_nonzero(truth & prediction) / total if total else np.nan)
            distances.append(hd95(prediction, truth, voxelspacing=(1.5, 1.5, 3.15))
                             if prediction.any() and truth.any() else np.nan)
        return {'labels': [int(k) for k in codes], 'scored_labels': len(codes),
                'DSC': {'mean': json_metric(np.mean(dice)) if dice else None,
                        'detailed': [json_metric(v) for v in dice]},
                'HD95': {'mean': json_metric(np.mean(distances)) if distances else None,
                         'detailed': [json_metric(v) for v in distances], 'units': 'mm'}}
    result = score(labels)
    result['fixed_label_diagnostics'] = score([1, 2, 3])
    result['missing_label_policy'] = 'Author intersection drops its first entry; fixed-label empty surfaces produce null HD95'
    return result


def evaluate(case, prediction, dataset, root, pair):
    labels = {k: np.asarray(nib.load(root / pair[f'{k}_label']).dataobj) for k in ('moving', 'fixed')}
    if dataset == 'neurite_oasis':
        scores = evaluate_labels(labels['fixed'], labels['moving'], prediction, list(range(1, 36)))
        scores.update(dataset=dataset, evaluation_convention='Repository Learn2Reg-style; GaussianDIR/FireANTs metric parity unestablished')
        return scores
    mapping = torch.as_tensor(prediction).detach()
    shape = labels['fixed'].shape
    if mapping.shape != (*shape, 3) or not torch.isfinite(mapping).all():
        raise ValueError('Expected a finite fixed-to-moving XYZ voxel map')
    extent = mapping.new_tensor(np.asarray(shape) - 1)
    grid = (2 * mapping / extent - 1)[..., [2, 1, 0]][None]
    moving = torch.as_tensor(labels['moving'].astype(np.float32), device=mapping.device)[None, None]
    warped = F.grid_sample(moving, grid, mode='nearest', padding_mode='zeros', align_corners=True)[0, 0].cpu().numpy()
    scores = acdc_metrics(labels['fixed'], warped)
    # det(d phi / dx) on the original prepared grid; explicit estimator adaptation.
    array = mapping.cpu().numpy().astype(np.float64)
    jacobian = np.stack([np.stack(np.gradient(array[..., component]), axis=-1) for component in range(3)], axis=-2)
    folding = float(np.mean(np.linalg.det(jacobian) < 0))
    scores.update(dataset=dataset, folding_fraction=folding, folding_percent=100 * folding,
                  folding_status='Full-grid finite-difference detJ<0; adaptation of author analytical Jacobian',
                  metric_revision='3ac12e200159811eafb6192f3f8db3afef2a9d1a')
    return scores
