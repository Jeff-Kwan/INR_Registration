"""Shared scalar evaluation. Dense outputs never reach artifact writers."""
import numpy as np
import torch

from medical.registration.geometry import identity_grid, sample
from medical.registration.metrics import (image_metrics, jacobian_determinant,
                                         label_metrics, voxel_vector_to_world)


@torch.no_grad()
def endpoint_gate(pair, moving, fixed, settings, chunk=262144):
    reconstructions = [pair.decode(tuple(moving.shape), t, chunk).numpy() for t in (0., 1.)]
    if any(not np.isfinite(a).all() for a in reconstructions):
        raise RuntimeError('Non-finite endpoint reconstruction')
    metrics = [image_metrics(image.detach().cpu().numpy(), reconstruction)
               for image, reconstruction in zip((moving, fixed), reconstructions)]
    threshold = settings['min_endpoint_psnr_db']
    if settings.get('skip_psnr_gate', False):
        for metric, reconstruction in zip(metrics, reconstructions):
            metric.update(mean=float(reconstruction.mean(dtype=np.float64)),
                          variance=float(reconstruction.var(dtype=np.float64)))
        reason = 'explicitly skipped for this experiment'
        return {'endpoint_source': metrics[0], 'endpoint_target': metrics[1],
                'endpoint_quality_pass': None, 'threshold_db': threshold,
                'psnr_gate_skipped': True, 'psnr_gate_reason': reason}
    return {'endpoint_source': metrics[0], 'endpoint_target': metrics[1],
            'endpoint_quality_pass': all(m['psnr_infinite'] or m['psnr_db'] >= threshold for m in metrics),
            'threshold_db': threshold}


@torch.no_grad()
def evaluate(moving, fixed, result, affine, *, moving_label=None, fixed_label=None, hd95=True):
    if moving.shape != fixed.shape or moving.ndim != 3:
        raise ValueError('Medical evaluation needs matching 3D endpoints')
    grid = identity_grid(moving.shape, moving.device)
    warped = sample(moving, result.inverse).cpu().numpy()
    displacement = (result.inverse - grid).cpu().numpy()
    determinant = jacobian_determinant(displacement)
    composition = (sample(result.forward, result.inverse) - grid).cpu().numpy()
    world = voxel_vector_to_world(composition, affine)
    bound = result.inverse.new_tensor([n - 1 for n in reversed(moving.shape)])
    outside = ((result.inverse < 0) | (result.inverse > bound)).any(-1).float().mean()
    metrics = {'metric_version': 2, 'before': image_metrics(fixed.cpu().numpy(), moving.cpu().numpy()),
               'after': image_metrics(fixed.cpu().numpy(), warped),
               'folding_fraction': float(np.mean(determinant <= 0)),
               'jacobian_min': float(determinant.min()), 'jacobian_mean': float(determinant.mean()),
               'inverse_composition_mean_mm': float(np.linalg.norm(world, axis=-1).mean()),
               'sampling_outside_fraction': float(outside), 'labels': None}
    per_label = None
    if moving_label is not None and fixed_label is not None:
        label = torch.as_tensor(moving_label, dtype=torch.float32, device=moving.device)
        warped_label = sample(label, result.inverse, mode='nearest', padding_mode='zeros').cpu().numpy().astype(np.int32)
        spacing = np.linalg.norm(np.asarray(affine)[:3, :3], axis=0)
        metrics['labels'], per_label = label_metrics(fixed_label, warped_label, spacing, compute_hd95=hd95)
    return metrics, per_label
