"""Frozen DIR-Lab landmark and crop diagnostics. No fitting or saved dense maps."""
import argparse
import json
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from medical.dirlab_evaluation import read_points, score_landmarks
from medical.models.siren import PairedSiren
from medical.registration.geometry import sample
from medical.registration.image_sampling import sample_points
from medical.registration.recovery import recover
from medical.registration.symmetric_registration import SymmetricPairRegistration, sample_crop_indices


BINS_MM = ((0, 2), (2, 5), (5, 10), (10, np.inf))


def motion_errors(start, target, predicted, basis):
    """Decompose endpoint error along observed motion in physical millimetres."""
    actual = (target - start) @ basis.T
    estimate = (predicted - start) @ basis.T
    residual = estimate - actual
    magnitude = np.linalg.norm(actual, axis=-1)
    unit = np.divide(actual, magnitude[:, None], out=np.zeros_like(actual),
                     where=magnitude[:, None] > 0)
    along = np.sum(residual * unit, axis=-1)
    perpendicular = np.linalg.norm(residual - along[:, None] * unit, axis=-1)
    def report(indices):
        nonzero = indices & (magnitude > 0)
        return {
            'count': int(indices.sum()), 'nonzero_motion_count': int(nonzero.sum()),
            'initial_motion_mm': float(magnitude[indices].mean()) if indices.any() else None,
            'continuous_tre_mm': float(np.linalg.norm(residual[indices], axis=-1).mean()) if indices.any() else None,
            'signed_along_error_mm': float(along[nonzero].mean()) if nonzero.any() else None,
            'perpendicular_error_mm': float(perpendicular[nonzero].mean()) if nonzero.any() else None,
            'signed_z_error_mm': float(residual[indices, 2].mean()) if indices.any() else None,
        }
    return {'all': report(np.ones(len(start), dtype=bool)),
            'motion_bins': {f'{low:g}-{high:g}': report((magnitude >= low) & (magnitude < high))
                            for low, high in BINS_MM},
            'per_landmark_motion_mm': magnitude.tolist(),
            'per_landmark_tre_mm': np.linalg.norm(residual, axis=-1).tolist()}


def local_correlation(image_a, image_b, starts, true_ends, predicted_ends, spacing_xyz):
    """Compare 9-mm physical endpoint windows with identical edge extension."""
    offsets = np.stack(np.meshgrid(*(np.linspace(-4.5, 4.5, 10, dtype=np.float32) for _ in range(3)),
                                   indexing='ij'), -1).reshape(-1, 3)
    offsets = torch.from_numpy(offsets).to(starts.device) / torch.as_tensor(
        spacing_xyz, dtype=starts.dtype, device=starts.device)
    def windows(image, centers):
        return sample_points(image, (centers[:, None] + offsets[None]).reshape(-1, 3)).reshape(len(centers), -1)
    source = windows(image_a, starts)
    target = windows(image_b, true_ends)
    predicted = windows(image_b, predicted_ends)
    source = source - source.mean(1, keepdim=True)
    target = target - target.mean(1, keepdim=True)
    predicted = predicted - predicted.mean(1, keepdim=True)
    source_var = source.square().mean(1)
    true_var = target.square().mean(1)
    pred_var = predicted.square().mean(1)
    low_var = (source_var < 1e-6) | (true_var < 1e-6) | (pred_var < 1e-6)
    def corr(values, variance):
        return (source * values).mean(1) / (source_var * variance).sqrt().clamp_min(1e-8)
    return corr(target, true_var).cpu().numpy(), corr(predicted, pred_var).cpu().numpy(), low_var.cpu().numpy()


def crop_coverage(branch, start, target, count, batch_size, generator, *, fraction_range=None):
    """Replay ordinary crop centers and count landmark starts in the particle region."""
    starts = torch.as_tensor(start, device=branch.moving.device, dtype=branch.moving.dtype)
    targets = torch.as_tensor(target, device=branch.moving.device, dtype=branch.moving.dtype)
    covered = torch.zeros(len(starts), dtype=torch.int32, device=starts.device)
    outside_target = torch.zeros_like(covered)
    nearest_target_cells = torch.full((len(starts),), float('inf'), device=starts.device)
    margin = (1 - branch.particle_sampling_fraction) / 2
    for draw in range(count):
        shape = None
        if fraction_range is not None:
            first, last = fraction_range
            progress = draw / (count - 1) if count > 1 else 0.5
            fraction = (1 - progress) * first + progress * last
            shape = tuple(max(2, int(np.ceil(n * fraction))) for n in branch.moving.shape)
        sample_crop_indices(branch, batch_size, generator, crop_shape=shape)
        lower = branch.origin + margin * branch.domain_extent
        upper = branch.origin + (1 - margin) * branch.domain_extent
        inside = ((starts >= lower) & (starts <= upper)).all(-1)
        covered += inside.int()
        target_outside = ((targets < branch.origin) | (targets > branch.origin + branch.domain_extent)).any(-1)
        outside_target += (inside & target_outside).int()
        cell_width = branch.domain_extent / branch.grid_size
        distance = torch.maximum(branch.origin - targets,
                                 targets - branch.origin - branch.domain_extent).clamp_min(0)
        nearest_target_cells = torch.minimum(nearest_target_cells,
            torch.where(inside, (distance / cell_width).amax(-1), float('inf')))
    nearest_target_cells = torch.where(torch.isfinite(nearest_target_cells), nearest_target_cells,
                                       torch.full_like(nearest_target_cells, float('nan')))
    return {'crop_draws': count, 'start_region_counts': covered.cpu().tolist(),
            'target_outside_crop_counts': outside_target.cpu().tolist(),
            'nearest_target_outside_cells': [float(value) if np.isfinite(value) else None
                                             for value in nearest_target_cells.cpu().tolist()]}


def run_case(case, checkpoint_root, output_root, device, crop_draws, *, allow_missing_metrics=False):
    case_name = f'case{case:02d}'
    case_dir = Path('Data/DIRLab4DCT_IDIR') / case_name
    payload = torch.load(checkpoint_root / case_name / 'checkpoint.pt',
                         map_location='cpu', weights_only=True)
    saved_path = checkpoint_root / case_name / 'metrics.json'
    saved = json.loads(saved_path.read_text()) if saved_path.exists() else None
    if saved is None and not allow_missing_metrics:
        raise FileNotFoundError(saved_path)
    config = payload['config']
    if config.get('spatial_units', 'voxel') != 'voxel' or 'working_grid' in config:
        raise ValueError('This diagnostic expects the current native voxel-unit checkpoint')
    model = PairedSiren(3, **payload['model']).to(device).eval().requires_grad_(False)
    model.load_state_dict(payload['state'], strict=True)
    arrays, masks = [], []
    for phase in ('T50', 'T00'):
        volume = nib.load(str(case_dir / f'{phase}.nii.gz'))
        image = np.asarray(volume.dataobj, dtype=np.float32)
        image = (image - image.min()) / (image.max() - image.min())
        arrays.append(torch.from_numpy(image.transpose(2, 1, 0).copy()).to(device))
        mask = np.asarray(nib.load(str(case_dir / f'{phase}_lung_mask.nii.gz')).dataobj)
        masks.append(torch.from_numpy(mask.transpose(2, 1, 0).copy()).to(device).nonzero().float())
    fixed_points = read_points(case_dir / 'T00_landmarks_voxel.csv')
    moving_points = read_points(case_dir / 'T50_landmarks_voxel.csv')
    basis = volume.affine[:3, :3]
    spacing = np.linalg.norm(basis, axis=0)
    affine = volume.affine.copy()
    affine[:3, :3] = affine[:3, :3][:, ::-1]
    with torch.no_grad(), tempfile.TemporaryDirectory(prefix='dirlab-gap-') as scratch:
        recovered = recover(model, arrays[0].shape, config['recovery'], scratch,
                            chunk=config['runtime']['inference_chunk'],
                            compile_derivatives=config['runtime']['compile_derivatives'],
                            max_axis=config['runtime'].get('recovery_max_axis'))
        predictions = {}
        if allow_missing_metrics:
            predictions['probe_metadata'] = {'saved_metrics_parity_checked': saved is not None,
                                             'recovery': recovered.diagnostics}
        for key, starts, targets, mapping, source_image, target_image in (
            ('forward', fixed_points, moving_points, recovered.inverse, arrays[1], arrays[0]),
            ('reverse', moving_points, fixed_points, recovered.forward, arrays[0], arrays[1])):
            start_tensor = torch.as_tensor(starts, device=device, dtype=torch.float32)
            target_tensor = torch.as_tensor(targets, device=device, dtype=torch.float32)
            predicted_tensor = sample(mapping, start_tensor.reshape(1, 1, -1, 3)).reshape(-1, 3)
            predicted = predicted_tensor.cpu().numpy()
            scores = score_landmarks(starts, targets, predicted, volume.affine)
            if saved is not None:
                saved_scores = saved if key == 'forward' else saved['reverse_T50_to_T00']
                for score in ('continuous_tre_mm', 'idir_rounded_tre_mm'):
                    delta = abs(scores[score]['mean_mm'] - saved_scores[score]['mean_mm'])
                    if delta > .01:
                        raise RuntimeError(f'{case_name} {key} {score} differs by {delta:.4f} mm')
            motion = motion_errors(starts, targets, predicted, basis)
            true_corr, pred_corr, low_var = local_correlation(
                source_image, target_image, start_tensor, target_tensor, predicted_tensor, spacing)
            bin_masks = [(np.linalg.norm((targets - starts) @ basis.T, axis=1) >= low) &
                         (np.linalg.norm((targets - starts) @ basis.T, axis=1) < high)
                         for low, high in BINS_MM]
            motion['image_evidence'] = {
                'low_variance_count': int(low_var.sum()),
                'true_minus_predicted_correlation_by_motion_bin': {
                    f'{low:g}-{high:g}': {'count': int((mask & ~low_var).sum()),
                        'mean': float((true_corr - pred_corr)[mask & ~low_var].mean())
                        if (mask & ~low_var).any() else None}
                    for (low, high), mask in zip(BINS_MM, bin_masks)},
                'per_landmark_true_correlation': true_corr.tolist(),
                'per_landmark_predicted_correlation': pred_corr.tolist(),
                'per_landmark_low_variance': low_var.tolist()}
            predictions[key] = {'scores': scores, 'motion': motion}
        if list(Path(scratch).iterdir()):
            raise RuntimeError('Recovery scratch was not cleaned')
        branch = SymmetricPairRegistration(*arrays, affine,
            seed=config['runtime']['seed'], c=config['objective']['c'], eta=config['objective']['eta'],
            compile_derivatives=False, **{k: v for k, v in config['objective']['registration'].items()
                                          if k != 'weight'})
        fraction_range = None
        if 'final_crop_fraction' in config['training']:
            fraction_range = (config['objective']['registration']['crop_fraction'],
                              config['training']['final_crop_fraction'])
        branch.crop_center_sites = masks
        for key, starts, targets in (('forward', fixed_points, moving_points),
                                     ('reverse', moving_points, fixed_points)):
            # Both directions use the same replayed crop sequence.
            branch_generator = torch.Generator(device=device).manual_seed(config['runtime']['seed'])
            coverage = crop_coverage(
                branch, starts, targets, crop_draws, config['training']['batch_size'], branch_generator,
                fraction_range=fraction_range)
            counts = np.asarray(coverage['start_region_counts'])
            magnitudes = np.asarray(predictions[key]['motion']['per_landmark_motion_mm'])
            errors = np.asarray(predictions[key]['motion']['per_landmark_tre_mm'])
            coverage['tre_by_motion_and_coverage'] = {}
            for lower, upper in BINS_MM:
                in_bin = (magnitudes >= lower) & (magnitudes < upper)
                if not in_bin.any():
                    continue
                median = float(np.median(counts[in_bin]))
                groups = {'lower_coverage': in_bin & (counts < median),
                          'higher_coverage': in_bin & (counts >= median)}
                coverage['tre_by_motion_and_coverage'][f'{lower:g}-{upper:g}'] = {
                    label: {'count': int(selected.sum()),
                            'mean_tre_mm': float(errors[selected].mean()) if selected.any() else None,
                            'mean_crop_count': float(counts[selected].mean()) if selected.any() else None}
                    for label, selected in groups.items()}
            predictions[key]['crop_coverage'] = coverage
    destination = output_root / case_name
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'diagnostics.json').write_text(json.dumps(predictions, indent=2, allow_nan=False) + '\n')
    print(case_name, 'forward TRE', predictions['forward']['scores']['idir_rounded_tre_mm']['mean_mm'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', type=int, nargs='+', default=[1, 3, 4, 5])
    parser.add_argument('--checkpoint-root', type=Path,
                        default=Path('results/dirlab_quick4_native_cropcenter_20261113'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--crop-draws', type=int, default=256)
    args = parser.parse_args()
    if args.crop_draws < 1 or args.output.exists():
        parser.error('Crop draws must be positive and output must be a fresh path')
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=False)
    for case in args.cases:
        run_case(case, args.checkpoint_root, args.output, torch.device(args.device), args.crop_draws)


if __name__ == '__main__':
    main()
