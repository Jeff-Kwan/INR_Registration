"""DIR-Lab adapter for canonical paired-SIREN fitting and native-grid evaluation."""
import tempfile
import time
import json
from pathlib import Path
import numpy as np
import nibabel as nib
import torch
from medical.registration.cuda_kernels import supported
from medical.configuration import validate_config
from medical.registration.training import fit_pair
from medical.registration.recovery import recover
from medical.registration.geometry import sample
from medical.dirlab_evaluation import read_points, score_landmarks


def run_case(case, config, device, *, checkpoint_callback=None):
    validate_config(config)
    fixed, moving = case['fixed'], case['moving']
    started = time.perf_counter()
    arrays, intensity, masks = [], [], []
    for item in (moving, fixed):
        data = item['data']
        lo, hi = float(data.min()), float(data.max())
        if hi <= lo:
            raise ValueError('Cannot normalize a constant volume')
        arrays.append(torch.from_numpy(((data-lo)/(hi-lo)).transpose(2, 1, 0).copy()).to(device))
        intensity.append({'min': lo, 'max': hi})
        image_path = Path(item['path'])
        stem = image_path.name.removesuffix('.gz').removesuffix('.nii')
        mask_image = nib.load(str(image_path.with_name(stem + '_lung_mask.nii.gz')))
        mask_data = np.asarray(mask_image.dataobj)
        if mask_data.shape != data.shape or not np.allclose(mask_image.affine, item['affine']):
            raise ValueError('Lung mask does not match its image grid')
        if not np.isfinite(mask_data).all() or not np.isin(mask_data, [0, 1]).all() or not mask_data.any():
            raise ValueError('Expected a nonempty binary lung mask')
        masks.append(torch.from_numpy(mask_data.transpose(2, 1, 0).copy()).to(device).bool())
    affine = moving['affine'].copy()
    affine[:3, :3] = affine[:3, :3][:, ::-1]
    preprocessing_seconds = time.perf_counter()-started
    started = time.perf_counter()
    pair = fit_pair(*arrays, config, affine=affine, moving_mask=masks[0], fixed_mask=masks[1],
                    report=lambda row: print(json.dumps({'training': row}), flush=True),
                    checkpoint_callback=checkpoint_callback)
    training_seconds = time.perf_counter()-started
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='paired-recovery-') as scratch:
        result = recover(pair, arrays[0].shape, config['recovery'], scratch,
                         chunk=config['runtime']['inference_chunk'],
                         compile_derivatives=config['runtime']['compile_derivatives'],
                         max_axis=config['runtime']['recovery_max_axis'])
        if result.inverse.is_cuda:
            torch.cuda.synchronize(result.inverse.device)
        recovery_seconds = time.perf_counter()-started
        evaluation_started = time.perf_counter()
        case_dir = Path(fixed['path']).parent
        fixed_points = read_points(case_dir/'T00_landmarks_voxel.csv')
        moving_points = read_points(case_dir/'T50_landmarks_voxel.csv')
        query = torch.as_tensor(fixed_points, dtype=result.inverse.dtype, device=device).reshape(1, 1, -1, 3)
        predicted = sample(result.inverse, query).reshape(-1, 3)
        scores = score_landmarks(fixed_points, moving_points, predicted, fixed['affine'], grid_shape=moving['data'].shape)
        if supported(result.inverse.device) and result.inverse.dtype == torch.float32:
            from medical.registration.dirlab_cuda import folding_fractions
            whole, lung = folding_fractions(result.inverse, masks[1])
        else:
            from medical.registration.metrics import jacobian_determinant
            grid = torch.stack(torch.meshgrid(*(torch.arange(n, device=device, dtype=torch.float32)
                                               for n in arrays[0].shape), indexing='ij')[::-1], -1)
            jac = jacobian_determinant((result.inverse-grid).cpu().numpy())
            whole = float(np.mean(jac <= 0))
            lung = float(np.mean(jac[masks[1].cpu().numpy()] <= 0))
        scores.update(folding_fraction_whole_grid=whole, folding_fraction_lung=lung, folding_fraction=lung)
        reverse_query = torch.as_tensor(moving_points, dtype=result.forward.dtype, device=device).reshape(1, 1, -1, 3)
        reverse_predicted = sample(result.forward, reverse_query).reshape(-1, 3)
        scores.update(reverse_T50_to_T00=score_landmarks(moving_points, fixed_points, reverse_predicted,
                      moving['affine'], grid_shape=fixed['data'].shape), primary_direction='T00_to_T50',
                      optimization_roi='shared crop; R231 masks select crop centers only',
                      seed=config['runtime']['seed'], recovery=result.diagnostics,
                      intensity_normalization={'method': 'independent_minmax', 'moving': intensity[0], 'fixed': intensity[1]})
    scores['training_execution_timing'] = getattr(pair, 'training_metadata', {}).get('execution_timings')
    scores['timings'] = {'preprocessing_seconds': preprocessing_seconds, 'training_seconds': training_seconds,
                         'recovery_seconds': recovery_seconds, 'evaluation_seconds': time.perf_counter()-evaluation_started,
                         'recovery_evaluation_seconds': time.perf_counter()-started}
    return {'metrics': scores, 'checkpoint': {'format_version': 1, 'model': pair.model_config,
            'training': getattr(pair, 'training_metadata', {}),
            'state': {k: v.detach().cpu() for k, v in pair.state_dict().items()}, 'config': config}}
