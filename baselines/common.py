"""Input/evaluation glue; method equations stay in upstream source."""
from contextlib import contextmanager
import json
from pathlib import Path
import time
import random

import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates
import torch

from medical.dirlab_evaluation import read_points, score_landmarks
from .sources import provenance

ROOT = Path(__file__).resolve().parent


def validate_config(config, name):
    reference = json.loads((ROOT / 'configs' / f'{name}.json').read_text())
    if not isinstance(config, dict) or config.get('method') != name:
        raise ValueError(f'Expected a {name} configuration')
    if set(config) != set(reference) or not isinstance(config.get('native'), dict):
        raise ValueError(f'Expected config sections {sorted(reference)}')
    if set(config['native']) != set(reference['native']):
        raise ValueError(f'Expected native options {sorted(reference["native"])}')
    if config['evaluation'] not in ('dirlab','abdomenctct','oasis','neurite_oasis','acdc','lpba40','ixi'):
        raise ValueError('Unknown evaluation protocol')
    native = config['native']
    if name == 'fireants':
        if len(native['scales']) != len(native['iterations']) or not native['scales']:
            raise ValueError('FireANTs scales and iterations must be nonempty and have equal lengths')
        if any(x <= 0 for x in native['scales'] + native['iterations']):
            raise ValueError('FireANTs scales and iterations must be positive')
    elif name == 'convexadam':
        if native['selected_niter'] <= 0 or native['grid_sp'] <= 0 or native['grid_sp_adam'] <= 0:
            raise ValueError('ConvexAdam iteration count and grid spacings must be positive')
        if native['use_mask']:
            raise ValueError('This connected MIND preset uses the native whole-image path')
    elif name == 'ccidir':
        if config['inference_order'] not in (1,2):
            raise ValueError('ccIDIR inference_order must be 1 or 2')
        if native['epochs'] <= 0 or native['batch_size'] <= 0:
            raise ValueError('ccIDIR epoch count and batch size must be positive')
    elif name == 'demons':
        if native['iterations'] <= 0 or native['standard_deviations'] <= 0:
            raise ValueError('Demons iterations and smoothing must be positive')
    elif name == 'nodeo':
        if native['epoches'] < 2 or native['ds'] < 0:
            raise ValueError('NODEO requires at least two epochs and nonnegative decoder levels')
    elif name == 'dnvf':
        if native['optimization_epoch'] < 1 or native['downsample_factor'] < 1:
            raise ValueError('DNVF requires positive iteration count and downsample factor')
    elif name == 'nir':
        if native['model_type'] not in {'gaussian_sine', 'ode_gaussian_sine'}:
            raise ValueError('Connected NIR variants: gaussian_sine or ode_gaussian_sine')
        if native['batch_size'] <= 0 or native['num_epochs'] <= 0:
            raise ValueError('NIR epoch count and batch size must be positive')
        if native['sample_mtd'] not in {'downsize_center', 'downsize_random', 'downsize_unified_random', 'patch', 'mae'}:
            raise ValueError('NIR sampler must produce spatial tensors for the released model')
        if not set(native['loss_type'].split('+')) <= {'image_mse', 'l1', 'ncc', 'discrete_jacobian'}:
            raise ValueError('Unsupported native NIR loss')
    return config


def adjacent(volume, suffix):
    path = Path(volume['path'])
    stem = path.name.removesuffix('.gz').removesuffix('.nii')
    return path.with_name(stem + suffix)


def load_mask(volume):
    path = adjacent(volume, '_lung_mask.nii.gz')
    if not path.is_file():
        raise ValueError(f'Native ccIDIR requires separate fixed and moving masks; missing {path}')
    image = nib.load(path)
    data = np.asarray(image.dataobj)
    if data.shape != volume['data'].shape or not np.allclose(image.affine, volume['affine']):
        raise ValueError(f'Mask grid differs from its image: {path}')
    if not np.isfinite(data).all() or not np.isin(data, [0, 1]).all() or not np.any(data):
        raise ValueError(f'Expected a finite nonempty binary mask: {path}')
    return data


def landmark_inputs(case):
    paths = [adjacent(case[key], '_landmarks_voxel.csv') for key in ('fixed', 'moving')]
    if not all(p.is_file() for p in paths):
        raise ValueError(f'Current baseline evaluator requires both DIR-Lab landmark CSVs: {paths}')
    points = [read_points(p) for p in paths]
    if any(p.shape != (300, 3) or not np.isfinite(p).all() for p in points):
        raise ValueError('Expected finite 300-by-3 DIR-Lab landmark arrays')
    return points


def preflight(case, config, name):
    validate_config(config, name)
    if config['evaluation'] in ('neurite_oasis','acdc','lpba40','ixi'):
        from medical.comparison_metrics import validate_case
        validate_case(case, config['evaluation'])
    elif config['evaluation'] in ('abdomenctct','oasis'):
        from medical.abdomen_evaluation import validate_case
        validate_case(case,config["evaluation"])
    else:
        landmark_inputs(case)  # Validate before any native optimizer starts.


@contextmanager
def execution(device, seed, cuda_required=False):
    random.seed(seed)
    device = torch.device(device)
    if device.type not in ('cuda', 'cpu') or cuda_required and device.type != 'cuda':
        raise ValueError('This native implementation requires a CUDA device')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable in the active Python environment')
        with torch.cuda.device(device):
            torch.manual_seed(seed)
            np.random.seed(seed)
            yield device
    else:
        torch.manual_seed(seed)
        np.random.seed(seed)
        yield device


def sample_xyz_field(field, points):
    """Field has NIfTI array order XYZ and vector component order XYZ."""
    return np.stack([map_coordinates(field[..., i], points.T, order=1, mode='nearest')
                     for i in range(3)], axis=-1)


def normalized_grid_predictions(grid, points, shape):
    """PyTorch grid [1,Z,Y,X,XYZ] -> continuous NIfTI voxel coordinates."""
    grid = grid[0].detach().cpu().float().numpy().transpose(2, 1, 0, 3)
    return (sample_xyz_field(grid, points) + 1) * ((np.asarray(shape) - 1) / 2)


def finish(name, case, config, predicted, checkpoint, started, **details):
    if config['evaluation'] in ('neurite_oasis','acdc','lpba40','ixi'):
        from medical.comparison_metrics import evaluate
        metrics=evaluate(case,predicted,config['evaluation'])
    elif config['evaluation'] in ('abdomenctct','oasis'):
        from medical.abdomen_evaluation import evaluate
        metrics=evaluate(case,predicted,config["evaluation"])
    else:
        fixed, moving = landmark_inputs(case)
        metrics = score_landmarks(fixed, moving, predicted, case['fixed']['affine'])
        shape = np.asarray(case['moving']['data'].shape)
        metrics['prediction_outside_voxel_grid'] = int(np.any((predicted < 0) | (predicted > shape - 1), axis=1).sum())
        metrics.update(folding_fraction=None,folding_status='not_evaluated')
    alignment=json.loads((ROOT/'audit.json').read_text())[name]
    metrics.update(alignment=alignment, method=name, seed=config['seed'], source=provenance(name), elapsed_seconds=time.monotonic()-started, **details)
    checkpoint = {'method': name, 'source': provenance(name), 'config': config, **checkpoint}
    return {'metrics': metrics, 'checkpoint': checkpoint}


def cpu_state(module):
    return {key: value.detach().cpu() for key, value in module.state_dict().items()}


def predict_for_case(case,config,predict):
    if config['evaluation'] != 'dirlab':
        return predict
    points,_=landmark_inputs(case)
    return predict(points)
