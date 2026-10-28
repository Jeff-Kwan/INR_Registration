"""LPBA registration through paired image SIRENs, analytical LK, and 3D optical flow.

Edit medical/configs/lpba_pdlk.json before running. No training runs on import.
"""
import argparse
import copy
import csv
import gc
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import re
import tempfile
import time

import nibabel as nib
import numpy as np
import torch
from tqdm.auto import tqdm

from medical.models.siren3d import Siren3D, coordinates_from_indices, decode
from medical.registration.configuration import load_config
from medical.registration.flow3d import (estimate_velocity, estimate_velocity_from_derivatives,
                                         identity_grid, integrate_interval, sample)
from medical.registration.io import (LABEL_METRIC_DEFINITION, image_metrics, jacobian_determinant,
                                     label_metrics, load_image, load_label, require_same_grid,
                                     save_volume, voxel_vector_to_world, write_json)
from medical.registration.pdlk import (analytical_siren_derivatives, lk_temporal_loss,
                                       sample_regularization_points)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / 'configs' / 'lpba_pdlk.json'


def resolve_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def resolve_dataset_path(config, value):
    """Resolve a dataset member beneath the configured local dataset root."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    root = Path(config['dataset']['root']).expanduser()
    if not root.is_absolute():
        root = ROOT / root
    return root / path


def validate_config(config):
    config.get('runtime',{}).setdefault('compile_derivatives',True)
    config.get('output', {}).setdefault('save_volumes', True)
    config.get('output', {}).setdefault('save_checkpoints', True)
    if isinstance(config.get('lk'),dict):
        config['lk'].setdefault('mode','pointwise')
        config['lk'].setdefault('affine_beta', None)
        config['lk'].setdefault('affine_rho', None)
        config['lk'].setdefault('points',4096)
        config['lk'].setdefault('weight_schedule','constant')
        config['lk'].setdefault('differentiate_solve',False)
    if 'h1' in config:
        raise ValueError('H1 configs are historical; use the pointwise PD-LK configuration')
    # Read older frame-based configs with their original recovery behavior.
    recovery = config.get('recovery', {})
    if 'frames' in recovery and 'intervals' not in recovery:
        frames = recovery.pop('frames')
        if type(frames) is not int or frames < 2:
            raise ValueError('recovery.frames must be an integer >=2')
        recovery.update(method='finite', intervals=frames-1)
    sections = {
        'dataset': {'root', 'fixed', 'moving_dir', 'label_dir', 'fixed_label', 'subjects', 'require_labels'},
        'runtime': {'device', 'seed', 'inference_chunk', 'compile_derivatives'},
        'model': {'width', 'layers', 'omega'},
        'training': {'steps', 'batch_size', 'lr', 'lr_final_factor', 'log_every', 'save_every', 'grad_clip'},
        'lk': {'weight', 'gradient_weight', 'ridge', 'mode', 'points', 'weight_schedule', 'differentiate_solve', 'affine_beta', 'affine_rho'},
        'recovery': {'method', 'intervals', 'smoothness', 'ridge', 'iterations', 'tolerance', 'substeps'},
        'evaluation': {'min_endpoint_psnr_db', 'hd95'},
        'output': {'root', 'save_frames', 'save_velocities', 'save_volumes', 'save_checkpoints'},
    }
    if set(config) != set(sections):
        raise ValueError(f'Expected configuration sections: {list(sections)}')
    for section, keys in sections.items():
        if not isinstance(config[section], dict) or set(config[section]) != keys:
            raise ValueError(f'Expected exactly these {section} keys: {sorted(keys)}')
    integers = {
        'runtime': {'seed': 0, 'inference_chunk': 0}, 'model': {'width': 1, 'layers': 1},
        'training': {'steps': 1, 'batch_size': 1, 'log_every': 1, 'save_every': 1},
        'lk': {'points': 1},
        'recovery': {'intervals': 1, 'iterations': 1, 'substeps': 1},
    }
    for section, settings in integers.items():
        for key, minimum in settings.items():
            value = config[section][key]
            if type(value) is not int or value < minimum:
                raise ValueError(f'{section}.{key} must be an integer >= {minimum}')
    for section, keys, strict in [
        ('model', ['omega'], True), ('training', ['lr', 'lr_final_factor'], True),
        ('recovery', ['smoothness', 'ridge', 'tolerance'], True),
        ('training', ['grad_clip'], False), ('lk', ['weight', 'gradient_weight'], False),
        ('lk', ['ridge'], True),
    ]:
        for key in keys:
            value = config[section][key]
            if type(value) not in (int, float) or not math.isfinite(value) or (value <= 0 if strict else value < 0):
                raise ValueError(f'Invalid {section}.{key}: {value}')
    if config['recovery']['method'] not in ('analytical', 'finite'):
        raise ValueError('recovery.method must be analytical or finite')
    if config['lk']['weight_schedule'] not in ('constant','cosine','linear'):
        raise ValueError('lk.weight_schedule must be constant, cosine or linear')
    if config['lk']['mode'] != 'pointwise':
        raise ValueError('Only pointwise PD-LK is maintained; patch LK is historical')
    beta = config['lk']['affine_beta']
    ratio = config['lk']['affine_rho']
    if ratio is not None and (type(ratio) not in (int, float) or not math.isfinite(ratio) or ratio < 0
                              or config['lk']['mode'] != 'pointwise' or beta is not None):
        raise ValueError('lk.affine_rho must be finite nonnegative in pointwise mode; do not also specify affine_beta')
    if beta is not None and (type(beta) not in (int, float) or not math.isfinite(beta) or beta <= 0
                             or config['lk']['mode'] != 'pointwise'):
        raise ValueError('lk.affine_beta must be null (standard PD-LK) or positive finite in pointwise mode')
    if not 0 < config['training']['lr_final_factor'] <= 1:
        raise ValueError('training.lr_final_factor must lie in (0,1]')
    if not math.isfinite(config['evaluation']['min_endpoint_psnr_db']):
        raise ValueError('Endpoint PSNR threshold must be finite')
    subjects = config['dataset']['subjects']
    if not isinstance(subjects, list) or not all(isinstance(s, str) and re.fullmatch(r'S\d{2}', s) for s in subjects):
        raise ValueError('dataset.subjects must be a list such as ["S02", "S03"]')
    for section, key in [('lk','differentiate_solve'), ('runtime','compile_derivatives'), ('dataset', 'require_labels'), ('evaluation', 'hd95'),
                         ('output', 'save_frames'), ('output', 'save_velocities'), ('output', 'save_volumes'), ('output', 'save_checkpoints')]:
        if type(config[section][key]) is not bool:
            raise ValueError(f'{section}.{key} must be true or false')
    for section, key in [('dataset', k) for k in ('root', 'fixed', 'moving_dir', 'label_dir', 'fixed_label')] + [('runtime', 'device'), ('output', 'root')]:
        if not isinstance(config[section][key], str) or not config[section][key]:
            raise ValueError(f'{section}.{key} must be a nonempty string')


def discover_subjects(config):
    directory = resolve_dataset_path(config, config['dataset']['moving_dir'])
    fixed = resolve_dataset_path(config, config['dataset']['fixed']).resolve()
    requested, subjects = set(config['dataset']['subjects']), {}
    for path in sorted(directory.glob('*.nii*')):
        match = re.match(r'(S\d{2})(?:\.|_)', path.name)
        if not match or path.resolve() == fixed:
            continue
        subject = match.group(1)
        if requested and subject not in requested:
            continue
        if subject in subjects:
            raise ValueError(f'Multiple MRI volumes found for {subject}')
        subjects[subject] = path
    if requested - subjects.keys():
        raise ValueError(f'Subjects not found: {sorted(requested - subjects.keys())}')
    if not subjects:
        raise ValueError(f'No moving subjects found in {directory}')
    return list(subjects.items())


def label_path(config, subject):
    candidates = sorted(resolve_dataset_path(config, config['dataset']['label_dir']).glob(f'{subject}*label*.nii*'))
    if len(candidates) > 1:
        raise ValueError(f'Multiple label files for {subject}')
    return candidates[0] if candidates else None


def duration(seconds):
    seconds = max(0, int(seconds))
    return f'{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}'


class RunETA:
    """Forecast the whole remaining run using measured costs of each stage."""
    def __init__(self, count, config):
        self.count, self.index = count, 0
        intervals = config['recovery']['intervals']
        decode_count = intervals+1 if config['recovery']['method'] == 'finite' or config['output']['save_frames'] else 2
        self.units = {'train': config['training']['steps'], 'decode': decode_count,
                      'flow': intervals, 'inverse': intervals, 'metrics': 1}
        self.costs, self.completed = {}, {}

    def begin_pair(self, index):
        self.index, self.completed = index, {}

    def update(self, stage, done, elapsed):
        self.completed[stage] = done
        if done:
            self.costs[stage] = elapsed / done

    def text(self):
        remaining = sum(self.costs.get(stage, 0) * (units - self.completed.get(stage, 0)
                        + units * (self.count - self.index - 1)) for stage, units in self.units.items())
        return duration(remaining) + ('' if len(self.costs) == len(self.units) else ' (partial; calibrating)')


def cpu_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def save_checkpoint(path, source, target, initial, optimizer, step, config):
    temporary = path.with_suffix('.tmp')
    torch.save({'format_version': 1, 'step': step, 'architecture': 'Siren3D',
                'source_subject': 'moving', 'target_subject': 'fixed',
                'initial': initial, 'source': cpu_state(source), 'target': cpu_state(target),
                'optimizer': optimizer.state_dict(), 'config': config}, temporary)
    temporary.replace(path)


def regularization_weight(settings,step,steps):
    """Optional rise from zero to configured weight across optimizer minibatches."""
    if settings.get('weight_schedule','constant') == 'constant':
        return settings['weight']
    if settings['weight_schedule'] not in ('cosine','linear'):
        raise ValueError('Unknown LK weight schedule')
    fraction=(step-1)/max(steps-1,1) if steps>1 else 1.
    factor=fraction if settings['weight_schedule']=='linear' else .5*(1-math.cos(math.pi*fraction))
    return settings['weight']*factor


def fit_pair(source_data, target_data, config, output, subject, eta):
    settings, lk = config['training'], config['lk']
    source = Siren3D(**config['model']).to(source_data.device)
    target = copy.deepcopy(source)
    initial = cpu_state(source)
    parameters = list(source.parameters()) + list(target.parameters())
    optimizer = torch.optim.Adam(parameters, lr=settings['lr'], fused=source_data.is_cuda)
    shape, voxel_count = source_data.shape, source_data.numel()
    started = time.perf_counter()
    columns = ['step', 'total', 'source_mse', 'target_mse', 'lk', 'weighted_lk',
               'time', 'lr', 'elapsed_seconds', 'lk_weight']
    with (output / 'training.csv').open('w', newline='') as handle, tqdm(
            total=settings['steps'], desc=f'{subject} fit', position=1,
            dynamic_ncols=True, leave=False) as bar:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for step in range(1, settings['steps'] + 1):
            lr = settings['lr'] * settings['lr_final_factor'] ** ((step - 1) / settings['steps'])
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            indices = torch.randint(voxel_count, (settings['batch_size'],), device=source_data.device)
            coordinates = coordinates_from_indices(indices, shape)
            source_loss = (source(coordinates).flatten() - source_data.flatten()[indices]).square().mean()
            target_loss = (target(coordinates).flatten() - target_data.flatten()[indices]).square().mean()
            lk_loss = source_loss.new_zeros(())
            # One t is shared across independent spatial queries on the trajectory.
            time_point = torch.rand((), device=source_data.device, dtype=source_data.dtype)
            if lk['weight']:
                queries = sample_regularization_points(shape, lk, source_data)
                lk_loss = lk_temporal_loss(source, target, queries, time_point, shape,
                    lk['ridge'], gradient_weight=lk['gradient_weight'],
                    affine_beta=lk.get('affine_beta'),
                    affine_rho=lk.get('affine_rho'),
                    compile_derivatives=config['runtime'].get('compile_derivatives',True),
                    differentiate_solve=lk['differentiate_solve'])
            lk_weight = regularization_weight(lk,step,settings['steps'])
            loss = source_loss + target_loss + lk_weight * lk_loss
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite training loss for {subject} at step {step}')
            loss.backward()
            if settings['grad_clip']:
                torch.nn.utils.clip_grad_norm_(parameters, settings['grad_clip'])
            optimizer.step()
            if step % settings['log_every'] == 0 or step == settings['steps']:
                total, src, tgt, patch = torch.stack(
                    [loss.detach(), source_loss.detach(), target_loss.detach(), lk_loss.detach()]).cpu().tolist()
                elapsed = time.perf_counter() - started
                writer.writerow(dict(zip(columns, [step, total, src, tgt, patch, lk_weight * patch,
                                                     float(time_point), lr, elapsed, lk_weight])))
                handle.flush()
                eta.update('train', step, elapsed)
                bar.set_postfix(src=f'{src:.4g}', tgt=f'{tgt:.4g}', lk=f'{patch:.4g}', lk_weight=f'{lk_weight:.4g}',
                                run_ETA=eta.text(), refresh=False)
                bar.update(step - bar.n)
            if config['output'].get('save_checkpoints', True) and (step % settings['save_every'] == 0 or step == settings['steps']):
                save_checkpoint(output / 'models.pt', source, target, initial, optimizer, step, config)
    source.eval()
    target.eval()
    return source, target


def prepare_recovery(source, target):
    """Drop training-only state before allocating full-volume activations."""
    for model in (source, target):
        model.zero_grad(set_to_none=True)
    gc.collect()
    device = next(source.parameters()).device
    if device.type == 'cuda':
        with torch.cuda.device(device):
            torch.cuda.empty_cache()


def stage_progress(items, desc, stage, eta):
    started = time.perf_counter()
    with tqdm(items, desc=desc, position=1, dynamic_ncols=True, leave=False) as bar:
        for index, item in enumerate(bar, 1):
            yield item
            eta.update(stage, index, time.perf_counter() - started)
            bar.set_postfix(run_ETA=eta.text(), refresh=False)


@torch.no_grad()
def recovery_derivatives(source, target, shape, time_point, chunk=0, compile_derivatives=True):
    """Exact g per voxel and q per unit time; chunks only bound model queries."""
    if chunk < 0:
        raise ValueError('inference_chunk must be nonnegative')
    parameter = next(source.parameters())
    total = math.prod(shape)
    gradient = parameter.new_empty((total, 3))
    temporal = parameter.new_empty(total)
    scale = [2/(n-1) for n in reversed(shape)]
    step = chunk or total
    for start in range(0, total, step):
        end = min(start+step, total)
        coordinates = coordinates_from_indices(torch.arange(start, end, device=parameter.device), shape)
        g, q = analytical_siren_derivatives(source, target, coordinates,
                                            parameter.new_tensor(time_point), scale,
                                            compile_derivatives=compile_derivatives)
        gradient[start:end], temporal[start:end] = g, q
    return gradient.reshape(*shape, 3), temporal.reshape(shape)


@torch.no_grad()
def recover(source, target, shape, reference, config, output, device, eta):
    settings = config['recovery']
    count = settings['intervals'] + 1
    dt = 1 / (count - 1)
    grid = identity_grid(shape, device)
    forward = grid.clone()
    diagnostics = []
    # Keep velocities on disk for inverse integration, bounding time-axis VRAM.
    with tempfile.TemporaryDirectory(prefix='recovery_', dir=output) as temporary:
        frame_path = output / 'frames.npy' if config['output']['save_frames'] else Path(temporary) / 'frames.npy'
        velocity_path = output / 'velocities.npy' if config['output']['save_velocities'] else Path(temporary) / 'velocities.npy'
        archive_frames = settings['method'] == 'finite' or config['output']['save_frames']
        frames = (np.lib.format.open_memmap(frame_path, mode='w+', dtype=np.float32, shape=(count, *shape))
                  if archive_frames else None)
        indices = range(count) if archive_frames else (0, count-1)
        endpoints = {}
        for index in stage_progress(indices, 'Decode trajectory', 'decode', eta):
            image = decode(source, target, shape, index * dt, config['runtime']['inference_chunk'])
            if frames is not None:
                frames[index] = image
            if index in (0, count-1):
                endpoints[index] = image
        if frames is not None:
            frames.flush()
        source_reconstruction, target_reconstruction = endpoints[0], endpoints[count-1]
        velocities = np.lib.format.open_memmap(velocity_path, mode='w+', dtype=np.float32,
                                               shape=(count - 1, *shape, 3))
        for index in stage_progress(range(count - 1), '3D optical flow', 'flow', eta):
            options = dict(smoothness=settings['smoothness'], ridge=settings['ridge'],
                           iterations=settings['iterations'], tolerance=settings['tolerance'])
            if settings['method'] == 'analytical':
                gradient, temporal = recovery_derivatives(source, target, shape, (index+.5)*dt,
                                                         config['runtime']['inference_chunk'],
                                                         config['runtime'].get('compile_derivatives',True))
                velocity, diagnostic = estimate_velocity_from_derivatives(gradient, temporal, **options)
                del gradient, temporal
            else:
                first = torch.from_numpy(np.array(frames[index])).to(device)
                second = torch.from_numpy(np.array(frames[index + 1])).to(device)
                velocity, diagnostic = estimate_velocity(first, second, dt, **options)
                del first, second
            diagnostics.append({'interval': index, 'time_a': index * dt,
                                'time_b': (index + 1) * dt, 'method': settings['method'], **diagnostic})
            velocities[index] = velocity.cpu().numpy()
            forward = integrate_interval(forward, velocity, dt, settings['substeps'])
            del velocity
        velocities.flush()
        inverse = grid.clone()
        for index in stage_progress(range(count - 2, -1, -1), 'Inverse map', 'inverse', eta):
            velocity = torch.from_numpy(np.array(velocities[index])).to(device)
            inverse = integrate_interval(inverse, velocity, -dt, settings['substeps'])
            del velocity
        del frames, velocities
    if config['output'].get('save_volumes', True):
        for direction, mapping in (('forward', forward), ('inverse', inverse)):
            field = (mapping - grid).cpu().numpy()
            save_volume(output / f'{direction}_displacement_voxels.nii.gz', field, reference, vector=True)
            save_volume(output / f'{direction}_displacement_world_mm.nii.gz',
                        voxel_vector_to_world(field, reference.affine), reference, vector=True)
        save_volume(output / 'source_reconstruction.nii.gz', source_reconstruction, reference)
        save_volume(output / 'target_reconstruction.nii.gz', target_reconstruction, reference)
    write_json(output / 'flow_diagnostics.json', diagnostics)
    return forward, inverse, source_reconstruction, target_reconstruction, diagnostics


@torch.no_grad()
def check_endpoint_reconstruction(source,target,moving,fixed,config,output):
    """Gate registration on original full-volume endpoint reconstructions."""
    shape=moving.shape
    images=[decode(source,target,shape,t,config['runtime']['inference_chunk']) for t in (0.,1.)]
    metrics=[image_metrics(truth.cpu().numpy(),image) for truth,image in zip((moving,fixed),images)]
    threshold=config['evaluation']['min_endpoint_psnr_db']
    result={'endpoint_source':metrics[0],'endpoint_target':metrics[1],
            'endpoint_quality_pass':all(m['psnr_infinite'] or m['psnr_db']>=threshold for m in metrics),
            'threshold_db':threshold}
    write_json(output/'endpoint_metrics.json',result)
    return result,images


@torch.no_grad()
def evaluate(source, target, source_label, target_label, forward, inverse,
             source_reconstruction, target_reconstruction, reference, config, output):
    grid = identity_grid(source.shape, source.device)
    warped = sample(source, inverse).cpu().numpy()
    fixed, moving = target.cpu().numpy(), source.cpu().numpy()
    source_metrics = image_metrics(moving, source_reconstruction)
    target_metrics = image_metrics(fixed, target_reconstruction)
    threshold = config['evaluation']['min_endpoint_psnr_db']
    quality = all(m['psnr_infinite'] or m['psnr_db'] >= threshold for m in (source_metrics, target_metrics))
    determinant = jacobian_determinant((inverse - grid).cpu().numpy())
    composition = sample(forward, inverse) - grid
    world_composition = voxel_vector_to_world(composition.cpu().numpy(), reference.affine)
    bound = inverse.new_tensor([source.shape[2] - 1, source.shape[1] - 1, source.shape[0] - 1])
    outside = ((inverse < 0) | (inverse > bound)).any(-1).float().mean()
    metrics = {'endpoint_source': source_metrics, 'endpoint_target': target_metrics,
               'endpoint_quality_pass': quality, 'before': image_metrics(fixed, moving),
               'after': image_metrics(fixed, warped), 'folding_fraction': float(np.mean(determinant <= 0)),
               'jacobian_min': float(determinant.min()), 'jacobian_mean': float(determinant.mean()),
               'inverse_composition_mean_mm': float(np.linalg.norm(world_composition, axis=-1).mean()),
               'sampling_outside_fraction': float(outside), 'labels': None,
               'ground_truth_epe': None}
    if config['output'].get('save_volumes', True):
        save_volume(output / 'warped_moving.nii.gz', warped, reference)
        save_volume(output / 'inverse_jacobian_det.nii.gz', determinant, reference)
    if source_label is not None and target_label is not None:
        label_tensor = torch.from_numpy(source_label.astype(np.float32)).to(source.device)
        warped_label = sample(label_tensor, inverse, mode='nearest', padding_mode='zeros').cpu().numpy().astype(np.int16)
        spacing = tuple(float(s) for s in reference.header.get_zooms()[:3])
        metrics['labels'], per_label = label_metrics(target_label, warped_label, spacing,
                                                     compute_hd95=config['evaluation']['hd95'])
        if config['output'].get('save_volumes', True):
            save_volume(output / 'warped_label.nii.gz', warped_label, reference)
        write_json(output / 'per_label_metrics.json', per_label)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--device', help='Runtime override: auto, cpu, cuda, cuda:0')
    parser.add_argument('--subjects', nargs='+', help='Runtime subset, e.g. S02 S03')
    parser.add_argument('--output-dir', type=Path, help='Use a new/empty run directory')
    args = parser.parse_args()
    config = load_config(args.config)
    if args.device is not None:
        config['runtime']['device'] = args.device
    if args.subjects is not None:
        config['dataset']['subjects'] = args.subjects
    validate_config(config)
    requested = config['runtime']['device']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if requested == 'auto' else torch.device(requested)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')
    subjects = discover_subjects(config)
    fixed_path = resolve_dataset_path(config, config['dataset']['fixed'])
    reference, fixed_np, fixed_normalization = load_image(fixed_path)
    if min(reference.shape) < 2:
        raise ValueError('Each spatial axis needs at least two voxels')
    basis = reference.affine[:3, :3]
    gram = basis.T @ basis
    if not np.allclose(gram, np.diag(np.diag(gram)), atol=1e-4):
        raise ValueError('Resample sheared voxel affines to an orthogonal grid first')
    fixed_label_path = resolve_dataset_path(config, config['dataset']['fixed_label'])
    fixed_label = load_label(fixed_label_path, reference) if fixed_label_path.exists() else None
    if config['dataset']['require_labels'] and fixed_label is None:
        raise FileNotFoundError(fixed_label_path)
    for subject, path in subjects:
        require_same_grid(reference, nib.load(str(path)), str(path))
        labels = label_path(config, subject)
        if config['dataset']['require_labels'] and labels is None:
            raise FileNotFoundError(f'Missing label for {subject}')
        if labels:
            require_same_grid(reference, nib.load(str(labels)), str(labels))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = args.output_dir if args.output_dir is not None else resolve_path(config['output']['root']) / stamp
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Output directory is not empty: {output}')
    output.mkdir(parents=True, exist_ok=True)
    config['runtime']['device'] = str(device)
    write_json(output / 'config.json', config)
    write_json(output / 'manifest.json', {
        'method': 'paired_3d_siren_relaxed_pdlk_optical_flow', 'label_metric_definition': LABEL_METRIC_DEFINITION, 'config_source': str(args.config.resolve()),
        'torch_version': str(torch.__version__), 'cuda_version': torch.version.cuda,
        'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
        'shape': list(reference.shape), 'subjects': [s for s, _ in subjects],
        'source': 'moving subject', 'target': 'fixed reference',
        'coordinates': 'NIfTI array [D,H,W], sampler/vector channels (x=W,y=H,z=D)',
        'recovery': config['recovery'],
        'derivatives': 'analytical g,q at interval midpoints (finite method uses adjacent frames)',
        'integration': 'piecewise constant Eulerian velocity; midpoint RK2',
        'frames': 'optional equally spaced archive including endpoints; analytical solve does not use frames',
        'forward_map': 'source to target; pixel identity + forward_displacement_voxels',
        'inverse_map': 'target to source; resample original moving image with this map',
        'world_vectors': 'RAS millimetres via NIfTI affine; displacements, not absolute coordinates',
        'lk_mode': config['lk']['mode'],
        'differentiate_velocity_solve': config['lk']['differentiate_solve'],
        'lk_sampling': f"{config['lk']['points']} independent uniform continuous full-volume xyz points; one shared uniform t; one velocity per point",
        'lk_objective': 'min_v (q+g.v)^2+c*|mixed+H.v|^2+eta*|v|^2; c=lk.gradient_weight; eta=lk.ridge; no temporal stationarity',
        'compile_derivatives': config['runtime'].get('compile_derivatives',True),
        'lk_derivatives': 'analytical sine chain rule for g,q; H,mixed only when lk.gradient_weight>0',
        'lk_units': 'gradients per voxel; H per voxel squared; mixed per voxel per unit time; velocity voxels per t in [0,1]',
        'ground_truth_epe': 'unavailable: LPBA provides labels, no true deformation field'})
    print(f'Device: {device}; subjects: {len(subjects)}; output: {output.resolve()}', flush=True)
    print('Full-run ETA updates per stage; initial estimates exclude unmeasured stages.', flush=True)
    target_data = torch.from_numpy(fixed_np).to(device)
    eta, summaries = RunETA(len(subjects), config), []
    with tqdm(total=len(subjects), desc='LPBA full run', position=0, dynamic_ncols=True) as overall:
        for index, (subject, path) in enumerate(subjects):
            eta.begin_pair(index)
            started = time.perf_counter()
            seed = config['runtime']['seed']
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if device.type == 'cuda':
                torch.cuda.manual_seed_all(seed)
            pair_output = output / subject
            pair_output.mkdir()
            write_json(pair_output / 'status.json', {'stage': 'training', 'complete': False})
            try:
                _, moving_np, moving_normalization = load_image(path)
                source_data = torch.from_numpy(moving_np).to(device)
                moving_label_path = label_path(config, subject)
                moving_label = load_label(moving_label_path, reference) if moving_label_path else None
                write_json(pair_output / 'pair.json', {'source': str(path.resolve()),
                    'target': str(fixed_path.resolve()), 'source_normalization': moving_normalization,
                    'target_normalization': fixed_normalization})
                source_model, target_model = fit_pair(source_data, target_data, config, pair_output, subject, eta)
                prepare_recovery(source_model, target_model)
                endpoint_metrics,_ = check_endpoint_reconstruction(source_model,target_model,source_data,target_data,config,pair_output)
                forward=inverse=None
                metrics_started = time.perf_counter()
                if endpoint_metrics['endpoint_quality_pass']:
                    write_json(pair_output / 'status.json', {'stage': 'recovery', 'complete': False})
                    forward, inverse, src_recon, tgt_recon, diagnostics = recover(
                        source_model, target_model, reference.shape, reference, config, pair_output, device, eta)
                    write_json(pair_output / 'status.json', {'stage': 'metrics', 'complete': False})
                    metrics_started = time.perf_counter()
                    metrics = evaluate(source_data, target_data, moving_label, fixed_label, forward,
                        inverse, src_recon, tgt_recon, reference, config, pair_output)
                    metrics.update(flow_all_intervals_converged=all(d['converged'] for d in diagnostics),
                                   flow_max_relative_residual=max(d['relative_linear_residual'] for d in diagnostics),
                                   registration_skipped=False)
                else:
                    metrics={**endpoint_metrics,'labels':None,'after':{'mse':None,'global_ncc':None},
                        'folding_fraction':None,'flow_all_intervals_converged':None,
                        'flow_max_relative_residual':None,'registration_skipped':True,
                        'skip_reason':'Endpoint PSNR below threshold; flow and registration metrics skipped'}
                metrics.update(subject=subject,time_seconds=time.perf_counter()-started)
                write_json(pair_output / 'metrics.json', metrics)
                eta.update('metrics', 1, time.perf_counter() - metrics_started)
                row = {'subject': subject, 'source_psnr_db': metrics['endpoint_source']['psnr_db'],
                    'target_psnr_db': metrics['endpoint_target']['psnr_db'],
                    'endpoint_quality_pass': metrics['endpoint_quality_pass'],
                    'dice': (metrics['labels'] or {}).get('dice'),
                    'hd95_mm': (metrics['labels'] or {}).get('hd95_mm'),
                    'warped_mse': metrics['after']['mse'], 'warped_ncc': metrics['after']['global_ncc'],
                    'folding_fraction': metrics['folding_fraction'],
                    'flow_all_intervals_converged': metrics['flow_all_intervals_converged'],
                    'time_seconds': metrics['time_seconds']}
                summaries.append(row)
                with (output / 'summary.csv').open('w', newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    writer.writeheader()
                    writer.writerows(summaries)
                aggregate = {key: (float(np.mean([r[key] for r in summaries]))
                                   if all(r[key] is not None for r in summaries) else None)
                             for key in ('dice', 'hd95_mm', 'warped_mse', 'folding_fraction')}
                write_json(output / 'summary.json', {'completed_subjects': len(summaries),
                           'total_subjects': len(subjects), 'label_metric_definition': LABEL_METRIC_DEFINITION,
                           'means': aggregate, 'subjects': summaries})
                write_json(pair_output / 'status.json', {'stage': 'complete', 'complete': True})
                folding_text=f"{row['folding_fraction']:.5g}" if row['folding_fraction'] is not None else 'skipped'
                tqdm.write(f"{subject}: Dice={row['dice']}, HD95={row['hd95_mm']} mm, "
                           f"folding={folding_text}, endpoint pass={row['endpoint_quality_pass']}")
                if not row['endpoint_quality_pass']:
                    tqdm.write(f'WARNING {subject}: endpoint PSNR below threshold; registration skipped.')
                if metrics['flow_all_intervals_converged'] is False:
                    tqdm.write(f'WARNING {subject}: some flow intervals did not reach the configured tolerance.')
                del source_data, source_model, target_model, forward, inverse
            except Exception as error:
                write_json(pair_output / 'status.json', {'stage': 'failed', 'complete': False, 'error': str(error)})
                raise
            overall.update(1)
            overall.set_postfix(run_ETA=eta.text())
    print(f'Results saved to {output.resolve()}', flush=True)


if __name__ == '__main__':
    main()
