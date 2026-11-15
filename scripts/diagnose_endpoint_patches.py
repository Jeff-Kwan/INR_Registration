"""Frozen endpoint fidelity on native images; scalar JSON outputs only."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medical.dirlab_evaluation import read_points
from medical.case_runner import load_volume
from medical.models.siren import PairedSiren, endpoint_rescaled_reconstruction
from medical.registration.geometry import sample
from medical.registration.image_sampling import sample_points
from scripts.diagnose_oasis_reconstruction import patch_statistics, predict


@torch.no_grad()
def endpoint(model, path, dataset, device):
    volume = load_volume(path)
    data = volume['data']
    if not np.isfinite(data).all() or data.max() <= data.min():
        raise ValueError('Expected a finite image with a nonconstant intensity range')
    image = torch.from_numpy(((data-data.min())/(data.max()-data.min())).T.copy()).to(device)
    extent = image.new_tensor(tuple(reversed(image.shape))) - 1
    spacing = np.linalg.norm(volume['affine'][:3, :3], axis=0)
    axes = [np.linspace(-4, 4, 9)] * 3
    offsets_mm = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
    offsets = torch.as_tensor(offsets_mm / spacing, dtype=image.dtype, device=device)
    generator = torch.Generator(device=device).manual_seed(42)

    def patches(centers, region=None):
        sites = centers[:, None] + offsets[None]
        inside = ((sites >= 0) & (sites <= extent)).all(-1)
        flat = sites.reshape(-1, 3)
        # Outside-image sites are excluded from moments and queried at the border.
        clipped = flat.clamp_min(0).minimum(extent)
        observed = sample_points(image, clipped).reshape(len(centers), -1).double()
        predicted = predict(model, clipped, extent).reshape_as(observed).double()
        result = {'full_patch': patch_statistics(observed, predicted, inside)}
        if region is not None:
            local = sample(region.float(), clipped.reshape(1, 1, -1, 3),
                           mode='nearest').reshape_as(observed).bool()
            result['within_region_patch'] = patch_statistics(observed, predicted, inside & local)
        return result

    centers = torch.rand((512, 3), generator=generator, device=device) * extent
    result = {'uniform_image': patches(centers)}
    if dataset == 'dirlab':
        points = read_points(path.with_name(path.name[:3]+'_landmarks_voxel.csv'))
        result['landmarks'] = patches(torch.as_tensor(points, dtype=image.dtype, device=device))
    else:
        labels = load_volume(path.parent.parent/'labelsTr'/path.name)
        if labels['data'].shape != data.shape or not np.allclose(labels['affine'], volume['affine']):
            raise ValueError('Labels do not match native image geometry')
        labels = torch.from_numpy(labels['data'].T.copy()).to(device)
        regions = {}
        label_ids = range(1, 36) if dataset == 'oasis' else range(1, 14)
        for label in label_ids:
            region = labels == label
            sites = region.nonzero()
            if not len(sites):
                regions[str(label)] = {'voxels': 0}
                continue
            selected = sites[torch.randint(len(sites), (64,), generator=generator, device=device)]
            regions[str(label)] = {'voxels': len(sites),
                                   **patches(selected.flip(-1).float(), region)}
        result['labels'] = regions
    return image, result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--worker', type=int, required=True)
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    if a.workers < 1 or not 0 <= a.worker < a.workers or a.gpu < 0:
        p.error('Use a positive worker count, a valid worker index and a nonnegative GPU')
    torch.set_num_threads(4)
    device = f'cuda:{a.gpu}'
    jobs = json.loads((a.run/'summary.json').read_text())['jobs']
    a.output.mkdir(parents=True, exist_ok=True)
    for job in jobs[a.worker::a.workers]:
        if job['returncode']:
            continue
        directory = Path(job['output'])
        target = a.output/f"{job['dataset']}_{job['pair_index']:02d}.json"
        if target.exists():
            raise FileExistsError(f'Diagnostic output already exists: {target}')
        payload = torch.load(directory/'checkpoint.pt', map_location='cpu', weights_only=True)
        cfg = payload['config']
        if 'working_grid' in cfg or cfg.get('spatial_units','voxel') != 'voxel':
            raise ValueError('This diagnostic requires native voxel checkpoints')
        pair = PairedSiren(3, **payload['model']).to(device).eval().requires_grad_(False)
        pair.load_state_dict(payload['state'])
        run = json.loads((directory/'run.json').read_text())
        result = {
            'dataset': job['dataset'],
            'pair_index': job['pair_index'],
            'checkpoint': str(directory/'checkpoint.pt'),
            'patch_span_mm': 8,
            'patch_samples_per_axis': 9,
            'seed': 42,
            'endpoints': {},
        }
        images = []
        started = time.perf_counter()
        for name, model in [('moving',pair.source),('fixed',pair.target)]:
            image, scores = endpoint(model, Path(run[name]), job['dataset'], device)
            images.append(image)
            result['endpoints'][name] = scores
        result['rescaled_psnr'] = endpoint_rescaled_reconstruction(pair, images)
        result['diagnostic_seconds'] = time.perf_counter()-started
        target.write_text(json.dumps(result, indent=2, allow_nan=False))
        print(job['dataset'], job['pair_index'], round(result['diagnostic_seconds'],2), flush=True)
        del pair, images, payload, image
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
