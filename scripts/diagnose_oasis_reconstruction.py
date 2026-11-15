"""Evaluate frozen OASIS endpoint reconstructions by anatomical label."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from medical.case_runner import load_volume
from medical.models.siren import PairedSiren, coordinates_from_indices
from medical.registration.geometry import sample
from medical.registration.image_sampling import sample_points


def patch_statistics(observed, predicted, mask):
    count = mask.sum(-1)
    weights = mask.to(observed.dtype)
    denominator = count.clamp_min(1)
    a = observed - (observed * weights).sum(-1, keepdim=True) / denominator[:, None]
    b = predicted - (predicted * weights).sum(-1, keepdim=True) / denominator[:, None]
    av = (a.square() * weights).sum(-1) / denominator
    bv = (b.square() * weights).sum(-1) / denominator
    valid = (count >= 8) & (av >= 1e-6) & (bv >= 1e-6)
    correlation = (a*b*weights).sum(-1) / denominator / (av*bv).sqrt().clamp_min(1e-12)
    return {'patches': len(count), 'valid_patches': int(valid.sum()),
            'low_variance_or_small_patches': int((~valid).sum()),
            'mean_ncc': float(correlation[valid].mean()) if valid.any() else None}


def predict(model, sites, extent):
    return torch.cat([model(2*piece/extent-1).flatten() for piece in sites.split(32768)])


@torch.no_grad()
def evaluate_endpoint(model, image_path, patch_count, seed, device, *, label_ids=range(1, 36)):
    volume = load_volume(image_path)
    data = volume['data']
    data = (data-data.min()) / (data.max()-data.min())
    image = torch.from_numpy(data.transpose(2, 1, 0).copy()).to(device)
    label_path = image_path.parent.parent / 'labelsTr' / image_path.name
    label_volume = load_volume(label_path)
    if (label_volume['data'].shape != volume['data'].shape
            or not np.allclose(label_volume['affine'], volume['affine'])):
        raise ValueError('Label grid differs from image grid')
    labels = torch.from_numpy(label_volume['data'].transpose(2, 1, 0).copy()).to(device)
    extent = image.new_tensor(tuple(reversed(image.shape))) - 1
    spacing = np.linalg.norm(volume['affine'][:3, :3], axis=0)
    offsets = np.stack(np.meshgrid(*(np.linspace(-4.5, 4.5, 10, dtype=np.float32)
                                     for _ in range(3)), indexing='ij'), -1).reshape(-1, 3)
    offsets = torch.as_tensor(offsets/spacing, dtype=image.dtype, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    regions = {}
    for label in label_ids:
        indices = (labels.flatten() == label).nonzero().flatten()
        if not len(indices):
            regions[str(label)] = {'voxels': 0}
            continue
        # Exact region moments, accumulated in float64, without a decoded volume.
        moments = torch.zeros(6, dtype=torch.float64, device=device)
        for selected in indices.split(32768):
            coordinates = coordinates_from_indices(selected, image.shape)
            a = image.flatten()[selected].double()
            b = model(coordinates).flatten().double()
            moments += torch.stack((a.sum(), b.sum(), a.square().sum(), b.square().sum(),
                                    (a*b).sum(), (a-b).square().sum()))
        sa, sb, saa, sbb, sab, error = moments.tolist()
        count = len(indices)
        av, bv = max(0, saa/count-(sa/count)**2), max(0, sbb/count-(sb/count)**2)
        ncc = (sab/count-sa*sb/count**2)/math.sqrt(av*bv) if av*bv > 1e-16 else None
        mse = error/count
        centers = indices[torch.randint(count, (min(patch_count, count),),
                                        generator=generator, device=device)]
        positions = (coordinates_from_indices(centers, image.shape)+1)*extent/2
        sites = (positions[:, None] + offsets[None]).reshape(-1, 3)
        observed = sample_points(image, sites).reshape(len(centers), -1)
        predicted = predict(model, sites, extent).reshape(len(centers), -1)
        local_labels = sample(labels, sites.reshape(1, 1, -1, 3), mode='nearest').reshape(len(centers), -1)
        regions[str(label)] = {
            'voxels': count, 'mse': mse, 'psnr_db': -10*math.log10(mse) if mse else None,
            'region_ncc': ncc,
            'full_patch': patch_statistics(observed, predicted, torch.ones_like(observed, dtype=torch.bool)),
            'within_label_patch': patch_statistics(observed, predicted, local_labels == label),
        }
    return {'image': str(image_path), 'regions': regions}


def mean(values):
    finite = [value for value in values if value is not None]
    return sum(finite)/len(finite) if finite else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--patches-per-label', type=int, default=128)
    args = parser.parse_args()
    if args.output.exists() or args.patches_per_label < 1:
        parser.error('Use a fresh output file and a positive patch count')
    torch.set_num_threads(4)
    protocol = json.loads(Path('medical/protocols/oasis_quick4.json').read_text())
    endpoints, reconstructions = [], []
    with torch.no_grad():
        for pair in protocol['pairs']:
            fixed = Path(pair['fixed']).name.split('_')[1]
            moving = Path(pair['moving']).name.split('_')[1]
            name = f'{fixed}_to_{moving}'
            directory = args.checkpoint_root / name
            payload = torch.load(directory/'checkpoint.pt', map_location='cpu', weights_only=True)
            model = PairedSiren(3, **payload['model']).to(args.device).eval()
            model.load_state_dict(payload['state'], strict=True)
            reconstructions.append(json.loads((directory/'metrics.json').read_text())['reconstruction'])
            for role, network in (('moving', model.source), ('fixed', model.target)):
                path = Path('Data/OASIS') / pair[role]
                result = evaluate_endpoint(network, path, args.patches_per_label, 42, args.device)
                result.update(pair=name, endpoint=role)
                endpoints.append(result)
                print(name, role, flush=True)
    per_label = {}
    for label in range(1, 36):
        scored = [row['regions'][str(label)] for row in endpoints if row['regions'][str(label)]['voxels']]
        per_label[str(label)] = {
            'scored_endpoints': len(scored), 'mean_region_ncc': mean([r['region_ncc'] for r in scored]),
            'mean_psnr_db': mean([r['psnr_db'] for r in scored]),
            'mean_full_patch_ncc': mean([r['full_patch']['mean_ncc'] for r in scored]),
            'mean_within_label_patch_ncc': mean([r['within_label_patch']['mean_ncc'] for r in scored]),
            'valid_full_patches': sum(r['full_patch']['valid_patches'] for r in scored),
            'valid_within_label_patches': sum(r['within_label_patch']['valid_patches'] for r in scored),
        }
    summary = {
        'whole_volume_mse': mean([r[e]['mse'] for r in reconstructions for e in ('moving', 'fixed')]),
        'whole_volume_psnr_db': mean([r[e]['psnr_db'] for r in reconstructions for e in ('moving', 'fixed')]),
        **{key: mean([r[key] for r in per_label.values()]) for key in
           ('mean_region_ncc', 'mean_psnr_db', 'mean_full_patch_ncc', 'mean_within_label_patch_ncc')},
    }
    result = {'summary': summary, 'per_label': per_label, 'endpoints': endpoints,
              'protocol': {'patch_width_mm': 9, 'samples_per_axis': 10,
                           'patches_per_label': args.patches_per_label, 'seed': 42,
                           'minimum_patch_voxels': 8, 'minimum_patch_variance': 1e-6,
                           'averaging': 'equal endpoints within label, then equal labels',
                           'endpoint_fits': len(endpoints), 'unique_images': len({r['image'] for r in endpoints})}}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
