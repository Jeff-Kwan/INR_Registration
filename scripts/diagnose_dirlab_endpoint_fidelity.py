"""Compare frozen DIR-Lab endpoint SIRENs with observed landmark patches.

This probe reads checkpoints and input images and writes scalar JSON only.
"""
import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from medical.dirlab_evaluation import read_points
from medical.models.siren import PairedSiren
from medical.registration.image_sampling import sample_points


def patch_ncc(a, b):
    a = a - a.mean(-1, keepdim=True)
    b = b - b.mean(-1, keepdim=True)
    av, bv = a.square().mean(-1), b.square().mean(-1)
    return (a*b).mean(-1) / (av*bv).sqrt().clamp_min(1e-8), (av < 1e-6) | (bv < 1e-6)


@torch.no_grad()
def endpoint_windows(case_dir, phase, centers, offsets, model, device):
    volume = nib.load(str(case_dir / f'{phase}.nii.gz'))
    observed = np.asarray(volume.dataobj, dtype=np.float32)
    observed = (observed - observed.min()) / (observed.max() - observed.min())
    observed = torch.from_numpy(observed.transpose(2, 1, 0).copy()).to(device)
    sites = (centers[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    actual = sample_points(observed, sites).reshape(len(centers), -1)
    shape = observed.shape
    scale = observed.new_tensor([2/(n-1) for n in reversed(shape)])
    estimated = []
    for piece in sites.split(32768):
        estimated.append(model(piece * scale - 1).flatten())
    estimated = torch.cat(estimated).reshape(len(centers), -1)
    return actual.cpu(), estimated.cpu()


@torch.no_grad()
def probe_case(case, root, device):
    name = f'case{case:02d}'
    case_dir = Path('Data/DIRLab4DCT_IDIR') / name
    payload = torch.load(root / name / 'checkpoint.pt', map_location='cpu', weights_only=True)
    config = payload['config']
    if config.get('spatial_units', 'voxel') != 'voxel' or 'working_grid' in config:
        raise ValueError(f'{name}: expected a native voxel-unit checkpoint')
    pair = PairedSiren(3, **payload['model']).to(device).eval().requires_grad_(False)
    pair.load_state_dict(payload['state'], strict=True)
    fixed = read_points(case_dir / 'T00_landmarks_voxel.csv')
    moving = read_points(case_dir / 'T50_landmarks_voxel.csv')
    affine = nib.load(str(case_dir / 'T50.nii.gz')).affine
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    offset_mm = np.stack(np.meshgrid(*(np.linspace(-4.5, 4.5, 10, dtype=np.float32)
                                      for _ in range(3)), indexing='ij'), -1).reshape(-1, 3)
    offsets = torch.from_numpy(offset_mm / spacing).float().to(device)
    fixed = torch.from_numpy(fixed).float().to(device)
    moving = torch.from_numpy(moving).float().to(device)
    observed_m, model_m = endpoint_windows(case_dir, 'T50', moving, offsets, pair.source, device)
    observed_f, model_f = endpoint_windows(case_dir, 'T00', fixed, offsets, pair.target, device)
    observed_identity, model_identity = endpoint_windows(
        case_dir, 'T50', fixed, offsets, pair.source, device)
    results = {}
    for label, a, b in (
        ('moving_fit', observed_m, model_m),
        ('fixed_fit', observed_f, model_f),
        ('observed_true_match', observed_m, observed_f),
        ('model_true_match', model_m, model_f),
        ('observed_identity_match', observed_identity, observed_f),
        ('model_identity_match', model_identity, model_f),
    ):
        ncc, low_var = patch_ncc(a, b)
        results[label] = {'mean_ncc': float(ncc.mean()), 'median_ncc': float(ncc.median()),
                          'low_variance_count': int(low_var.sum())}
    results['case'] = case
    results['checkpoint_root'] = str(root)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', type=int, nargs='+', default=[1, 8])
    parser.add_argument('--checkpoint-root', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= len(args.cases) <= 10 or any(not 1 <= c <= 10 for c in args.cases):
        parser.error('Choose 1–10 DIR-Lab cases and a fresh output file')
    torch.set_num_threads(4)
    result = [probe_case(case, args.checkpoint_root, torch.device(args.device)) for case in args.cases]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    for case in result:
        print(case['case'], case['moving_fit']['mean_ncc'], case['fixed_fit']['mean_ncc'],
              case['model_true_match']['mean_ncc'], case['model_identity_match']['mean_ncc'])


if __name__ == '__main__':
    main()
