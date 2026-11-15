"""Probe frozen DIR-Lab SIREN transport along observed landmark paths.

Writes scalar summaries only; no model fitting, recovery, or dense images.
"""
import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from medical.dirlab_evaluation import read_points
from medical.models.siren import PairedSiren
from medical.models.siren_functional import coordinate_scale


def summarize(values):
    return float(np.mean(values)) if len(values) else None


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
    moving = read_points(case_dir / 'T50_landmarks_voxel.csv')
    fixed = read_points(case_dir / 'T00_landmarks_voxel.csv')
    shape = tuple(reversed(nib.load(str(case_dir / 'T50.nii.gz')).shape))
    affine = nib.load(str(case_dir / 'T50.nii.gz')).affine
    basis = affine[:3, :3]
    displacement = fixed - moving
    magnitude = np.linalg.norm(displacement @ basis.T, axis=1)
    start = torch.as_tensor(moving, dtype=torch.float32, device=device)
    delta = torch.as_tensor(displacement, dtype=torch.float32, device=device)
    scale = coordinate_scale(shape, start)
    prepared = pair.prepare().at
    temporal, spatial_projection, residual, intensity = [], [], [], []
    times = np.linspace(0.0, 1.0, 33)
    for time in times:
        location = start + float(time) * delta
        coordinates = location * scale - 1
        frame = prepared(float(time))
        g, q = frame.derivatives(coordinates, scale, compile_derivatives=False)
        directional = (g * delta).sum(-1)
        temporal.append(q.cpu().numpy())
        spatial_projection.append((-directional).cpu().numpy())
        residual.append((q + directional).cpu().numpy())
        intensity.append(frame.image(coordinates).flatten().cpu().numpy())
    temporal = np.stack(temporal)
    spatial_projection = np.stack(spatial_projection)
    residual = np.stack(residual)
    intensity = np.stack(intensity)
    # Endpoint mismatch is an independent check of the path derivative integral.
    image_delta = intensity[-1] - intensity[0]
    integrated_residual = np.trapezoid(residual, x=times, axis=0)
    result = {'case': case, 'checkpoint_root': str(root), 'count': len(moving), 'bins': {}}
    for label, mask in {'all': np.ones(len(moving), bool), '<2': magnitude < 2,
                        '2-5': (magnitude >= 2) & (magnitude < 5),
                        '5-10': (magnitude >= 5) & (magnitude < 10),
                        '>=10': magnitude >= 10}.items():
        if not mask.any():
            continue
        q, target, r = temporal[:, mask].ravel(), spatial_projection[:, mask].ravel(), residual[:, mask].ravel()
        result['bins'][label] = {
            'landmarks': int(mask.sum()), 'mean_motion_mm': summarize(magnitude[mask]),
            'q_rms': float(np.sqrt(np.mean(q*q))),
            'target_rms': float(np.sqrt(np.mean(target*target))),
            'transport_residual_rms': float(np.sqrt(np.mean(r*r))),
            'q_on_transport_target_slope': float(np.dot(q, target) / np.dot(target, target)),
            'q_transport_target_correlation': float(np.corrcoef(q, target)[0, 1]),
            'mean_endpoint_model_intensity_delta': summarize(image_delta[mask]),
            'mean_abs_endpoint_model_intensity_delta': summarize(np.abs(image_delta[mask])),
            'mean_abs_integrated_residual': summarize(np.abs(integrated_residual[mask])),
            'max_endpoint_integral_difference': float(np.max(np.abs(image_delta[mask] - integrated_residual[mask]))),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', type=int, nargs='+', default=[1, 3, 4, 5])
    parser.add_argument('--checkpoint-root', type=Path,
                        default=Path('results/dirlab_quick4_native_cropcenter_20261113'))
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= len(args.cases) <= 10 or any(not 1 <= c <= 10 for c in args.cases):
        parser.error('Choose 1–10 DIR-Lab cases and a fresh output file')
    torch.set_num_threads(4)
    result = [probe_case(case, args.checkpoint_root, torch.device(args.device)) for case in args.cases]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    for case in result:
        high = case['bins'].get('>=10', {})
        print(case['case'], high.get('landmarks'), high.get('q_on_transport_target_slope'),
              high.get('transport_residual_rms'))


if __name__ == '__main__':
    main()
