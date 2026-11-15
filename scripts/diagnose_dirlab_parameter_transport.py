"""Decompose frozen SIREN temporal transport by parameter group.

Uses observed landmark paths and scalar summaries only. No fitting or recovery.
"""
import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from medical.dirlab_evaluation import read_points
from medical.models.siren import PairedSiren
from medical.models.siren_functional import coordinate_scale


def parameter_contributions(frame, coordinates):
    values = coordinates
    history = []
    for (weight, bias), omega in zip(frame.parameters[:-1], frame.omegas):
        phase = omega * F.linear(values, weight, bias)
        history.append((values, omega * phase.cos(), weight))
        values = phase.sin()
    delta_weight, delta_bias = frame.direction[-1]
    contributions = [None] * (2 * len(frame.parameters))
    contributions[-2] = F.linear(values, delta_weight).squeeze(-1)
    contributions[-1] = delta_bias.squeeze().expand(len(coordinates))
    adjoint = frame.parameters[-1][0]
    for index in reversed(range(len(history))):
        values, factor, weight = history[index]
        delta_weight, delta_bias = frame.direction[index]
        adjoint = adjoint * factor
        contributions[2 * index] = (adjoint * F.linear(values, delta_weight)).sum(-1)
        contributions[2 * index + 1] = (adjoint * delta_bias).sum(-1)
        adjoint = adjoint @ weight
    return torch.stack(contributions)


@torch.no_grad()
def probe_case(case, checkpoint_root, device):
    directory = Path('Data/DIRLab4DCT_IDIR') / f'case{case:02d}'
    payload = torch.load(checkpoint_root / f'case{case:02d}' / 'checkpoint.pt',
                         weights_only=True, map_location='cpu')
    config = payload['config']
    if config.get('spatial_units', 'voxel') != 'voxel' or 'working_grid' in config:
        raise ValueError('This probe requires native voxel-unit checkpoints')
    pair = PairedSiren(3, **payload['model']).to(device).eval().requires_grad_(False)
    pair.load_state_dict(payload['state'], strict=True)
    image = nib.load(str(directory / 'T50.nii.gz'))
    moving = read_points(directory / 'T50_landmarks_voxel.csv')
    fixed = read_points(directory / 'T00_landmarks_voxel.csv')
    displacement = fixed - moving
    magnitude = np.linalg.norm(displacement @ image.affine[:3, :3].T, axis=1)
    start = torch.as_tensor(moving, dtype=torch.float32, device=device)
    delta = torch.as_tensor(displacement, dtype=torch.float32, device=device)
    scale = coordinate_scale(tuple(reversed(image.shape)), start)
    endpoints = pair.prepare()
    components, targets, totals = [], [], []
    maximum_error = 0.
    for time in np.linspace(0, 1, 33):
        coordinates = (start + float(time) * delta) * scale - 1
        frame = endpoints.at(float(time))
        gradient, temporal = frame.derivatives(coordinates, scale, compile_derivatives=False)
        contribution = parameter_contributions(frame, coordinates)
        torch.testing.assert_close(contribution.sum(0), temporal, rtol=2e-4, atol=2e-5)
        maximum_error = max(maximum_error, float((contribution.sum(0) - temporal).abs().max()))
        components.append(contribution.cpu().numpy())
        targets.append((-(gradient * delta).sum(-1)).cpu().numpy())
        totals.append(temporal.cpu().numpy())
    components = np.stack(components)
    targets = np.stack(targets)
    totals = np.stack(totals)
    layers = ['first', *[f'hidden{index}' for index in range(1, payload['model']['layers'])], 'output']
    names = [f'{layer}.{kind}' for layer in layers for kind in ('weight', 'bias')]
    result = {'case': case, 'checkpoint_root': str(checkpoint_root),
              'maximum_component_sum_error': maximum_error, 'bins': {}}
    for label, mask in {'all': np.ones(len(moving), dtype=bool), '>=10': magnitude >= 10}.items():
        target = targets[:, mask].reshape(-1)
        total = totals[:, mask].reshape(-1)
        denominator = float(np.dot(target, target))
        groups = {}
        for index, name in enumerate(names):
            values = components[:, index, mask].reshape(-1)
            groups[name] = {'q_rms': float(np.sqrt(np.mean(values ** 2))),
                           'transport_target_slope': float(np.dot(values, target) / denominator)
                           if denominator else None}
        result['bins'][label] = {
            'landmarks': int(mask.sum()),
            'total_transport_target_slope': float(np.dot(total, target) / denominator)
            if denominator else None,
            'transport_residual_rms': float(np.sqrt(np.mean((total - target) ** 2))),
            'groups': groups}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-root', required=True, type=Path)
    parser.add_argument('--cases', nargs='+', type=int, default=[1, 3, 4, 5])
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output file')
    torch.set_num_threads(4)
    results = [probe_case(case, args.checkpoint_root, torch.device(args.device)) for case in args.cases]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
