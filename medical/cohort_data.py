"""Frozen ACDC/KAN-IDIR and Neurite-OASIS preparation records and pair lists."""
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from medical.brain_data import sha256

DATASETS = {'acdc': 'ACDC', 'neurite_oasis': 'NeuriteOASIS'}


def definition(root, dataset):
    root = Path(root).resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    if dataset == 'acdc':
        if (manifest['protocol'] != 'acdc_kan_idir_v1' or
                manifest['source_revision'] != '3ac12e200159811eafb6192f3f8db3afef2a9d1a' or
                manifest['spacing'] != [1.5, 1.5, 3.15]):
            raise ValueError('ACDC preparation does not match KAN-IDIR')
        expected = [(f'patient{p}_{a}', f'patient{p}_{b}')
                    for a, b in [('ED', 'ES'), ('ES', 'ED')] for p in range(101, 151)]
        for row in manifest['subjects']:
            name = f'{row["subject"]}_{row["phase"]}'
            if row['image'] != f'images/{name}.nii.gz' or row['label'] != f'labels/{name}.nii.gz':
                raise ValueError('Unexpected ACDC prepared paths')
        identities = [(r['subject'], r['phase']) for r in manifest['subjects']]
        expected_phases = [(f'patient{p}', phase) for p in range(101, 151) for phase in ('ED', 'ES')]
        if sorted(identities) != sorted(expected_phases):
            raise ValueError('Expected exactly 100 selected ACDC phase records')
    elif dataset == 'neurite_oasis':
        subjects = manifest['all_subjects']
        selected = manifest['selected_subjects']
        if (manifest['protocol'] != 'neurite_oasis_gaussiandir_last50_consecutive49_v1' or
                manifest['release_commit'] != 'd3702232241b81a716f1db3dfa6c8502ad6d8027' or
                manifest['archive_sha256'] != '86dd117dda17f736ade8a4088d7e98e066e1181950fe8b406f1a35f7fb743e78' or
                len(set(subjects)) != 414 or subjects != sorted(subjects) or selected != subjects[-50:] or
                selected[0] != 'OASIS_OAS1_0404_MR1' or selected[-1] != 'OASIS_OAS1_0457_MR1'):
            raise ValueError('Unexpected Neurite-OASIS cohort')
        expected = [(b, a) for a, b in zip(selected[:-1], selected[1:])]
        if [r['subject'] for r in manifest['subjects']] != selected:
            raise ValueError('Neurite preparation records differ from the selected cohort')
    else:
        raise ValueError('Unknown cohort')
    if [(p['moving'], p['fixed']) for p in manifest['pairs']] != expected:
        raise ValueError('Ordered pairs differ from the frozen cohort')
    for index, pair in enumerate(manifest['pairs'], 1):
        if pair['pair_index'] != index:
            raise ValueError('Unexpected pair index')
        for key in ('moving', 'fixed'):
            if dataset == 'acdc':
                image, label = f'images/{pair[key]}.nii.gz', f'labels/{pair[key]}.nii.gz'
            else:
                image, label = f'{pair[key]}/aligned_norm.nii.gz', f'{pair[key]}/aligned_seg35.nii.gz'
            if pair[f'{key}_image'] != image or pair[f'{key}_label'] != label:
                raise ValueError('Unexpected prepared pair paths')
    return manifest


def validate_case(case, dataset):
    roots = {Path(v['path']).resolve().parent.parent for v in case.values()}
    if len(roots) != 1:
        raise ValueError('Expected one prepared release')
    root = roots.pop()
    manifest = definition(root, dataset)
    matches = [p for p in manifest['pairs'] if all(
        Path(case[k]['path']).resolve() == root / p[f'{k}_image'] for k in ('moving', 'fixed'))]
    if len(matches) != 1:
        raise ValueError('Pair is outside the frozen cohort')
    pair = matches[0]
    records = {f'{r["subject"]}_{r["phase"]}' if dataset == 'acdc' else r['subject']: r for r in manifest['subjects']}
    for key in ('moving', 'fixed'):
        volume, rec = case[key], records[pair[key]]
        shape = tuple(rec['shape'])
        if dataset == 'acdc':
            if shape[:2] != (128, 128) or shape[2] not in (19, 24, 25, 27, 29, 30, 32):
                raise ValueError('Unexpected KAN-IDIR shape')
            hashes = {pair[f'{key}_image']: rec['image_sha256'], pair[f'{key}_label']: rec['label_sha256']}
            spacing = (1.5, 1.5, 3.15)
        else:
            if shape != (160, 192, 224):
                raise ValueError('Unexpected Neurite shape')
            if set(rec['sha256']) != {'aligned_norm.nii.gz', 'aligned_seg35.nii.gz'}:
                raise ValueError('Both released Neurite file hashes are required')
            hashes = {f'{rec["subject"]}/{name}': h for name, h in rec['sha256'].items()}
            spacing = (1., 1., 1.)
        for path, expected in hashes.items():
            if sha256(root / path) != expected:
                raise ValueError(f'Prepared checksum mismatch: {path}')
        label = nib.load(root / pair[f'{key}_label'])
        codes = np.asarray(label.dataobj)
        if (volume['data'].shape != shape or label.shape != shape or
                not np.allclose(volume['affine'], rec['affine'], rtol=0, atol=1e-5) or
                not np.allclose(label.affine, volume['affine'], rtol=0, atol=1e-5) or
                not np.allclose(np.linalg.norm(volume['affine'][:3, :3], axis=0), spacing) or
                not np.isfinite(volume['data']).all() or np.ptp(volume['data']) <= 0 or
                not np.isin(codes, range(4 if dataset == 'acdc' else 36)).all()):
            raise ValueError('Prepared intensity, labels or geometry changed')
    if not np.allclose(case['moving']['affine'], case['fixed']['affine'], atol=1e-5):
        raise ValueError('Prepared phases must share a benchmark grid')
    return root, pair
