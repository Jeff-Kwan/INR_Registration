"""Add missing T50 R231 masks to prepared DIR-Lab cases; preserve existing inputs."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import os

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch

ROOT = Path('Data/DIRLab4DCT_IDIR')
WEIGHTS_SHA256 = 'd5d2fc3df432115933491f115b283e94d1f5c841c0e82f6e7bd2d65d3ccde69f'


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    from lungmask import LMInferer
    weights = Path(torch.hub.get_dir()) / 'checkpoints/unet_r231-d5d2fc3d.pth'
    if importlib.metadata.version('lungmask') != '0.2.22' or digest(weights) != WEIGHTS_SHA256:
        raise RuntimeError('Expected the pinned lungmask 0.2.22 / R231 weights')
    manifest = {'protocol': 'ccIDIR phase-specific foreground masks', 'lungmask_version': '0.2.22',
                'model': 'R231', 'weights_sha256': WEIGHTS_SHA256,
                'inference_intensity': 'prepared raw int16 minus 1024 HU', 'cases': []}
    inferer = None
    for number in range(1, 11):
        case = ROOT / f'case{number:02d}'
        record = {'case': number, 'phases': {}}
        for phase in ('T00', 'T50'):
            image_path = case / f'{phase}.nii.gz'
            mask_path = case / f'{phase}_lung_mask.nii.gz'
            image = nib.load(image_path)
            if not mask_path.exists():
                if phase == 'T00':
                    raise RuntimeError(f'Expected preserved T00 mask: {mask_path}')
                if inferer is None:
                    inferer = LMInferer(modelname='R231', batch_size=8)
                data = np.asarray(image.dataobj, dtype=np.float32).transpose(2, 1, 0).copy() - 1024
                hu = sitk.GetImageFromArray(data)
                hu.SetSpacing(tuple(np.linalg.norm(image.affine[:3, :3], axis=0)))
                mask = (inferer.apply(hu) > 0).astype(np.uint8).transpose(2, 1, 0)
                if mask.shape != image.shape or not mask.any():
                    raise ValueError(f'Invalid inferred mask: {case}')
                temporary = case / f'.{phase}_lung_mask.partial.nii.gz'
                try:
                    nib.save(nib.Nifti1Image(mask, image.affine), temporary)
                    os.replace(temporary, mask_path)
                finally:
                    temporary.unlink(missing_ok=True)
            mask_image = nib.load(mask_path)
            mask = np.asarray(mask_image.dataobj)
            if (mask.shape != image.shape or not np.allclose(mask_image.affine, image.affine)
                    or not np.isin(mask, [0, 1]).all() or not mask.any()):
                raise ValueError(f'Invalid phase mask: {mask_path}')
            record['phases'][phase] = {'image_sha256': digest(image_path), 'mask_sha256': digest(mask_path),
                                      'mask_voxels': int(mask.sum())}
        manifest['cases'].append(record)
        (ROOT / 'ccidir_masks.json').write_text(json.dumps(manifest, indent=2) + '\n')
        print(f'case {number:02d}: both masks verified', flush=True)


if __name__ == '__main__':
    main()
