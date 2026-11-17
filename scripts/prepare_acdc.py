"""Prepare official ACDC testing inputs using the pinned KAN-IDIR array recipe."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.metadata
import json
from pathlib import Path
import sys
import tempfile

import nibabel as nib
import numpy as np
from skimage.transform import rescale

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medical.brain_data import sha256

REVISION = '3ac12e200159811eafb6192f3f8db3afef2a9d1a'
SPACING = (1.5, 1.5, 3.15)


def preprocess(image, labels, spacing):
    # Preserve the author's float32 header-scalar divisions before rescale
    # rounds the requested output dimensions. Promoting these can shift a crop.
    factors = tuple(spacing[i] / SPACING[i] for i in range(3))
    scaled = rescale(np.asarray(image, dtype=np.float32), factors, order=3, mode='edge')
    segmentation = rescale(np.asarray(labels, dtype=np.float32), factors, order=0, mode='edge')
    if scaled.shape != segmentation.shape or not np.isfinite(scaled).all() or scaled.max() <= scaled.min():
        raise ValueError('Invalid resampled MRI or label grid')
    scaled = (scaled - scaled.min()) / (scaled.max() - scaled.min())
    size = np.array([128, 128, min(scaled.shape[2], 32)])
    offset = (np.asarray(scaled.shape) - size) // 2
    if np.any(offset < 0):
        raise ValueError('Selected ACDC cohort requires resampled XY extents of at least 128')
    crop = tuple(slice(int(o), int(o + n)) for o, n in zip(offset, size))
    return scaled[crop], segmentation[crop], offset


def prepare(source, output, workers=4):
    if workers < 1:
        raise ValueError('workers must be positive')
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    acquisition = json.loads((source / 'testing_download_manifest.json').read_text())
    if not acquisition['complete'] or len(acquisition['files']) != 252:
        raise ValueError('Expected the complete official testing acquisition')
    for record in acquisition['files']:
        path = source / record['path']
        if path.stat().st_size != record['bytes'] or sha256(path) != record['sha256']:
            raise ValueError(f'Acquisition checksum mismatch: {path}')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.acdc-preparation-', dir=output.parent) as temporary:
        staging = Path(temporary) / 'prepared'
        (staging / 'images').mkdir(parents=True)
        (staging / 'labels').mkdir()
        records = []
        def prepare_patient(patient):
            patient_records = []
            directory = source / 'testing' / f'patient{patient}'
            info = dict(line.split(':', 1) for line in (directory / 'Info.cfg').read_text().splitlines() if ':' in line)
            frames = {phase: int(info[phase]) for phase in ('ED', 'ES')}
            es = nib.load(directory / f'patient{patient}_frame{frames["ES"]:02d}.nii.gz')
            spacing = es.header.get_zooms()[:3]
            for phase, frame in frames.items():
                image_path = directory / f'patient{patient}_frame{frame:02d}.nii.gz'
                label_path = directory / f'patient{patient}_frame{frame:02d}_gt.nii.gz'
                image, label = nib.load(image_path), nib.load(label_path)
                labels = label.get_fdata(dtype=np.float32)
                if image.shape != label.shape or not np.isin(labels, range(4)).all():
                    raise ValueError(f'Invalid raw ACDC labels: {label_path}')
                values, codes, offset = preprocess(image.get_fdata(dtype=np.float32), labels, spacing)
                affine = np.diag([*SPACING, 1.])
                affine[:3, 3] = offset * SPACING
                name = f'patient{patient}_{phase}.nii.gz'
                for folder, array in [('images', values.astype(np.float32)), ('labels', codes.astype(np.uint8))]:
                    nib.save(nib.Nifti1Image(array, affine), staging / folder / name)
                patient_records.append(dict(subject=f'patient{patient}', phase=phase, frame=frame,
                                    image=f'images/{name}', label=f'labels/{name}', shape=list(values.shape),
                                    affine=affine.tolist(), es_spacing=list(map(float, spacing)), crop_offset=offset.tolist(),
                                    image_sha256=sha256(staging / 'images' / name), label_sha256=sha256(staging / 'labels' / name),
                                    source_image_sha256=sha256(image_path), source_label_sha256=sha256(label_path)))
            print(f'Prepared patient{patient}', flush=True)
            return patient_records

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for patient_records in pool.map(prepare_patient, range(101, 151)):
                records.extend(patient_records)
        pairs = []
        for moving, fixed in [('ED', 'ES'), ('ES', 'ED')]:
            for patient in range(101, 151):
                pairs.append(dict(pair_index=len(pairs) + 1, moving=f'patient{patient}_{moving}', fixed=f'patient{patient}_{fixed}',
                                  moving_image=f'images/patient{patient}_{moving}.nii.gz', fixed_image=f'images/patient{patient}_{fixed}.nii.gz',
                                  moving_label=f'labels/patient{patient}_{moving}.nii.gz', fixed_label=f'labels/patient{patient}_{fixed}.nii.gz'))
        manifest = dict(protocol='acdc_kan_idir_v1', source_revision=REVISION, spacing=SPACING,
                        geometry='Array-rescaled benchmark frame with crop offset; original patient affine preserved in raw inputs',
                        preprocessing='ES spacing for both phases; cubic/edge MRI, nearest/edge labels; full resampled minmax before center crop; no padding',
                        acquisition_sha256=sha256(source / 'testing_download_manifest.json'), subjects=records, pairs=pairs,
                        environment={n: importlib.metadata.version(n) for n in ('numpy', 'scipy', 'scikit-image', 'nibabel')})
        (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        if output.exists():
            raise FileExistsError(output)
        staging.rename(output)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('Data/ACDC'))
    parser.add_argument('--output', type=Path, default=Path('Data/ACDC/prepared'))
    parser.add_argument('--workers', type=int, default=4, help='Concurrent patient preparations')
    args = parser.parse_args()
    prepare(args.source, args.output, args.workers)
