"""Prepare frozen IXI/TransMorph and LPBA40/SITReg inputs; no fitting/recovery."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nibabel as nib
import numpy as np
import torch
from medical.brain_data import (DATASET_NAMES, PROTOCOL_PATH, protocols, read_ixi,
                                read_analyze, sha256, preparation_fingerprint,
                                require_sitreg_dependencies)


def save_volume(path, values, affine):
    nib.save(nib.Nifti1Image(values, affine), path)
    check = nib.load(path)
    if not np.array_equal(np.asarray(check.dataobj), values):
        raise RuntimeError(f'NIfTI conversion changed values: {path}')
    return sha256(path)


def convert_ixi(root, output, workers):
    raw = root / 'raw/IXI_data'
    verified = json.loads((root / 'verification.json').read_text())
    hashes = {row['path']: row['sha256'] for row in verified['files']}
    spec = protocols()['ixi_transmorph']
    members = [(folder, name) for folder in ('Train', 'Val', 'Test')
               for name in spec['split_membership'][folder]] + [('atlas', 'atlas.pkl')]
    if any(set(p.name for p in (raw / folder).glob('*.pkl')) != set(spec['split_membership'][folder])
           for folder in ('Train', 'Val', 'Test')):
        raise ValueError('IXI split membership differs from the author release')

    def convert(item):
        split, name = item
        source = raw / name if split == 'atlas' else raw / split / name
        relative = source.relative_to(raw).as_posix()
        checksum = sha256(source)
        if checksum != hashes[relative]:
            raise ValueError(f'Original IXI input changed: {source}')
        image, labels = read_ixi(source)
        if image.shape != (160, 192, 224) or image.min() < 0 or image.max() > 1:
            raise ValueError('Unexpected IXI release geometry/intensities')
        if not np.isin(np.unique(labels), list(map(int, spec['freesurfer_id_to_consecutive_id']))).all():
            raise ValueError('Unexpected IXI label IDs')
        stem = Path(name).stem
        affine = np.eye(4)
        paths = {key: f'{key}/{stem}.nii' for key in ('images', 'labels')}
        image_hash = save_volume(output / paths['images'], image, affine)
        label_hash = save_volume(output / paths['labels'], labels.astype(np.uint8), affine)
        return {'subject': stem, 'split': split, 'source': relative, 'source_sha256': checksum,
                **paths, 'image_sha256': image_hash, 'labels_sha256': label_hash,
                'range': [float(image.min()), float(image.max())]}

    records = []
    with ThreadPoolExecutor(workers) as pool:
        for index, row in enumerate(pool.map(convert, members), 1):
            records.append(row)
            if index % 50 == 0:
                print(json.dumps({'dataset': 'ixi', 'prepared': index, 'intended': 577}), flush=True)
    return records, {'shape': [160, 192, 224], 'affine': np.eye(4).tolist(),
                     'intensity': 'author released values unchanged',
                     'geometry': 'released array grid, unit voxel frame; no original world affine supplied'}


@torch.no_grad()
def lpba_arrays(image_path, tissue_path, label_path, air_path):
    """Author inverse AIR interpolation and NumPy normalization, before crop."""
    require_sitreg_dependencies()
    from composable_mapping import LinearInterpolator
    from medical.protocols.sitreg_air import (read_air_file, inverse_transform_volume_by_air,
                                              get_source_voxel_size)
    with air_path.open('rb') as stream:
        air = read_air_file(stream)
    outputs = []
    for path in (image_path, tissue_path):
        image = read_analyze(path)
        source = torch.from_numpy(image.get_fdata()[None, ..., 0])
        transformed = inverse_transform_volume_by_air(
            air, source, image.header['pixdim'][1:4],
            LinearInterpolator(mask_extrapolated_regions=False))
        outputs.append(transformed.numpy()[0])
    values, tissue = outputs
    white_matter = np.abs(tissue - 3) < 1e-5
    if not white_matter.any():
        raise ValueError('No white matter after the author AIR transform')
    mean = np.sum(values * white_matter) / np.sum(white_matter)
    if not np.isfinite(mean) or mean <= 0:
        raise ValueError('Invalid white-matter mean')
    values = values / mean
    values = (values * (tissue > 0)).astype(values.dtype)
    labels = read_analyze(label_path).get_fdata()[..., 0]
    spacing = np.asarray(get_source_voxel_size(air))
    affine = np.diag([float(-spacing[0]), float(-spacing[1]), float(spacing[2]), 1.])
    label_spacing = read_analyze(label_path).header['pixdim'][1:4]
    if not np.array_equal(spacing, label_spacing) or labels.shape != values.shape:
        raise ValueError('AIR reference and manual label grids disagree')
    return values, labels, affine, float(mean)


def convert_lpba(root, output, workers):
    spec = protocols()['lpba40_sitreg']
    crop = spec['data_config']['crop']
    records = []
    # Match author CPU float64 interpolation; no model, optimizer, or CUDA work.
    for index, subject in enumerate(sorted(spec['train_subjects'] + spec['validation_subjects'] + spec['test_subjects']), 1):
        raw = root / 'raw'
        inputs = {}
        for key, suffix in [('image', 'native.brain.bfc.hdr'), ('tissue', 'native.tissue.hdr'),
                            ('labels', 'delineation.structure.label.hdr'), ('air', 'delineation.to.native.air')]:
            matches = list(raw.rglob(f'{subject}.{suffix}'))
            if len(matches) != 1:
                raise ValueError(f'Missing or ambiguous LPBA40 source: {subject}.{suffix}')
            inputs[key] = matches[0]
        values, labels, affine, wm_mean = lpba_arrays(inputs['image'], inputs['tissue'], inputs['labels'], inputs['air'])
        slices = tuple(slice(left, -right if right else None) for left, right in crop)
        affine[:3, 3] = affine[:3, :3] @ np.asarray([pair[0] for pair in crop])
        values, labels = values[slices].copy(), labels[slices].astype(np.uint8)
        if values.shape != (160, 192, 160) or not np.isfinite(values).all():
            raise ValueError('Unexpected prepared LPBA40 grid or intensities')
        allowed = [0] + [i for group in spec['labels'].values() for i in group]
        if not np.isin(np.unique(labels), allowed).all():
            raise ValueError('Unexpected LPBA40 label IDs')
        paths = {key: f'{key}/{subject}.nii' for key in ('images', 'labels')}
        source_records = {}
        for key, path in inputs.items():
            source_records[key] = {'path': str(path.relative_to(root)), 'sha256': sha256(path)}
            if path.suffix == '.hdr':
                payload = path.with_suffix('.img.gz')
                source_records[key]['payload_sha256'] = sha256(payload)
        records.append({'subject': subject, 'sources': source_records, **paths,
                        'image_sha256': save_volume(output / paths['images'], values, affine),
                        'labels_sha256': save_volume(output / paths['labels'], labels, affine),
                        'white_matter_mean_before_normalization': wm_mean,
                        'range': [float(values.min()), float(values.max())]})
        print(json.dumps({'dataset': 'lpba40', 'prepared': index, 'intended': 40}), flush=True)
    return records, {'shape': [160, 192, 160], 'affine': affine.tolist(),
                     'intensity': 'SITReg white-matter mean normalization and tissue foreground masking',
                     'geometry': 'author inverse AIR native-to-delineation preparation, exact crop with affine origin shifted'}


def prepare(dataset, root, workers=2):
    root = Path(root).resolve()
    output = root / 'prepared'
    staging = root / 'prepared.preparing'
    if output.exists() or staging.exists():
        raise FileExistsError('Refusing to overwrite an existing preparation')
    acquisition = json.loads((root / 'acquisition.json').read_text())
    for archive in acquisition['archives']:
        if sha256(archive['path']) != archive['sha256']:
            raise ValueError('Original source archive checksum changed')
    staging.mkdir()
    for name in ('images', 'labels'):
        (staging / name).mkdir()
    try:
        records, geometry = (convert_ixi if dataset == 'ixi' else convert_lpba)(root, staging, workers)
        metadata = {'format_version': 1, 'dataset': dataset, **geometry,
                    'protocol_sha256': sha256(PROTOCOL_PATH),
                    'preparation_protocol_sha256': preparation_fingerprint(dataset),
                    'source_revision': protocols()['source_pins']['transmorph' if dataset == 'ixi' else 'sitreg']['commit']}
        (staging / 'protocol.json').write_text(json.dumps(metadata, indent=2) + '\n')
        packages = ['numpy', 'nibabel', 'torch']
        if dataset == 'lpba40':
            packages += ['composable-mapping', 'deformation-inversion-layer']
        provenance = {**metadata, 'volumes': records,
                      'acquisition_sha256': sha256(root / 'acquisition.json'),
                      'software': {name: importlib.metadata.version(name)
                                   for name in packages},
                      'registration_training_or_recovery_executed': False}
        (staging / 'preparation.json').write_text(json.dumps(provenance, indent=2) + '\n')
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging)
        raise
    print(json.dumps({'prepared_dataset': dataset, 'volumes': len(records), 'output': str(output)}), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=DATASET_NAMES, default=list(DATASET_NAMES))
    parser.add_argument('--data-root', type=Path, default=Path('Data'))
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args(argv)
    if args.workers < 1 or len(set(args.datasets)) != len(args.datasets):
        parser.error('Positive workers and distinct datasets required')
    torch.set_num_threads(4)
    for dataset in args.datasets:
        prepare(dataset, args.data_root / DATASET_NAMES[dataset], args.workers)


if __name__ == '__main__':
    main()
