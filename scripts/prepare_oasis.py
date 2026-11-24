"""Extract the official Learn2Reg OASIS release and verify the validation inputs."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import nibabel as nib
import numpy as np


REVISION = '88475095e35442087a24439ff34ec09f54bc7dac'


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    pair_count = 19
    label_count = 35
    archive = Path('Data/_downloads/OASIS.zip')
    destination = Path('Data/OASIS')
    if destination.exists():
        raise FileExistsError('Refusing to overwrite the prepared release')
    with ZipFile(archive) as release:
        if release.testzip() is not None:
            raise ValueError('Archive CRC failure')
        for info in release.infolist():
            relative = Path(info.filename)
            if relative.is_absolute() or '..' in relative.parts or relative.parts[:1] != (destination.name,):
                raise ValueError('Unexpected archive path')
        release.extractall(destination.parent)

    protocol = json.loads(Path('medical/protocols/oasis_val.json').read_text())
    pairs = protocol['eval_pairs']
    assert len(pairs) == pair_count and len({(pair['fixed'], pair['moving']) for pair in pairs}) == pair_count
    paths = sorted({pair[role] for pair in pairs for role in ('fixed', 'moving')})
    records = []
    for path in paths:
        image_path = destination / path
        label_path = destination / path.replace('imagesTr', 'labelsTr')
        image, label = nib.load(image_path), nib.load(label_path)
        values, labels = np.asarray(image.dataobj), np.asarray(label.dataobj)
        assert image.shape == tuple(protocol['expected_shape'][:3]) and image.shape == label.shape
        assert np.allclose(image.affine, label.affine) and np.allclose(image.header.get_zooms()[:3], 1)
        assert np.isfinite(values).all() and np.isin(labels, range(label_count + 1)).all()
        records.append({
            'image': path,
            'image_sha256': sha256(image_path),
            'label_sha256': sha256(label_path),
            'affine': image.affine.tolist(),
            'range': [float(values.min()), float(values.max())],
        })
    assert all(np.allclose(record['affine'], records[0]['affine']) for record in records)

    manifest = {
        'archive_sha256': sha256(archive),
        'official_evaluator_revision': REVISION,
        'preprocessing': 'official release unchanged; no resampling, clipping, masks or affine fitting',
        'evaluation_pairs': pairs,
        'subjects': records,
    }
    (destination / 'preparation.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Verified unchanged OASIS release: {len(records)} validation subjects, {pair_count} directed pairs, {label_count} labels')


if __name__ == '__main__':
    main()
