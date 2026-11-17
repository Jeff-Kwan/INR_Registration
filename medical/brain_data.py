"""Frozen IXI/TransMorph and LPBA40/SITReg inputs, without model execution."""
from functools import lru_cache
import hashlib
import importlib.metadata
import json
from pathlib import Path
import pickle

import nibabel as nib
import numpy as np

PROTOCOL_PATH = Path(__file__).parent / 'protocols/brain_benchmarks.json'
DATASET_NAMES = {'ixi': 'IXI_TransMorph', 'lpba40': 'LPBA40'}
EXPECTED_SHAPES = {'ixi': (160, 192, 224), 'lpba40': (160, 192, 160)}


def require_sitreg_dependencies():
    """Freeze the mapping implementation used by author preparation and scoring."""
    for package, expected in [('composable-mapping', '0.1.2'),
                              ('deformation-inversion-layer', '1.1.2')]:
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError('Install the brain-protocols extra for LPBA40') from exc
        if actual != expected:
            raise RuntimeError(f'LPBA40 protocol requires {package}=={expected}; found {actual}')


@lru_cache(maxsize=1)
def protocols():
    result = json.loads(PROTOCOL_PATH.read_text())
    for name, key in [('sitreg_air.py', 'vendored_air_sha256'),
                      ('sitreg_reference.py', 'reference_metrics_sha256')]:
        if sha256(PROTOCOL_PATH.parent / name) != result['sitreg_reference'][key]:
            raise RuntimeError(f'Pinned SITReg protocol helper changed: {name}')
    return result


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def preparation_fingerprint(dataset):
    """Image preparation identity, independent of changes to evaluation code."""
    spec = protocols()['ixi_transmorph' if dataset == 'ixi' else 'lpba40_sitreg']
    if dataset == 'ixi':
        data = {key: spec[key] for key in ('split_membership', 'shape', 'freesurfer_id_to_consecutive_id')}
    else:
        data = {key: spec[key] for key in ('train_subjects', 'validation_subjects', 'test_subjects')}
        data['preparation_config'] = {key: spec['data_config'][key] for key in
                                      ('crop', 'normalize', 'downsampling_factor', 'file_type', 'segmentation_file_type')}
        data['air_sha256'] = protocols()['sitreg_reference']['vendored_air_sha256']
    data['source_revision'] = protocols()['source_pins']['transmorph' if dataset == 'ixi' else 'sitreg']['commit']
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


class ArraysOnlyUnpickler(pickle.Unpickler):
    """The author IXI release contains only tuples of NumPy arrays."""
    def find_class(self, module, name):
        allowed = {('numpy', 'ndarray'), ('numpy', 'dtype')}
        allowed |= {(prefix, operation)
                    for prefix in ('numpy.core.multiarray', 'numpy._core.multiarray')
                    for operation in ('_reconstruct', 'scalar')}
        allowed |= {(prefix, '_frombuffer')
                    for prefix in ('numpy.core.numeric', 'numpy._core.numeric')}
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f'Unexpected class {module}.{name}')
        return super().find_class(module, name)


def read_ixi(path):
    with Path(path).open('rb') as stream:
        result = ArraysOnlyUnpickler(stream).load()
    if type(result) is not tuple or len(result) != 2:
        raise ValueError('Expected the released (image, segmentation) tuple')
    image, labels = result
    if not isinstance(image, np.ndarray) or not isinstance(labels, np.ndarray):
        raise ValueError('Expected NumPy image and label arrays')
    if image.ndim != 3 or labels.shape != image.shape:
        raise ValueError('Image and labels must have matching 3D shapes')
    if not np.isfinite(image).all() or not np.isfinite(labels).all():
        raise ValueError('Nonfinite input')
    if not np.equal(labels, np.floor(labels)).all():
        raise ValueError('Noninteger segmentation')
    return image, labels


def read_analyze(path):
    """Read LONI's .hdr/.img.gz pairs without modifying/decompressing sources."""
    image = nib.load(str(path))
    if not Path(image.file_map['image'].filename).exists():
        mapping = image.file_map
        mapping['image'].filename = str(mapping['image'].filename) + '.gz'
        image = type(image).from_file_map(mapping)
    return image


def pair_list(dataset, split='test'):
    if dataset not in DATASET_NAMES:
        raise ValueError('Unknown brain benchmark')
    if split not in ('test', 'validation'):
        raise ValueError('Only frozen test and validation pairs are supported')
    key = 'ixi_transmorph' if dataset == 'ixi' else 'lpba40_sitreg'
    spec = protocols()[key]
    if dataset == 'ixi':
        folder = 'Test' if split == 'test' else 'Val'
        return [{'moving': 'atlas', 'fixed': Path(name).stem}
                for name in spec['split_membership'][folder]]
    return [{'moving': row['image_1'], 'fixed': row['image_2']}
            for row in spec[f'{split}_pairs']]


def prepared_paths(root, pair):
    root = Path(root)
    return {key: root / 'images' / f'{pair[key]}.nii' for key in ('moving', 'fixed')}


def label_path(volume):
    path = Path(volume['path'])
    if path.parent.name != 'images' or path.suffix != '.nii':
        raise ValueError('Expected a prepared brain benchmark image')
    return path.parent.parent / 'labels' / path.name


def validate_case(case, dataset):
    """Check preparation provenance and membership before any fitting."""
    if dataset not in DATASET_NAMES:
        raise ValueError('Unknown brain benchmark')
    roots = {Path(case[key]['path']).resolve().parent.parent for key in ('moving', 'fixed')}
    if len(roots) != 1:
        raise ValueError('Images must come from one prepared release')
    root = roots.pop()
    meta = json.loads((root / 'protocol.json').read_text())
    if (meta['dataset'] != dataset or meta['format_version'] != 1
            or meta.get('preparation_protocol_sha256') != preparation_fingerprint(dataset)):
        raise ValueError('Prepared release protocol does not match the pinned specification')
    if tuple(meta['shape']) != EXPECTED_SHAPES[dataset]:
        raise ValueError('Prepared shape differs from the benchmark protocol')
    ids = {key: Path(case[key]['path']).stem for key in ('moving', 'fixed')}
    matches = [(split, i) for split in ('test', 'validation')
               for i, pair in enumerate(pair_list(dataset, split)) if pair == ids]
    if len(matches) != 1:
        raise ValueError('Pair is not in a frozen test/validation list')
    for key in ('moving', 'fixed'):
        volume = case[key]
        if list(volume['data'].shape) != meta['shape']:
            raise ValueError('Prepared image shape changed')
        if not np.allclose(volume['affine'], meta['affine'], rtol=0, atol=1e-6):
            raise ValueError('Prepared image affine changed')
        labels = nib.load(label_path(volume))
        if labels.shape != volume['data'].shape or not np.allclose(labels.affine, volume['affine']):
            raise ValueError('Evaluation label geometry does not match its image')
    split, index = matches[0]
    return {'split': split, 'pair_index': index + 1, 'preparation': meta}
