#!/usr/bin/env python3
"""Fetch an LPBA40 archive and prepare the project's fixed/test/train layout.

Never overwrites a destination. Imports no training code. Prepared NIfTI files
are copied byte-for-byte; Analyze files are converted only on a compatible grid.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import urllib.parse
import urllib.request
import zipfile

import nibabel as nib
import numpy as np

SHAPE = (160, 192, 160)
AFFINE = np.diag([-1., -1., 1., 1.])
OFFICIAL_PAGE = 'https://loni.usc.edu/research/atlas_downloads'


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def unpack(archive, destination):
    """Extract regular files only, rejecting links and traversal paths."""
    def safe(name):
        # Backslashes can be interpreted as separators on other platforms.
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or '\\' in name:
            raise ValueError(f'Unsafe archive member: {name}')
        return destination.joinpath(*path.parts)

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as handle:
            for member in handle.infolist():
                target = safe(member.filename)
                if (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError('Archive symlinks are not supported')
                if member.is_dir():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with handle.open(member) as src, target.open('wb') as dst:
                    shutil.copyfileobj(src, dst)
    else:
        with tarfile.open(archive) as handle:
            for member in handle:
                target = safe(member.name)
                if member.isdir():
                    continue
                if not member.isfile():
                    raise ValueError(f'Unsupported archive member: {member.name}')
                target.parent.mkdir(parents=True, exist_ok=True)
                with handle.extractfile(member) as src, target.open('wb') as dst:
                    shutil.copyfileobj(src, dst)


def locate(root, subject, label=False):
    stem = f'{subject}.delineation.' + ('structure.label' if label else 'skullstripped')
    matches = []
    for extension in ('.nii.gz', '.nii', '.hdr'):
        matches.extend(root.rglob(stem + extension))
    if subject == 'S01' and not label and not matches:
        matches = list(root.rglob('fixed.nii.gz'))
    # Existing data may include derived labels in a SyNCC subfolder.
    matches = [p for p in matches if not any(x.lower() == 'syncc' for x in p.parts)]
    if len(matches) != 1:
        raise ValueError(f'Expected one input for {stem}, found {len(matches)}')
    return matches[0]


def inspect(path, label=False):
    image = nib.load(str(path))
    if image.shape != SHAPE or not np.allclose(image.affine, AFFINE, atol=1e-5):
        raise ValueError(f'{path.name}: incompatible grid. Expected {SHAPE} and '
                         'diag(-1,-1,1,1). Original raw-to-grid preprocessing is '
                         'not recorded; supply the prepared release, not an atlas '
                         'or unprocessed native/delineation release.')
    data = image.get_fdata(dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError(f'{path.name}: nonfinite values')
    if label:
        if np.any(data < 0) or not np.array_equal(data, np.rint(data)):
            raise ValueError(f'{path.name}: labels must be nonnegative integers')
    elif data.min() < 0 or data.max() > 1.000001 or data.max() <= 0:
        raise ValueError(f'{path.name}: expected prepared intensities in [0,1]')
    return image


def prepare(source, destination):
    source, destination = source.resolve(), destination.resolve()
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite existing destination: {destination}')
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError('Source and destination must be separate, non-nested directories')
    inputs = []
    # Validate all 80 inputs before writing any prepared output.
    for number in range(1, 41):
        subject = f'S{number:02}'
        for label in (False, True):
            path = locate(source, subject, label)
            inspect(path, label)
            if label:
                relative = Path('label') / f'{subject}.delineation.structure.label.nii.gz'
            elif number == 1:
                relative = Path('delineation_l_norm/fixed.nii.gz')
            else:
                split = 'test' if number <= 10 else 'train'
                relative = Path('delineation_l_norm') / split / f'{subject}.delineation.skullstripped.nii.gz'
            inputs.append((path, relative))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.lpba40-prepare-', dir=destination.parent) as tmp:
        staging = Path(tmp) / 'prepared'
        staging.mkdir()
        records = []
        for src, relative in inputs:
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if src.name.endswith('.nii.gz'):
                shutil.copyfile(src, target)
            else:
                image = nib.load(str(src))
                converted = nib.Nifti1Image(np.asanyarray(image.dataobj), image.affine)
                nib.save(converted, str(target))
            records.append({'source_name': src.name, 'path': str(relative),
                            'sha256': digest(target)})
        (staging / 'setup_manifest.json').write_text(json.dumps({
            'dataset': 'LPBA40', 'shape': SHAPE, 'affine': AFFINE.tolist(),
            'fixed': 'S01', 'test': 'S02-S10', 'train': 'S11-S40',
            'processing': 'Prepared inputs preserved; no resampling or intensity changes',
            'runtime_normalization': 'Pipeline applies nonzero foreground 1st/99th percentiles',
            'files': records}, indent=2) + '\n')
        # Destination stays absent until the validated layout is complete.
        if destination.exists():
            raise FileExistsError(destination)
        staging.rename(destination)
    return len(inputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--source', type=Path, help='Extracted prepared dataset or local zip/tar archive')
    source.add_argument('--url', help='Authorized direct archive URL; obtain access from LONI first')
    parser.add_argument('--sha256', help='Expected archive checksum (required with --url)')
    parser.add_argument('--destination', type=Path, required=True,
                        help='New LPBA40_delineation directory; existing paths are refused')
    args = parser.parse_args()
    if args.destination.exists():
        parser.error('Destination already exists; choose a new path. Existing data is never overwritten.')
    if args.url and (not args.sha256 or urllib.parse.urlparse(args.url).scheme != 'https'):
        parser.error('--url requires HTTPS and --sha256; no credentials or agreement acceptance are automated')
    with tempfile.TemporaryDirectory(prefix='lpba40-source-') as tmp:
        temporary = Path(tmp)
        path = args.source
        if args.url:
            path = temporary / 'download.archive'
            with urllib.request.urlopen(args.url, timeout=120) as response, path.open('wb') as out:
                shutil.copyfileobj(response, out)
        if path.is_dir():
            if args.sha256:
                parser.error('--sha256 applies to archives, not directories')
            root = path
        else:
            if args.sha256 and digest(path).lower() != args.sha256.lower():
                raise ValueError('Archive SHA256 mismatch')
            root = temporary / 'extracted'
            root.mkdir()
            unpack(path, root)
        count = prepare(root, args.destination)
    print(f'Prepared {count} files at {args.destination}')


if __name__ == '__main__':
    main()
