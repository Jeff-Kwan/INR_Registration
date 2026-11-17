"""Verify Neurite-OASIS v1.0 and copy the frozen last-50/consecutive-49 cohort.

No resampling, normalization, training, or recovery. Source archive is retained.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile

import nibabel as nib
import numpy as np

ARCHIVE_SHA256 = '86dd117dda17f736ade8a4088d7e98e066e1181950fe8b406f1a35f7fb743e78'
RELEASE_COMMIT = 'd3702232241b81a716f1db3dfa6c8502ad6d8027'
FILES = ('aligned_norm.nii.gz', 'aligned_seg35.nii.gz')


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare(archive, output):
    archive, output = Path(archive), Path(output)
    if output.exists():
        raise FileExistsError(output)
    digest = sha256(archive)
    if digest != ARCHIVE_SHA256:
        raise ValueError(f'Archive checksum mismatch: {digest}')
    print('Archive SHA-256 verified', flush=True)
    with tarfile.open(archive, 'r:') as source:
        members = source.getmembers()
        subjects = set()
        lookup = {}
        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or '..' in path.parts:
                raise ValueError(f'Unsafe archive path: {member.name}')
            ids = [part for part in path.parts if part.startswith('OASIS_OAS1_')]
            if ids:
                if len(ids) != 1:
                    raise ValueError(member.name)
                subjects.add(ids[0])
                if path.name in FILES and member.isfile():
                    key = (ids[0], path.name)
                    if key in lookup:
                        raise ValueError(f'Duplicate member: {key}')
                    lookup[key] = member
        subjects = sorted(subjects)
        if len(subjects) != 414 or any((sid, name) not in lookup for sid in subjects for name in FILES):
            raise ValueError('Expected 414 complete subject directories')
        selected = subjects[-50:]
        if selected[0] != 'OASIS_OAS1_0404_MR1' or selected[-1] != 'OASIS_OAS1_0457_MR1':
            raise ValueError(f'Unexpected cohort endpoints: {selected[0]}, {selected[-1]}')
        output.mkdir(parents=True)
        records = []
        for sid in selected:
            directory = output / sid
            directory.mkdir()
            checksums = {}
            archive_members = {}
            for name in FILES:
                with source.extractfile(lookup[sid, name]) as src, (directory / name).open('xb') as dst:
                    shutil.copyfileobj(src, dst)
                checksums[name] = sha256(directory / name)
                archive_members[name] = lookup[sid, name].name
            image, label = [nib.load(directory / name) for name in FILES]
            values = image.get_fdata(dtype=np.float32)
            labels = np.asanyarray(label.dataobj)
            codes = np.unique(labels)
            if image.shape != (160, 192, 224) or label.shape != image.shape:
                raise ValueError(f'{sid}: unexpected shape')
            if not np.isfinite(image.affine).all() or not np.allclose(np.linalg.norm(image.affine[:3,:3],axis=0),1) or not np.allclose(image.affine, label.affine, atol=1e-6) or not np.allclose(image.header.get_zooms()[:3], 1) or not np.allclose(label.header.get_zooms()[:3], 1):
                raise ValueError(f'{sid}: mismatched geometry')
            if not np.isfinite(values).all() or not values.max() > values.min():
                raise ValueError(f'{sid}: invalid intensity values')
            if not np.isin(codes, np.arange(36)).all():
                raise ValueError(f'{sid}: invalid labels {codes}')
            records.append(dict(subject=sid, shape=list(image.shape), spacing=list(map(float,image.header.get_zooms()[:3])), affine=image.affine.tolist(), intensity_range=[float(values.min()),float(values.max())], labels=codes.tolist(), sha256=checksums, archive_members=archive_members))
            print(f'Validated {sid}', flush=True)
        metadata = output / 'source_metadata'
        metadata.mkdir()
        # Retain non-subject release documents verbatim, if present.
        for index, member in enumerate(members):
            path = PurePosixPath(member.name)
            if member.isfile() and not any(part.startswith('OASIS_OAS1_') for part in path.parts):
                with source.extractfile(member) as src, (metadata / f'{index}_{path.name}').open('xb') as dst:
                    shutil.copyfileobj(src, dst)
    pairs = [dict(pair_index=i+1, fixed=a, moving=b, fixed_image=f'{a}/{FILES[0]}', moving_image=f'{b}/{FILES[0]}', fixed_label=f'{a}/{FILES[1]}', moving_label=f'{b}/{FILES[1]}') for i,(a,b) in enumerate(zip(selected[:-1],selected[1:]))]
    manifest = dict(protocol='neurite_oasis_gaussiandir_last50_consecutive49_v1', created_utc=datetime.now(timezone.utc).isoformat(), source_url=f'https://huggingface.co/datasets/DalcaLab/neurite-oasis/resolve/{RELEASE_COMMIT}/neurite-oasis.v1.0.tar', release_commit=RELEASE_COMMIT, archive_sha256=digest, archive_subject_count=len(subjects), all_subjects=subjects, selected_subjects=selected, subjects=records, pairs=pairs, preprocessing='Byte-identical copies of released aligned_norm and aligned_seg35; no spatial or intensity changes', labels_evaluation_only=True, evaluation=dict(labels=list(range(1,36)), convention='Repository Learn2Reg-style Dice and voxel HD95; no GaussianDIR/FireANTs metric parity claim'), provenance=dict(gaussiandir='https://pmc.ncbi.nlm.nih.gov/articles/PMC12361785/', fireants='https://github.com/rohitrango/FireANTs/blob/c8f466cb790043253ed91c0e923658c5dd76bba2/fireants/scripts/pairwise/oasis.py'), environment={name:__import__('importlib.metadata',fromlist=['version']).version(name) for name in ('numpy','nibabel')})
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(subjects=50, images=50, labels=50, directed_pairs=len(pairs), manifest=str(output/'manifest.json'))),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    prepare(args.archive,args.output)
