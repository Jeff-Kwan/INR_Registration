"""Acquire all official ACDC training files for patients001–100; no preparation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request

PAGE = 'https://www.creatis.insa-lyon.fr/Challenge/acdc/databasesTraining.html'
API = 'https://humanheart-project.creatis.insa-lyon.fr/database/api/v1'
TRAINING = '63721d7073e9f0047faa0525'


def fetch(route, params=None):
    url = API + route + ('?' + urllib.parse.urlencode(params) if params else '')
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return response.read()
        except OSError:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def frame_numbers(path):
    fields = dict(line.split(':', 1) for line in Path(path).read_text().splitlines() if ':' in line)
    ed, es = int(fields['ED']), int(fields['ES'])
    if ed < 1 or es < 1 or ed == es:
        raise ValueError(f'Invalid ED/ES in {path}')
    return ed, es


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('Data/ACDC'))
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / 'training_download_manifest.json'
    prior = json.loads(manifest.read_text()) if manifest.exists() else {'files': []}
    known = {r['path']: r for r in prior['files']}
    folders = json.loads(fetch('/folder', {'parentType': 'folder', 'parentId': TRAINING, 'limit': 0}))
    assert {f['name'] for f in folders} == {f'patient{i:03}' for i in range(1, 101)}
    folders.sort(key=lambda f: f['name'])

    def listing(folder):
        items = json.loads(fetch('/item', {'folderId': folder['_id'], 'limit': 0}))
        return folder['name'], {i['name']: i for i in items}

    with ThreadPoolExecutor(args.workers) as pool:
        inventory = dict(pool.map(listing, folders))
    records = []

    def download(job):
        relative, item = job
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            old = known.get(str(relative))
            if old:
                verified = path.stat().st_size == item['size'] and sha256(path) == old['sha256']
            else:
                payload = fetch('/item/' + item['_id'] + '/download')
                verified = len(payload) == item['size'] and sha256(path) == hashlib.sha256(payload).hexdigest()
            if not verified:
                raise FileExistsError(f'Refusing to replace unverified existing source: {path}')
        else:
            payload = fetch('/item/' + item['_id'] + '/download')
            if len(payload) != item['size']:
                raise RuntimeError(f'Incorrect download size: {relative}')
            part = path.with_name(path.name + '.part')
            try:
                with part.open('xb') as stream:
                    stream.write(payload)
                part.rename(path)
            finally:
                part.unlink(missing_ok=True)
        return {'path': str(relative), 'item_id': item['_id'], 'bytes': item['size'],
                'sha256': sha256(path), 'url': API + '/item/' + item['_id'] + '/download'}

    def batch(jobs):
        with ThreadPoolExecutor(args.workers) as pool:
            for record in pool.map(download, jobs):
                records.append(record)
                # Retain progress for safe source verification on a later retry.
                manifest.write_text(json.dumps({'complete': False, 'files': records}, indent=2) + '\n')
                if len(records) % 25 == 0:
                    print('Downloaded', len(records), 'files', flush=True)

    info_jobs = [(Path('training') / name / 'Info.cfg', items['Info.cfg'])
                 for name, items in inventory.items()]
    terms = [('LICENSE_TERMS.md', '66e290e0961576b1bad4ee3c'),
             ('MANDATORY_CITATION.md', '63721b4873e9f0047faa051f')]
    info_jobs += [(Path(name), json.loads(fetch('/item/' + ident))) for name, ident in terms]
    batch(info_jobs)
    remaining = []
    phases = {}
    for name, items in inventory.items():
        ed, es = frame_numbers(root / 'training' / name / 'Info.cfg')
        phases[name] = {'ED': ed, 'ES': es}
        for frame in (ed, es):
            for suffix in ('.nii.gz', '_gt.nii.gz'):
                if f'{name}_frame{frame:02}{suffix}' not in items:
                    raise ValueError(f'Missing ED/ES endpoint in source: {name}')
        if f'{name}_4d.nii.gz' not in items:
            raise ValueError(f'Missing full cine sequence: {name}')
        for filename, item in sorted(items.items()):
            if Path(filename).name != filename:
                raise ValueError('Unsafe source filename')
            if filename != 'Info.cfg':
                remaining.append((Path('training') / name / filename, item))
    print('Full training inventory:', len(folders), 'patients,', len(info_jobs)+len(remaining),
          'files,', sum(item['size'] for _,item in info_jobs+remaining), 'bytes', flush=True)
    batch(remaining)
    # Read headers and endpoint labels; keep original source arrays unchanged.
    import nibabel as nib
    import numpy as np
    validation = []
    for name, phase in phases.items():
        folder = root / 'training' / name
        cine = nib.load(folder/f'{name}_4d.nii.gz')
        if len(cine.shape) != 4 or max(phase.values()) > cine.shape[3]:
            raise ValueError(f'Invalid cine/frame metadata: {name}')
        for tag, frame in phase.items():
            image = nib.load(folder/f'{name}_frame{frame:02}.nii.gz')
            label = nib.load(folder/f'{name}_frame{frame:02}_gt.nii.gz')
            values = np.unique(np.asarray(label.dataobj))
            if image.shape != label.shape or image.shape != cine.shape[:3] or not np.isin(values,[0,1,2,3]).all():
                raise ValueError(f'Endpoint geometry/label mismatch: {name} {tag}')
            if not np.allclose(image.affine,label.affine,rtol=1e-5,atol=1e-5):
                raise ValueError(f'Endpoint image/label affine mismatch: {name} {tag}')
        validation.append({'patient': name, 'cine_shape': list(cine.shape), **phase})
    manifest.write_text(json.dumps({'complete': True, 'official_page': PAGE, 'api': API,
        'training_folder_id': TRAINING, 'patients': phases, 'validation': validation,
        'scope': 'all released patient files including full cine sequences, ED/ES MRI and GT; no preparation or fitting', 'files': records,
        'checksum_note': 'Locally computed SHA-256, not independently published checksums.'}, indent=2) + '\n')
    print('Complete:', len(phases), 'patients,', len(records), 'files', flush=True)


if __name__ == '__main__':
    main()
