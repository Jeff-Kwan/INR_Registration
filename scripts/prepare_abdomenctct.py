"""Extract official release unchanged, verify validation inputs, record provenance."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile
import nibabel as nib
import numpy as np

REVISION='88475095e35442087a24439ff34ec09f54bc7dac'

def main(dataset="abdomenctct"):
    name={"abdomenctct":"AbdomenCTCT","oasis":"OASIS"}[dataset]
    count=45 if dataset=="abdomenctct" else 19
    label_count=13 if dataset=="abdomenctct" else 35
    archive=Path(f'Data/_downloads/{name}.zip')
    destination=Path(f'Data/{name}')
    if destination.exists():
        raise FileExistsError('Refusing to overwrite the prepared release')
    with ZipFile(archive) as z:
        if z.testzip() is not None: raise ValueError('Archive CRC failure')
        for info in z.infolist():
            relative=Path(info.filename)
            if relative.is_absolute() or '..' in relative.parts or relative.parts[0]!=name:
                raise ValueError('Unexpected archive path')
        z.extractall('Data')
    protocol=json.loads(Path(f'medical/protocols/{dataset}_val.json').read_text())
    pairs=protocol['eval_pairs']
    assert len(pairs)==count and len({(p['fixed'],p['moving']) for p in pairs})==count
    records=[]
    paths=sorted({p[k] for p in pairs for k in ('fixed','moving')})
    for path in paths:
        image_path=destination/path
        label_path=destination/path.replace('imagesTr','labelsTr')
        image,label=nib.load(image_path),nib.load(label_path)
        values=np.asarray(image.dataobj); labels=np.asarray(label.dataobj)
        assert image.shape==tuple(protocol["expected_shape"][:3]) and label.shape==image.shape
        assert np.allclose(image.affine,label.affine) and np.allclose(image.header.get_zooms()[:3],2 if dataset=="abdomenctct" else 1)
        assert np.isfinite(values).all() and np.isin(labels,range(label_count+1)).all()
        records.append({'image':path,'image_sha256':hashlib.file_digest(image_path.open('rb'),'sha256').hexdigest(),
                        'label_sha256':hashlib.file_digest(label_path.open('rb'),'sha256').hexdigest(),
                        'affine':image.affine.tolist(),'range':[float(values.min()),float(values.max())]})
    assert all(np.allclose(r['affine'],records[0]['affine']) for r in records)
    manifest={'archive_sha256':hashlib.file_digest(archive.open('rb'),'sha256').hexdigest(),
              'official_evaluator_revision':REVISION,'preprocessing':'official release unchanged; no resampling, clipping, masks or affine fitting',
              'evaluation_pairs':pairs,'subjects':records}
    (destination/'preparation.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Verified unchanged {name} release: {len(records)} validation subjects, {count} directed pairs, {label_count} labels')

if __name__=='__main__': main()
