"""Prepare DIR-Lab 4DCT extreme phases on their native voxel grids."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import importlib.metadata
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from lungmask import LMInferer
import torch

SOURCE = Path('Data/DIRLab4DCT')
DEST = Path('Data/DIRLab4DCT_IDIR')
# Native arrays are stored in z,y,x order. Physical origin is an arbitrary
# common local frame: original packets do not provide patient-world origin.
SHAPES = [(94,256,256),(112,256,256),(104,256,256),(99,256,256),(106,256,256),
          (128,512,512),(136,512,512),(128,512,512),(128,512,512),(120,512,512)]
SPACING = [(0.97,0.97,2.5),(1.16,1.16,2.5),(1.15,1.15,2.5),(1.13,1.13,2.5),
           (1.10,1.10,2.5)]+[(0.97,0.97,2.5)]*5


def only(paths):
    paths=list(paths)
    if len(paths)!=1: raise ValueError(f'Expected one source file, found {paths}')
    return paths[0]


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def main():
    if DEST.exists(): raise FileExistsError(f'Refusing to overwrite {DEST}')
    DEST.mkdir(parents=True)
    inferer=LMInferer(modelname='R231', batch_size=8)
    manifest={'reference':'IDIR MIDL 2022; lungmask R231', 'lungmask_version':importlib.metadata.version('lungmask'),
              'torch_version':torch.__version__, 'model_weights_sha256':digest(Path(torch.hub.get_dir())/'checkpoints/unet_r231-d5d2fc3d.pth'), 'cases':[],
              'coordinates':'NIfTI array axes x=right-to-left, y=anterior-to-posterior, z=superior-to-inferior; arbitrary local RAS origin; landmark indices zero-based',
              'intensity':'NIfTI retains raw int16 values as in IDIR; lungmask inference input uses raw minus 1024 HU',
              'mask_note':'R231 model is reference approach; exact IDIR mask model version and weights are not specified'}
    for n,(shape,spacing) in enumerate(zip(SHAPES,SPACING),1):
        source=only(SOURCE.glob(f'Case{n}Pack')) if n!=8 else only(SOURCE.glob('Case8Deploy'))
        target=DEST/f'case{n:02d}';target.mkdir()
        affine=np.diag([-spacing[0],-spacing[1],-spacing[2],1.0])
        record={'case':n,'shape_xyz':list(shape[::-1]),'spacing_mm':spacing,'phases':{}}
        for phase in ('00','50'):
            raw=only((source/'Images').glob(f'case{n}_T{phase}*.img'))
            vals=np.fromfile(raw,dtype='<i2')
            if vals.size != np.prod(shape): raise ValueError(f'Unexpected raw size: {raw}')
            arr=vals.reshape(shape).transpose(2,1,0)
            name=f'T{phase}.nii.gz'; nib.save(nib.Nifti1Image(arr,affine),target/name)
            marks=only(p for p in source.rglob('*300*') if p.is_file() and f'T{phase}' in p.name and p.suffix=='.txt')
            pts=np.loadtxt(marks,dtype=np.int64)-1
            if pts.shape!=(300,3) or (pts<0).any() or (pts>=np.array(shape[::-1])).any(): raise ValueError(f'Invalid landmarks: {marks}')
            np.savetxt(target/f'T{phase}_landmarks_voxel.csv',pts,fmt='%d',delimiter=',',header='x,y,z',comments='')
            record['phases'][phase]={'raw':str(raw),'raw_sha256':digest(raw),'landmarks':str(marks),'landmarks_sha256':digest(marks),'image':name}
            if phase=='00':
                hu=sitk.GetImageFromArray(vals.reshape(shape).astype(np.float32)-1024.0)
                hu.SetSpacing(spacing)
                mask=(inferer.apply(hu)>0).astype(np.uint8)
                if mask.shape!=shape or not mask.any(): raise ValueError(f'Invalid mask for case {n}')
                nib.save(nib.Nifti1Image(mask.transpose(2,1,0),affine),target/'T00_lung_mask.nii.gz')
                record['mask_voxels']=int(mask.sum())
        manifest['cases'].append(record)
        print(f'case {n}: prepared, mask voxels {record["mask_voxels"]}',flush=True)
        (DEST/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Preparation complete',flush=True)


if __name__=='__main__': main()
