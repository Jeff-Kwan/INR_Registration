"""Freeze 128 patient-disjoint development pairs and prepare their ACDC inputs only."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import importlib.metadata
import json
from pathlib import Path
import sys
import tempfile

import nibabel as nib
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from medical.brain_data import protocols,sha256
from scripts.prepare_acdc import preprocess,SPACING,REVISION

ROOT=Path(__file__).resolve().parents[1]
ACDC_PREP=ROOT/'Data/ACDC/prepared_training_tuning32'
OUTPUT=ROOT/'Data/Tuning32'
OASIS_OVERLAP = Path('medical/protocols/oasis_subject_overlap.json')


def select(seed=42):
    spec=protocols();rows={}
    def rng(index):return np.random.default_rng(np.random.SeedSequence([seed,index]))
    generator=rng(0)
    ids=generator.choice(np.arange(1,365),64,replace=False).tolist()
    rows['oasis']=[dict(moving=f'OASIS_{a:04d}',fixed=f'OASIS_{b:04d}') for a,b in zip(ids[::2],ids[1::2])]
    generator=rng(1)
    ids=generator.choice(np.arange(1,101),32,replace=False).tolist()
    directions=generator.permutation(['ED_to_ES']*16+['ES_to_ED']*16).tolist()
    rows['acdc']=[dict(moving=f'patient{i:03d}_{d.split("_to_")[0]}',fixed=f'patient{i:03d}_{d.split("_to_")[1]}',patient=f'patient{i:03d}') for i,d in zip(ids,directions)]
    generator=rng(2)
    ids=generator.permutation(spec['lpba40_sitreg']['train_subjects']).tolist()
    edges=[(ids[i],ids[(i+1)%len(ids)]) for i in range(len(ids))]
    present={frozenset(e) for e in edges}
    # A random Hamiltonian cycle covers everyone twice; seven disjoint extra
    # edges give fourteen subjects degree3 and eleven degree2, with no repeats.
    for attempt in range(1000):
        extra=generator.choice(ids,14,replace=False).tolist()
        matches=list(zip(extra[::2],extra[1::2]))
        if not any(frozenset(e) in present for e in matches):break
    else:raise RuntimeError('Could not form balanced distinct LPBA edges')
    edges+=matches
    edges=[e if generator.integers(2) else e[::-1] for e in edges]
    rows['lpba40']=[dict(moving=edges[i][0],fixed=edges[i][1]) for i in generator.permutation(32)]
    generator=rng(3)
    ids=generator.choice(spec['ixi_transmorph']['split_membership']['Train'],32,replace=False)
    rows['ixi']=[dict(moving='atlas',fixed=Path(i).stem) for i in ids]
    for dataset,pairs in rows.items():
        for i,pair in enumerate(pairs,1):pair.update(dataset=dataset,pair_index=i)
    validate_selection(rows)
    return rows


def validate_selection(rows):
    spec=protocols()
    if set(rows)!={'oasis','acdc','lpba40','ixi'} or any(len(p)!=32 for p in rows.values()):
        raise ValueError('Exactly32 pairs per dataset required')
    oasis=[p[k] for p in rows['oasis'] for k in ('moving','fixed')]
    if len(set(oasis))!=64 or any(not 1<=int(s.split('_')[1])<=364 for s in oasis):raise ValueError('OASIS development/test overlap or reused subject')
    acdc=rows['acdc'];patients=[p['patient'] for p in acdc]
    if len(set(patients))!=32 or any(not 1<=int(p[7:])<=100 for p in patients):raise ValueError('ACDC development/test overlap or reused patient')
    for p in acdc:
        if {p['moving'],p['fixed']}!={p['patient']+'_ED',p['patient']+'_ES'}:raise ValueError('Expected same-patient phase pair')
    if sum(p['moving'].endswith('_ED') for p in acdc)!=16:raise ValueError('ACDC phase directions must balance16/16')
    lpba=rows['lpba40'];uses=Counter(p[k] for p in lpba for k in ('moving','fixed'))
    if set(uses)!=set(spec['lpba40_sitreg']['train_subjects']) or Counter(uses.values())!={2:11,3:14}:raise ValueError('LPBA development coverage/degree mismatch')
    if len({frozenset((p['moving'],p['fixed'])) for p in lpba})!=32 or any(p['moving']==p['fixed'] for p in lpba):raise ValueError('Repeated LPBA pair')
    targets=[p['fixed'] for p in rows['ixi']]
    train={Path(x).stem for x in spec['ixi_transmorph']['split_membership']['Train']}
    if len(set(targets))!=32 or not set(targets)<=train or any(p['moving']!='atlas' for p in rows['ixi']):raise ValueError('IXI split/task mismatch')


def prepare_acdc(selection,workers):
    acquisition_path=ROOT/'Data/ACDC/training_download_manifest.json'
    acquisition=json.loads(acquisition_path.read_text())
    if not acquisition['complete'] or len(acquisition['patients'])!=100:raise ValueError('Complete ACDC training sources required')
    patients=[p['patient'] for p in selection]
    known={r['path']:r for r in acquisition['files']}
    for patient in patients:
        for name in ('Info.cfg',*[f'{patient}_frame{acquisition["patients"][patient][phase]:02d}{suffix}' for phase in ('ED','ES') for suffix in ('.nii.gz','_gt.nii.gz')]):
            relative=f'training/{patient}/{name}';path=ROOT/'Data/ACDC'/relative
            if sha256(path)!=known[relative]['sha256']:raise ValueError(f'Source changed: {relative}')
    if ACDC_PREP.exists():
        manifest=json.loads((ACDC_PREP/'manifest.json').read_text())
        if manifest['selected_patients']!=patients or manifest['source_acquisition_sha256']!=sha256(acquisition_path):raise ValueError('Existing training preparation is a different selection/source')
        return manifest
    with tempfile.TemporaryDirectory(prefix='.acdc-tuning-',dir=ACDC_PREP.parent) as scratch:
        staging=Path(scratch)/'prepared';(staging/'images').mkdir(parents=True);(staging/'labels').mkdir()
        def patient_record(patient):
            raw=ROOT/'Data/ACDC/training'/patient
            fields=dict(line.split(':',1) for line in (raw/'Info.cfg').read_text().splitlines() if ':' in line)
            frames={phase:int(fields[phase]) for phase in ('ED','ES')}
            es=nib.load(raw/f'{patient}_frame{frames["ES"]:02d}.nii.gz');spacing=es.header.get_zooms()[:3]
            records=[]
            for phase,frame in frames.items():
                image_path=raw/f'{patient}_frame{frame:02d}.nii.gz';label_path=raw/f'{patient}_frame{frame:02d}_gt.nii.gz'
                image=nib.load(image_path);label=nib.load(label_path)
                values,codes,offset=preprocess(image.get_fdata(dtype=np.float32),label.get_fdata(dtype=np.float32),spacing)
                affine=np.diag([*SPACING,1.]);affine[:3,3]=offset*SPACING
                name=f'{patient}_{phase}.nii.gz'
                for folder,array in [('images',values.astype(np.float32)),('labels',codes.astype(np.uint8))]:nib.save(nib.Nifti1Image(array,affine),staging/folder/name)
                records.append(dict(subject=patient,phase=phase,frame=frame,shape=list(values.shape),affine=affine.tolist(),es_spacing=list(map(float,spacing)),crop_offset=offset.tolist(),image=f'images/{name}',label=f'labels/{name}',image_sha256=sha256(staging/'images'/name),label_sha256=sha256(staging/'labels'/name),source_image_sha256=sha256(image_path),source_label_sha256=sha256(label_path)))
            print('Prepared development',patient,flush=True)
            return records
        with ThreadPoolExecutor(workers) as pool:records=[r for group in pool.map(patient_record,patients) for r in group]
        manifest=dict(protocol='acdc_kan_idir_training_tuning32_v1',source_revision=REVISION,spacing=SPACING,selected_patients=patients,source_acquisition_sha256=sha256(acquisition_path),preprocessing='same preprocess() as frozen KAN-IDIR test preparation; ES spacing, cubic/edge MRI, nearest/edge labels, minmax before crop, no padding',subjects=records)
        (staging/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');staging.rename(ACDC_PREP)
    return manifest


def paths(pair):
    ds=pair['dataset']
    if ds=='oasis':
        root=ROOT/'Data/OASIS'
        return {f'{k}_{kind}':str((root/folder/f'{pair[k]}_0000.nii.gz').relative_to(ROOT)) for k in ('moving','fixed') for kind,folder in [('image','imagesTr'),('label','labelsTr')]}
    root=ACDC_PREP if ds=='acdc' else ROOT/('Data/IXI_TransMorph/prepared' if ds=='ixi' else 'Data/LPBA40/prepared')
    suffix='.nii.gz' if ds=='acdc' else '.nii'
    return {f'{k}_{kind}':str((root/folder/(pair[k]+suffix)).relative_to(ROOT)) for k in ('moving','fixed') for kind,folder in [('image','images'),('label','labels')]}


def verify_files(pairs,workers):
    image_paths={p[k+'_image'] for p in pairs for k in ('moving','fixed')}
    label_paths={p[k+'_label'] for p in pairs for k in ('moving','fixed')}
    def record(relative):
        path=ROOT/relative;image=nib.load(path);array=np.asarray(image.dataobj)
        if array.ndim!=3 or not np.isfinite(array).all():raise ValueError(f'Invalid prepared volume: {relative}')
        if relative in image_paths and np.ptp(array)<=0:raise ValueError('Constant prepared image')
        if relative in label_paths and (np.any(array<0) or not np.equal(array,np.rint(array)).all()):raise ValueError(f'Invalid label codes: {relative}')
        return relative,dict(sha256=sha256(path),bytes=path.stat().st_size,shape=list(image.shape),affine=image.affine.tolist())
    with ThreadPoolExecutor(workers) as pool:records=dict(pool.map(record,sorted(image_paths|label_paths)))
    for p in pairs:
        a,b=[records[p[k+'_image']] for k in ('moving','fixed')]
        if a['shape']!=b['shape'] or not np.allclose(a['affine'],b['affine'],atol=1e-5):raise ValueError('Pair geometry mismatch')
        for k in ('moving','fixed'):
            i,l=records[p[k+'_image']],records[p[k+'_label']]
            if i['shape']!=l['shape'] or not np.allclose(i['affine'],l['affine'],atol=1e-5):raise ValueError('Image/label geometry mismatch')
    return records


def verify_frozen(workers=4):
    manifest=json.loads((OUTPUT/'manifest.json').read_text())
    validate_selection(manifest['selection'])
    expected=[dict(r,**paths(r)) for group in manifest['selection'].values() for r in group]
    if manifest['seed']!=42 or manifest['selection']!=select() or manifest['pairs']!=expected:
        raise ValueError('Frozen selection changed')
    for relative,digest in manifest['dependencies'].items():
        # Existing frozen manifests record the original location of the same file.
        path = OASIS_OVERLAP if relative == 'results/data_audit_20261117/oasis_subject_overlap.json' else relative
        if sha256(ROOT/path)!=digest:raise ValueError(f'Dependency changed: {relative}')
    def check(item):
        relative,record=item
        if sha256(ROOT/relative)!=record['sha256']:raise ValueError(f'Input changed: {relative}')
    with ThreadPoolExecutor(workers) as pool:list(pool.map(check,manifest['files'].items()))
    print('Verified frozen cohort: 128 pairs,',len(manifest['files']),'files; dependencies and selection unchanged')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workers',type=int,default=4);p.add_argument('--plan-only',action='store_true');p.add_argument('--verify-only',action='store_true');a=p.parse_args(argv)
    if a.workers<1:p.error('workers must be positive')
    if a.verify_only:
        verify_frozen(a.workers);return
    rows=select()
    if a.plan_only:
        print(json.dumps(dict(seed=42,selection=rows,total_pairs=128),indent=2));return
    if OUTPUT.exists():raise FileExistsError('Frozen cohort already exists; validate/reuse it rather than redraw')
    overlap = ROOT / OASIS_OVERLAP
    audit = json.loads(overlap.read_text())
    if {r['l2r_id'] for r in audit['mapping']}!=set(range(365,415)) or not all(r['exact_label_array_match'] for r in audit['mapping']):raise ValueError('Unexpected OASIS overlap audit')
    acdc=prepare_acdc(rows['acdc'],a.workers)
    pairs=[dict(r,**paths(r)) for group in rows.values() for r in group]
    files=verify_files(pairs,a.workers)
    counts={ds:len({r.get('patient',r[k]) for r in group for k in ('moving','fixed')}) for ds,group in rows.items()}
    dependencies=[ROOT/'medical/protocols/brain_benchmarks.json',ROOT/'Data/OASIS/preparation.json',ROOT/'Data/ACDC/training_download_manifest.json',ACDC_PREP/'manifest.json',overlap]
    manifest=dict(protocol='sirenpath_tuning32_v1',purpose='development/hyperparameter selection only; never final evaluation',seed=42,rng='numpy PCG64 SeedSequence([42,dataset_index]), order oasis/acdc/lpba40/ixi',total_pairs=128,pairs_per_dataset=32,training_seed=42,unique_subjects=counts,selection=rows,pairs=pairs,files=files,dependencies={str(x.relative_to(ROOT)):sha256(x) for x in dependencies},selection_rules=dict(oasis='64 distinct L2R subjects0001–0364;32 disjoint random directed pairs',acdc='32 distinct training patients001–100;16 ED→ES,16 ES→ED;both phases prepared',lpba40='25 training subjects only;random25-edge cycle plus7 disjoint extra edges;degrees2/3',ixi='32 distinct author Train subjects with fixed atlas;atlas shared by evaluation protocol'),validation_subjects_reserved=True,normalization=dict(oasis='independent minmax at fitting, same L2R adapter recipe',acdc='prepared KAN-IDIR values',lpba40='preserve prepared SITReg values',ixi='preserve prepared TransMorph values'),environment={n:importlib.metadata.version(n) for n in ('numpy','nibabel','scikit-image')})
    OUTPUT.mkdir();(OUTPUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Frozen tuning cohort:',counts,'128 pairs;',len(files),'verified files;',OUTPUT,flush=True)


if __name__=='__main__':main()
