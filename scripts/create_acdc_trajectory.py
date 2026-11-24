"""Compare the frozen SIRENPath trajectory, fitted crossfade and ACDC cine."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

import nibabel as nib
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
TIMES=(0.,.25,.5,.75,1.)


def median_pair(root):
    rows=[]
    for path in (root/'acdc/ours').glob('pair*/metrics.json'):
        rows.append((json.loads(path.read_text())['Dice']['mean'],int(path.parent.name[4:])))
    if len(rows)!=100:raise ValueError('Require all100 official ACDC cases')
    rows.sort();return rows[49] # Lower median, ties broken by pair index.


def bracket(ed,es,t,frames):
    end=es if es>ed else es+frames
    coordinate=ed+t*(end-ed)
    low=int(np.floor(coordinate));high=int(np.ceil(coordinate));weight=coordinate-low
    return (low-1)%frames,(high-1)%frames,weight,coordinate


def patch_ncc_2d(prediction,truth):
    import torch
    import torch.nn.functional as F
    a=torch.as_tensor(prediction.copy(),dtype=torch.float64)[None,None]
    b=torch.as_tensor(truth.copy(),dtype=torch.float64)[None,None]
    mean=lambda x:F.avg_pool2d(x,9,stride=1)
    ma,mb=mean(a),mean(b);va=(mean(a*a)-ma*ma).clamp_min(0);vb=(mean(b*b)-mb*mb).clamp_min(0)
    valid=vb>1e-8;values=((mean(a*b)-ma*mb)/(va*vb).sqrt().clamp_min(1e-8)).clamp(-1,1)
    return dict(mean=float(values[valid].mean()) if valid.any() else None,scored_windows=int(valid.sum()))


def run(args):
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from medical.models.siren import PairedSiren
    from medical.sirenpath_reconstruction import patch_ncc
    from scripts.prepare_acdc import preprocess
    from scripts.download_acdc_testing import fetch,API,TESTING
    from scripts.create_case_media import label_color
    score,index=median_pair(args.sirenpath)
    source=args.sirenpath/'acdc/ours'/f'pair{index:03d}'
    info=json.loads((source/'run.json').read_text());moving=Path(info['moving']);fixed=Path(info['fixed'])
    patient=moving.name.split('_')[0]
    raw=ROOT/'Data/ACDC/testing'/patient
    fields=dict(line.split(':',1) for line in (raw/'Info.cfg').read_text().splitlines() if ':' in line)
    ed,es=int(fields['ED']),int(fields['ES'])
    model_reversed=moving.name.split('_')[1].split('.')[0]=='ES'
    args.artifacts.mkdir(parents=True,exist_ok=False)
    args.output.mkdir(parents=True,exist_ok=False)
    folders=json.loads(fetch('/folder',{'parentType':'folder','parentId':TESTING,'limit':0}))
    folder=next(f for f in folders if f['name']==patient)
    items=json.loads(fetch('/item',{'folderId':folder['_id'],'limit':0}))
    item=next(i for i in items if i['name']==patient+'_4d.nii.gz')
    payload=fetch('/item/'+item['_id']+'/download')
    if len(payload)!=item['size']:raise ValueError('Cine byte count mismatch')
    cine_path=args.artifacts/item['name'];cine_path.write_bytes(payload)
    cine_image=nib.load(cine_path);cine=cine_image.get_fdata(dtype=np.float32)
    es_path=raw/f'{patient}_frame{es:02d}.nii.gz';spacing=nib.load(es_path).header.get_zooms()[:3]
    ed_image=nib.load(ROOT/'Data/ACDC/prepared/images'/f'{patient}_ED.nii.gz').get_fdata(dtype=np.float32)
    es_image=nib.load(ROOT/'Data/ACDC/prepared/images'/f'{patient}_ES.nii.gz').get_fdata(dtype=np.float32)
    z=ed_image.shape[2]//2
    acquired={}
    def frame(i):
        if i not in acquired:
            acquired[i]=preprocess(cine[:,:,:,i],np.zeros(cine.shape[:3],dtype=np.float32),spacing)[0]
        return acquired[i]
    if not np.allclose(frame(ed-1),ed_image,atol=1e-6) or not np.allclose(frame(es-1),es_image,atol=1e-6):
        raise ValueError('Prepared cine endpoints do not match frozen benchmark endpoint arrays')
    checkpoint=torch.load(source/'checkpoint.pt',map_location='cpu',weights_only=False)
    if checkpoint.get('training',{}).get('spatial_units')!='voxel':raise ValueError('Unsupported checkpoint geometry')
    pair=PairedSiren(3,**checkpoint['model']).to(args.device);pair.load_state_dict(checkpoint['state']);pair.eval().requires_grad_(False)
    predictions=[]
    with torch.no_grad():
        for t in TIMES:
            value=pair.decode(ed_image.shape[::-1],1-t if model_reversed else t,chunk=checkpoint['config']['runtime']['inference_chunk']).numpy().transpose(2,1,0)
            predictions.append(value)
    # Fixed temporal display calibration from both reconstructions; never normalize
    # each intermediate independently. NCC uses original, uncalibrated predictions.
    endpoints=np.concatenate([predictions[0].ravel(),predictions[-1].ravel()]);lo,hi=np.percentile(endpoints,[1,99])
    recon_span=max(float(hi-lo),1e-8)
    def save(name,image,window=(0.,1.)):
        a,b=window;values=np.clip((image[:,:,z]-a)/max(float(b-a),1e-8),0,1)
        plt.imsave(args.output/name,np.rot90(values),cmap='gray',vmin=0,vmax=1)
    def labeled(name,image,codes):
        gray=np.clip(image[:,:,z],0,1);rgba=np.dstack([gray,gray,gray,np.ones_like(gray)])
        labels=codes[:,:,z]
        for code in np.unique(labels):
            if code:
                mask=labels==code;rgba[mask,:3]=.55*rgba[mask,:3]+.45*np.array(label_color(code)[:3])
        plt.imsave(args.output/name,np.rot90(rgba))
    fixed_labels=nib.load(ROOT/'Data/ACDC/prepared/labels'/fixed.name).get_fdata(dtype=np.float32)
    moving_labels=nib.load(ROOT/'Data/ACDC/prepared/labels'/moving.name).get_fdata(dtype=np.float32)
    fixed_array=es_image if not model_reversed else ed_image;moving_array=ed_image if not model_reversed else es_image
    save('fixed.png',fixed_array);save('moving.png',moving_array)
    labeled('fixed_labels_overlay.png',fixed_array,fixed_labels);labeled('moving_labels_overlay.png',moving_array,moving_labels)
    for name,codes in [('fixed',fixed_labels),('moving',moving_labels)]:
        labels=codes[:,:,z];colors=np.zeros((*labels.shape,4),dtype=np.float32)
        for code in np.unique(labels):
            if code:colors[labels==code]=label_color(code)
        plt.imsave(args.output/(name+'_labels.png'),np.rot90(colors))
    shutil.copyfile(ROOT/'docs/media/acdc/label_legend.png',args.output/'label_legend.png')
    rows=[];volumes={}
    for t,prediction in zip(TIMES,predictions):
        low,high,weight,coordinate=bracket(ed,es,t,cine.shape[3])
        real=(1-weight)*frame(low)+weight*frame(high)
        nearest_index=int(np.floor(coordinate+.5)-1)%cine.shape[3];nearest=frame(nearest_index)
        crossfade=(1-t)*predictions[0]+t*predictions[-1]
        tag=f't{int(t*100):03d}'
        save('siren_'+tag+'.png',prediction,(lo,hi));save('crossfade_'+tag+'.png',crossfade,(lo,hi))
        save('real_'+tag+'.png',real);save('real_nearest_'+tag+'.png',nearest)
        row=dict(t=t,model_t=1-t if model_reversed else t,frame_coordinate_one_based=coordinate,frame_low=low+1,frame_high=high+1,high_weight=weight,nearest_frame=nearest_index+1)
        for method,array in [('siren',prediction),('crossfade',crossfade)]:
            for reference,target in [('cine_linear',real),('cine_nearest',nearest)]:
                row[method+'_'+reference+'_patch_ncc_2d']=patch_ncc_2d(array[:,:,z],target[:,:,z])['mean']
                row[method+'_'+reference+'_patch_ncc_3d']=patch_ncc(torch.from_numpy(array.transpose(2,1,0).copy()),torch.from_numpy(target.transpose(2,1,0).copy()))['mean']
        rows.append(row);volumes[tag]=dict(siren=prediction,crossfade=crossfade,cine_linear=real,cine_nearest=nearest)
    for tag,arrays in volumes.items():
        for name,array in arrays.items():nib.save(nib.Nifti1Image(array,nib.load(fixed).affine),args.artifacts/(name+'_'+tag+'.nii.gz'))
    metadata=dict(patient=patient,pair_index=index,Dice=score,selection='lower median of all100 official original-default ACDC Dice values',ED=ed,ES=es,contraction='ED to ES in cyclic frame order',model_time_reversed=model_reversed,slice_axis=2,slice_index=z,shape=list(ed_image.shape),
                  cine_url=API+'/item/'+item['_id']+'/download',cine_sha256=hashlib.sha256(payload).hexdigest(),checkpoint=str(source/'checkpoint.pt'),checkpoint_sha256=hashlib.sha256((source/'checkpoint.pt').read_bytes()).hexdigest(),
                  preprocessing='same preprocess() as frozen KAN-IDIR endpoints, ES spacing, full per-frame minmax before crop',reference='real_t PNGs are linear interpolation of independently preprocessed neighboring acquired cine frames; real_nearest_t PNGs are actual acquired frames',
                  patch_ncc='signed Pearson; 9x9 displayed slice and 9x9x9 whole volume; valid stride1; constant truth windows variance<=1e-8 excluded; FP64 moments, denominator max(sqrt(var_a var_b),1e-8)',
                  reconstruction_display_window=[float(lo),float(hi)],reference_display_window=[0,1],rows=rows)
    (args.artifacts/'report.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
    with (args.artifacts/'patch_ncc.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# ACDC contraction trajectory','',f'Patient {patient}, pair{index}, original Dice {score:.6f}; ED frame{ed} to ES frame{es}; native axis2 middle slice{z}.',
           '', 'real_t panels interpolate neighboring acquired cine frames at exact normalized times; real_nearest_t panels show actual acquired frames. No intermediate labels are fabricated.',
           '', '| t | Cine frame | SIRENPath Patch NCC 2D | Fitted crossfade Patch NCC 2D | SIRENPath Patch NCC 3D | Fitted crossfade Patch NCC 3D |','|---:|---:|---:|---:|---:|---:|']
    for row in rows:lines.append(f"| {row['t']:.2f} | {row['frame_coordinate_one_based']:.2f} | {row['siren_cine_linear_patch_ncc_2d']:.5f} | {row['crossfade_cine_linear_patch_ncc_2d']:.5f} | {row['siren_cine_linear_patch_ncc_3d']:.5f} | {row['crossfade_cine_linear_patch_ncc_3d']:.5f} |")
    lines+=['','NCC is measured without display rescaling or fitted calibration. Reconstruction and crossfade use one shared endpoint-derived display window; cine uses its prepared0–1 scale. See report.json for both linear-reference and nearest-acquired-reference scores.','']
    (args.artifacts/'REPORT.md').write_text('\n'.join(lines));print('\n'.join(lines))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sirenpath','--ours',metavar='DIR',type=Path,help='Directory containing the official SIRENPath results',default=ROOT/'results/official_ours_cosine_20261116');p.add_argument('--output',type=Path,default=ROOT/'docs/media/acdc_trajectory');p.add_argument('--artifacts',type=Path,default=ROOT/'results/acdc_trajectory_original_20261117');p.add_argument('--device',default='cuda:0');p.add_argument('--run',action='store_true');args=p.parse_args()
    if not args.run:print(dict(median_pair=median_pair(args.sirenpath),times=TIMES,output=str(args.output),artifacts=str(args.artifacts)));return
    if args.output.exists() or args.artifacts.exists():raise FileExistsError('Use fresh output and artifact paths')
    run(args)


if __name__=='__main__':main()
