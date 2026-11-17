"""Shared four-contour inspection with a10% larger crop and anatomical Dice legend.

Uses existing saved labels/images only; no training or map recovery.
"""
import argparse
import json
import math
from pathlib import Path
import shutil
import sys
import textwrap

import nibabel as nib
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
NAMES=dict(ours='SIRENPath',convexadam='ConvexAdam',fireants='FireANTs',niftyreg='NiftyReg',demons='Demons',dnvf='DNVF',nodeo='NODEO',nir='NIR',ccidir='ccIDIR')


def expanded_crop(crop,shape,factor=1.1):
    result=[]
    for low,high,length in [(crop[0],crop[1],shape[0]),(crop[2],crop[3],shape[1])]:
        size=min(length,math.ceil((high-low)*factor));start=int(math.floor((low+high-size)/2))
        start=max(0,min(start,length-size));result.extend([start,start+size])
    return result


def anatomy_name(dataset,code,artifacts):
    if dataset=='neurite_oasis':
        path=ROOT/'Data/NeuriteOASIS/prepared/source_metadata/5384_seg35_labels.txt'
        for line in path.read_text().splitlines():
            values=line.split()
            if values and values[0]==str(code):
                name=values[1].replace('Inf-Lat','Inferior-Lateral');return name.replace('-',' '),str(path)
        raise ValueError('Label missing from released Neurite lookup table')
    values=json.loads((artifacts/'recovery'/dataset/'label_colors.json').read_text())['classes']
    name=next(v['name'] for v in values if v['code']==code)
    if name.startswith('L '):name='Left '+name[2:]
    if name.startswith('R '):name='Right '+name[2:]
    return name.replace('-',' '),'medical/protocols/brain_benchmarks.json' if dataset!='acdc' else 'docs/DATASETS.md ACDC labels'


def dice(truth,prediction):
    count=truth.sum()+prediction.sum()
    return float(2*np.count_nonzero(truth&prediction)/count) if count else None


def signed_targets(labels,ours,baseline):
    rows=[dict(code=int(code),ours=float(a),baseline=float(b),gap=float(a-b)) for code,a,b in zip(labels,ours,baseline) if a is not None and b is not None]
    wins=sorted((r for r in rows if r['gap']>0),key=lambda r:(-r['gap'],r['code']))
    losses=sorted((r for r in rows if r['gap']<0),key=lambda r:(r['gap'],r['code']))
    return dict(win=wins[0] if wins else None,loss=losses[0] if losses else None)


def target_slice(mask):
    counts=mask.sum(axis=(0,1));best=int(counts.max())
    if best==0:raise ValueError('Target has no fixed-label voxels in the volume')
    return min(range(len(counts)),key=lambda z:(-int(counts[z]),abs(z-(len(counts)-1)/2),z))


def render_all(artifacts,media_root,datasets=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scripts.create_median_dice_media import crop_bounds
    artifacts=Path(artifacts);media_root=Path(media_root)
    selections=json.loads((artifacts/'selection.json').read_text())
    source=json.loads((ROOT/'results/case_advantage_original_20261117/advantage.json').read_text())
    archive=artifacts/'superseded_structure_pngs';summaries=[]
    previous=json.loads((artifacts/'four_contour_summary.json').read_text()) if (artifacts/'four_contour_summary.json').exists() else []
    for selection in selections:
        dataset=selection['dataset']
        if datasets is not None and dataset not in datasets:
            summaries.extend(r for r in previous if r['dataset']==dataset);continue
        case=f"median_dice_pair{selection['pair_index']:03d}"
        data=artifacts/'recovery'/dataset/case;images=media_root/dataset/case
        means=selection['baseline_dataset_means']
        classic=min(source['groups']['classical'],key=lambda m:(-means[m],m));neural=min(source['groups']['neural'],key=lambda m:(-means[m],m))
        strongest=min(means,key=lambda m:(-means[m],m));compared=['ours',classic,neural]
        def original(method):
            group='ours' if method=='ours' else 'classical' if method in source['groups']['classical'] else 'neural'
            return json.loads((Path(source['roots'][group])/dataset/method/f"pair{selection['pair_index']:03d}"/'metrics.json').read_text())
        own=original('ours');best=original(strongest)
        if own['labels']!=best['labels']:raise ValueError('Mismatched label order')
        targets=signed_targets(own['labels'],own['Dice']['detailed'],best['Dice']['detailed'])
        if dataset=='acdc' and targets['loss'] is None:
            positive=[dict(code=int(code),ours=float(a),baseline=float(b),gap=float(a-b)) for code,a,b in zip(own['labels'],own['Dice']['detailed'],best['Dice']['detailed']) if a is not None and b is not None and a>b]
            if positive:
                targets={'win':targets['win'],'smallest_win':min(positive,key=lambda r:(r['gap'],r['code']))}
        fixed=nib.load(data/'fixed_image.nii.gz').get_fdata(dtype=np.float32)
        fixed_codes=np.asarray(nib.load(data/'fixed_labels.nii.gz').dataobj)
        predictions={m:np.asarray(nib.load(data/m/'warped_labels.nii.gz').dataobj) for m in compared}
        limits=np.percentile(fixed,[1,99]);gray=np.clip((fixed-limits[0])/max(float(limits[1]-limits[0]),1e-8),0,1)
        styles={'ground_truth':('#00ff66','-',2.5),'ours':('#ff00ff','-',2.1),classic:('#00bfff','--',2.1),neural:('#ffb000','-.',2.1)}
        for kind,target in targets.items():
            record=dict(dataset=dataset,pair_index=selection['pair_index'],gap_type=kind,selection_comparator=strongest,
                selection='smallest strictly positive original per-class Dice gap (ACDC-only replacement for unavailable loss)' if kind=='smallest_win' else 'largest strictly positive gap for win; most negative gap for loss; original full-volume class Dice; tie lowest label code',
                classical=classic,neural=neural,baseline_selection='highest dataset-wide mean Dice separately in classical/neural groups')
            if target is None:
                record.update(available=False,anatomy=None,reason=f'No strictly {"positive" if kind=="win" else "negative"} per-label Dice gap against {NAMES[strongest]}')
                fig,ax=plt.subplots(figsize=(7,1.4),facecolor='black');ax.set_facecolor('black');ax.axis('off')
                ax.text(.5,.55,'No qualifying structure',ha='center',va='center',fontsize=18,color='white')
                ax.text(.5,.25,f"No {'positive' if kind=='win' else 'negative'} per-label gap against {NAMES[strongest]}",ha='center',va='center',fontsize=15,color='white')
                fig.savefig(images/f'{dataset}_target_structure_{kind}.png',dpi=180,bbox_inches='tight',pad_inches=.1);plt.close(fig)
                summaries.append(record);continue
            code=target['code'];truth=fixed_codes==code
            try:z=target_slice(truth)
            except ValueError:
                record.update(available=False,anatomy=None,reason='Selected label has no fixed-reference voxels; no maximum-occupancy reference slice exists',label=target)
                summaries.append(record);continue
            masks={'ground_truth':truth,**{m:predictions[m]==code for m in compared}}
            base_crop=crop_bounds([a[:,:,z] for a in masks.values()],padding=12)
            crop=expanded_crop(base_crop,truth.shape[:2],factor=1.0);x0,x1,y0,y1=crop;roi=(slice(x0,x1),slice(y0,y1))
            name,name_source=anatomy_name(dataset,code,artifacts)
            width=7.;image_height=width*(y1-y0)/(x1-x0);legend_height=2.1
            fig,ax=plt.subplots(figsize=(width,image_height),dpi=180,facecolor='black')
            legend_fig,legend_axis=plt.subplots(figsize=(width,legend_height),dpi=180,facecolor='black')
            ax.set_facecolor('black');legend_axis.set_facecolor('black');legend_axis.axis('off')
            ax.imshow(np.rot90(gray[:,:,z][roi]),cmap='gray',vmin=0,vmax=1,interpolation='nearest')
            for method in [classic,neural,'ours','ground_truth']:
                a=np.rot90(masks[method][:,:,z][roi]);color,style,line=styles[method]
                if a.any() and not a.all():ax.contour(a.astype(float),levels=[.5],colors=[color],linestyles=[style],linewidths=line)
            ax.axis('off');rows=[]
            handles=[Line2D([],[],color=styles['ground_truth'][0],linewidth=2.5,label='Ground truth')]
            for method in [classic,neural,'ours']:
                score=original(method);k=score['labels'].index(code)
                row=dict(method=method,name=NAMES[method],original_full_volume_Dice=score['Dice']['detailed'][k],
                    recovered_full_volume_Dice=dice(truth,masks[method]),shown_slice_Dice=dice(truth[:,:,z],masks[method][:,:,z]));rows.append(row)
                color,style,line=styles[method]
                handles.append(Line2D([],[],color=color,linestyle=style,linewidth=line,label=f"{NAMES[method]} {row['recovered_full_volume_Dice']:.4f}"))
            legend_axis.text(.025,.5,textwrap.fill(name,width=10),transform=legend_axis.transAxes,
                             ha='left',va='center',fontsize=29.75,fontfamily='DejaVu Sans Condensed',color='white',linespacing=1.05)
            legend_axis.legend(handles=handles,loc='center right',bbox_to_anchor=(.985,.5),frameon=False,
                               prop={'size':26.25,'family':'DejaVu Sans Condensed'},labelcolor='white',borderpad=0,borderaxespad=0,
                               labelspacing=.2,handlelength=1.0,handletextpad=.3)
            fig.subplots_adjust(0,0,1,1);fig.savefig(images/f'{dataset}_target_structure_{kind}.png',pad_inches=0);plt.close(fig)
            legend_fig.subplots_adjust(0,0,1,1);legend_fig.savefig(images/f'{dataset}_target_structure_{kind}_legend.png',pad_inches=0);plt.close(legend_fig)
            record.update(available=True,label=target,label_code=code,anatomy=name,label_name_source=name_source,slice=z,
                slice_selection='maximum fixed target-label voxel count; ties nearest volume centre then lowest index',
                target_voxels_on_slice=int(truth[:,:,z].sum()),crop=crop,base_crop=base_crop,crop_factor=1.0,
                crop_definition='union of all four masks on selected slice, plus12pixels, then1.0 width/height, clamped to image',
                scores=rows,background='shared fixed image',legend_Dice='recovered full-volume class Dice; selection uses original recorded Dice')
            summaries.append(record)
        obsolete=[images/'target_structure_contours.png',images/'target_structure_legend.png',images/'contour_legend.png',*images.glob('*/target_structure_contours.png')]
        obsolete.extend(images.glob('target_structure_*.png'))
        if 'smallest_win' in targets:obsolete.append(images/f'{dataset}_target_structure_loss.png')
        for old in obsolete:
            if old.exists():
                destination=archive/dataset/old.relative_to(images);destination.parent.mkdir(parents=True,exist_ok=True)
                if destination.exists():old.unlink()
                else:old.rename(destination)
        (data/'signed_contour_inspection.json').write_text(json.dumps([r for r in summaries if r['dataset']==dataset],indent=2)+'\n')
    (artifacts/'four_contour_summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    return summaries


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifacts',type=Path,default=ROOT/'results/median_dice_media_20261117');p.add_argument('--media',type=Path,default=ROOT/'docs/media');p.add_argument('--datasets',nargs='+',choices=['acdc','neurite_oasis','lpba40','ixi']);a=p.parse_args()
    for row in render_all(a.artifacts,a.media,a.datasets):print(row['dataset'],row['anatomy'],row['classical'],row['neural'],row.get('slice'),row.get('reason',''))


if __name__=='__main__':main()
