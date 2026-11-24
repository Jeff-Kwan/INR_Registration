"""Median SIRENPath Dice cases, four strongest baselines and label-gap inspection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def selections(report):
    result=[]
    for dataset in report['datasets']:
        rows=[r for r in report['rows'] if r['dataset']==dataset]
        ranked=sorted(rows,key=lambda r:(r['ours_dice'],r['pair_index']))
        row=dict(ranked[(len(rows)-1)//2],rank='median_dice')
        means={m:statistics.mean(r[m+'_dice'] for r in rows) for names in report['groups'].values() for m in names}
        classical=sorted(report['groups']['classical'],key=lambda m:(-means[m],m))[:2]
        neural=sorted(report['groups']['neural'],key=lambda m:(-means[m],m))[:2]
        row.update(methods=['ours',*classical,*neural],baseline_dataset_means=means,strongest_baseline=max(means,key=means.get))
        result.append(row)
    return result


def crop_bounds(masks,padding=12):
    union=np.logical_or.reduce(masks);coords=np.argwhere(union)
    if not len(coords):return (0,union.shape[0],0,union.shape[1])
    lo=coords.min(0)-padding;hi=coords.max(0)+padding+1
    return max(0,int(lo[0])),min(union.shape[0],int(hi[0])),max(0,int(lo[1])),min(union.shape[1],int(hi[1]))


def render(output,case,fixed_labels,moving_labels,methods,selection,media_output=None):
    import nibabel as nib
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from matplotlib.patches import Patch
    from baselines.comparison_report import NAMES
    from scripts.create_case_media import label_color,save_legend
    images=Path(media_output);images.mkdir(parents=True,exist_ok=True)
    fixed=case['fixed']['data'];moving=case['moving']['data'];z=fixed.shape[2]//2
    lo,hi=np.percentile(fixed,[1,99]);span=max(float(hi-lo),1e-8)
    def gray(a):return np.clip((a-lo)/span,0,1)
    def rgb(a):return np.dstack([a,a,a])
    def labels_overlay(a,codes):
        image=rgb(a)
        for code in np.unique(codes):
            if code:
                mask=codes==code;image[mask]=.55*image[mask]+.45*np.asarray(label_color(code)[:3])
        return image
    def save(path,array):plt.imsave(path,np.rot90(array),cmap='gray',vmin=0,vmax=1)
    save(images/'fixed_image.png',gray(fixed[:,:,z]));save(images/'moving_image.png',gray(moving[:,:,z]))
    save(images/'fixed_labels_overlay.png',labels_overlay(gray(fixed[:,:,z]),fixed_labels[:,:,z]))
    save(images/'moving_labels_overlay.png',labels_overlay(gray(moving[:,:,z]),moving_labels[:,:,z]))
    metadata=[]
    for method in methods:
        source=output/method;target=images/method;target.mkdir(exist_ok=True)
        warped=np.asarray(nib.load(source/'warped_image.nii.gz').dataobj)
        codes=np.asarray(nib.load(source/'warped_labels.nii.gz').dataobj)
        mapping=np.asarray(nib.load(source/'fixed_to_moving_voxel_map.nii.gz').dataobj)
        error=np.clip(np.abs(fixed[:,:,z]-warped[:,:,z])/span,0,1)
        cmap=plt.get_cmap('magma');opacity=.65
        overlay=(1-opacity)*rgb(gray(fixed[:,:,z]))+opacity*cmap(error)[...,:3]
        save(target/'intensity_error_overlay.png',overlay)
        save(target/'warped_labels_overlay.png',labels_overlay(gray(warped[:,:,z]),codes[:,:,z]))
        # Pull-map grid: contours where moving X or Y is a grid multiple,
        # shown on the fixed slice. This does not silently drop out-of-plane flow.
        fig,ax=plt.subplots(figsize=(5,5*fixed.shape[1]/fixed.shape[0]),dpi=180);ax.imshow(np.rot90(gray(fixed[:,:,z])),cmap='gray',vmin=0,vmax=1)
        for component in (0,1):
            levels=np.arange(0,case['moving']['data'].shape[component],8)
            ax.contour(np.rot90(mapping[:,:,z,component]),levels=levels,colors=['#00ffff'],linewidths=1.8)
        ax.set_xlim(-.5,fixed.shape[0]-.5);ax.set_ylim(fixed.shape[1]-.5,-.5);ax.axis('off');fig.subplots_adjust(0,0,1,1)
        fig.savefig(target/'deformation_grid.png',bbox_inches=None,pad_inches=0);plt.close(fig)
        metadata.append(dict(method=method,**json.loads((source/'recovery.json').read_text())))
    # A separate bar shows the actual composited overlay colour on the same
    # colormap/range. Alpha is fixed, so hue consistently encodes magnitude.
    fig,ax=plt.subplots(figsize=(5,.7),layout='constrained');fig.colorbar(ScalarMappable(norm=Normalize(0,1),cmap='magma'),cax=ax,orientation='horizontal',label='Absolute intensity error / fixed 1st–99th percentile span (clipped at 1)')
    fig.savefig(images/'intensity_error_colorbar.png',dpi=200,bbox_inches='tight');plt.close(fig)
    codes=sorted((set(np.unique(fixed_labels))|set(np.unique(moving_labels)))-{0})
    save_legend(images.parent,selection['dataset'],codes,metadata_output=output.parent)
    report=json.loads((ROOT/'results/case_advantage_original_20261117/advantage.json').read_text())
    def recorded(method):
        group='ours' if method=='ours' else 'classical' if method in report['groups']['classical'] else 'neural'
        return json.loads((Path(report['roots'][group])/selection['dataset']/method/f"pair{selection['pair_index']:03d}"/'metrics.json').read_text())
    baseline=selection['strongest_baseline'];sirenpath=recorded('ours');other=recorded(baseline)
    if sirenpath['labels']!=other['labels']:raise ValueError('Label order differs')
    gaps=[dict(code=int(code),ours=float(a),baseline=float(b),difference=float(a-b)) for code,a,b in zip(sirenpath['labels'],sirenpath['Dice']['detailed'],other['Dice']['detailed']) if a is not None and b is not None]
    gaps.sort(key=lambda r:(-abs(r['difference']),r['code']));gap=gaps[0];code=gap['code']
    warped_labels={method:np.asarray(nib.load(output/method/'warped_labels.nii.gz').dataobj) for method in ('ours',baseline)}
    contour_z=z
    if not np.any(fixed_labels[:,:,z]==code):
        counts=(fixed_labels==code).sum(axis=(0,1));contour_z=int(np.argmax(counts)) if counts.max()>0 else z
    masks=[fixed_labels[:,:,contour_z]==code,*[warped_labels[m][:,:,contour_z]==code for m in ('ours',baseline)]]
    x0,x1,y0,y1=crop_bounds(masks);roi=(slice(x0,x1),slice(y0,y1))
    for method in ('ours',baseline):
        image=np.asarray(nib.load(output/method/'warped_image.nii.gz').dataobj)
        fig,ax=plt.subplots(figsize=(5,5*(y1-y0)/(x1-x0)),dpi=180)
        ax.imshow(np.rot90(gray(image[:,:,contour_z][roi])),cmap='gray',vmin=0,vmax=1)
        for mask,color,style in [(fixed_labels[:,:,contour_z][roi]==code,'#00ff66','-'),(warped_labels[method][:,:,contour_z][roi]==code,'#ff00ff','--')]:
            if mask.any() and not mask.all():ax.contour(np.rot90(mask.astype(float)),levels=[.5],colors=[color],linestyles=[style],linewidths=2.)
        ax.axis('off');fig.subplots_adjust(0,0,1,1);fig.savefig(images/method/'target_structure_contours.png',pad_inches=0);plt.close(fig)
    fig,ax=plt.subplots(figsize=(5,1));ax.axis('off');ax.legend(handles=[plt.Line2D([],[],color='#00ff66',linewidth=2,label='Fixed label contour'),plt.Line2D([],[],color='#ff00ff',linestyle='--',linewidth=2,label='Warped moving label contour')],loc='center',frameon=False)
    fig.savefig(images/'contour_legend.png',dpi=180,bbox_inches='tight');plt.close(fig)
    replay_label={}
    for method in ('ours',baseline):
        truth=fixed_labels==code;predicted=warped_labels[method]==code;total=truth.sum()+predicted.sum()
        a=truth[:,:,contour_z];b=predicted[:,:,contour_z];total2=a.sum()+b.sum()
        replay_label[method]=dict(full_volume_Dice=float(2*np.count_nonzero(truth&predicted)/total) if total else None,
                                 shown_slice_Dice=float(2*np.count_nonzero(a&b)/total2) if total2 else None)
    label_names=json.loads((output.parent/'label_colors.json').read_text())['classes']
    label_name=next((r['name'] for r in label_names if r['code']==code),f'Class {code}')
    details=dict(selection=selection,middle_slice=z,slice_axis=2,grid='iso-contours of absolute moving X/Y voxel coordinates sampled on fixed slice; step8voxels; linewidth1.8; out-of-plane component retained in saved3D map but not separately plotted',
                 intensity_error='absolute original-intensity difference normalized by shared fixed1st–99th percentile span; clip1; magma colormap,fixed alpha0.65',recovery=metadata,
                 targeted=dict(baseline=baseline,selection='largest absolute per-label Dice gap in original recorded results; label-code tie break',label=gap,label_name=label_name,recovered_per_label=replay_label,all_labels=gaps,winner='SIRENPath' if gap['difference']>0 else baseline,
                               slice=contour_z,middle_slice_used=contour_z==z,crop=[x0,x1,y0,y1],crop_selection='union of target fixed and both warped masks plus12pixels, same crop and slice for both methods',
                               caveat='full-volume Dice gap may not be visible in this single slice; compare recovered scores in audit before interpreting contours'))
    (output/'inspection.json').write_text(json.dumps(details,indent=2)+'\n')
    baseline_name=NAMES.get(baseline,baseline)
    winner=details['targeted']['winner']
    lines=['# Median-Dice targeted structure check','',f"{selection['dataset']} pair{selection['pair_index']}; strongest dataset-mean baseline: {baseline_name}.",
           '',f"Selected label{code} ({label_name}) by largest absolute full-volume per-label Dice gap: SIRENPath {gap['ours']:.6f}, {baseline_name} {gap['baseline']:.6f}; winner: {NAMES.get(winner,winner)}.",
           '',f"General panels: middle slice{z}. Targeted contour panels: slice{contour_z}, common crop{x0,x1,y0,y1}.",
           '', 'Selection is explicitly post-hoc and highlights the largest disagreement; it is not a typical-structure claim. No anatomical boundary explanation is inferred automatically from these plots. Full-volume gaps do not guarantee an obvious difference on a single slice.','']
    (output/'INSPECTION.md').write_text('\n'.join(lines))


def worker(args,selection,report):
    from scripts import create_case_media as recovery
    recovery.METHODS=tuple(selection['methods']);recovery.plot_case=render
    directory=args.artifacts/'recovery'/selection['dataset']/f"median_dice_pair{selection['pair_index']:03d}"
    old=ROOT/'results/case_media_recovery_original_20261117'/selection['dataset']
    matches=list(old.glob(f"*_pair{selection['pair_index']:03d}"))
    directory.mkdir(parents=True,exist_ok=True)
    if matches:
        for method in selection['methods']:
            source=matches[0]/method;target=directory/method
            if source.exists() and not target.exists():
                def copy(a,b):
                    if str(a).endswith(('.nii.gz','.npz')):os.link(a,b)
                    else:shutil.copy2(a,b)
                    return b
                shutil.copytree(source,target,copy_function=copy)
    recovery.run_case(selection,report,args.artifacts/'recovery',args.artifacts/'png_stage')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifacts',type=Path,default=ROOT/'results/median_dice_media_20261117');p.add_argument('--gpus',type=int,nargs='+',default=[0,1,2,3]);p.add_argument('--task-index',type=int);p.add_argument('--render-only',action='store_true');p.add_argument('--run',action='store_true');a=p.parse_args()
    report=json.loads((ROOT/'results/case_advantage_original_20261117/advantage.json').read_text());selected=selections(report)
    if a.task_index is not None:worker(a,selected[a.task_index],report);return
    if a.render_only:
        import nibabel as nib
        from medical.case_runner import load_volume
        from medical.comparison_metrics import label_path
        for selection in selected:
            case={key:load_volume(selection[key]) for key in ('fixed','moving')}
            labels=[np.asarray(nib.load(label_path(case[k],selection['dataset'])).dataobj) for k in ('fixed','moving')]
            directory=a.artifacts/'recovery'/selection['dataset']/f"median_dice_pair{selection['pair_index']:03d}"
            render(directory,case,*labels,selection['methods'],selection,media_output=ROOT/'docs/media'/selection['dataset']/directory.name)
        from scripts.create_structure_contours import render_all
        render_all(a.artifacts,ROOT/'docs/media')
        return
    if not a.run:print(json.dumps(selected,indent=2));return
    a.artifacts.mkdir(parents=True,exist_ok=False);(a.artifacts/'selection.json').write_text(json.dumps(selected,indent=2)+'\n')
    # Preserve a hash inventory proving the separate trajectory subtree is untouched.
    trajectory=ROOT/'docs/media/acdc_trajectory'
    before={str(p.relative_to(trajectory)):hashlib.sha256(p.read_bytes()).hexdigest() for p in trajectory.rglob('*.png')}
    import concurrent.futures
    def task(index):
        with (a.artifacts/f'task{index}.log').open('w') as log:
            r=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--artifacts',str(a.artifacts),'--task-index',str(index)],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpus[index%len(a.gpus)]),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT)
        return dict(index=index,returncode=r.returncode)
    with concurrent.futures.ThreadPoolExecutor(min(4,len(a.gpus))) as pool:results=list(pool.map(task,range(4)))
    (a.artifacts/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
    if any(r['returncode'] for r in results):raise SystemExit(1)
    archive=a.artifacts/'previous_media_pngs';archive.mkdir()
    for row in selected:
        dataset=row['dataset'];old=ROOT/'docs/media'/dataset;new=a.artifacts/'png_stage'/dataset
        if old.exists():old.rename(archive/dataset)
        new.rename(old)
    from scripts.create_structure_contours import render_all
    render_all(a.artifacts,ROOT/'docs/media')
    after={str(p.relative_to(trajectory)):hashlib.sha256(p.read_bytes()).hexdigest() for p in trajectory.rglob('*.png')}
    if before!=after:raise RuntimeError('Trajectory subtree changed unexpectedly')
    if any(p.suffix!='.png' and p not in {ROOT/f'docs/media/{ds}_pair{index:03d}_centroid/REPORT.md' for ds,index in [('ixi',79),('neurite_oasis',10),('acdc',1),('lpba40',40)]} for p in (ROOT/'docs/media').rglob('*') if p.is_file()):raise RuntimeError('Unexpected non-PNG media file')
    print('Installed four median-Dice datasets; trajectory unchanged; PNG-only media.',flush=True)


if __name__=='__main__':main()
