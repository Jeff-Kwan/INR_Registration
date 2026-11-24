"""Dataset-aware centroid-plane comparison PNGs and metrics from saved maps."""
import argparse
import json
from pathlib import Path
import statistics
import sys

import nibabel as nib
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


VIEWS={'ixi':[('axial',1,'rotate90'),('coronal',2,'transpose')],
       'neurite_oasis':[('axial',1,'rotate90'),('coronal',2,'transpose')],
       'lpba40':[('axial',2,'transpose'),('coronal',1,'rotate90')],
       'acdc':[('short_axis',2,'rotate90'),('long_axis_reformat',1,'rotate90')]}


def transform(array,view,rotation=None):
    # Released IXI arrays: axis1 is superior/inferior; axis2 anterior/posterior.
    # The artificial unit NIfTI affine is not an anatomical orientation source.
    rotate=(view=='axial') if rotation is None else rotation=='rotate90'
    return np.rot90(array) if rotate else array.transpose(1,0,*range(2,array.ndim))


def run(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize,LinearSegmentedColormap
    from matplotlib.cm import ScalarMappable
    from matplotlib.lines import Line2D
    from scipy.ndimage import center_of_mass
    from medical.comparison_metrics import evaluate_arrays,labels_for
    from scripts.create_case_media import label_color,save_legend
    from scripts.create_median_dice_media import crop_bounds
    report=json.loads((ROOT/'results/case_advantage_original_20261117/advantage.json').read_text())
    dataset=args.dataset
    cohort=sorted((r for r in report['rows'] if r['dataset']==dataset),key=lambda r:(r['ours_dice'],r['pair_index']))
    row=cohort[(len(cohort)-1)//2];pair_index=row['pair_index']
    if args.output is None:args.output=ROOT/f'docs/media/{dataset}_pair{pair_index:03d}_centroid'
    if args.artifacts is None:args.artifacts=ROOT/f'results/{dataset}_pair{pair_index:03d}_centroid_20261117'
    if (args.output.exists() or args.artifacts.exists()) and not args.refresh:raise FileExistsError('Use fresh output paths or --refresh')
    classical=min(report['groups']['classical'],key=lambda m:(-row[m+'_dice'],m));neural=min(report['groups']['neural'],key=lambda m:(-row[m+'_dice'],m));methods=['ours',classical,neural]
    data=ROOT/'results/median_dice_media_20261117/recovery'/dataset/f'median_dice_pair{pair_index:03d}'
    fixed=nib.load(data/'fixed_image.nii.gz').get_fdata(dtype=np.float32);moving=nib.load(data/'moving_image.nii.gz').get_fdata(dtype=np.float32)
    fixed_labels=np.asarray(nib.load(data/'fixed_labels.nii.gz').dataobj);moving_labels=np.asarray(nib.load(data/'moving_labels.nii.gz').dataobj)
    voxel_spacing=np.linalg.norm(nib.load(data/'fixed_image.nii.gz').affine[:3,:3],axis=0)
    centroid=np.asarray(center_of_mass(fixed_labels>0));indices=np.floor(centroid+.5).astype(int)
    lo,hi=np.percentile(fixed,[1,99]);span=max(float(hi-lo),1e-8)
    window=lambda a:np.clip((a-lo)/span,0,1)
    def rgb(a):return np.dstack([a,a,a])
    def labels_overlay(a,codes):
        image=rgb(a)
        for code in np.unique(codes):
            if code:
                mask=codes==code;image[mask]=.55*image[mask]+.45*np.asarray(label_color(code)[:3])
        return image
    args.output.mkdir(parents=True,exist_ok=args.refresh);args.artifacts.mkdir(parents=True,exist_ok=args.refresh)
    error_cmap=LinearSegmentedColormap.from_list('black_red',['black','#ff0000'])
    metadata=dict(dataset=dataset,pair_index=pair_index,fixed=row['fixed'],moving=row['moving'],methods=methods,baseline_selection='highest per-case original Dice within each group, not dataset-wide means',original_Dice={m:row[m+'_dice'] for m in methods},
                  centroid=centroid.tolist(),rounded_centroid=indices.tolist(),foreground='all nonzero fixed-label voxels, unweighted binary centroid',intensity_window=[float(lo),float(hi)],
                  orientation='dataset-specific planes verified from image anatomy and available source geometry; artificial benchmark affines are not assumed anatomical; ACDC short-axis stack and orthogonal long-axis reformat are not claimed to be standard2CH/4CH; L/R markers omitted',views={},metrics={})
    if args.refresh:
        metadata['metrics']=json.loads((args.artifacts/'report.json').read_text())['metrics']
    else:
        mapping_identity=np.indices(fixed.shape,dtype=np.float32).transpose(1,2,3,0)
        spacing=np.linalg.norm(nib.load(data/'fixed_image.nii.gz').affine[:3,:3],axis=0)
        scores=evaluate_arrays(fixed_labels,moving_labels,mapping_identity,labels_for(dataset),spacing,dataset);metadata['metrics']['identity']=scores
        del mapping_identity
        for method in methods:
            folder=data/method
            mapping=np.asarray(nib.load(folder/'fixed_to_moving_voxel_map.nii.gz').dataobj)
            metadata['metrics'][method]=evaluate_arrays(fixed_labels,moving_labels,mapping,labels_for(dataset),spacing,dataset)
            del mapping
    for view,axis,rotation in VIEWS[dataset]:
        index=int(indices[axis]);remaining=[a for a in range(3) if a!=axis]
        f=np.take(fixed,index,axis=axis);m=np.take(moving,index,axis=axis);fl=np.take(fixed_labels,index,axis=axis);ml=np.take(moving_labels,index,axis=axis)
        crop=crop_bounds([fl>0],padding=8);x0,x1,y0,y1=crop;roi=(slice(x0,x1),slice(y0,y1))
        from scipy.ndimage import zoom
        inplane_spacing=voxel_spacing[remaining];unit=float(inplane_spacing.min())
        display_scale=[float(inplane_spacing[1]/unit),float(inplane_spacing[0]/unit)]
        def cut(a):
            display=transform(a[roi],view,rotation)
            if not np.allclose(display_scale,[1,1]):
                display=zoom(display,display_scale+[1.]*(display.ndim-2),order=0,prefilter=False)
            return display
        directory=args.output/view;directory.mkdir(exist_ok=args.refresh);audit=args.artifacts/view;audit.mkdir(exist_ok=args.refresh)
        def save(name,array):plt.imsave(directory/name,array,cmap='gray',vmin=0,vmax=1)
        save('fixed_image.png',cut(window(f)));save('moving_image.png',cut(window(m)))
        save('fixed_labels_overlay.png',cut(labels_overlay(window(f),fl)));save('moving_labels_overlay.png',cut(labels_overlay(window(m),ml)))
        for name,codes in [('fixed',fl),('moving',ml)]:
            colors=np.zeros((*codes.shape,4),np.float32)
            for code in np.unique(codes):
                if code:colors[codes==code]=label_color(code)
            save(name+'_labels.png',cut(colors))
        for method in methods:
            folder=data/method;target=directory/method;target.mkdir(exist_ok=args.refresh)
            warped=np.take(np.asarray(nib.load(folder/'warped_image.nii.gz').dataobj),index,axis=axis)
            codes=np.take(np.asarray(nib.load(folder/'warped_labels.nii.gz').dataobj),index,axis=axis)
            mapping=np.take(np.asarray(nib.load(folder/'fixed_to_moving_voxel_map.nii.gz').dataobj),index,axis=axis)
            error=np.clip(np.abs(f-warped)/span,0,1)
            plt.imsave(target/'intensity_error.png',cut(np.dstack([error,np.zeros_like(error),np.zeros_like(error)])))
            wrong=codes!=fl
            label_error=rgb(window(f));label_error[wrong]=[1.,0.,0.]
            plt.imsave(target/'label_error_overlay.png',cut(label_error))
            (target/'error.png').unlink(missing_ok=True)
            warped_labels_image=cut(labels_overlay(window(warped),codes))
            plt.imsave(target/'warped_labels_overlay.png',warped_labels_image)
            for filename,background in [('deformation_grid.png',cut(window(f))),('warped_labels_grid.png',warped_labels_image)]:
                height,width=background.shape[:2]
                fig,ax=plt.subplots(figsize=(5,5*height/width),dpi=180,facecolor='black')
                ax.imshow(background,cmap='gray',vmin=0,vmax=1)
                for component in remaining:
                    ax.contour(cut(mapping[...,component]),levels=np.arange(0,moving.shape[component],8),colors=['#66e5ff'],linewidths=.7)
                ax.set(xlim=(-.5,width-.5),ylim=(height-.5,-.5));ax.axis('off');fig.subplots_adjust(0,0,1,1)
                fig.savefig(target/filename,pad_inches=0);plt.close(fig)
            old=target/'intensity_error_overlay.png'
            if old.exists():
                backup=args.artifacts/'superseded_error_pngs'/view/method/old.name;backup.parent.mkdir(parents=True,exist_ok=True)
                if backup.exists():old.unlink()
                else:old.rename(backup)
        codes=sorted((set(np.unique(fixed_labels))|set(np.unique(moving_labels)))-{0});save_legend(directory,dataset,codes,metadata_output=audit)
        fig,ax=plt.subplots(figsize=(6,.8),layout='constrained');fig.colorbar(ScalarMappable(norm=Normalize(0,1),cmap=error_cmap),cax=ax,orientation='horizontal',label='Intensity error')
        fig.savefig(directory/'intensity_error_colorbar.png',dpi=180,bbox_inches='tight');plt.close(fig)
        (directory/'error_colorbar.png').unlink(missing_ok=True)
        fig,ax=plt.subplots(figsize=(6,.7));ax.axis('off')
        ax.legend(handles=[Line2D([],[],marker='s',color='red',linestyle='None',markersize=12,label='Wrong warped label voxel')],loc='center',frameon=False)
        fig.savefig(directory/'label_error_legend.png',dpi=180,bbox_inches='tight');plt.close(fig)
        fig,ax=plt.subplots(figsize=(6,.7));ax.axis('off');ax.legend(handles=[Line2D([],[],color='#66e5ff',linewidth=.7,label='Moving-coordinate grid,8voxel spacing')],loc='center',frameon=False)
        fig.savefig(directory/'deformation_grid_legend.png',dpi=180,bbox_inches='tight');plt.close(fig)
        metadata['views'][view]=dict(array_axis=axis,index=index,remaining_axes=remaining,crop=crop,crop_selection='fixed foreground bounding box on this slice plus8pixels; same crop for all methods and moving/fixed references',display=rotation,voxel_spacing=voxel_spacing.tolist(),display_row_column_scale=display_scale,display_resampling='nearest-neighbor2D display only to preserve spacing aspect;3D metrics and maps unchanged',grid='in-plane moving-coordinate pull-map iso-contours on fixed plane; full3D warp used for images/labels, out-of-plane motion not separately drawn')
    originals={}
    for method in methods:
        group='ours' if method=='ours' else 'classical' if method==classical else 'neural'
        originals[method]=json.loads((Path(report['roots'][group])/dataset/method/f'pair{pair_index:03d}/metrics.json').read_text())
    metadata['error_display']=dict(name='Intensity error',colormap='linear black to bright red',scale=[0,1],definition='abs(fixed-warped)/fixed1st–99th percentile span; clip1',shared='same scale across both views and all methods')
    metadata['label_error_display']=dict(definition='warped moving label != fixed label; all released label codes, including foreground/background errors',color='opaque red on fixed grayscale image',note='not an intensity residual; includes all released label codes, which may exceed the scored subset')
    metadata['grid_display']=dict(color='#66e5ff',linewidth_points=.7,spacing_voxels=8,backgrounds=['fixed image','warped labels overlay'])
    metadata['original_recorded_metrics']=originals
    (args.artifacts/'report.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
    names={'identity':'Identity','ours':'SIRENPath','convexadam':'ConvexAdam','fireants':'FireANTs-SyN','niftyreg':'NiftyReg','demons':'Demons','dnvf':'DNVF','nodeo':'NODEO','nir':'NIR','ccidir':'ccIDIR'}
    lines=[f'# {dataset} pair{pair_index}: foreground-centroid comparisons','',f"Fixed: {Path(row['fixed']).name}; moving: {Path(row['moving']).name}. Binary fixed-foreground centroid: {centroid.tolist()}.",'',
           'Views: '+', '.join(f'{view}:array axis{axis},slice{indices[axis]}' for view,axis,_ in VIEWS[dataset])+' (zero-based).', '',
           f'Baselines selected by original **case{pair_index}** Dice: {names.get(classical,classical)} (classical), {names.get(neural,neural)} (neural). SIRENPath included.',
           '', 'All methods share each view’s slice/crop and one fixed-image1st–99th percentile intensity window across both views. PNGs are standalone panels and legends; this requested report is included alongside them. Volumes and machine-readable metadata remain outside media.',
           '', '## Metrics recomputed from the saved maps used for plotting','', '| Method | Dice ↑ | HD95 ↓ | NPJ (%) ↓ | SDlogJ ↓ |','|---|---:|---:|---:|---:|']
    for method,s in metadata['metrics'].items():lines.append('| '+names.get(method,method)+' | '+' | '.join(f"{s[k]['mean']:.6f}" if s[k]['mean'] is not None else 'undefined' for k in ('Dice','HD95','NPJ','SDlogJ'))+' |')
    lines+=['', 'Metrics are whole-volume standardized_v1 with the dataset-specific scored labels; identity is the unwarped moving segmentation. HD95 is reported in mm on the prepared grid. For IXI, the benchmark assigns nominal 1 × 1 × 1 mm spacing. NPJ is the percentage of nonpositive interior Jacobians.','', '## Original recorded case metrics','', '| Method | Dice | HD95 | NPJ (%) | SDlogJ |','|---|---:|---:|---:|---:|']
    for method,s in originals.items():lines.append('| '+names.get(method,method)+' | '+' | '.join(f"{s[k]['mean']:.6f}" if s[k]['mean'] is not None else 'undefined' for k in ('Dice','HD95','NPJ','SDlogJ'))+' |')
    lines+=['','Any replay difference is visible by comparing these tables. This rendering/metric script reused existing maps with no fitting or recovery. ACDC FireANTs-SyN required one separately documented native replay before rendering because that map had not been retained. Orientation follows dataset-specific source geometry and image checks; artificial benchmark affines are not treated as anatomical. Left/right markers are intentionally omitted. Full score definitions,slice/crop coordinates and intensity window are in report.json.','']
    lines+=['','## Plot styling','','Intensity error panels are standalone black-to-bright-red maps, with a common **linear0–1 scale** across both views and all methods. Error=absolute original-image residual divided by the shared fixed-image display span, clipped at1. The colour bar is labelled **Intensity error**. Linear scaling keeps magnitude comparisons direct and does not amplify small background residuals as a log scale would.','','Each method has `intensity_error.png`, `label_error_overlay.png`, `deformation_grid.png`, `warped_labels_overlay.png`, and `warped_labels_grid.png`. Display resampling preserves each plane’s physical spacing aspect (notably ACDC’s through-plane spacing); this is2D display only. Both grids use thin0.7pt bright light-blue lines (#66e5ff),8voxel spacing. `label_error_overlay.png` paints every unequal warped/fixed label voxel opaque red over the fixed image, including missed foreground and spurious foreground; equal labels retain grayscale. Its legend is separate. All other source panels and legends remain separate.','']
    (args.output/'REPORT.md').write_text('\n'.join(lines))
    (args.artifacts/'REPORT.md').write_text('\n'.join(lines));(ROOT/'docs'/f'{dataset.upper()}_PAIR{pair_index:03d}_CENTROID.md').write_text('\n'.join(lines)+f'\nPNG location: `{args.output.relative_to(ROOT)}`.\nData/provenance: `{args.artifacts.relative_to(ROOT)}`.\nAutomation: `scripts/create_ixi_centroid_media.py --dataset {dataset} --run`.\n')
    print('\n'.join(lines))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',choices=list(VIEWS),default='ixi');p.add_argument('--output',type=Path);p.add_argument('--artifacts',type=Path);p.add_argument('--refresh',action='store_true',help='Regenerate figures from existing artifacts, retaining cached metrics');p.add_argument('--run',action='store_true');a=p.parse_args()
    if not a.run:print({'dataset':a.dataset,'views':VIEWS[a.dataset],'selection':'median original SIRENPath Dice, best case-specific classical/neural','mode':'existing maps only'});return
    run(a)


if __name__=='__main__':main()
