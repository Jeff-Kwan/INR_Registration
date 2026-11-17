"""Recover selected original-default cases and save consistent multi-method figures.

The user explicitly authorized saved maps/warps for these selected cases.
"""
import argparse
from contextlib import ExitStack,redirect_stdout,redirect_stderr
import gc
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
METHODS=('ours','convexadam','fireants','demons','niftyreg','ccidir','nir','nodeo','dnvf')


def select_cases(report):
    selected=[]
    for dataset in report['datasets']:
        rows=[dict(row,average_advantage=(row['classical_advantage']+row['neural_advantage'])/2) for row in report['rows'] if row['dataset']==dataset]
        rows.sort(key=lambda row:(row['average_advantage'],row['pair_index']))
        for rank,index in [('worst',0),('median',(len(rows)-1)//2),('best',len(rows)-1)]:
            selected.append(dict(rows[index],rank=rank,selection_rank=index+1,cohort_size=len(rows)))
    return selected


def choose_slice(labels,axis=2):
    counts=[len(set(np.unique(np.take(labels,i,axis=axis)))-{0}) for i in range(labels.shape[axis])]
    maximum=max(counts)
    if maximum==0:raise ValueError('No foreground labels')
    index=min(range(len(counts)),key=lambda i:(-counts[i],abs(i-(len(counts)-1)/2),i))
    return index,counts


def display_scale(image):
    lo,hi=np.percentile(image,[1,99])
    return np.clip((image-lo)/max(float(hi-lo),1e-8),0,1)


def label_color(code):
    import colorsys
    return (*colorsys.hsv_to_rgb((float(code)*.618033988749895)%1,.75,.95),1.)


def save_legend(output,dataset,codes,metadata_output=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    names={}
    if dataset=='acdc':names={1:'Right ventricle',2:'Myocardium',3:'Left ventricle'}
    else:
        spec=json.loads((ROOT/'medical/protocols/brain_benchmarks.json').read_text())
        if dataset=='lpba40':
            names={int(code):name for name,values in spec['lpba40_sitreg']['labels'].items() for code in values}
        elif dataset=='ixi':
            names={int(v['freesurfer_id']):v['name'] for values in spec['ixi_transmorph']['score_groups'].values() for v in values}
        elif dataset=='neurite_oasis':
            for line in (ROOT/'Data/NeuriteOASIS/prepared/source_metadata/5384_seg35_labels.txt').read_text().splitlines():
                fields=line.split()
                if fields and fields[0].isdigit():names[int(fields[0])]=fields[1].replace('-', ' ')
    entries=[dict(code=int(code),name=names.get(int(code),f'Class {int(code)}'),rgba=list(label_color(code))) for code in codes]
    columns=1 if len(entries)<=6 else 2 if len(entries)<=36 else 3
    rows=int(np.ceil(len(entries)/columns));fig,ax=plt.subplots(figsize=(columns*4,max(1,rows*.30+.3)))
    ax.axis('off');handles=[Patch(facecolor=e['rgba'],label=f"{e['code']}: {e['name']}" if e['name']!=f"Class {e['code']}" else e['name']) for e in entries]
    ax.legend(handles=handles,loc='center',ncol=columns,frameon=False,fontsize=10)
    fig.savefig(Path(output)/'label_legend.png',dpi=200,bbox_inches='tight',pad_inches=.05);plt.close(fig)
    (Path(metadata_output if metadata_output is not None else output)/'label_colors.json').write_text(json.dumps(dict(background='class0 transparent; omitted from foreground legend',classes=entries),indent=2)+'\n')


def plot_case(output,case,fixed_labels,moving_labels,methods,selection,media_output=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    images=Path(media_output) if media_output is not None else output
    images.mkdir(parents=True,exist_ok=True)
    index,counts=choose_slice(fixed_labels)
    def plane(array):return np.rot90(array[:,:,index] if array.ndim==3 else array)
    def rgba(gray):
        values=plane(gray);return np.dstack([values,values,values,np.ones_like(values)])
    def overlay(gray,labels=None,error=None):
        image=rgba(gray)
        if labels is not None:
            codes=plane(labels)
            for code in np.unique(codes):
                if code==0:continue
                mask=codes==code;image[mask,:3]=.55*image[mask,:3]+.45*np.asarray(label_color(code)[:3])
        if error is not None:
            values=plane(error);alpha=np.clip(.8*values,0,.8)[...,None]
            image[...,:3]=(1-alpha)*image[...,:3]+alpha*plt.get_cmap('Reds')(values)[...,:3]
        return image
    def save(path,array):plt.imsave(path,np.clip(array,0,1))
    window_lo,window_hi=np.percentile(case['fixed']['data'],[1,99]);window_span=max(float(window_hi-window_lo),1e-8)
    fixed=np.clip((case['fixed']['data']-window_lo)/window_span,0,1);moving=np.clip((case['moving']['data']-window_lo)/window_span,0,1)
    save(images/'fixed_image.png',rgba(fixed));save(images/'moving_image.png',rgba(moving))
    save(images/'fixed_labels_overlay.png',overlay(fixed,labels=fixed_labels))
    save(images/'moving_labels_overlay.png',overlay(moving,labels=moving_labels))
    metadata=[]
    for method in methods:
        folder=output/method;image_folder=images/method;image_folder.mkdir(parents=True,exist_ok=True)
        with np.load(folder/'plot_slices.npz') as warp:
            warped_labels=warp['warped_labels'];warped_image=warp['warped_image']
        row=json.loads((folder/'recovery.json').read_text());metadata.append(dict(method=method,**row))
        label_error=((warped_labels!=fixed_labels[:,:,index])&((warped_labels!=0)|(fixed_labels[:,:,index]!=0))).astype(np.float32)
        lo,hi=np.percentile(case['fixed']['data'],[1,99]);span=max(float(hi-lo),1e-8)
        intensity_error=np.clip(np.abs(case['fixed']['data'][:,:,index]-warped_image)/span,0,1)
        warped=np.clip((warped_image-window_lo)/window_span,0,1)
        save(image_folder/'warped_image.png',rgba(warped))
        save(image_folder/'warped_labels_overlay.png',overlay(warped,labels=warped_labels))
        save(image_folder/'warped_label_error_overlay.png',overlay(fixed,error=label_error))
        save(image_folder/'warped_intensity_error_overlay.png',overlay(fixed,error=intensity_error))
        (image_folder/'comparison.png').unlink(missing_ok=True)
    for name in ('inputs.png','methods.png'):(images/name).unlink(missing_ok=True)
    (output/'slice.json').write_text(json.dumps(dict(axis=2,index=index,selection='maximum distinct nonzero fixed label classes; ties nearest centre then lowest index',class_counts=counts,recovery=metadata,
        files='separate PNG panels, native slice resolution, no titles, axes or margins; label legend in dataset parent'),indent=2)+'\n')


def save_prediction(prediction,output,case,fixed_labels,moving_labels,recorded,mode):
    import nibabel as nib
    from scipy.ndimage import map_coordinates
    shape=fixed_labels.shape
    if callable(prediction):
        mapping=np.empty((*shape,3),dtype=np.float32)
        for start in range(0,fixed_labels.size,65536):
            flat=np.arange(start,min(start+65536,fixed_labels.size))
            points=np.stack(np.unravel_index(flat,shape),axis=1).astype(np.float32)
            mapping.reshape(-1,3)[start:start+len(flat)]=prediction(points)
    else:
        import torch
        mapping=prediction.detach().cpu().numpy() if isinstance(prediction,torch.Tensor) else np.asarray(prediction)
        mapping=mapping.astype(np.float32)
    if mapping.shape!=(*shape,3) or not np.isfinite(mapping).all():raise ValueError('Invalid absolute XYZ map')
    warped_labels=map_coordinates(moving_labels,mapping.transpose(3,0,1,2),order=0,mode='constant',cval=0)
    warped=map_coordinates(case['moving']['data'],mapping.transpose(3,0,1,2),order=1,mode='constant',cval=0)
    dice=[]
    for code in recorded['labels']:
        a,b=fixed_labels==code,warped_labels==code;total=a.sum()+b.sum()
        if total:dice.append(float(2*np.count_nonzero(a&b)/total))
    measured=float(np.mean(dice));delta=measured-recorded['Dice']['mean']
    output.mkdir(exist_ok=False)
    for name,array in [('fixed_to_moving_voxel_map',mapping),('warped_image',warped.astype(np.float32)),('warped_labels',warped_labels.astype(np.uint16))]:
        nib.save(nib.Nifti1Image(array,case['fixed']['affine']),output/(name+'.nii.gz'))
    index,_=choose_slice(fixed_labels)
    np.savez_compressed(output/'plot_slices.npz',warped_image=warped[:,:,index],warped_labels=warped_labels[:,:,index])
    (output/'recovery.json').write_text(json.dumps(dict(mode=mode,recorded_Dice=recorded['Dice']['mean'],recovered_Dice=measured,delta_Dice=delta,
        matches_recorded_within_1e_4=abs(delta)<=1e-4,map='absolute fixed-to-moving XYZ native voxel coordinates; affine defines fixed sampling grid',image_warp='linear zero outside',label_warp='nearest zero outside'),indent=2)+'\n')


def recover_ours(checkpoint,case,consume):
    import torch
    from medical.models.siren import PairedSiren
    from medical.registration.recovery import recover
    config=checkpoint['config']
    if checkpoint.get('training',{}).get('spatial_units')!='voxel':raise ValueError('Unsupported checkpoint geometry')
    pair=PairedSiren(3,**checkpoint['model']).cuda();pair.load_state_dict(checkpoint['state']);pair.eval().requires_grad_(False)
    pair.training_metadata=checkpoint['training']
    with tempfile.TemporaryDirectory(prefix='case-media-recovery-') as scratch:
        result=recover(pair,case['fixed']['data'].shape[::-1],config['recovery'],scratch,chunk=config['runtime']['inference_chunk'],compile_derivatives=config['runtime']['compile_derivatives'],max_axis=config['runtime']['recovery_max_axis'])
        consume(result.inverse.permute(2,1,0,3),'frozen SIRENPath checkpoint recovery')


def recover_dnvf(checkpoint,case,consume):
    import torch
    import torch.nn.functional as F
    from baselines.sources import upstream_imports,native_modules
    from baselines.precision import ieee_fp32
    from baselines.common import execution
    config=checkpoint['config'];native=config['native'];shape=case['fixed']['data'].shape
    with upstream_imports('dnvf'),execution('cuda:0',config['seed'],cuda_required=True),ieee_fp32(),torch.no_grad():
        network,_,_=native_modules('dnvf')
        train_utils=importlib.import_module('utils.train_utils')
        small=np.asarray(shape)//native['downsample_factor']
        grid=train_utils.get_mgrid(small).permute(0,2,3,4,1)
        normalized=grid.clone().float()
        for axis in range(3):normalized[...,axis]=2*(grid[...,axis]/(small[axis]-1)-.5)
        model=network.Siren(in_features=3,hidden_features=native['hidden_features'],hidden_layers=native['hidden_layers'],out_features=3,outermost_linear=True).cuda()
        model.load_state_dict(checkpoint['state']);model.eval().requires_grad_(False)
        velocity=F.interpolate(model(normalized.cuda()).permute(0,4,1,2,3),size=shape,mode='trilinear',align_corners=True)
        for axis in range(3):velocity[:,axis]=velocity[:,axis]/2*(shape[axis]-1)
        flow=network.VecInt(shape).cuda()(velocity)[0].movedim(0,-1).cpu().numpy()
        consume(np.indices(shape,dtype=np.float32).transpose(1,2,3,0)+flow,'frozen native DNVF checkpoint; pre-last-update validation snapshot')


def recover_external(method,checkpoint,case,consume,device):
    from unittest.mock import patch
    adapter=importlib.import_module('baselines.'+method)
    with ExitStack() as stack:
        def finish(name,case,config,predicted,*args,**kwargs):
            consume(predicted,'frozen checkpoint inference' if method in ('nir','ccidir') else 'repeat original native registration; original parameters not retained')
            return {}
        stack.enter_context(patch.object(adapter,'finish',side_effect=finish))
        if method in ('nir','ccidir'):
            original=adapter.native_modules
            def modules(name):
                loaded=original(name)
                if method=='ccidir':
                    def fit(reg):
                        reg.network.load_state_dict(checkpoint['forward']);reg.network_rev.load_state_dict(checkpoint['reverse'])
                    stack.enter_context(patch.object(loaded[0].ImplicitRegistrator,'fit',fit))
                else:
                    def fit(model,*args,**kwargs):model.load_state_dict(checkpoint['state'])
                    stack.enter_context(patch.object(loaded[-1],'train_efficient_pairwise',side_effect=fit))
                return loaded
            stack.enter_context(patch.object(adapter,'native_modules',side_effect=modules))
        adapter.run_case(case,checkpoint['config'],device)


def run_case(selection,report,output,media_root):
    import nibabel as nib
    import torch
    from medical.case_runner import load_volume
    from medical.comparison_metrics import label_path
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    dataset=selection['dataset'];index=selection['pair_index'];directory=output/dataset/f"{selection['rank']}_pair{index:03d}";directory.mkdir(parents=True,exist_ok=True)
    case={key:load_volume(selection[key]) for key in ('fixed','moving')}
    fixed_labels,moving_labels=[np.asarray(nib.load(label_path(case[k],dataset)).dataobj) for k in ('fixed','moving')]
    for key,labels in [('fixed',fixed_labels),('moving',moving_labels)]:
        nib.save(nib.Nifti1Image(case[key]['data'],case[key]['affine']),directory/(key+'_image.nii.gz'))
        nib.save(nib.Nifti1Image(labels.astype(np.uint16),case[key]['affine']),directory/(key+'_labels.nii.gz'))
    for method in METHODS:
        if (directory/method/'recovery.json').exists():
            print(dataset,selection['rank'],method,'reused completed recovery',flush=True);continue
        group='ours' if method=='ours' else 'classical' if method in ('convexadam','fireants','demons','niftyreg') else 'neural'
        source=Path(report['roots'][group])/dataset/method/f'pair{index:03d}'
        checkpoint=torch.load(source/'checkpoint.pt',map_location='cpu',weights_only=False)
        recorded=json.loads((source/'metrics.json').read_text())
        def consume(prediction,mode):
            with tempfile.TemporaryDirectory(prefix='.'+method+'-',dir=directory) as scratch:
                artifacts=Path(scratch)/'artifacts'
                save_prediction(prediction,artifacts,case,fixed_labels,moving_labels,recorded,mode)
                artifacts.rename(directory/method)
        with (directory/(method+'.log')).open('w') as log,redirect_stdout(log),redirect_stderr(log):
            if method=='ours':recover_ours(checkpoint,case,consume)
            elif method=='dnvf':recover_dnvf(checkpoint,case,consume)
            else:recover_external(method,checkpoint,case,consume,'cpu' if method in ('demons','niftyreg') else 'cuda:0')
        del checkpoint;gc.collect();torch.cuda.empty_cache()
        print(dataset,selection['rank'],method,'done',flush=True)
    plot_case(directory,case,fixed_labels,moving_labels,METHODS,selection,media_output=Path(media_root)/dataset/directory.name)


def render_selected(selection,output,artifacts):
    import nibabel as nib
    from medical.case_runner import load_volume
    from medical.comparison_metrics import label_path
    directory=Path(artifacts)/selection['dataset']/f"{selection['rank']}_pair{selection['pair_index']:03d}"
    case={key:load_volume(selection[key]) for key in ('fixed','moving')}
    labels=[np.asarray(nib.load(label_path(case[k],selection['dataset'])).dataobj) for k in ('fixed','moving')]
    plot_case(directory,case,*labels,METHODS,selection,media_output=Path(output)/selection['dataset']/directory.name)
    return str(directory)


def prepare_legends(report,selected,output,artifacts):
    import nibabel as nib
    from medical.comparison_metrics import label_path
    for dataset in report['datasets']:
        codes=set()
        for selection in selected:
            if selection['dataset']!=dataset:continue
            for key in ('fixed','moving'):
                path=label_path(dict(path=selection[key]),dataset)
                codes.update(map(int,np.unique(np.asarray(nib.load(path).dataobj))))
        (Path(output)/dataset).mkdir(parents=True,exist_ok=True)
        (Path(artifacts)/dataset).mkdir(parents=True,exist_ok=True)
        codes.discard(0);save_legend(Path(output)/dataset,dataset,sorted(codes),metadata_output=Path(artifacts)/dataset)

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--advantage',type=Path,default=ROOT/'results/case_advantage_original_20261117/advantage.json')
    parser.add_argument('--output',type=Path,default=ROOT/'docs/media',help='PNG panels and legends only')
    parser.add_argument('--artifacts',type=Path,default=ROOT/'results/case_media_recovery_original_20261117',help='All volumes, recovery data, caches, metadata and logs')
    parser.add_argument('--gpus',type=int,nargs='+',default=list(range(8)))
    parser.add_argument('--render-only',action='store_true');parser.add_argument('--resume',action='store_true');parser.add_argument('--run',action='store_true');parser.add_argument('--task-index',type=int,help=argparse.SUPPRESS)
    args=parser.parse_args(argv);report=json.loads(args.advantage.read_text());selected=select_cases(report)
    if args.task_index is not None:
        if args.render_only:render_selected(selected[args.task_index],args.output,args.artifacts)
        else:run_case(selected[args.task_index],report,args.artifacts,args.output)
        return
    plan=dict(selection='mean of classical and neural signed relative advantages; sort ascending; lower median for even cohort sizes',slice='maximum distinct foreground fixed label classes on native array axis2; ties closest centre then lowest index',selected=selected,methods=METHODS,
              saved='fixed/moving images and labels, absolute fixed-to-moving voxel maps, warped images and labels; PNG figures',sources=report['roots'])
    if args.render_only:
        prepare_legends(report,selected,args.output,args.artifacts)
        import concurrent.futures,multiprocessing
        with concurrent.futures.ProcessPoolExecutor(4,mp_context=multiprocessing.get_context('spawn')) as pool:
            tasks=[pool.submit(render_selected,selection,args.output,args.artifacts) for selection in selected]
            for task in concurrent.futures.as_completed(tasks):print('Rendered',task.result(),flush=True)
        return
    if not args.run:print(json.dumps(plan,indent=2));return
    if args.artifacts.exists():
        if not args.resume:raise FileExistsError('Output exists; use --resume only for this same selection')
        if json.dumps(json.loads((args.artifacts/'selection.json').read_text()),sort_keys=True)!=json.dumps(plan,sort_keys=True):raise ValueError('Selection changed')
    else:
        args.artifacts.mkdir(parents=True);(args.artifacts/'selection.json').write_text(json.dumps(plan,indent=2)+'\n')
    prepare_legends(report,selected,args.output,args.artifacts)
    import concurrent.futures
    def worker(gpu,indices):
        results=[]
        for index in indices:
            cmd=[sys.executable,'-B',str(Path(__file__).resolve()),'--advantage',str(args.advantage.resolve()),'--output',str(args.output.resolve()),'--artifacts',str(args.artifacts.resolve()),'--task-index',str(index)]
            with (args.artifacts/f'task{index:02d}.log').open('w') as log:
                p=subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT)
            results.append(dict(index=index,gpu=gpu,returncode=p.returncode))
        return results
    results=[]
    with concurrent.futures.ThreadPoolExecutor(len(args.gpus)) as pool:
        futures=[pool.submit(worker,gpu,list(range(i,len(selected),len(args.gpus)))) for i,gpu in enumerate(args.gpus)]
        for future in concurrent.futures.as_completed(futures):
            results+=future.result();(args.artifacts/'progress.json').write_text(json.dumps(results,indent=2)+'\n')
    (args.artifacts/'summary.json').write_text(json.dumps(dict(tasks=results,successful=sum(r['returncode']==0 for r in results)),indent=2)+'\n')
    if any(r['returncode'] for r in results):raise SystemExit(1)


if __name__=='__main__':main()
