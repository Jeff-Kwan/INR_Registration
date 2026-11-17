"""Endpoint-anchored monotone temporal alignment, with identical crossfade control.

Only frozen network inference and post-hoc diagnostics; no optimizer steps.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def monotone_alignment(similarity,*,speed_penalty=.01,smooth_penalty=.01,band=.25,max_step=16):
    """Choose strictly increasing model indices for uniformly sampled cine frames.

    Matrix axes: model-time, acquired cine-time. Slopes are bounded by positive
    index steps1..max_step; with4x sampling this gives cine/model speed[.25,4].
    """
    scores=np.asarray(similarity,dtype=np.float64);n,m=scores.shape
    if n<2 or m<2 or not np.isfinite(scores).all():raise ValueError('Finite2D similarity matrix required')
    if min(speed_penalty,smooth_penalty,band)<0 or max_step<1:raise ValueError('Invalid constraints')
    expected=(n-1)/(m-1)
    state=np.full((m,n,max_step+1),-np.inf);parents=np.full(state.shape,-1,np.int16)
    state[0,0,0]=scores[0,0]
    for j in range(1,m):
        for i in range(1,n):
            if abs(i/(n-1)-j/(m-1))>band+1e-12 or (j==m-1 and i!=n-1):continue
            for step in range(1,min(max_step,i)+1):
                previous=state[j-1,i-step].copy()
                transition=np.zeros(max_step+1)
                transition[1:]=smooth_penalty*np.log(step/np.arange(1,max_step+1))**2
                if j>1:previous[0]=-np.inf
                values=previous-transition
                best=int(np.argmax(values))
                if not np.isfinite(values[best]):continue
                state[j,i,step]=values[best]+scores[i,j]-speed_penalty*np.log(step/expected)**2
                parents[j,i,step]=best
    step=int(np.argmax(state[-1,-1]));objective=state[-1,-1,step]
    if not np.isfinite(objective):raise ValueError('No feasible monotone alignment')
    indices=np.empty(m,dtype=np.int32);i=n-1
    for j in range(m-1,0,-1):
        indices[j]=i;previous=int(parents[j,i,step]);i-=step;step=previous
    indices[0]=i
    if indices[0]!=0 or indices[-1]!=n-1 or not (np.diff(indices)>0).all():raise RuntimeError('Invalid backtrace')
    return dict(indices=indices.tolist(),model_times=(indices/(n-1)).tolist(),cine_times=np.linspace(0,1,m).tolist(),
                regularized_objective_mean=float(objective/m))


def score_alignment(scores,alignment):
    n,m=scores.shape;j=np.arange(m);indices=np.asarray(alignment['indices']);linear=np.rint(j*(n-1)/(m-1)).astype(int)
    values={'unaligned':scores[linear,j],'aligned':scores[indices,j],'independent_best':scores.max(axis=0)}
    return {name:dict(mean=float(v.mean()),interior_mean=float(v[1:-1].mean()),per_frame=v.tolist()) for name,v in values.items()}


def matrix(predictions,reference,device):
    import torch
    from medical.sirenpath_reconstruction import _box_mean
    with torch.no_grad():
        truths=[]
        for a in reference:
            b=torch.from_numpy(a.transpose(2,1,0).copy()).to(device=device,dtype=torch.float64)
            mb=_box_mean(b,9);vb=(_box_mean(b*b,9)-mb*mb).clamp_min(0)
            truths.append((b,mb,vb,vb>1e-8))
        result=np.empty((len(predictions),len(truths)))
        for i,volume in enumerate(predictions):
            a=torch.from_numpy(volume.transpose(2,1,0).copy()).to(device=device,dtype=torch.float64)
            ma=_box_mean(a,9);va=(_box_mean(a*a,9)-ma*ma).clamp_min(0)
            for j,(b,mb,vb,valid) in enumerate(truths):
                corr=((_box_mean(a*b,9)-ma*mb)/(va*vb).sqrt().clamp_min(1e-8)).clamp(-1,1)
                if not valid.any():raise ValueError('No nonconstant reference patches')
                result[i,j]=float(corr[valid].mean())
    return result


def plot_pncc_curves(report,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    colors={'siren':'tab:blue','crossfade':'tab:orange'}
    names={'siren':'SIREN path','crossfade':'Reconstruction crossfade'}
    cine_times=np.linspace(0,1,report['cine_samples'])
    for methods,filename in [(['siren','crossfade'],'pncc_over_time.png'),(['siren'],'siren_pncc_over_time.png'),(['crossfade'],'crossfade_pncc_over_time.png')]:
        fig,ax=plt.subplots(figsize=(8,5),layout='constrained')
        for method in methods:
            scores=report['results'][method]['scores']
            for alignment,style in [('unaligned','--'),('aligned','-')]:
                ax.plot(cine_times,scores[alignment]['per_frame'],style,marker='o',color=colors[method],
                        label=f"{names[method]} {alignment} (mean {scores[alignment]['mean']:.5f})")
        ax.set(xlabel='Normalized acquired cine time (ED to ES)',ylabel='Whole-volume signed Patch NCC',xlim=(0,1))
        ax.grid(alpha=.25);ax.legend(fontsize=8);fig.savefig(output/filename,dpi=200);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained');ax.axhline(0,color='black',linestyle='--',linewidth=1)
    for method in colors:
        scores=report['results'][method]['scores']
        delta=np.asarray(scores['aligned']['per_frame'])-np.asarray(scores['unaligned']['per_frame'])
        ax.plot(cine_times,delta,'o-',color=colors[method],label=names[method])
    ax.set(xlabel='Normalized acquired cine time (ED to ES)',ylabel='Aligned minus unaligned Patch NCC',xlim=(0,1))
    ax.grid(alpha=.25);ax.legend();fig.savefig(output/'pncc_alignment_gain.png',dpi=200);plt.close(fig)


def run(args):
    import nibabel as nib
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from medical.models.siren import PairedSiren
    from scripts.prepare_acdc import preprocess
    source=args.source;metadata=json.loads((source/'report.json').read_text())
    checkpoint_path=Path(metadata['checkpoint'])
    if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()!=metadata['checkpoint_sha256']:raise ValueError('Checkpoint changed')
    cine_path=source/(metadata['patient']+'_4d.nii.gz')
    if hashlib.sha256(cine_path.read_bytes()).hexdigest()!=metadata['cine_sha256']:raise ValueError('Cine changed')
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    if checkpoint.get('training',{}).get('spatial_units')!='voxel':raise ValueError('Unsupported geometry')
    cine=nib.load(cine_path).get_fdata(dtype=np.float32);ed,es=metadata['ED'],metadata['ES']
    end=es if es>ed else es+cine.shape[3];frame_indices=[(i-1)%cine.shape[3] for i in range(ed,end+1)]
    spacing=nib.load(ROOT/'Data/ACDC/testing'/metadata['patient']/f'{metadata["patient"]}_frame{es:02d}.nii.gz').header.get_zooms()[:3]
    reference=[preprocess(cine[:,:,:,i],np.zeros(cine.shape[:3],dtype=np.float32),spacing)[0] for i in frame_indices]
    count=4*(len(reference)-1)+1;times=np.linspace(0,1,count)
    pair=PairedSiren(3,**checkpoint['model']).to(args.device);pair.load_state_dict(checkpoint['state']);pair.eval().requires_grad_(False)
    predictions=[]
    with torch.no_grad():
        for t in times:
            predictions.append(pair.decode(reference[0].shape[::-1],1-float(t) if metadata['model_time_reversed'] else float(t),chunk=65536).numpy().transpose(2,1,0))
    del pair
    crossfade=[(1-t)*predictions[0]+t*predictions[-1] for t in times]
    args.artifacts.mkdir(parents=True,exist_ok=False);args.output.mkdir(parents=True,exist_ok=False)
    report=dict(patient=metadata['patient'],pair_index=metadata['pair_index'],model_samples=count,model_dt=1/(count-1),cine_frames_one_based=[i+1 for i in frame_indices],
                cine_samples=len(reference),cine_normalized_dt=1/(len(reference)-1),selection='same overall lower-median Dice case as source trajectory',
                scope='post-hoc whole-volume signed9x9x9 Patch NCC, valid stride1, exclude constant truth patches; not independent motion validation',
                constraints=dict(endpoint_anchors=True,strict_order=True,speed_penalty=args.speed_penalty,smooth_penalty=args.smooth_penalty,max_index_step=16,local_cine_per_model_speed_bounds=[.25,4.],max_normalized_time_deviation=.25),
                source_report_sha256=hashlib.sha256((source/'report.json').read_bytes()).hexdigest(),results={})
    scores_by_method={}
    for method,volumes in [('siren',predictions),('crossfade',crossfade)]:
        scores=matrix(volumes,reference,args.device);alignment=monotone_alignment(scores,speed_penalty=args.speed_penalty,smooth_penalty=args.smooth_penalty)
        summary=score_alignment(scores,alignment);report['results'][method]=dict(alignment=alignment,scores=summary)
        scores_by_method[method]=scores;np.save(args.artifacts/(method+'_similarity.npy'),scores)
        print(method,json.dumps(summary),flush=True)
        directory=args.output/method;directory.mkdir()
        lo,hi=metadata['reconstruction_display_window'];z=metadata['slice_index']
        # Save paired panels for every observed cine time, with the same window
        # across all model frames and both trajectory methods.
        for j,(real,i) in enumerate(zip(reference,alignment['indices'])):
            target=directory/f'frame{frame_indices[j]+1:03d}';target.mkdir()
            unaligned_index=int(round(j*(count-1)/(len(reference)-1)))
            for name,volume,window in [('aligned',volumes[i],(lo,hi)),('unaligned',volumes[unaligned_index],(lo,hi)),('cine',real,(0.,1.))]:
                a,b=window;image=np.clip((volume[:,:,z]-a)/max(b-a,1e-8),0,1)
                plt.imsave(target/(name+'.png'),np.rot90(image),cmap='gray',vmin=0,vmax=1)
        fig,ax=plt.subplots(figsize=(7,5),layout='constrained')
        ax.imshow(scores.T,origin='lower',aspect='auto',extent=[0,1,0,1],cmap='viridis',vmin=.5,vmax=1)
        ax.plot([0,1],[0,1],'w--',label='Linear time');ax.plot(alignment['model_times'],alignment['cine_times'],'r.-',label='Constrained alignment')
        ax.set(xlabel='Model time',ylabel='Normalized acquired cine time',title=method+' Patch NCC and alignment');ax.legend()
        fig.savefig(args.output/(method+'_similarity.png'),dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,6),layout='constrained');ax.plot([0,1],[0,1],'k--',label='Linear alignment')
    for method,style in [('siren','o-'),('crossfade','s-')]:
        alignment=report['results'][method]['alignment'];ax.plot(alignment['model_times'],alignment['cine_times'],style,label='SIREN path' if method=='siren' else 'Reconstruction crossfade')
    ax.set(xlabel='Model time',ylabel='Normalized acquired cine time',xlim=(0,1),ylim=(0,1));ax.set_aspect('equal');ax.grid(alpha=.25);ax.legend()
    fig.savefig(args.output/'time_warp.png',dpi=200);plt.close(fig)
    plot_pncc_curves(report,args.output)
    (args.artifacts/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines=['# Monotone ACDC temporal alignment','','53 model times (delta1/52),14 acquired cine frames from ED1 to ES14. No temporally interpolated reference frames are used for scoring.',
           '', '| Method | Unaligned mean PNCC | Aligned mean PNCC | Change | Independent-frame ceiling |','|---|---:|---:|---:|---:|']
    for method,result in report['results'].items():
        s=result['scores'];lines.append(f"| {method} | {s['unaligned']['mean']:.6f} | {s['aligned']['mean']:.6f} | {s['aligned']['mean']-s['unaligned']['mean']:+.6f} | {s['independent_best']['mean']:.6f} |")
    lines+=['','Means use the same14 acquired frames including fixed endpoints; report.json also gives12-frame interior means, every selected time pair, all constraints and per-frame PNCC. The independent-frame ceiling ignores order and endpoint constraints. Alignment is selected and scored on the same data; gains are descriptive, not held-out evidence. No training or registration optimizer runs.','']
    (args.artifacts/'REPORT.md').write_text('\n'.join(lines))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=ROOT/'results/acdc_trajectory_original_20261117');p.add_argument('--output',type=Path,default=ROOT/'docs/media/acdc_trajectory/aligned');p.add_argument('--artifacts',type=Path,default=ROOT/'results/acdc_trajectory_alignment_20261117');p.add_argument('--device',default='cuda:0');p.add_argument('--speed-penalty',type=float,default=.01);p.add_argument('--smooth-penalty',type=float,default=.01);p.add_argument('--plots-only',action='store_true');p.add_argument('--run',action='store_true');a=p.parse_args()
    if a.plots_only:
        plot_pncc_curves(json.loads((a.artifacts/'report.json').read_text()),a.output);return
    if not a.run:print(dict(model_samples=53,cine_samples=14,model_dt=1/52,output=str(a.output),artifacts=str(a.artifacts)));return
    if a.output.exists() or a.artifacts.exists():raise FileExistsError('Use fresh destinations')
    run(a)


if __name__=='__main__':main()
