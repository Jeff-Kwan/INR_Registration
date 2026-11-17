"""Canonical SIRENPath on the frozen development pairs, never official test pairs."""
import json
from pathlib import Path
import tempfile
import time

import nibabel as nib
import numpy as np
import torch

from medical.brain_data import sha256
from medical.comparison_metrics import evaluate_arrays, labels_for
from medical.configuration import validate_config
from medical.registration.training import fit_pair
from medical.registration.recovery import recover
from scripts.prepare_tuning_cohort import OUTPUT, ROOT, select, paths, validate_selection


def membership(case):
    manifest=json.loads((OUTPUT/'manifest.json').read_text())
    validate_selection(manifest['selection'])
    expected=[dict(r,**paths(r)) for group in select().values() for r in group]
    if manifest['pairs']!=expected:
        raise ValueError('Frozen development pairs changed')
    found=[p for p in manifest['pairs'] if all(Path(case[k]['path']).resolve()==(ROOT/p[k+'_image']).resolve() for k in ('moving','fixed'))]
    if len(found)!=1:raise ValueError('Input pair is not in the frozen tuning cohort')
    row=found[0]
    for key in ('moving_image','fixed_image','moving_label','fixed_label'):
        if sha256(ROOT/row[key])!=manifest['files'][row[key]]['sha256']:
            raise ValueError(f'Tuning input changed: {row[key]}')
    return row


def run_case(case,config,device,*,checkpoint_callback=None):
    validate_config(config)
    if config['runtime']['seed']!=42:raise ValueError('Tuning remains single-seed42')
    row=membership(case);dataset=row['dataset']
    started=time.perf_counter();arrays=[]
    for key in ('moving','fixed'):
        values=case[key]['data']
        if dataset=='oasis':
            lo,hi=float(values.min()),float(values.max())
            if hi<=lo:raise ValueError('Constant OASIS image')
            values=(values-lo)/(hi-lo)
        arrays.append(torch.from_numpy(values.transpose(2,1,0).copy()).to(device))
    affine=case['moving']['affine'].copy();affine[:3,:3]=affine[:3,:3][:,::-1]
    timings={'preprocessing_seconds':time.perf_counter()-started}
    started=time.perf_counter()
    pair=fit_pair(*arrays,config,affine=affine,report=lambda r:print(json.dumps({'training':r}),flush=True),checkpoint_callback=checkpoint_callback)
    timings['training_seconds']=time.perf_counter()-started;started=time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='tuning-recovery-') as scratch:
        recovered=recover(pair,arrays[0].shape,config['recovery'],scratch,
                          chunk=config['runtime']['inference_chunk'],compile_derivatives=config['runtime']['compile_derivatives'],max_axis=config['runtime']['recovery_max_axis'])
        if recovered.inverse.is_cuda:torch.cuda.synchronize(recovered.inverse.device)
        timings['recovery_seconds']=time.perf_counter()-started;started=time.perf_counter()
        labels=[np.asarray(nib.load(ROOT/row[k+'_label']).dataobj) for k in ('fixed','moving')]
        scores=evaluate_arrays(*labels,recovered.inverse.detach().permute(2,1,0,3).cpu().numpy(),labels_for(dataset),np.linalg.norm(case['fixed']['affine'][:3,:3],axis=0),dataset)
        timings['evaluation_seconds']=time.perf_counter()-started
        scores.update(split='development_tuning32',pair_index=row['pair_index'],seed=42,timings=timings,recovery=recovered.diagnostics,
                      intensity_normalization='independent_minmax_0_1' if dataset=='oasis' else 'preserve_protocol_prepared_values')
    scores['training_execution_timing']=getattr(pair,'training_metadata',{}).get('execution_timings')
    return dict(metrics=scores,checkpoint=dict(format_version=1,model=pair.model_config,training=getattr(pair,'training_metadata',{}),
        state={k:v.detach().cpu() for k,v in pair.state_dict().items()},config=config,tuning_pair=row,tuning_manifest_sha256=sha256(OUTPUT/'manifest.json')))
