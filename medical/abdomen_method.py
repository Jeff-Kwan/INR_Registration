"""Our unchanged paired method connected to official AbdomenCTCT validation."""
import tempfile
import json
import time
import numpy as np
import torch
from medical.configuration import validate_config
from medical.registration.training import fit_pair
from medical.registration.recovery import recover
from medical.abdomen_evaluation import validate_case,evaluate
from medical.models.siren import endpoint_reconstruction

def run_case(case,config,device,*,protocol="abdomenctct",checkpoint_callback=None):
    validate_config(config)
    validate_case(case,protocol)
    preprocessing_started=time.perf_counter()
    working=None
    arrays=[]
    normalization={}
    if 'working_grid' not in config:
        for key in ('moving','fixed'):
            data=case[key]['data']; lo,hi=float(data.min()),float(data.max())
            if hi<=lo: raise ValueError('Cannot normalize constant input')
            arrays.append(torch.from_numpy(((data-lo)/(hi-lo)).transpose(2,1,0).copy()).to(device))
            normalization[key]={'min':lo,'max':hi}
        affine=case['moving']['affine'].copy(); affine[:3,:3]=affine[:3,:3][:,::-1]
    if 'working_grid' in config:
        from medical.working_grid import prepare
        arrays,affine,_,working=prepare(case,config['working_grid'],device)
    preprocessing_seconds=time.perf_counter()-preprocessing_started
    started=time.perf_counter()
    pair=fit_pair(*arrays,config,affine=affine,
                  report=lambda row: print(json.dumps({'training':row}),flush=True),
                  checkpoint_callback=checkpoint_callback)
    training_seconds=time.perf_counter()-started
    if working is not None:pair.working_grid_metadata=working
    started=time.perf_counter()
    reconstruction=endpoint_reconstruction(pair,arrays,config['runtime']['inference_chunk'])
    if working is not None:
        from medical.working_grid import original_reconstruction
        reconstruction={'working_grid':reconstruction,'original_grid':original_reconstruction(pair,case,working,config['runtime']['inference_chunk'])}
    reconstruction_seconds=time.perf_counter()-started
    started=time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=f'{protocol}-recovery-') as scratch:
        recovered=recover(pair,arrays[0].shape,config['recovery'],scratch,
            chunk=config['runtime']['inference_chunk'],compile_derivatives=config['runtime']['compile_derivatives'],
            max_axis=config['runtime'].get('recovery_max_axis'))
        if recovered.inverse.is_cuda:
            torch.cuda.synchronize(recovered.inverse.device)
        recovery_seconds=time.perf_counter()-started
        evaluation_started=time.perf_counter()
        if working is None:
            mapping=recovered.inverse.detach().permute(2,1,0,3)
        else:
            from medical.working_grid import predictor
            mapping=predictor(recovered.inverse,working,case['fixed']['affine'],case['moving']['affine'])
        metrics=evaluate(case,mapping,protocol)
        metrics.update(recovery=recovered.diagnostics,intensity_normalization={'method':'independent_minmax_0_1',**normalization},
                       optimization_roi='whole_image',seed=config['runtime']['seed'],reconstruction=reconstruction,
                       timings={'preprocessing_seconds':preprocessing_seconds,
                                'training_seconds':training_seconds,'reconstruction_seconds':reconstruction_seconds,
                                'recovery_seconds':recovery_seconds,
                                'evaluation_seconds':time.perf_counter()-evaluation_started,
                                'recovery_evaluation_seconds':time.perf_counter()-started})
    extra={}
    if working is not None:
        metrics['working_grid']=working
        metrics['working_grid_shape_xyz']=working['working_geometry']['shape']
        metrics['intensity_normalization']={'method':'working_grid_independent_minmax_0_1',**{k:v['normalization'] for k,v in working['endpoints'].items()}}
        extra={'working_grid':working}
    if config.get('spatial_units','voxel')=='mm':
        extra['physical_units']=pair.physical_units_metadata
        metrics['physical_units']=pair.physical_units_metadata
    return {'metrics':metrics,'checkpoint':{'format_version':1,'model':pair.model_config,'training':getattr(pair,'training_metadata',{}),
            'state':{k:v.detach().cpu() for k,v in pair.state_dict().items()},'config':config,**extra}}
