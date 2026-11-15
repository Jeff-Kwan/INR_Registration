"""Single-case adapter for paired-SIREN fitting, PCG recovery and evaluation."""
import tempfile
import time
import json
from pathlib import Path
import numpy as np
import nibabel as nib
import torch
from medical.registration.cuda_kernels import supported
from medical.configuration import validate_config
from medical.registration.training import fit_pair
from medical.registration.recovery import recover
from medical.registration.geometry import sample
from medical.dirlab_evaluation import read_points, score_landmarks


def run_case(case, config, device, *, checkpoint_callback=None):
    validate_config(config)
    fixed, moving = case['fixed'], case['moving']
    preprocessing_started=time.perf_counter()
    working=None
    arrays=[]; intensity=[]
    if 'working_grid' not in config:
        for item in (moving,fixed):
            data=item['data']; lo,hi=float(data.min()),float(data.max())
            if hi<=lo: raise ValueError('Cannot normalize a constant volume')
            arrays.append(torch.from_numpy(((data-lo)/(hi-lo)).transpose(2,1,0).copy()).to(device))
            intensity.append({'min':lo,'max':hi})
        # Torch arrays are z,y,x; the registration code expects affine columns in array-axis order.
        affine=moving['affine'].copy(); affine[:3,:3]=affine[:3,:3][:,::-1]
    masks = []
    for volume in (moving, fixed):
        image_path = Path(volume['path'])
        stem = image_path.name.removesuffix('.gz').removesuffix('.nii')
        mask_image = nib.load(str(image_path.with_name(stem + '_lung_mask.nii.gz')))
        mask_data = np.asarray(mask_image.dataobj)
        if mask_data.shape != volume['data'].shape or not np.allclose(mask_image.affine, volume['affine']):
            raise ValueError('Lung mask does not match its image grid')
        if not np.isfinite(mask_data).all() or not np.isin(mask_data, [0, 1]).all() or not mask_data.any():
            raise ValueError('Expected a nonempty binary lung mask')
        masks.append(mask_data if 'working_grid' in config else
                     torch.from_numpy(mask_data.transpose(2, 1, 0).copy()).to(device).bool())
    if 'working_grid' in config:
        from medical.working_grid import prepare
        raw_masks=masks
        arrays,affine,masks,working=prepare(case,config['working_grid'],device,raw_masks)
    preprocessing_seconds=time.perf_counter()-preprocessing_started
    started=time.perf_counter()
    pair=fit_pair(*arrays,config,affine=affine,moving_mask=masks[0],fixed_mask=masks[1],
                  report=lambda row: print(json.dumps({'training':row}),flush=True),
                  checkpoint_callback=checkpoint_callback)
    training_seconds=time.perf_counter()-started
    if working is not None:
        from medical.models.siren import endpoint_reconstruction
        from medical.working_grid import original_reconstruction
        pair.working_grid_metadata=working
        started=time.perf_counter()
        reconstruction={'working_grid':endpoint_reconstruction(pair,arrays,config['runtime']['inference_chunk']),
                        'original_grid':original_reconstruction(pair,case,working,config['runtime']['inference_chunk'])}
        reconstruction_seconds=time.perf_counter()-started
    started=time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='paired-recovery-') as scratch:
        result=recover(pair,arrays[0].shape,config['recovery'],scratch,
                       chunk=config['runtime']['inference_chunk'],
                       compile_derivatives=config['runtime']['compile_derivatives'],
                       max_axis=config['runtime'].get('recovery_max_axis'))
        if result.inverse.is_cuda:
            torch.cuda.synchronize(result.inverse.device)
        recovery_seconds=time.perf_counter()-started
        evaluation_started=time.perf_counter()
        case_dir=Path(fixed['path']).parent
        fixed_points=read_points(case_dir/'T00_landmarks_voxel.csv')
        moving_points=read_points(case_dir/'T50_landmarks_voxel.csv')
        query=torch.as_tensor(fixed_points,dtype=result.inverse.dtype,device=device).reshape(1,1,-1,3)
        if working is None:
            predicted=sample(result.inverse,query).reshape(-1,3)
        else:
            from medical.working_grid import predictor
            predict=predictor(result.inverse,working,fixed['affine'],moving['affine'])
            predicted=predict(fixed_points)
        scores=score_landmarks(fixed_points,moving_points,predicted,fixed['affine'], grid_shape=moving['data'].shape)
        if working is None and supported(result.inverse.device) and result.inverse.dtype == torch.float32:
            from medical.registration.dirlab_cuda import folding_fractions
            whole, lung = folding_fractions(result.inverse, masks[1])
        else:
            from medical.registration.metrics import jacobian_determinant
            if working is None:
                displacement=result.inverse-torch.stack(torch.meshgrid(*(torch.arange(n,device=device,dtype=torch.float32) for n in arrays[0].shape),indexing='ij')[::-1],-1)
                jac=jacobian_determinant(displacement.cpu().numpy())
                evaluation_mask=masks[1].cpu().numpy()
            else:
                from medical.abdomen_evaluation import dense_mapping
                mapping=dense_mapping(predict,fixed['data'].shape)
                identity=np.stack(np.meshgrid(*(np.arange(n) for n in fixed['data'].shape),indexing='ij'),-1)
                jac=jacobian_determinant((mapping-identity).transpose(2,1,0,3))
                evaluation_mask=raw_masks[1].transpose(2,1,0).astype(bool)
            whole, lung = float(np.mean(jac<=0)), float(np.mean(jac[evaluation_mask]<=0))
        scores['folding_fraction_whole_grid'] = whole
        scores['folding_fraction_lung'] = lung
        scores['folding_fraction']=scores['folding_fraction_lung']
        reverse_query=torch.as_tensor(moving_points,dtype=result.forward.dtype,device=device).reshape(1,1,-1,3)
        reverse_predicted=(sample(result.forward,reverse_query).reshape(-1,3) if working is None else
                           predictor(result.forward,working,moving['affine'],fixed['affine'])(moving_points))
        reverse_scores=score_landmarks(moving_points,fixed_points,reverse_predicted,moving['affine'], grid_shape=fixed['data'].shape)
        scores['reverse_T50_to_T00']=reverse_scores
        scores['primary_direction']='T00_to_T50'
        scores['optimization_roi']='shared crop; R231 masks select crop centers only'
        scores['seed']=config['runtime']['seed']
        scores['recovery']=result.diagnostics
        if working is None:
            scores['intensity_normalization']={'method':'independent_minmax','moving':intensity[0],'fixed':intensity[1]}
    extra={}
    scores['timings'] = {
        'preprocessing_seconds': preprocessing_seconds,
        'training_seconds': training_seconds,
        'recovery_seconds': recovery_seconds,
        'evaluation_seconds': time.perf_counter() - evaluation_started,
        'recovery_evaluation_seconds': time.perf_counter() - started}
    if working is not None:
        scores['working_grid']=working
        scores['working_grid_shape_xyz']=working['working_geometry']['shape']
        scores['reconstruction']=reconstruction
        scores['timings']={'preprocessing_seconds':preprocessing_seconds,'training_seconds':training_seconds,
                          'reconstruction_seconds':reconstruction_seconds,'recovery_seconds':recovery_seconds,
                          'evaluation_seconds':time.perf_counter()-evaluation_started,'recovery_evaluation_seconds':time.perf_counter()-started}
        scores['intensity_normalization']={'method':'working_grid_independent_minmax_0_1',**{k:v['normalization'] for k,v in working['endpoints'].items()}}
        extra={'working_grid':working}
    if config.get('spatial_units','voxel')=='mm':
        extra['physical_units']=pair.physical_units_metadata
        scores['physical_units']=pair.physical_units_metadata
    return {'metrics':scores,'checkpoint':{'format_version':1,'model':pair.model_config,'training':getattr(pair,'training_metadata',{}),
            'state':{k:v.detach().cpu() for k,v in pair.state_dict().items()},'config':config,**extra}}
