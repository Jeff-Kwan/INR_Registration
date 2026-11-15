"""Official ccIDIR training and Taylor cycle-consensus landmark inference."""
from tempfile import TemporaryDirectory
import time

import numpy as np
import torch
from medical.dirlab_evaluation import score_landmarks

from .common import predict_for_case, cpu_state, execution, finish, landmark_inputs, load_mask, preflight
from .sources import native_modules, upstream_imports


def run_case(case, config, device):
    preflight(case, config, 'ccidir')
    label_evaluation=config['evaluation'] in ('abdomenctct','oasis')
    fixed_mask, moving_mask = ((np.ones_like(case[key]['data']) if label_evaluation else load_mask(case[key])).transpose(2,1,0).copy() for key in ('fixed','moving'))
    started = time.monotonic()
    with upstream_imports('ccidir'), execution(device, config.get('seed', 42), cuda_required=True):
        models, general = native_modules('ccidir')
        fixed, moving = (torch.from_numpy(case[key]['data'].transpose(2, 1, 0).copy()) for key in ('fixed', 'moving'))
        with TemporaryDirectory(prefix='ccidir-') as scratch:
            reg = models.ImplicitRegistrator(moving, fixed, mask=fixed_mask, mask_2=moving_mask,
                                             save_folder=scratch, seed=config.get('seed', 42), **config['native'])
            reg.fit()
            reg.network.eval().requires_grad_(False)
            reg.network_rev.eval().requires_grad_(False)
            def predict(points, reverse=False):
                if label_evaluation:
                    coords=2*points[:,::-1].copy()/(np.asarray(fixed.shape)-1)-1
                    coordinates=torch.as_tensor(coords,dtype=torch.float32).cuda().requires_grad_(True)
                else:
                    coordinates=general.landmarks_to_coordmarks(points[:,::-1]+1,fixed.shape).cuda().requires_grad_(True)
                with torch.enable_grad():
                    forward,_,offset=general.compute_coordinates_cycle_autoretry(
                        reg.network_rev if reverse else reg.network, reg.network if reverse else reg.network_rev,
                        coordinates,inf_order=config["inference_order"])
                    consensus=forward+0.5*offset
                    if label_evaluation:
                        return ((consensus.detach().cpu().numpy()+1)*(np.asarray(fixed.shape)-1)/2)[:,::-1]
                    return general.coordmarks_to_landmarks(consensus,fixed.shape).detach().cpu().numpy()[:,::-1]-1
            predicted=predict_for_case(case,config,predict)
            extra={}
            if not label_evaluation:
                fixed_points,moving_points=landmark_inputs(case)
                reverse_predicted=predict(moving_points,reverse=True)
                reverse_scores=score_landmarks(moving_points,fixed_points,reverse_predicted,case['moving']['affine'])
                shape=np.asarray(case['fixed']['data'].shape)
                reverse_scores['prediction_outside_voxel_grid']=int(np.any((reverse_predicted<0)|(reverse_predicted>shape-1),axis=1).sum())
                extra['reverse_T50_to_T00']=reverse_scores
                extra['dirlab_robustness']={'threshold_mm':2.,'metric':'idir_rounded_tre_mm.mean_mm',
                    'forward_failed':bool(score_landmarks(fixed_points,moving_points,predicted,case['fixed']['affine'])['idir_rounded_tre_mm']['mean_mm']>2),
                    'reverse_failed':bool(reverse_scores['idir_rounded_tre_mm']['mean_mm']>2)}
            checkpoint = {'kind': 'network_weights', 'forward': cpu_state(reg.network), 'reverse': cpu_state(reg.network_rev)}
        return finish('ccidir', case, config, predicted, checkpoint, started,
                      variant=f'cycle-trained + order-{config["inference_order"]} cycle-consensus inference',
                      inference_order=config['inference_order'], **extra,
                      preprocessing='native raw intensities; dense normalized coordinates' if label_evaluation else 'native raw intensities, one-based DIR-Lab landmark convention',
                      optimization_roi='whole_image' if label_evaluation else 'separate fixed/moving lung masks',
                      cycle_voxel_size=config['native'].get('voxel_size', [1, 1, 1]))
