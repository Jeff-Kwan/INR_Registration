"""Native FireANTs SyN; no affine pre-registration or image writers."""
import time

import torch

from .common import predict_for_case, execution, finish, landmark_inputs, normalized_grid_predictions, preflight
from .sources import native_modules, upstream_imports


def run_case(case, config, device):
    preflight(case, config, 'fireants')
    started = time.monotonic()
    with upstream_imports('fireants'), execution(device, config.get('seed', 42)) as dev:
        image_module, syn_module = native_modules('fireants')
        from fireants.interpolator import fireants_interpolator
        fixed = image_module.BatchedImages([image_module.Image.load_file(case['fixed']['path'], device=dev)])
        moving = image_module.BatchedImages([image_module.Image.load_file(case['moving']['path'], device=dev)])
        reg = syn_module.SyNRegistration(fixed_images=fixed, moving_images=moving, **config['native'])
        reg.optimize()
        with torch.no_grad():
            # Native composition includes numerical inversion of the reverse warp.
            grid = reg.get_warped_coordinates(fixed, moving)
            field=grid[0].detach().cpu().float().numpy().transpose(2,1,0,3)
            from .common import sample_xyz_field
            import numpy as np
            predicted=predict_for_case(case,config,lambda points: (sample_xyz_field(field,points)+1)*(np.asarray(case['moving']['data'].shape)-1)/2)
        # For this dense method the optimized parameters ARE map artifacts.
        # Preserve compact provenance only; do not serialize those fields.
        checkpoint = {'kind': 'provenance_only', 'parameter_checkpoint': False,
                      'reason': 'Native parameters are dense deformation fields; map persistence is disabled'}
        return finish('fireants', case, config, predicted, checkpoint, started,
                      variant='SyN', fused_operations=fireants_interpolator.use_ffo,
                      preprocessing='native Image.load_file, no extra scaling or orientation', optimization_roi='whole_image')
