"""Native ccIDIR fitting and Taylor cycle-consensus inference."""
from tempfile import TemporaryDirectory
import time

import numpy as np
import torch

from .common import cpu_state, execution, finish, preflight
from .sources import native_modules, upstream_imports


def run_case(case, config, device):
    preflight(case, config, 'ccidir')
    fixed_mask, moving_mask = (
        np.ones_like(case[key]['data']).transpose(2, 1, 0).copy()
        for key in ('fixed', 'moving')
    )
    started = time.monotonic()
    with upstream_imports('ccidir'), execution(device, config.get('seed', 42), cuda_required=True):
        models, general = native_modules('ccidir')
        fixed, moving = (
            torch.from_numpy(case[key]['data'].transpose(2, 1, 0).copy())
            for key in ('fixed', 'moving')
        )
        with TemporaryDirectory(prefix='ccidir-') as scratch:
            reg = models.ImplicitRegistrator(moving, fixed, mask=fixed_mask, mask_2=moving_mask,
                                             save_folder=scratch, seed=config.get('seed', 42), **config['native'])
            reg.fit()
            reg.network.eval().requires_grad_(False)
            reg.network_rev.eval().requires_grad_(False)
            extent = np.asarray(fixed.shape) - 1

            def predict(points):
                coords = 2 * points[:, ::-1].copy() / extent - 1
                coordinates = torch.as_tensor(coords, dtype=torch.float32).cuda().requires_grad_(True)
                with torch.enable_grad():
                    forward, _, offset = general.compute_coordinates_cycle_autoretry(
                        reg.network, reg.network_rev, coordinates, inf_order=config['inference_order'])
                    consensus = forward + 0.5 * offset
                return ((consensus.detach().cpu().numpy() + 1) * extent / 2)[:, ::-1]

            checkpoint = {'kind': 'network_weights', 'forward': cpu_state(reg.network),
                          'reverse': cpu_state(reg.network_rev)}
        return finish('ccidir', case, config, predict, checkpoint, started,
                      variant=f'cycle-trained + order-{config["inference_order"]} cycle-consensus inference',
                      inference_order=config['inference_order'],
                      preprocessing='native raw intensities; dense normalized coordinates',
                      optimization_roi='whole_image',
                      cycle_voxel_size=config['native'].get('voxel_size', [1, 1, 1]))
