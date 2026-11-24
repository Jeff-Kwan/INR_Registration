"""Native ConvexAdam MIND-SSC, using its in-memory return API."""
import time

import numpy as np
import torch

from .common import execution, finish, preflight, sample_xyz_field
from .sources import native_modules, upstream_imports


def run_case(case, config, device):
    preflight(case, config, 'convexadam')
    started = time.monotonic()
    with upstream_imports('convexadam'), execution(device, config.get('seed', 42)) as dev:
        module, = native_modules('convexadam')
        # Native nibabel path uses XYZ array axes, and returns displacements in
        # those same axes, in voxels. No physical/normalized displacement swap.
        displacement = module.convex_adam_pt(
            img_fixed=case['fixed']['data'], img_moving=case['moving']['data'],
            device=dev, dtype=torch.float16 if dev.type == 'cuda' else torch.float32,
            **config['native'])
        expected = (*case['fixed']['data'].shape, 3)
        if displacement.shape != expected or not np.isfinite(displacement).all():
            raise ValueError(f'Invalid native displacement: expected {expected}')
        def predicted(points):
            return points + sample_xyz_field(displacement, points)
        checkpoint = {'kind': 'provenance_only', 'parameter_checkpoint': False,
                      'reason': 'Native result is a dense displacement field; map persistence is disabled'}
        return finish('convexadam', case, config, predicted, checkpoint, started,
                      variant='MIND-SSC', preprocessing='native, no extra scaling', optimization_roi='whole_image')
